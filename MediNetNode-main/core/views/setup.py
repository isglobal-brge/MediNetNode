from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render, redirect
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from core.models import SystemConfiguration
from users.models import Role
from users.decorators import require_role
import json
import os
import re
import logging

User = get_user_model()
logger = logging.getLogger(__name__)


class RootRedirectView(View):
    """
    Root redirect view that intelligently routes users.

    If no users exist: redirect to setup
    If users exist: redirect to login
    """

    def get(self, request):
        # Check if any users exist
        if not User.objects.exists() and not SystemConfiguration.is_setup_completed():
            # No users, redirect to setup
            return redirect('initial-setup-page')
        else:
            # Users exist, redirect to login
            return redirect('login')


class InitialSetupPageView(View):
    """
    Django template view for initial setup page.

    Renders the HTML form for initial setup or redirects to login
    if setup is already completed.
    """

    def get(self, request):
        # Check if setup is already completed
        if User.objects.exists() or SystemConfiguration.is_setup_completed():
            # Setup already completed, redirect to login
            return redirect('login')

        # Get suggested port from server
        suggested_port = request.META.get('SERVER_PORT', '8000')

        return render(request, 'setup/initial_setup.html', {
            'suggested_port': suggested_port
        })


@method_decorator(csrf_protect, name='dispatch')
class InitialSetupView(APIView):
    """
    Initial configuration view - ONLY accessible if no users exist.

    Security Features:
    - CSRF protection via decorator
    - No authentication required (only in initial state)
    - Database-level locking to prevent race conditions
    - Automatic blocking if any user exists
    - Automatic blocking if SystemConfiguration exists
    - Atomic transaction for all-or-nothing
    - Strict data validation with Django validators
    - Automatic ADMIN role assignment

    Initial setup performs:
    1. Create first superuser with ADMIN role
    2. Create global system configuration (SystemConfiguration)
    """

    permission_classes = []
    authentication_classes = []

    @staticmethod
    def check_setup_allowed():
        """
        CRITICAL SECURITY CHECK - Triple verification

        Verifies that no users OR system configuration exist.
        Returns tuple: (allowed: bool, error_response: dict|None)
        """
        if User.objects.exists():
            return False, {
                "error": "Setup already completed. Access denied.",
                "code": "SETUP_ALREADY_COMPLETED",
                "reason": "Users exist in the system"
            }

        if SystemConfiguration.is_setup_completed():
            return False, {
                "error": "Setup already completed. Access denied.",
                "code": "SETUP_ALREADY_COMPLETED",
                "reason": "System configuration exists"
            }

        return True, None

    def get(self, request):
        """
        Get information for initial setup.

        Returns:
            - suggested_port: Current server port
            - setup_required: Always True
            - tips: Tips to complete the setup
        """
        # Security check
        allowed, error_data = self.check_setup_allowed()
        if not allowed:
            return Response(error_data, status=status.HTTP_403_FORBIDDEN)

        current_port = request.META.get('SERVER_PORT', '8000')

        return Response({
            "message": "Welcome to MediNet! Let's set up your medical center.",
            "suggested_port": current_port,
            "setup_required": True,
            "tips": {
                "center_id": "Use a unique identifier (lowercase, no spaces). E.g.: 'central', 'norte', 'pediatrico'",
                "network": "This ID will identify your center when connecting to other MediNet nodes",
                "password": "Minimum 8 characters for security",
                "role": "The first user will automatically receive ADMIN role with full permissions"
            }
        })

    @staticmethod
    def validate_center_id(center_id):
        """
        Validate center identifier format.

        Rules:
        - Only lowercase, numbers, hyphens
        - No spaces or special characters
        - 3-20 characters
        - Cannot start or end with hyphen

        Args:
            center_id (str): Identifier to validate

        Returns:
            tuple: (is_valid: bool, error_message: str|None)
        """
        if not center_id:
            return False, "Center ID is required"

        if not re.match(r'^[a-z0-9-]{3,20}$', center_id):
            return False, "Center ID must be 3-20 characters: lowercase letters, numbers, hyphens only"

        if center_id.startswith('-') or center_id.endswith('-'):
            return False, "Center ID cannot start or end with hyphen"

        return True, None

    @staticmethod
    def get_or_create_admin_role():
        """
        Get or create ADMIN role with all permissions.

        Returns:
            Role: ADMIN role instance
        """
        admin_permissions = {
            'api.access': True,
            'user.create': True,
            'user.view': True,
            'user.edit': True,
            'user.delete': True,
            'dataset.view': True,
            'dataset.create': True,
            'dataset.edit': True,
            'dataset.delete': True,
            'dataset.train': True,
            'audit.view': True,
            'training.view': True,
            'training.manage': True,
            'system.admin': True,
        }

        admin_role, created = Role.objects.get_or_create(
            name='ADMIN',
            defaults={'permissions': admin_permissions}
        )

        # If role already existed, update permissions just in case
        if not created:
            admin_role.permissions = admin_permissions
            admin_role.save()

        return admin_role

    @transaction.atomic
    def post(self, request):
        """
        Complete initial medical center setup.

        Process (atomic transaction with database-level locking):
        1. Acquire database lock to prevent race conditions
        2. Security check (with lock held)
        3. Validate form data with Django validators
        4. Validate email format
        5. Get/create ADMIN role
        6. Create superuser with ADMIN role
        7. Create system configuration (SystemConfiguration)
        8. Return confirmation

        The atomic transaction with select_for_update ensures everything completes or nothing does,
        and prevents race conditions where multiple simultaneous requests could create multiple admins.
        """
        # 0. CRITICAL: ACQUIRE DATABASE LOCK TO PREVENT RACE CONDITION
        # This prevents multiple simultaneous setup requests from creating multiple admins
        with transaction.atomic():
            # Use select_for_update on both tables to acquire row-level locks
            # This blocks concurrent transactions until this one commits or rolls back
            user_exists = User.objects.select_for_update().exists()
            config_exists = SystemConfiguration.objects.select_for_update().exists()

            # Security check with lock held
            if user_exists or config_exists:
                logger.warning(
                    f"Setup attempt blocked - Users exist: {user_exists}, "
                    f"Config exists: {config_exists}, IP: {request.META.get('REMOTE_ADDR')}"
                )
                return Response({
                    "error": "Setup already completed. Access denied.",
                    "code": "SETUP_ALREADY_COMPLETED",
                    "reason": "System is already configured"
                }, status=status.HTTP_403_FORBIDDEN)

        # 1. DATA EXTRACTION AND CLEANING
        username = request.data.get('username', '').strip()
        password = request.data.get('password')
        email = request.data.get('email', '').strip()
        center_id = request.data.get('center_id', '').lower().strip()
        center_display_name = request.data.get('center_display_name', '').strip()

        # 2. BASIC VALIDATIONS
        if not all([username, password, email, center_id, center_display_name]):
            return Response(
                {
                    "error": "All fields are required",
                    "code": "MISSING_FIELDS",
                    "missing": [
                        field for field, value in {
                            'username': username,
                            'password': password,
                            'email': email,
                            'center_id': center_id,
                            'center_display_name': center_display_name
                        }.items() if not value
                    ]
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate email format
        try:
            validate_email(email)
        except ValidationError:
            logger.warning(f"Invalid email format in setup: {email}")
            return Response(
                {
                    "error": "Invalid email format",
                    "code": "INVALID_EMAIL"
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate password with Django's password validators
        try:
            # Create temporary user instance for password validation
            temp_user = User(username=username, email=email)
            validate_password(password, user=temp_user)
        except ValidationError as e:
            logger.warning(f"Password validation failed for user {username}: {e.messages}")
            return Response(
                {
                    "error": "Password does not meet security requirements",
                    "code": "WEAK_PASSWORD",
                    "details": e.messages
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate center_id format
        is_valid, error_msg = InitialSetupView.validate_center_id(center_id)
        if not is_valid:
            return Response(
                {
                    "error": error_msg,
                    "code": "INVALID_CENTER_ID"
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3. SYSTEM SETUP
        try:
            # 3.1. Get/create ADMIN role
            admin_role = InitialSetupView.get_or_create_admin_role()

            # 3.2. Create superuser with ADMIN role
            user = User.objects.create_superuser(
                username=username,
                email=email,
                password=password
            )
            # Assign ADMIN role
            user.role = admin_role
            user.save(update_fields=['role'])

            # 3.3. Create system configuration
            system_config = SystemConfiguration.objects.create(
                center_id=center_id,
                center_display_name=center_display_name,
                center_email=email,
                setup_completed_by=user,
                extra_settings={}  # Scalable for future configurations
            )

            # 4. SUCCESS RESPONSE
            return Response({
                "message": "Medical center setup completed successfully!",
                "center": {
                    "id": center_id,
                    "name": center_display_name,
                    "email": email
                },
                "admin_user": {
                    "username": username,
                    "email": email,
                    "role": admin_role.name
                },
                "next_step": "Please login with your credentials at /auth/login"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Transaction rolls back automatically
            # Log full error details server-side for debugging
            logger.error(
                f"Setup failed for user {username}, center {center_id}: {str(e)}",
                exc_info=True
            )
            # Return sanitized error message to client (no internal details)
            return Response(
                {
                    "error": "Setup failed due to an internal error. Please try again or contact support.",
                    "code": "SETUP_FAILED"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@require_role('ADMIN')
def system_settings_view(request):
    """
    System settings view - Display current system configuration.

    Shows:
    - Center information (ID, name, email)
    - API configuration (URL, port)
    - Setup metadata (when, by whom)
    - Extra settings (JSON)

    Only accessible by ADMIN users.
    """
    system_config = SystemConfiguration.get_instance()

    if not system_config:
        # This should not happen if setup was completed, but handle it
        return render(request, 'core/system_settings.html', {
            'error': 'System configuration not found. Please complete initial setup.',
            'setup_completed': False
        })

    context = {
        'system_config': system_config,
        'setup_completed': True,
        'api_access_config': system_config.get_api_access_config(),
    }

    return render(request, 'core/system_settings.html', context)
