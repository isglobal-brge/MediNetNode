from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import get_user_model
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.db.models import Q, Count, Sum
from django.utils import timezone
from datetime import timedelta
from django.db import transaction
import csv

from .models import Role, PasswordHistory
from .forms import (
    SecureUserCreationForm,
    UserUpdateForm,
    UserPasswordChangeForm,
    UserSearchForm
)
from .decorators import admin_required, require_permission, require_role
from django.contrib.auth.decorators import login_required
from audit.models import AuditLog
from core.models import SystemConfiguration

User = get_user_model()
@login_required
def researcher_info(request):
    """Informational page for non-admin roles explaining limited access."""
    if request.user.role and request.user.role.name == 'RESEARCHER':
        return render(request, 'users/researcher_info.html', {})
    return redirect('admin_dashboard')


@admin_required
def admin_dashboard(request):
    """Dashboard principal de administración con estadísticas."""
    
    # Estadísticas básicas
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    
    # Usuarios por rol
    users_by_role = User.objects.values('role__name', 'role_id').annotate(
        count=Count('id')
    ).order_by('role__name')
    
    # Actividad reciente (últimas 24 horas)
    yesterday = timezone.now() - timedelta(days=1)
    recent_logins = User.objects.filter(
        last_login__gte=yesterday
    ).count()
    
    # Logs recientes
    recent_logs = AuditLog.objects.select_related('user').order_by('-timestamp')[:10]
    
    # Usuarios con sesión activa
    active_sessions = User.objects.filter(is_active_session=True).count()
    
    # Cuentas bloqueadas
    locked_accounts = User.objects.filter(
        account_locked_until__gt=timezone.now()
    ).count()
    
    # User activity data for the last 30 days - REAL DATA ONLY
    activity_data = []
    today = timezone.now().date()
    
    # Get real activity data - no fake data generation
    for i in range(29, -1, -1):
        date = today - timedelta(days=i)
        
        # Count ONLY real logins from audit logs
        daily_activity = 0
        try:
            # Get actual login activity from audit logs
            daily_activity = AuditLog.objects.filter(
                timestamp__date=date,
                action__in=['LOGIN_SUCCESS']  # Only successful logins
            ).values('user_id').distinct().count()
            
            # If no audit activity, check last_login field (fallback only)
            if daily_activity == 0:
                daily_activity = User.objects.filter(
                    last_login__date=date
                ).count()
                
        except Exception:
            # No fake data - if there's an exception, show 0
            daily_activity = 0
        
        # Count real registrations only
        daily_registrations = User.objects.filter(
            date_joined__date=date
        ).count()
        
        activity_data.append({
            'date': date.strftime('%b %d'),
            'logins': daily_activity,  # Real data only - 0 if no activity
            'registrations': daily_registrations
        })
    
    # Dataset metrics (import here to avoid circular imports)
    try:
        from dataset.models import Dataset
        
        # Total datasets
        total_datasets = Dataset.objects.using('datasets_db').filter(is_active=True).count()
        
        # Total size
        total_size_bytes = Dataset.objects.using('datasets_db').filter(is_active=True).aggregate(
            total_size=Sum('file_size')
        )['total_size'] or 0
        
        # Format file size
        def format_file_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB", "TB"]
            import math
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_names[i]}"
        
        total_size_formatted = format_file_size(total_size_bytes)
        
    except ImportError:
        # Dataset app not available
        total_datasets = 0
        total_size_formatted = "0 B"
    
    context = {
        'total_users': total_users,
        'active_users': active_users,
        'inactive_users': total_users - active_users,
        'users_by_role': users_by_role,
        'recent_logins': recent_logins,
        'recent_logs': recent_logs,
        'active_sessions': active_sessions,
        'locked_accounts': locked_accounts,
        'total_datasets': total_datasets,
        'total_size_formatted': total_size_formatted,
        'activity_data': activity_data,
    }
    
    return render(request, 'users/admin_dashboard.html', context)


@require_permission('user.view')
def user_list(request):
    """User list with search and filters."""
    
    form = UserSearchForm(request.GET)
    users = User.objects.select_related('role', 'created_by').all()
    
    # Apply filters
    if form.is_valid():
        # Search query
        q = form.cleaned_data.get('q')
        if q:
            users = users.filter(
                Q(username__icontains=q) |
                Q(email__icontains=q) |
                Q(first_name__icontains=q) |
                Q(last_name__icontains=q)
            )
        
        # Role filter
        role = form.cleaned_data.get('role')
        if role:
            users = users.filter(role=role)
        
        # Active filter
        is_active = form.cleaned_data.get('is_active')
        if is_active == 'True':
            users = users.filter(is_active=True)
        elif is_active == 'False':
            users = users.filter(is_active=False)
        
        # Ordering
        ordering = form.cleaned_data.get('ordering')
        if ordering:
            users = users.order_by(ordering)
    
    # Pagination
    paginator = Paginator(users, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'form': form,
        'page_obj': page_obj,
        'total_users': users.count(),
    }
    
    return render(request, 'users/user_list.html', context)


@require_permission('user.create')
@csrf_protect
def create_user(request):
    """Create a new user."""
    
    if request.method == 'POST':
        form = SecureUserCreationForm(request.POST, created_by=request.user)
        
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save()
                    
                    # Auto-generate API key for RESEARCHER users
                    api_key = None
                    if user.role and user.role.name == 'RESEARCHER':
                        from .models import APIKey
                        api_key = APIKey.objects.create(
                            user=user,
                            name=f"Auto-generated API Key",
                            ip_whitelist=['0.0.0.0/0'],  # Allow all IPs by default - can be configured later
                            expires_at=None  # No expiration for now
                        )
                    
                    # Crear log de auditoría
                    AuditLog.objects.create(
                        user=request.user,
                        action='USER_CREATE',
                        resource=f'user:{user.username}',
                        ip_address=request.META.get('REMOTE_ADDR'),
                        success=True,
                        details={
                            'created_user_id': user.id,
                            'created_username': user.username,
                            'assigned_role': user.role.name if user.role else None,
                            'api_key_generated': bool(api_key),
                        }
                    )
                    
                    # Store user data in session for success modal (one-time display)
                    if api_key:
                        request.session['new_user_data'] = {
                            'username': user.username,
                            'password': form.cleaned_data['password1'],
                            'email': user.email,
                            'first_name': user.first_name,
                            'last_name': user.last_name,
                            'role': user.role.name,
                            'api_key': api_key.key,
                            'api_key_created': api_key.created_at.isoformat(),
                            'user_id': user.id
                        }
                    
                        print(request.session['new_user_data'])
                        return redirect('user_created_success')
                    
                    messages.success(
                        request,
                        f'User {user.username} created successfully with role {user.role.name if user.role else "No Role"}.'
                    )
                    return redirect('user_detail', user_id=user.id)
                    
            except Exception as e:
                messages.error(request, f'Error creating user: {str(e)}')
    else:
        form = SecureUserCreationForm()
    
    return render(request, 'users/create_user.html', {
        'form': form,
        'roles': Role.objects.all()
    })


@require_permission('user.create')
@csrf_protect
def user_created_success(request):
    """Display user creation success modal with API key (one-time view)."""
    
    user_data = request.session['new_user_data']
    if not user_data:
        messages.error(request, 'No user creation data found.')
        return redirect('user_list')
    
    context = {
        'user_data': user_data
    }
    
    # Clear session data after first access (one-time view)
    #del request.session['new_user_data']
    
    return render(request, 'users/user_created_success.html', context)


@require_role('ADMIN')
@csrf_protect
def download_user_info(request):
    """Download user credentials and API key information."""
    print(request.session.keys())
    user_data = request.session.get('new_user_data')
    print("User data:", user_data)
    if not user_data:
        return HttpResponseForbidden("No user data available for download")

    # Get system configuration for center info
    system_config = SystemConfiguration.get_instance()
    if system_config:
        api_access_config = system_config.get_api_access_config()
        base_url = api_access_config.get('url', 'http://localhost:8000')
        center_name = api_access_config['name']
        center_display = api_access_config['display_name']
    else:
        # Fallback if no system configuration exists
        base_url = "http://localhost:8000"
        center_name = "unknown"
        center_display = "MediNet Center"

    # Create downloadable content
    download_content = {
        "user_credentials": {
            "username": user_data['username'],
            "password": user_data.get('password', 'Not available'),
            "email": user_data['email'],
            "full_name": f"{user_data['first_name']} {user_data['last_name']}",
            "role": user_data['role'],
            "created_at": user_data['api_key_created']
        },
        "api_access": {
            "name": center_name,
            "display_name": center_display,
            "url": base_url,
            "api_key": user_data['api_key'],
            "created_at": user_data['api_key_created'],
            "expires_at": None,
            "endpoints": {
                "base_url": base_url,
                "ping": "/api/v1/ping",
                "get_data": "/api/v1/get-data-info",
                "start_training": "/api/v1/start-client",
                "cancel_training": "/api/v1/cancel-training/<session_id>"
            }
        },
        "usage_instructions": {
            "authentication": "Include 'X-API-Key' header in all requests",
            "client_ip": "Include 'X-Client-IP' header with your IP address",
            "example_curl": f"curl -H 'X-API-Key: {user_data['api_key']}' -H 'X-Client-IP: YOUR_IP' {base_url}/api/v1/ping"
        }
    }
    
    # Generate JSON response for download
    response = JsonResponse(download_content, json_dumps_params={'indent': 2})
    response['Content-Disposition'] = f'attachment; filename="user_{user_data["username"]}_credentials.json"'
    
    # Clear session data after download (one-time use)
    del request.session['new_user_data']
    
    return response


@require_permission('user.view')
def user_detail(request, user_id):
    """View user details."""
    
    user = get_object_or_404(User, id=user_id)
    
    # Logs de actividad del usuario (últimos 10)
    user_logs = AuditLog.objects.filter(user=user).order_by('-timestamp')[:10]
    
    # Historial de contraseñas
    password_history = PasswordHistory.objects.filter(user=user).order_by('-created_at')[:5]
    
    # API Keys information
    api_keys = user.api_keys.all() if hasattr(user, 'api_keys') else []
    
    context = {
        'user_detail': user,
        'user_logs': user_logs,
        'password_history': password_history,
        'api_keys': api_keys,
        'is_locked': user.is_account_locked(),
        'session_expired': user.is_session_expired(),
        'can_edit': request.user.has_permission('user.edit'),
        'can_delete': request.user.has_permission('user.delete'),
    }
    
    return render(request, 'users/user_detail.html', context)


@require_permission('user.edit')
@csrf_protect
def update_user(request, user_id):
    """Update user."""
    
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        form = UserUpdateForm(
            request.POST, 
            instance=user, 
            request_user=request.user
        )
        
        if form.is_valid():
            try:
                with transaction.atomic():
                    # Capture changes
                    old_role = user.role.name if user.role else None
                    old_active = user.is_active
                    
                    updated_user = form.save()
                    
                    # Log changes
                    changes = {}
                    if old_role != (updated_user.role.name if updated_user.role else None):
                        changes['role_changed'] = {
                            'from': old_role,
                            'to': updated_user.role.name if updated_user.role else None
                        }
                    if old_active != updated_user.is_active:
                        changes['active_status_changed'] = {
                            'from': old_active,
                            'to': updated_user.is_active
                        }
                    
                    AuditLog.objects.create(
                        user=request.user,
                        action='USER_UPDATE',
                        resource=f'user:{user.username}',
                        ip_address=request.META.get('REMOTE_ADDR'),
                        success=True,
                        details={
                            'updated_user_id': user.id,
                            'changes': changes
                        }
                    )
                    
                    messages.success(request, f'User {user.username} updated successfully.')
                    return redirect('user_detail', user_id=user.id)
                    
            except Exception as e:
                messages.error(request, f'Error updating user: {str(e)}')
    else:
        form = UserUpdateForm(instance=user, request_user=request.user)
    
    return render(request, 'users/update_user.html', {
        'form': form,
        'user_to_update': user,
    })


@require_permission('user.delete')
@csrf_protect
@require_http_methods(["POST"])
def delete_user(request, user_id):
    """Delete user with confirmation."""
    
    user = get_object_or_404(User, id=user_id)
    
    # Prevent self-deletion
    if request.user.id == user.id:
        messages.error(request, "You cannot delete your own account.")
        return HttpResponseForbidden("Cannot delete own account")
    
    try:
        with transaction.atomic():
            username = user.username
            
            # Log deletion before deleting
            AuditLog.objects.create(
                user=request.user,
                action='USER_DELETE',
                resource=f'user:{username}',
                ip_address=request.META.get('REMOTE_ADDR'),
                success=True,
                details={
                    'deleted_user_id': user.id,
                    'deleted_username': username,
                    'had_role': user.role.name if user.role else None,
                }
            )
            
            user.delete()
            messages.success(request, f'User {username} deleted successfully.')
            
    except Exception as e:
        messages.error(request, f'Error deleting user: {str(e)}')
    
    return redirect('user_list')


@require_permission('user.view')
def export_users_csv(request):
    """Export users to CSV."""
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="users.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Username', 'Email', 'First Name', 'Last Name',
        'Role', 'Active', 'Created At', 'Last Login'
    ])
    
    users = User.objects.select_related('role').all()
    
    for user in users:
        writer.writerow([
            user.username,
            user.email,
            user.first_name,
            user.last_name,
            user.role.name if user.role else 'No Role',
            'Yes' if user.is_active else 'No',
            user.date_joined.strftime('%Y-%m-%d %H:%M:%S'),
            user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never',
        ])
    
    return response


@require_permission('user.edit')
@csrf_protect
def change_user_password(request, user_id):
    """Change user password."""
    
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        form = UserPasswordChangeForm(user, request.POST)
        
        if form.is_valid():
            try:
                form.save()
                
                AuditLog.objects.create(
                    user=request.user,
                    action='PASSWORD_CHANGE',
                    resource=f'user:{user.username}',
                    ip_address=request.META.get('REMOTE_ADDR'),
                    success=True,
                    details={'target_user_id': user.id}
                )
                
                messages.success(request, f'Password for {user.username} changed successfully.')
                return redirect('user_detail', user_id=user.id)
                
            except Exception as e:
                messages.error(request, f'Error changing password: {str(e)}')
    else:
        form = UserPasswordChangeForm(user)
    
    return render(request, 'users/change_password.html', {
        'form': form,
        'user_to_update': user,
    })


@require_permission('user.view')
def user_activity_logs(request, user_id):
    """Ver logs de actividad de un usuario específico."""
    
    user = get_object_or_404(User, id=user_id)
    logs = AuditLog.objects.filter(user=user).order_by('-timestamp')
    
    # Filters
    action_filter = request.GET.get('action')
    if action_filter:
        logs = logs.filter(action=action_filter)
    
    success_filter = request.GET.get('success')
    if success_filter:
        logs = logs.filter(success=success_filter == 'true')
    
    # Pagination
    paginator = Paginator(logs, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Available actions for filter
    available_actions = AuditLog.objects.filter(user=user).values_list(
        'action', flat=True
    ).distinct()
    
    context = {
        'user_detail': user,
        'page_obj': page_obj,
        'available_actions': available_actions,
        'current_filters': {
            'action': action_filter,
            'success': success_filter,
        }
    }
    
    return render(request, 'users/user_activity_logs.html', context)


@require_role('ADMIN', 'AUDITOR')
def system_audit_logs(request):
    """View all system audit logs - accessible to admins and auditors."""
    
    logs = AuditLog.objects.select_related('user').order_by('-timestamp')
    
    # Filters
    action_filter = request.GET.get('action')
    if action_filter:
        logs = logs.filter(action=action_filter)
    
    user_filter = request.GET.get('user')
    if user_filter:
        # Try to filter by user ID first, then fall back to username
        if user_filter.isdigit():
            logs = logs.filter(user_id=user_filter)
        else:
            logs = logs.filter(user__username__icontains=user_filter)
    
    success_filter = request.GET.get('success')
    if success_filter:
        logs = logs.filter(success=success_filter == 'true')
    
    # Date range filter
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    if date_from:
        try:
            from_date = timezone.datetime.strptime(date_from, '%Y-%m-%d').date()
            logs = logs.filter(timestamp__date__gte=from_date)
        except ValueError:
            pass
    if date_to:
        try:
            to_date = timezone.datetime.strptime(date_to, '%Y-%m-%d').date()
            logs = logs.filter(timestamp__date__lte=to_date)
        except ValueError:
            pass
    
    # Pagination
    paginator = Paginator(logs, 100)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Available actions and users for filter
    available_actions = AuditLog.objects.values_list('action', flat=True).distinct().order_by('action')
    available_users = User.objects.filter(
        audit_logs__isnull=False
    ).values_list('username', flat=True).distinct().order_by('username')
    
    context = {
        'page_obj': page_obj,
        'available_actions': available_actions,
        'available_users': available_users,
        'current_filters': {
            'action': action_filter,
            'user': user_filter,
            'success': success_filter,
            'date_from': date_from,
            'date_to': date_to,
        },
        'total_logs': logs.count(),
    }
    
    return render(request, 'audit/system_logs.html', context)
