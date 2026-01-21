"""
Pytest configuration for MediNet-Node.

This file ensures Django is properly configured before running tests.
"""

import os
import django
from django.conf import settings
import pytest

# Set environment variables for tests BEFORE Django setup
os.environ.setdefault('SECRET_KEY', 'django-insecure-test-key-for-running-tests-only')
os.environ.setdefault('DEBUG', 'True')
os.environ.setdefault('ALLOWED_HOSTS', 'localhost,127.0.0.1')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings_test')

# Setup Django
if not settings.configured:
    django.setup()


@pytest.fixture(scope='session')
def django_db_setup(django_db_setup, django_db_blocker):
    """
    Create default roles once per test session.
    This prevents UNIQUE constraint errors.
    """
    from users.models import Role

    with django_db_blocker.unblock():
        # Create roles if they don't exist
        # Using dot-notation permissions to match production code
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

        # ADMIN role - full permissions
        Role.objects.get_or_create(
            name='ADMIN',
            defaults={'permissions': admin_permissions}
        )

        # RESEARCHER role - minimal permissions, API-only (NO web access)
        Role.objects.get_or_create(
            name='RESEARCHER',
            defaults={
                'permissions': {
                    'api.access': True,       # API access only
                    'dataset.view': True,
                    'dataset.train': True,
                    # NO 'web.access' - researchers are API-only
                }
            }
        )

        # AUDITOR role - audit and view permissions only
        Role.objects.get_or_create(
            name='AUDITOR',
            defaults={
                'permissions': {
                    'dataset.view': True,    # Read-only dataset access
                    'audit.view': True,
                    'training.view': True,
                    'user.view': True,       # Read-only user view
                }
            }
        )


@pytest.fixture
def admin_role(db):
    """Fixture to get ADMIN role (already created in session)."""
    from users.models import Role
    return Role.objects.get(name='ADMIN')


@pytest.fixture
def researcher_role(db):
    """Fixture to get RESEARCHER role (already created in session)."""
    from users.models import Role
    return Role.objects.get(name='RESEARCHER')


@pytest.fixture
def auditor_role(db):
    """Fixture to get AUDITOR role (already created in session)."""
    from users.models import Role
    return Role.objects.get(name='AUDITOR')


@pytest.fixture
def admin_user(db, admin_role):
    """Fixture to create an admin user."""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    return User.objects.create_user(
        username='admin_test',
        password='AdminPass123!',
        email='admin@test.com',
        role=admin_role
    )


@pytest.fixture
def researcher_user(db, researcher_role):
    """Fixture to create a researcher user."""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    return User.objects.create_user(
        username='researcher_test',
        password='ResearcherPass123!',
        email='researcher@test.com',
        role=researcher_role
    )


@pytest.fixture
def auditor_user(db, auditor_role):
    """Fixture to create an auditor user."""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    return User.objects.create_user(
        username='auditor_test',
        password='AuditorPass123!',
        email='auditor@test.com',
        role=auditor_role
    )
