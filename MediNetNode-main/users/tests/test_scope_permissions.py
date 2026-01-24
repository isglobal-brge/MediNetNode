"""
Tests for scope-based permissions system.
"""
import pytest
from django.core.management import call_command
from users.models import Role, CustomUser


@pytest.fixture(scope='class')
def setup_roles(django_db_setup, django_db_blocker):
    """Setup roles before running tests."""
    with django_db_blocker.unblock():
        call_command('setup_roles', '--force')


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestScopeBasedPermissions:
    """Test scope-based permission system."""

    def test_member_role_exists(self):
        """Test that MEMBER role is created."""
        member_role = Role.objects.get(name='MEMBER')
        assert member_role is not None
        assert 'inference.execute' in member_role.permissions
        assert 'inference.upload' in member_role.permissions

    def test_simple_boolean_permission(self):
        """Test backward compatible boolean permissions."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_test',
            password='testpass123',
            role=admin_role
        )

        assert admin_user.has_permission('api.access') is True
        assert admin_user.has_permission('system.admin') is True
        assert admin_user.has_permission('nonexistent.permission') is False

    def test_scope_all_permission(self):
        """Test scope: ALL permission."""
        member_role = Role.objects.get(name='MEMBER')
        member_user = CustomUser.objects.create_user(
            username='member_test',
            password='testpass123',
            role=member_role
        )

        # Basic permission check (no domain)
        assert member_user.has_permission('inference.execute') is True

        # Domain check with scope ALL
        assert member_user.has_permission('inference.execute', domain='cardiology') is True
        assert member_user.has_permission('inference.execute', domain='neurology') is True
        assert member_user.has_permission('inference.execute', domain='oncology') is True

    def test_scope_list_permission(self):
        """Test scope with specific domain list."""
        # Create a role with limited domain scope
        limited_role = Role.objects.create(
            name='LIMITED_RESEARCHER',
            permissions={
                'api.access': True,
                'inference.execute': {'scope': ['cardiology', 'neurology']},
            }
        )
        limited_user = CustomUser.objects.create_user(
            username='limited_test',
            password='testpass123',
            role=limited_role
        )

        # Has permission without domain check
        assert limited_user.has_permission('inference.execute') is True

        # Has permission for allowed domains
        assert limited_user.has_permission('inference.execute', domain='cardiology') is True
        assert limited_user.has_permission('inference.execute', domain='neurology') is True

        # No permission for disallowed domain
        assert limited_user.has_permission('inference.execute', domain='oncology') is False
        assert limited_user.has_permission('inference.execute', domain='radiology') is False

    def test_get_permission_scope(self):
        """Test get_permission_scope method."""
        member_role = Role.objects.get(name='MEMBER')
        member_user = CustomUser.objects.create_user(
            username='scope_test',
            password='testpass123',
            role=member_role
        )

        # Scope ALL
        scope = member_user.get_permission_scope('inference.execute')
        assert scope == 'ALL'

        # Boolean permission has no scope
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_scope_test',
            password='testpass123',
            role=admin_role
        )
        scope = admin_user.get_permission_scope('inference.approve')
        assert scope is None

    def test_superuser_has_all_permissions(self):
        """Test that superuser bypasses all permission checks."""
        superuser = CustomUser.objects.create_superuser(
            username='superuser_test',
            password='testpass123',
            email='super@test.com'
        )

        assert superuser.has_permission('any.permission') is True
        assert superuser.has_permission('inference.execute', domain='any_domain') is True
        assert superuser.get_permission_scope('any.permission') is None

    def test_user_without_role_has_no_permissions(self):
        """Test that user without role has no permissions."""
        user_no_role = CustomUser.objects.create_user(
            username='no_role_test',
            password='testpass123'
        )

        assert user_no_role.has_permission('api.access') is False
        assert user_no_role.has_permission('inference.execute') is False
        assert user_no_role.get_permission_scope('inference.execute') is None

    def test_researcher_inference_permissions(self):
        """Test RESEARCHER role has correct inference permissions."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            password='testpass123',
            role=researcher_role
        )

        # RESEARCHER can execute inference
        assert researcher_user.has_permission('inference.execute') is True
        assert researcher_user.has_permission('inference.execute', domain='cardiology') is True

        # RESEARCHER cannot upload models
        assert researcher_user.has_permission('inference.upload') is False
        assert researcher_user.has_permission('inference.approve') is False

    def test_member_inference_permissions(self):
        """Test MEMBER role has correct inference permissions."""
        member_role = Role.objects.get(name='MEMBER')
        member_user = CustomUser.objects.create_user(
            username='member_test2',
            password='testpass123',
            role=member_role
        )

        # MEMBER can execute inference
        assert member_user.has_permission('inference.execute') is True
        assert member_user.has_permission('inference.execute', domain='cardiology') is True

        # MEMBER can upload models
        assert member_user.has_permission('inference.upload') is True

        # MEMBER cannot approve models
        assert member_user.has_permission('inference.approve') is False
        assert member_user.has_permission('inference.admin') is False

    def test_admin_inference_permissions(self):
        """Test ADMIN role has all inference permissions."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_test2',
            password='testpass123',
            role=admin_role
        )

        # ADMIN has all inference permissions
        assert admin_user.has_permission('inference.execute') is True
        assert admin_user.has_permission('inference.upload') is True
        assert admin_user.has_permission('inference.approve') is True
        assert admin_user.has_permission('inference.admin') is True

        # Scope is ALL
        scope = admin_user.get_permission_scope('inference.execute')
        assert scope == 'ALL'

    def test_auditor_inference_permissions(self):
        """Test AUDITOR role has view-only inference permissions."""
        auditor_role = Role.objects.get(name='AUDITOR')
        auditor_user = CustomUser.objects.create_user(
            username='auditor_test',
            password='testpass123',
            role=auditor_role
        )

        # AUDITOR can only view inference logs
        assert auditor_user.has_permission('inference.view') is True

        # AUDITOR cannot execute, upload, or approve
        assert auditor_user.has_permission('inference.execute') is False
        assert auditor_user.has_permission('inference.upload') is False
        assert auditor_user.has_permission('inference.approve') is False
