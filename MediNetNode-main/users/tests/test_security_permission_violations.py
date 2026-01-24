"""
Aggressive security tests for MEMBER and RESEARCHER roles.
Tests attempt to violate permission boundaries and verify proper access control.
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
class TestMemberSecurityViolations:
    """Aggressive security tests for MEMBER role - attempting permission violations."""

    @pytest.fixture(autouse=True)
    def setup_member(self):
        """Create MEMBER user for each test."""
        member_role = Role.objects.get(name='MEMBER')
        self.member = CustomUser.objects.create_user(
            username='member_attacker',
            password='testpass123',
            email='member@test.com',
            role=member_role
        )

    # ============================================================================
    # ADMIN-ONLY PERMISSION VIOLATION ATTEMPTS
    # ============================================================================

    def test_member_cannot_access_system_admin(self):
        """MEMBER should NOT have system.admin permission."""
        assert self.member.has_permission('system.admin') is False

    def test_member_cannot_create_users(self):
        """MEMBER should NOT be able to create users."""
        assert self.member.has_permission('user.create') is False

    def test_member_cannot_view_users(self):
        """MEMBER should NOT be able to view user list."""
        assert self.member.has_permission('user.view') is False

    def test_member_cannot_edit_users(self):
        """MEMBER should NOT be able to edit users."""
        assert self.member.has_permission('user.edit') is False

    def test_member_cannot_delete_users(self):
        """MEMBER should NOT be able to delete users."""
        assert self.member.has_permission('user.delete') is False

    def test_member_cannot_edit_datasets(self):
        """MEMBER should NOT be able to edit datasets (only create their own)."""
        assert self.member.has_permission('dataset.edit') is False

    def test_member_cannot_delete_datasets(self):
        """MEMBER should NOT be able to delete datasets."""
        assert self.member.has_permission('dataset.delete') is False

    def test_member_cannot_manage_trainings(self):
        """MEMBER should NOT be able to manage training sessions."""
        assert self.member.has_permission('training.manage') is False

    def test_member_cannot_view_audit_logs(self):
        """MEMBER should NOT be able to view audit logs."""
        assert self.member.has_permission('audit.view') is False

    # ============================================================================
    # INFERENCE PERMISSION VIOLATIONS
    # ============================================================================

    def test_member_cannot_approve_models(self):
        """MEMBER should NOT be able to approve models."""
        assert self.member.has_permission('inference.approve') is False

    def test_member_cannot_access_inference_admin(self):
        """MEMBER should NOT have inference.admin permission."""
        assert self.member.has_permission('inference.admin') is False

    def test_member_cannot_view_inference_logs_as_auditor(self):
        """MEMBER should NOT have inference.view (only AUDITOR)."""
        assert self.member.has_permission('inference.view') is False

    # ============================================================================
    # VALID MEMBER PERMISSIONS (Verify what they SHOULD have)
    # ============================================================================

    def test_member_can_access_api(self):
        """MEMBER SHOULD be able to access API."""
        assert self.member.has_permission('api.access') is True

    def test_member_can_view_datasets(self):
        """MEMBER SHOULD be able to view datasets."""
        assert self.member.has_permission('dataset.view') is True

    def test_member_can_create_datasets(self):
        """MEMBER SHOULD be able to create datasets."""
        assert self.member.has_permission('dataset.create') is True

    def test_member_can_train(self):
        """MEMBER SHOULD be able to train with datasets."""
        assert self.member.has_permission('dataset.train') is True

    def test_member_can_view_trainings(self):
        """MEMBER SHOULD be able to view training sessions."""
        assert self.member.has_permission('training.view') is True

    def test_member_can_execute_inference(self):
        """MEMBER SHOULD be able to execute inference."""
        assert self.member.has_permission('inference.execute') is True

    def test_member_can_upload_models(self):
        """MEMBER SHOULD be able to upload models (for approval)."""
        assert self.member.has_permission('inference.upload') is True

    # ============================================================================
    # SCOPE BOUNDARY TESTS
    # ============================================================================

    def test_member_inference_scope_is_all(self):
        """MEMBER should have ALL scope for inference.execute."""
        scope = self.member.get_permission_scope('inference.execute')
        assert scope == 'ALL'

    def test_member_dataset_scope_is_all(self):
        """MEMBER should have ALL scope for dataset operations."""
        scope = self.member.get_permission_scope('dataset.view')
        assert scope == 'ALL'
        scope = self.member.get_permission_scope('dataset.train')
        assert scope == 'ALL'

    # ============================================================================
    # PRIVILEGE ESCALATION ATTEMPTS
    # ============================================================================

    def test_member_cannot_escalate_to_admin_permissions(self):
        """MEMBER should not be able to access any admin-only permissions."""
        admin_only_permissions = [
            'system.admin',
            'user.create',
            'user.edit',
            'user.delete',
            'dataset.edit',
            'dataset.delete',
            'training.manage',
            'audit.view',
            'inference.approve',
            'inference.admin',
        ]

        for permission in admin_only_permissions:
            assert self.member.has_permission(permission) is False, \
                f"SECURITY VIOLATION: MEMBER has {permission} permission!"

    def test_member_permission_count(self):
        """MEMBER should have exactly 7 permissions, no more."""
        expected_permissions = {
            'api.access',
            'dataset.view',
            'dataset.create',
            'dataset.train',
            'training.view',
            'inference.execute',
            'inference.upload',
        }

        actual_permissions = set(self.member.role.permissions.keys())
        assert actual_permissions == expected_permissions, \
            f"SECURITY VIOLATION: MEMBER has unexpected permissions: {actual_permissions - expected_permissions}"


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestResearcherSecurityViolations:
    """Aggressive security tests for RESEARCHER role - attempting permission violations."""

    @pytest.fixture(autouse=True)
    def setup_researcher(self):
        """Create RESEARCHER user for each test."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        self.researcher = CustomUser.objects.create_user(
            username='researcher_attacker',
            password='testpass123',
            email='researcher@test.com',
            role=researcher_role
        )

    # ============================================================================
    # ADMIN-ONLY PERMISSION VIOLATION ATTEMPTS
    # ============================================================================

    def test_researcher_cannot_access_system_admin(self):
        """RESEARCHER should NOT have system.admin permission."""
        assert self.researcher.has_permission('system.admin') is False

    def test_researcher_cannot_create_users(self):
        """RESEARCHER should NOT be able to create users."""
        assert self.researcher.has_permission('user.create') is False

    def test_researcher_cannot_view_users(self):
        """RESEARCHER should NOT be able to view user list."""
        assert self.researcher.has_permission('user.view') is False

    def test_researcher_cannot_edit_users(self):
        """RESEARCHER should NOT be able to edit users."""
        assert self.researcher.has_permission('user.edit') is False

    def test_researcher_cannot_delete_users(self):
        """RESEARCHER should NOT be able to delete users."""
        assert self.researcher.has_permission('user.delete') is False

    def test_researcher_cannot_create_datasets(self):
        """RESEARCHER should NOT be able to create datasets."""
        assert self.researcher.has_permission('dataset.create') is False

    def test_researcher_cannot_edit_datasets(self):
        """RESEARCHER should NOT be able to edit datasets."""
        assert self.researcher.has_permission('dataset.edit') is False

    def test_researcher_cannot_delete_datasets(self):
        """RESEARCHER should NOT be able to delete datasets."""
        assert self.researcher.has_permission('dataset.delete') is False

    def test_researcher_cannot_manage_trainings(self):
        """RESEARCHER should NOT be able to manage training sessions."""
        assert self.researcher.has_permission('training.manage') is False

    def test_researcher_cannot_view_trainings(self):
        """RESEARCHER should NOT be able to view training sessions (API-only access)."""
        assert self.researcher.has_permission('training.view') is False

    def test_researcher_cannot_view_audit_logs(self):
        """RESEARCHER should NOT be able to view audit logs."""
        assert self.researcher.has_permission('audit.view') is False

    # ============================================================================
    # INFERENCE PERMISSION VIOLATIONS
    # ============================================================================

    def test_researcher_cannot_upload_models(self):
        """RESEARCHER should NOT be able to upload models."""
        assert self.researcher.has_permission('inference.upload') is False

    def test_researcher_cannot_approve_models(self):
        """RESEARCHER should NOT be able to approve models."""
        assert self.researcher.has_permission('inference.approve') is False

    def test_researcher_cannot_access_inference_admin(self):
        """RESEARCHER should NOT have inference.admin permission."""
        assert self.researcher.has_permission('inference.admin') is False

    def test_researcher_cannot_view_inference_logs(self):
        """RESEARCHER should NOT have inference.view (only AUDITOR)."""
        assert self.researcher.has_permission('inference.view') is False

    # ============================================================================
    # VALID RESEARCHER PERMISSIONS (Verify what they SHOULD have)
    # ============================================================================

    def test_researcher_can_access_api(self):
        """RESEARCHER SHOULD be able to access API."""
        assert self.researcher.has_permission('api.access') is True

    def test_researcher_can_view_datasets(self):
        """RESEARCHER SHOULD be able to view datasets."""
        assert self.researcher.has_permission('dataset.view') is True

    def test_researcher_can_train(self):
        """RESEARCHER SHOULD be able to train with datasets."""
        assert self.researcher.has_permission('dataset.train') is True

    def test_researcher_can_execute_inference(self):
        """RESEARCHER SHOULD be able to execute inference."""
        assert self.researcher.has_permission('inference.execute') is True

    # ============================================================================
    # SCOPE BOUNDARY TESTS
    # ============================================================================

    def test_researcher_inference_scope_is_all(self):
        """RESEARCHER should have ALL scope for inference.execute."""
        scope = self.researcher.get_permission_scope('inference.execute')
        assert scope == 'ALL'

    def test_researcher_dataset_scope_is_all(self):
        """RESEARCHER should have ALL scope for dataset operations."""
        scope = self.researcher.get_permission_scope('dataset.view')
        assert scope == 'ALL'
        scope = self.researcher.get_permission_scope('dataset.train')
        assert scope == 'ALL'

    # ============================================================================
    # PRIVILEGE ESCALATION ATTEMPTS
    # ============================================================================

    def test_researcher_cannot_escalate_to_admin_permissions(self):
        """RESEARCHER should not be able to access any admin-only permissions."""
        admin_only_permissions = [
            'system.admin',
            'user.create',
            'user.edit',
            'user.delete',
            'user.view',
            'dataset.create',
            'dataset.edit',
            'dataset.delete',
            'training.manage',
            'training.view',
            'audit.view',
            'inference.upload',
            'inference.approve',
            'inference.admin',
            'inference.view',
        ]

        for permission in admin_only_permissions:
            assert self.researcher.has_permission(permission) is False, \
                f"SECURITY VIOLATION: RESEARCHER has {permission} permission!"

    def test_researcher_cannot_escalate_to_member_permissions(self):
        """RESEARCHER should not have MEMBER-specific permissions."""
        member_specific_permissions = [
            'dataset.create',
            'training.view',
            'inference.upload',
        ]

        for permission in member_specific_permissions:
            assert self.researcher.has_permission(permission) is False, \
                f"SECURITY VIOLATION: RESEARCHER has MEMBER permission {permission}!"

    def test_researcher_permission_count(self):
        """RESEARCHER should have exactly 4 permissions, no more."""
        expected_permissions = {
            'api.access',
            'dataset.view',
            'dataset.train',
            'inference.execute',
        }

        actual_permissions = set(self.researcher.role.permissions.keys())
        assert actual_permissions == expected_permissions, \
            f"SECURITY VIOLATION: RESEARCHER has unexpected permissions: {actual_permissions - expected_permissions}"


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestCrossRoleSecurityBoundaries:
    """Test security boundaries between different roles."""

    @pytest.fixture(autouse=True)
    def setup_users(self):
        """Create users for all roles."""
        self.admin = CustomUser.objects.create_user(
            username='admin_user',
            password='testpass123',
            role=Role.objects.get(name='ADMIN')
        )
        self.member = CustomUser.objects.create_user(
            username='member_user',
            password='testpass123',
            role=Role.objects.get(name='MEMBER')
        )
        self.researcher = CustomUser.objects.create_user(
            username='researcher_user',
            password='testpass123',
            role=Role.objects.get(name='RESEARCHER')
        )
        self.auditor = CustomUser.objects.create_user(
            username='auditor_user',
            password='testpass123',
            role=Role.objects.get(name='AUDITOR')
        )

    # ============================================================================
    # ADMIN EXCLUSIVE PERMISSIONS
    # ============================================================================

    def test_only_admin_can_approve_models(self):
        """Only ADMIN should be able to approve models."""
        assert self.admin.has_permission('inference.approve') is True
        assert self.member.has_permission('inference.approve') is False
        assert self.researcher.has_permission('inference.approve') is False
        assert self.auditor.has_permission('inference.approve') is False

    def test_only_admin_has_inference_admin(self):
        """Only ADMIN should have inference.admin permission."""
        assert self.admin.has_permission('inference.admin') is True
        assert self.member.has_permission('inference.admin') is False
        assert self.researcher.has_permission('inference.admin') is False
        assert self.auditor.has_permission('inference.admin') is False

    def test_only_admin_can_manage_users(self):
        """Only ADMIN should be able to manage users."""
        user_permissions = ['user.view', 'user.create', 'user.edit', 'user.delete']

        for perm in user_permissions:
            assert self.admin.has_permission(perm) is True, f"ADMIN should have {perm}"
            assert self.member.has_permission(perm) is False, f"MEMBER should NOT have {perm}"
            assert self.researcher.has_permission(perm) is False, f"RESEARCHER should NOT have {perm}"

    # ============================================================================
    # MEMBER EXCLUSIVE PERMISSIONS (vs RESEARCHER)
    # ============================================================================

    def test_only_member_can_create_datasets(self):
        """Only ADMIN and MEMBER should be able to create datasets."""
        assert self.admin.has_permission('dataset.create') is True
        assert self.member.has_permission('dataset.create') is True
        assert self.researcher.has_permission('dataset.create') is False
        assert self.auditor.has_permission('dataset.create') is False

    def test_only_member_can_upload_models(self):
        """Only ADMIN and MEMBER should be able to upload models."""
        assert self.admin.has_permission('inference.upload') is True
        assert self.member.has_permission('inference.upload') is True
        assert self.researcher.has_permission('inference.upload') is False
        assert self.auditor.has_permission('inference.upload') is False

    def test_only_member_can_view_trainings_dashboard(self):
        """Only ADMIN and MEMBER should be able to view training dashboard."""
        assert self.admin.has_permission('training.view') is True
        assert self.member.has_permission('training.view') is True
        assert self.researcher.has_permission('training.view') is False
        assert self.auditor.has_permission('training.view') is True  # AUDITOR can also view

    # ============================================================================
    # AUDITOR EXCLUSIVE PERMISSIONS
    # ============================================================================

    def test_only_admin_and_auditor_can_view_audit_logs(self):
        """Only ADMIN and AUDITOR should be able to view audit logs."""
        assert self.admin.has_permission('audit.view') is True
        assert self.member.has_permission('audit.view') is False
        assert self.researcher.has_permission('audit.view') is False
        assert self.auditor.has_permission('audit.view') is True

    def test_only_auditor_has_inference_view(self):
        """Only ADMIN and AUDITOR should have inference.view."""
        assert self.admin.has_permission('inference.view') is False  # ADMIN doesn't need it
        assert self.member.has_permission('inference.view') is False
        assert self.researcher.has_permission('inference.view') is False
        assert self.auditor.has_permission('inference.view') is True

    # ============================================================================
    # SHARED PERMISSIONS (All can execute inference and train)
    # ============================================================================

    def test_all_roles_can_execute_inference_except_auditor(self):
        """ADMIN, MEMBER, RESEARCHER should be able to execute inference."""
        assert self.admin.has_permission('inference.execute') is True
        assert self.member.has_permission('inference.execute') is True
        assert self.researcher.has_permission('inference.execute') is True
        assert self.auditor.has_permission('inference.execute') is False

    def test_all_roles_can_train_except_auditor(self):
        """ADMIN, MEMBER, RESEARCHER should be able to train."""
        assert self.admin.has_permission('dataset.train') is True
        assert self.member.has_permission('dataset.train') is True
        assert self.researcher.has_permission('dataset.train') is True
        assert self.auditor.has_permission('dataset.train') is False

    def test_all_roles_can_view_datasets(self):
        """All roles should be able to view datasets."""
        assert self.admin.has_permission('dataset.view') is True
        assert self.member.has_permission('dataset.view') is True
        assert self.researcher.has_permission('dataset.view') is True
        assert self.auditor.has_permission('dataset.view') is True

    # ============================================================================
    # PERMISSION COUNT VERIFICATION
    # ============================================================================

    def test_permission_count_hierarchy(self):
        """ADMIN should have most permissions, RESEARCHER least."""
        admin_count = len(self.admin.role.permissions)
        member_count = len(self.member.role.permissions)
        researcher_count = len(self.researcher.role.permissions)
        auditor_count = len(self.auditor.role.permissions)

        # ADMIN should have the most
        assert admin_count > member_count
        assert admin_count > researcher_count
        assert admin_count > auditor_count

        # MEMBER should have more than RESEARCHER
        assert member_count > researcher_count

        # Exact counts
        assert admin_count == 18
        assert member_count == 7
        assert researcher_count == 4
        assert auditor_count == 5


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestDomainScopeSecurityViolations:
    """Test scope-based permission violations for domain restrictions."""

    @pytest.fixture(autouse=True)
    def setup_limited_user(self):
        """Create a user with limited domain scope."""
        self.limited_role = Role.objects.create(
            name='LIMITED_CARDIOLOGIST',
            permissions={
                'api.access': True,
                'inference.execute': {'scope': ['cardiology']},
                'dataset.view': {'scope': ['cardiology', 'general']},
            }
        )
        self.limited_user = CustomUser.objects.create_user(
            username='limited_cardio',
            password='testpass123',
            role=self.limited_role
        )

    def test_limited_user_can_access_allowed_domain(self):
        """User should access domains in their scope."""
        assert self.limited_user.has_permission('inference.execute', domain='cardiology') is True

    def test_limited_user_cannot_access_disallowed_domain(self):
        """User should NOT access domains outside their scope."""
        assert self.limited_user.has_permission('inference.execute', domain='neurology') is False
        assert self.limited_user.has_permission('inference.execute', domain='oncology') is False
        assert self.limited_user.has_permission('inference.execute', domain='radiology') is False

    def test_limited_user_cannot_bypass_scope_with_all(self):
        """User with limited scope should not have ALL access."""
        scope = self.limited_user.get_permission_scope('inference.execute')
        assert scope != 'ALL'
        assert scope == ['cardiology']

    def test_limited_user_different_scopes_for_different_permissions(self):
        """User can have different scopes for different permissions."""
        inference_scope = self.limited_user.get_permission_scope('inference.execute')
        dataset_scope = self.limited_user.get_permission_scope('dataset.view')

        assert inference_scope == ['cardiology']
        assert dataset_scope == ['cardiology', 'general']

        # Can view general datasets
        assert self.limited_user.has_permission('dataset.view', domain='general') is True
        # Cannot execute inference on general models
        assert self.limited_user.has_permission('inference.execute', domain='general') is False

    def test_scope_violations_attempt_escalation(self):
        """Attempt to escalate scope through various methods."""
        restricted_domains = ['neurology', 'oncology', 'radiology', 'pathology']

        for domain in restricted_domains:
            # Direct domain check
            assert self.limited_user.has_permission('inference.execute', domain=domain) is False, \
                f"SECURITY VIOLATION: Limited user accessed {domain}!"

            # Try with uppercase
            assert self.limited_user.has_permission('inference.execute', domain=domain.upper()) is False

            # Try with mixed case
            assert self.limited_user.has_permission('inference.execute', domain=domain.title()) is False
