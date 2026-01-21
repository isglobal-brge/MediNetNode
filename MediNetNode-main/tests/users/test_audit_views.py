from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import datetime, timedelta

from audit.models import AuditLog
from users.models import CustomUser, Role


User = get_user_model()


class SystemAuditLogsViewTests(TestCase):
    """Comprehensive tests for system audit logs view."""

    def setUp(self):
        """Set up test users with different roles."""
        # Create roles
        self.admin_role = Role.objects.get(name='ADMIN')
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.researcher_role = Role.objects.get(name='RESEARCHER')

        # Create users
        self.admin_user = CustomUser.objects.create_user(
            username='admin_user',
            email='admin@test.com',
            password='AdminPass123!',
            role=self.admin_role,
            first_name='Admin',
            last_name='User'
        )
        
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            email='auditor@test.com',
            password='AuditorPass123!',
            role=self.auditor_role,
            first_name='Auditor',
            last_name='User'
        )
        
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_user',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role,
            first_name='Researcher',
            last_name='User'
        )
        
        self.no_role_user = CustomUser.objects.create_user(
            username='no_role_user',
            email='norole@test.com',
            password='NoRolePass123!',
            role=None,
            first_name='No Role',
            last_name='User'
        )

        # Create sample audit logs
        self.create_sample_audit_logs()
        
        self.client = Client()
        self.url = reverse('system_audit_logs')

    def create_sample_audit_logs(self):
        """Create sample audit logs for testing."""
        base_time = timezone.now() - timedelta(days=5)
        
        # Create logs for different users and actions
        self.audit_logs = []
        
        # Admin user logs
        self.audit_logs.append(AuditLog.objects.create(
            user=self.admin_user,
            action='USER_CREATE',
            resource='user:test_user',
            success=True,
            ip_address='192.168.1.100',
            timestamp=base_time + timedelta(hours=1),
            details={'created_user_id': 1}
        ))
        
        self.audit_logs.append(AuditLog.objects.create(
            user=self.admin_user,
            action='USER_DELETE',
            resource='user:old_user',
            success=True,
            ip_address='192.168.1.100',
            timestamp=base_time + timedelta(hours=2)
        ))
        
        # Auditor user logs
        self.audit_logs.append(AuditLog.objects.create(
            user=self.auditor_user,
            action='USER_VIEW',
            resource='user:profile',
            success=True,
            ip_address='192.168.1.101',
            timestamp=base_time + timedelta(hours=3)
        ))
        
        # Failed login attempt
        self.audit_logs.append(AuditLog.objects.create(
            user=self.researcher_user,
            action='LOGIN_FAIL',
            resource='auth:login',
            success=False,
            ip_address='192.168.1.102',
            timestamp=base_time + timedelta(hours=4)
        ))
        
        # System logs (no user)
        self.audit_logs.append(AuditLog.objects.create(
            user=None,
            action='SYSTEM_STARTUP',
            resource='system:boot',
            success=True,
            ip_address='127.0.0.1',
            timestamp=base_time + timedelta(hours=5)
        ))

    def test_admin_user_can_access(self):
        """Test that admin users can access system audit logs."""
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'System Audit Logs')
        self.assertContains(response, 'Search & Filter Logs')

    def test_auditor_user_can_access(self):
        """Test that auditor users can access system audit logs."""
        self.client.force_login(self.auditor_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'System Audit Logs')
        self.assertContains(response, 'Search & Filter Logs')

    def test_researcher_user_redirected_from_web_access(self):
        """Test that researcher users are redirected from web access (API-only)."""
        self.client.force_login(self.researcher_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 302)  # Redirected to researcher info page
        
    def test_no_role_user_cannot_access(self):
        """Test that users without roles cannot access system audit logs."""
        self.client.force_login(self.no_role_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 403)

    def test_anonymous_user_redirected_to_login(self):
        """Test that anonymous users are redirected to login."""
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, f'/auth/login/?next={self.url}')

    def test_superuser_can_access(self):
        """Test that superusers can access regardless of role."""
        superuser = CustomUser.objects.create_superuser(
            username='superuser',
            email='super@test.com',
            password='SuperPass123!'
        )
        
        self.client.force_login(superuser)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'System Audit Logs')

    def test_all_logs_displayed(self):
        """Test that all audit logs are displayed when no filters applied."""
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        
        # Check that all our test logs are in the response
        for log in self.audit_logs:
            if log.user:
                self.assertContains(response, log.user.username)
            else:
                self.assertContains(response, 'System')
            self.assertContains(response, log.action)

    def test_filter_by_user_id(self):
        """Test filtering by user ID (when clicking from user detail page)."""
        self.client.force_login(self.admin_user)
        
        # Filter by admin user ID
        response = self.client.get(self.url, {'user': str(self.admin_user.id)})
        
        self.assertEqual(response.status_code, 200)
        
        # Should contain admin user logs
        self.assertContains(response, 'USER_CREATE')
        self.assertContains(response, 'USER_DELETE')
        
        # Check that only admin user logs are displayed
        page_obj = response.context['page_obj']
        displayed_logs = page_obj.object_list
        
        # All displayed logs should belong to admin user
        for log in displayed_logs:
            self.assertEqual(log.user, self.admin_user)

    def test_filter_by_username(self):
        """Test filtering by username (text search)."""
        self.client.force_login(self.admin_user)
        
        # Filter by username substring
        response = self.client.get(self.url, {'user': 'auditor'})
        
        self.assertEqual(response.status_code, 200)
        
        # Should contain auditor user logs
        self.assertContains(response, 'USER_VIEW')
        self.assertContains(response, self.auditor_user.username)
        
        # Check that only auditor user logs are displayed
        page_obj = response.context['page_obj']
        displayed_logs = page_obj.object_list
        
        # All displayed logs should belong to users with 'auditor' in username
        for log in displayed_logs:
            if log.user:
                self.assertIn('auditor', log.user.username.lower())

    def test_filter_by_action(self):
        """Test filtering by action type."""
        self.client.force_login(self.admin_user)
        
        # Filter by USER_CREATE action
        response = self.client.get(self.url, {'action': 'USER_CREATE'})
        
        self.assertEqual(response.status_code, 200)
        
        # Should contain USER_CREATE logs in the table rows
        self.assertContains(response, 'USER_CREATE')
        
        # Check that the logs displayed are filtered correctly
        page_obj = response.context['page_obj']
        displayed_actions = [log.action for log in page_obj.object_list]
        
        # All displayed logs should be USER_CREATE
        self.assertTrue(all(action == 'USER_CREATE' for action in displayed_actions))
        
        # Should not contain other actions in the actual log entries
        self.assertNotIn('USER_DELETE', displayed_actions)
        self.assertNotIn('LOGIN_FAIL', displayed_actions)

    def test_filter_by_success_status(self):
        """Test filtering by success/failure status."""
        self.client.force_login(self.admin_user)
        
        # Filter by successful actions
        response = self.client.get(self.url, {'success': 'true'})
        
        self.assertEqual(response.status_code, 200)
        
        # Should show success indicators
        self.assertContains(response, 'Success')
        
        # Test failed actions
        response = self.client.get(self.url, {'success': 'false'})
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Failed')
        self.assertContains(response, 'LOGIN_FAIL')

    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        self.client.force_login(self.admin_user)
        
        # Get tomorrow's date (no logs should exist)
        tomorrow = (timezone.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        response = self.client.get(self.url, {
            'date_from': tomorrow,
            'date_to': tomorrow
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Should show no logs in the table
        page_obj = response.context['page_obj']
        self.assertEqual(len(page_obj.object_list), 0)

    def test_multiple_filters_combined(self):
        """Test using multiple filters together."""
        self.client.force_login(self.admin_user)
        
        response = self.client.get(self.url, {
            'user': str(self.admin_user.id),
            'action': 'USER_CREATE',
            'success': 'true'
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Should contain only matching logs
        self.assertContains(response, 'USER_CREATE')
        self.assertContains(response, self.admin_user.username)
        
        # Check that all displayed logs match the filters
        page_obj = response.context['page_obj']
        displayed_logs = page_obj.object_list
        
        for log in displayed_logs:
            self.assertEqual(log.user, self.admin_user)
            self.assertEqual(log.action, 'USER_CREATE')
            self.assertTrue(log.success)

    def test_invalid_date_format_ignored(self):
        """Test that invalid date formats are ignored gracefully."""
        self.client.force_login(self.admin_user)
        
        response = self.client.get(self.url, {
            'date_from': 'invalid-date',
            'date_to': 'also-invalid'
        })
        
        # Should not crash, should show all logs (invalid dates ignored)
        self.assertEqual(response.status_code, 200)
        
        # Check that we still have logs (since invalid dates are ignored)
        page_obj = response.context['page_obj']
        self.assertGreater(len(page_obj.object_list), 0)

    def test_pagination_works(self):
        """Test that pagination works correctly."""
        # Create many audit logs to test pagination
        for i in range(150):
            AuditLog.objects.create(
                user=self.admin_user,
                action=f'TEST_ACTION_{i}',
                resource=f'test:resource_{i}',
                success=True,
                ip_address='127.0.0.1'
            )
        
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        
        # Check pagination context
        self.assertTrue('page_obj' in response.context)
        page_obj = response.context['page_obj']
        
        # Should be paginated (100 per page by default)
        self.assertTrue(page_obj.has_next())
        self.assertEqual(len(page_obj.object_list), 100)

    def test_context_variables_present(self):
        """Test that all required context variables are present."""
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        
        # Check required context variables
        context = response.context
        self.assertIn('page_obj', context)
        self.assertIn('available_actions', context)
        self.assertIn('available_users', context)
        self.assertIn('current_filters', context)
        self.assertIn('total_logs', context)

    def test_available_actions_populated(self):
        """Test that available actions are populated correctly."""
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        
        self.assertEqual(response.status_code, 200)
        
        available_actions = response.context['available_actions']
        
        # Should contain actions from our test data
        expected_actions = {'USER_CREATE', 'USER_DELETE', 'USER_VIEW', 'LOGIN_FAIL', 'SYSTEM_STARTUP'}
        self.assertTrue(expected_actions.issubset(set(available_actions)))

    def test_filter_status_displayed(self):
        """Test that filter status is displayed when filters are applied."""
        self.client.force_login(self.admin_user)
        
        response = self.client.get(self.url, {'user': 'admin'})
        
        self.assertEqual(response.status_code, 200)
        
        # Should show filtered status (template shows "filtered")
        self.assertContains(response, 'filtered')
        self.assertContains(response, 'Clear Filters')  # Button should be present

    def test_clear_filters_works(self):
        """Test that clear filters functionality works."""
        self.client.force_login(self.admin_user)
        
        # First apply some filters
        response = self.client.get(self.url, {'user': 'admin', 'action': 'USER_CREATE'})
        self.assertEqual(response.status_code, 200)
        
        # Then clear them by visiting the base URL
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        
        # Should show all logs again
        page_obj = response.context['page_obj']
        all_actions = [log.action for log in page_obj.object_list]
        
        # Should contain various actions from our test data
        self.assertIn('USER_CREATE', all_actions)
        self.assertIn('USER_VIEW', all_actions)
        self.assertIn('LOGIN_FAIL', all_actions)

    def test_user_detail_link_works(self):
        """Test that clicking from user detail page works correctly."""
        # This simulates the link from user_detail.html template
        self.client.force_login(self.admin_user)
        
        # URL with user ID as parameter (simulating the template link)
        user_logs_url = f"{self.url}?user={self.admin_user.id}"
        response = self.client.get(user_logs_url)
        
        self.assertEqual(response.status_code, 200)
        
        # Should show only logs for that specific user
        logs_for_user = AuditLog.objects.filter(user=self.admin_user).count()
        self.assertGreater(logs_for_user, 0)
        
        # Check that the filtered count matches
        displayed_count = response.context['page_obj'].paginator.count
        self.assertEqual(displayed_count, logs_for_user)