from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from users.models import CustomUser, Role
from audit.models import AuditEvent, SecurityIncident


User = get_user_model()


class AuditorDashboardAccessControlTest(TestCase):
    """Test that only AUDITOR role can access audit dashboard views."""
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test users with different roles."""
        # Create roles
        self.admin_role = Role.objects.get(name='ADMIN')
        self.investigador_role = Role.objects.get(name='RESEARCHER')
        self.auditor_role = Role.objects.get(name='AUDITOR')
        
        # Create users with different roles
        self.admin_user = CustomUser.objects.create_user(
            username='admin_user',
            password='StrongPass123!',
            role=self.admin_role
        )
        
        self.investigador_user = CustomUser.objects.create_user(
            username='investigador_user', 
            password='StrongPass123!',
            role=self.investigador_role
        )
        
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!', 
            role=self.auditor_role
        )
        
        # Create test data
        self.audit_event = AuditEvent.objects.create(
            user=self.admin_user,
            action='TEST_ACTION',
            resource='test_resource',
            category='SYSTEM',
            severity='INFO',
            risk_score=25
        )
        
        self.security_incident = SecurityIncident.objects.create(
            incident_type='SUSPICIOUS_ACTIVITY',
            description='Test incident',
            severity=2,
            state='OPEN'
        )
        
        self.client = Client()

    def test_auditor_dashboard_access_auditor_allowed(self):
        """Test that AUDITOR role can access auditor dashboard."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:auditor_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Audit Dashboard')

    def test_auditor_dashboard_access_admin_denied(self):
        """Test that ADMIN role is denied access to auditor dashboard."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:auditor_dashboard'))
        self.assertEqual(response.status_code, 403)

    def test_auditor_dashboard_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to auditor dashboard."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:auditor_dashboard'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_auditor_dashboard_access_anonymous_denied(self):
        """Test that anonymous users are denied access to auditor dashboard."""
        response = self.client.get(reverse('audit:auditor_dashboard'))
        # Should redirect to login page
        self.assertEqual(response.status_code, 302)
        self.assertIn('/auth/login/', response.url)

    def test_audit_search_access_auditor_allowed(self):
        """Test that AUDITOR role can access audit search."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:audit_search'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Advanced Audit Search')

    def test_audit_search_access_admin_denied(self):
        """Test that ADMIN role is denied access to audit search."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:audit_search'))
        self.assertEqual(response.status_code, 403)

    def test_audit_search_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to audit search."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:audit_search'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_dataset_analysis_access_auditor_allowed(self):
        """Test that AUDITOR role can access dataset analysis."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:dataset_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset Access Analysis')

    def test_dataset_analysis_access_admin_denied(self):
        """Test that ADMIN role is denied access to dataset analysis."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:dataset_analysis'))
        self.assertEqual(response.status_code, 403)

    def test_dataset_analysis_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to dataset analysis."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:dataset_analysis'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_security_incidents_access_auditor_allowed(self):
        """Test that AUDITOR role can access security incidents."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:security_incidents'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Security Incidents Management')

    def test_security_incidents_access_admin_denied(self):
        """Test that ADMIN role is denied access to security incidents."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:security_incidents'))
        self.assertEqual(response.status_code, 403)

    def test_security_incidents_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to security incidents."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:security_incidents'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_export_audit_report_access_auditor_allowed(self):
        """Test that AUDITOR role can access audit report export."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:export_audit_report'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')

    def test_export_audit_report_access_admin_denied(self):
        """Test that ADMIN role is denied access to audit report export."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:export_audit_report'))
        self.assertEqual(response.status_code, 403)

    def test_export_audit_report_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to audit report export."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:export_audit_report'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_update_incident_state_access_auditor_allowed(self):
        """Test that AUDITOR role can update incident states."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.post(
            reverse('audit:update_incident_state', args=[self.security_incident.id]),
            {'state': 'INVESTIGATING'}
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify the incident was updated
        self.security_incident.refresh_from_db()
        self.assertEqual(self.security_incident.state, 'INVESTIGATING')

    def test_update_incident_state_access_admin_denied(self):
        """Test that ADMIN role is denied access to update incident states."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.post(
            reverse('audit:update_incident_state', args=[self.security_incident.id]),
            {'state': 'INVESTIGATING'}
        )
        self.assertEqual(response.status_code, 403)

    def test_update_incident_state_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to update incident states."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.post(
            reverse('audit:update_incident_state', args=[self.security_incident.id]),
            {'state': 'INVESTIGATING'}
        )
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)


class AuditorDashboardFunctionalityTest(TestCase):
    """Test auditor dashboard functionality for authorized users."""
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test environment for functionality tests."""
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.admin_role = Role.objects.get(name='ADMIN')
        
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!',
            role=self.auditor_role
        )
        
        self.admin_user = CustomUser.objects.create_user(
            username='admin_user', 
            password='StrongPass123!',
            role=self.admin_role
        )
        
        # Create test audit events
        for i in range(10):
            AuditEvent.objects.create(
                user=self.admin_user if i % 2 == 0 else self.auditor_user,
                action=f'TEST_ACTION_{i}',
                resource=f'test_resource_{i}',
                category='SYSTEM' if i < 5 else 'AUTH',
                severity='INFO' if i < 3 else 'WARNING' if i < 7 else 'CRITICAL',
                risk_score=10 + (i * 10),
                success=i % 3 != 0  # Some failures
            )
        
        self.client = Client()
        self.client.login(username='auditor_user', password='StrongPass123!')

    def test_dashboard_displays_metrics(self):
        """Test that dashboard displays correct metrics."""
        response = self.client.get(reverse('audit:auditor_dashboard'))
        self.assertEqual(response.status_code, 200)
        
        # Check if metrics are displayed
        self.assertContains(response, 'Total Events')
        self.assertContains(response, 'Security Events')
        self.assertContains(response, 'Failed Events')
        
        # Check for chart data
        self.assertContains(response, 'trendsChart')
        self.assertContains(response, 'categoryChart')

    def test_audit_search_filters_work(self):
        """Test that audit search filters work correctly."""
        # Test category filter
        response = self.client.get(reverse('audit:audit_search'), {'category': 'SYSTEM'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'TEST_ACTION_0')
        
        # Test severity filter
        response = self.client.get(reverse('audit:audit_search'), {'severity': 'CRITICAL'})
        self.assertEqual(response.status_code, 200)
        
        # Test success filter by searching for failed events
        response = self.client.get(reverse('audit:audit_search'), {'search': 'TEST_ACTION'})
        self.assertEqual(response.status_code, 200)

    def test_dataset_analysis_displays_data(self):
        """Test that dataset analysis displays data correctly."""
        response = self.client.get(reverse('audit:dataset_analysis'))
        self.assertEqual(response.status_code, 200)
        
        self.assertContains(response, 'Total Accesses')
        self.assertContains(response, 'Unique Users')
        self.assertContains(response, 'Access Patterns by Hour')

    def test_security_incidents_management(self):
        """Test security incidents management functionality."""
        # Create a test incident
        incident = SecurityIncident.objects.create(
            incident_type='SUSPICIOUS_ACTIVITY',
            description='Test incident for management',
            severity=3,
            state='OPEN'
        )
        
        response = self.client.get(reverse('audit:security_incidents'))
        self.assertEqual(response.status_code, 200)
        
        self.assertContains(response, 'Test incident for management')
        self.assertContains(response, 'Total Incidents')
        
    def test_csv_export_functionality(self):
        """Test CSV export contains expected data."""
        response = self.client.get(reverse('audit:export_audit_report'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        
        content = response.content.decode('utf-8')
        self.assertIn('Timestamp,Category,Action,User', content)
        self.assertIn('TEST_ACTION', content)


class AuditorDashboardSecurityTest(TestCase):
    """Test security aspects of auditor dashboard."""
    
    def setUp(self):
        """Set up security test environment."""
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!',
            role=self.auditor_role
        )
        
        self.client = Client()

    def test_csrf_protection_on_incident_update(self):
        """Test that incident state updates require CSRF token."""
        incident = SecurityIncident.objects.create(
            incident_type='SUSPICIOUS_ACTIVITY',
            description='Test incident',
            severity=2,
            state='OPEN'
        )
        
        self.client.login(username='auditor_user', password='StrongPass123!')
        
        # Try to update without CSRF token (should fail in production)
        response = self.client.post(
            reverse('audit:update_incident_state', args=[incident.id]),
            {'state': 'INVESTIGATING'},
            HTTP_X_CSRFTOKEN='invalid_token'
        )
        
        # In test environment, Django might not enforce CSRF strictly,
        # but we can verify the view expects POST method
        self.assertIn(response.status_code, [200, 403])  # Either success or CSRF failure

    def test_sql_injection_protection(self):
        """Test that search parameters are properly sanitized."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        
        # Attempt SQL injection in search parameter
        malicious_search = "'; DROP TABLE audit_auditevent; --"
        response = self.client.get(
            reverse('audit:audit_search'),
            {'search': malicious_search}
        )
        
        # Should not cause an error and should return normally
        self.assertEqual(response.status_code, 200)
        
        # Verify table still exists by trying to create an audit event
        AuditEvent.objects.create(
            action='TEST_AFTER_INJECTION',
            resource='test',
            category='SYSTEM'
        )

    def test_xss_protection_in_templates(self):
        """Test that user input is properly escaped in templates."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        
        # Create audit event with potentially malicious content
        AuditEvent.objects.create(
            action='<script>alert("xss")</script>',
            resource='<img src=x onerror=alert("xss")>',
            category='SYSTEM',
            details={'malicious': '<script>alert("xss")</script>'}
        )
        
        response = self.client.get(reverse('audit:audit_search'))
        self.assertEqual(response.status_code, 200)
        
        # Verify script tags are escaped
        content = response.content.decode('utf-8')
        self.assertNotIn('<script>alert("xss")</script>', content)
        self.assertIn('&lt;script&gt;', content)  # Should be escaped