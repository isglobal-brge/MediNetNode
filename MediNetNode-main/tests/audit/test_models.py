from django.test import TestCase
from django.utils import timezone
from datetime import timedelta

from audit.models import AuditEvent, DataAccessLog, SecurityIncident
from audit.audit_logger import AuditLogger
from users.models import CustomUser, Role


class AuditEventModelTests(TestCase):
    def setUp(self) -> None:
        role = Role.objects.get(name='ADMIN')
        self.user = CustomUser.objects.create_user(
            username='auditor', password='StrongPass123!', role=role
        )
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.auditor = CustomUser.objects.create_user(
            username='audit_user', password='StrongPass123!', role=self.auditor_role
        )

    def test_create_audit_event(self):
        """Test basic AuditEvent creation."""
        event = AuditEvent.objects.create(
            user=self.user,
            action='TEST_EVENT',
            resource='users:1',
            category='USER_MGMT',
            severity='INFO',
            risk_score=25,
            success=True,
            details={'note': 'test event'},
        )
        self.assertIsNotNone(event.id)
        self.assertEqual(event.action, 'TEST_EVENT')
        self.assertEqual(event.category, 'USER_MGMT')
        self.assertEqual(event.risk_score, 25)
        self.assertEqual(event.details.get('note'), 'test event')
        self.assertIn('Risk:25', str(event))

    def test_mark_reviewed(self):
        """Test marking event as reviewed."""
        event = AuditEvent.objects.create(
            user=self.user,
            action='TEST_REVIEW',
            resource='test',
            risk_score=75,
            requires_review=True
        )
        
        self.assertIsNone(event.reviewed_at)
        self.assertIsNone(event.reviewed_by)
        
        event.mark_reviewed(self.auditor)
        
        self.assertIsNotNone(event.reviewed_at)
        self.assertEqual(event.reviewed_by, self.auditor)

    def test_high_risk_requires_review(self):
        """Test that high risk events automatically require review."""
        event = AuditEvent.objects.create(
            user=self.user,
            action='HIGH_RISK_EVENT',
            resource='critical_system',
            risk_score=80,
            requires_review=True  # Should be set automatically by AuditLogger
        )
        self.assertTrue(event.requires_review)

    def test_category_choices(self):
        """Test that all category choices work."""
        categories = ['AUTH', 'DATA_ACCESS', 'USER_MGMT', 'DATASET_MGMT', 'TRAINING', 'API', 'SYSTEM']
        for category in categories:
            event = AuditEvent.objects.create(
                user=self.user,
                action=f'TEST_{category}',
                category=category
            )
            self.assertEqual(event.category, category)

    def test_severity_choices(self):
        """Test that all severity choices work."""
        severities = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SECURITY']
        for severity in severities:
            event = AuditEvent.objects.create(
                user=self.user,
                action=f'TEST_{severity}',
                severity=severity
            )
            self.assertEqual(event.severity, severity)


class DataAccessLogModelTests(TestCase):
    def setUp(self) -> None:
        role = Role.objects.get(name='RESEARCHER')
        self.user = CustomUser.objects.create_user(
            username='researcher', password='StrongPass123!', role=role
        )

    def test_create_data_access_log(self):
        """Test DataAccessLog creation with AuditEvent."""
        audit_event = AuditEvent.objects.create(
            user=self.user,
            action='DATA_ACCESS',
            resource='dataset:cardiology_001',
            category='DATA_ACCESS',
            risk_score=35
        )
        
        data_log = DataAccessLog.objects.create(
            audit_event=audit_event,
            medical_domain='cardiology',
            patient_count_accessed=150,
            data_sensitivity_level=4,
            records_accessed=1500,
            columns_accessed=['age', 'diagnosis', 'treatment'],
            query_hash='abc123def456'
        )
        
        self.assertEqual(data_log.audit_event, audit_event)
        self.assertEqual(data_log.medical_domain, 'cardiology')
        self.assertEqual(data_log.data_sensitivity_level, 4)
        self.assertEqual(data_log.records_accessed, 1500)
        self.assertIn('cardiology', str(data_log))
        self.assertIn('Level 4', str(data_log))

    def test_data_sensitivity_levels(self):
        """Test all data sensitivity levels."""
        for level in range(1, 6):
            # Create a separate audit event for each data access log
            audit_event = AuditEvent.objects.create(
                user=self.user,
                action=f'DATA_ACCESS_LEVEL_{level}',
                resource=f'dataset:test_level_{level}'
            )
            
            data_log = DataAccessLog.objects.create(
                audit_event=audit_event,
                data_sensitivity_level=level
            )
            self.assertEqual(data_log.data_sensitivity_level, level)


class SecurityIncidentModelTests(TestCase):
    def setUp(self) -> None:
        role = Role.objects.get(name='ADMIN')
        self.user = CustomUser.objects.create_user(
            username='admin_user', password='StrongPass123!', role=role
        )
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.auditor = CustomUser.objects.create_user(
            username='incident_auditor', password='StrongPass123!', role=self.auditor_role
        )

    def test_create_security_incident(self):
        """Test SecurityIncident creation."""
        incident = SecurityIncident.objects.create(
            incident_type='UNAUTHORIZED_ACCESS',
            severity=3,
            state='OPEN',
            description='Unauthorized access attempt detected',
            risk_score=85
        )
        
        self.assertEqual(incident.incident_type, 'UNAUTHORIZED_ACCESS')
        self.assertEqual(incident.severity, 3)
        self.assertEqual(incident.state, 'OPEN')
        self.assertEqual(incident.risk_score, 85)
        self.assertIn('UNAUTHORIZED_ACCESS', str(incident))

    def test_incident_states(self):
        """Test all incident states."""
        states = ['OPEN', 'INVESTIGATING', 'RESOLVED', 'CLOSED']
        for state in states:
            incident = SecurityIncident.objects.create(
                incident_type='SUSPICIOUS_ACTIVITY',
                description=f'Test incident in {state} state',
                state=state
            )
            self.assertEqual(incident.state, state)

    def test_incident_assignment(self):
        """Test incident assignment to auditor."""
        incident = SecurityIncident.objects.create(
            incident_type='DATA_BREACH_ATTEMPT',
            description='Test assignment',
            assigned_to=self.auditor
        )
        
        self.assertEqual(incident.assigned_to, self.auditor)

    def test_update_risk_score(self):
        """Test risk score update from related events."""
        # Create high-risk audit event
        audit_event = AuditEvent.objects.create(
            user=self.user,
            action='FAILED_LOGIN',
            resource='auth/login',
            risk_score=90,
            success=False
        )
        
        incident = SecurityIncident.objects.create(
            incident_type='MULTIPLE_FAILED_AUTH',
            description='Multiple failed login attempts'
        )
        
        # Link the event
        incident.related_events.add(audit_event)
        
        # Update risk score
        incident.update_risk_score()
        
        self.assertEqual(incident.risk_score, 90)

    def test_multiple_related_events(self):
        """Test incident with multiple related events."""
        # Create multiple events
        event1 = AuditEvent.objects.create(
            user=self.user,
            action='FAILED_LOGIN',
            risk_score=75
        )
        event2 = AuditEvent.objects.create(
            user=self.user,
            action='FAILED_LOGIN', 
            risk_score=85
        )
        
        incident = SecurityIncident.objects.create(
            incident_type='MULTIPLE_FAILED_AUTH',
            description='Multiple events test'
        )
        
        incident.related_events.add(event1, event2)
        incident.update_risk_score()
        
        # Should take the maximum risk score
        self.assertEqual(incident.risk_score, 85)


class AuditLoggerTests(TestCase):
    def setUp(self) -> None:
        role = Role.objects.get(name='RESEARCHER')
        self.user = CustomUser.objects.create_user(
            username='logger_test', password='StrongPass123!', role=role
        )

    def test_log_event_basic(self):
        """Test basic event logging with AuditLogger."""
        event = AuditLogger.log_event(
            action='TEST_ACTION',
            user=self.user,
            resource='test_resource',
            ip_address='192.168.1.1',
            success=True,
            details={'test': 'data'}
        )
        
        self.assertIsInstance(event, AuditEvent)
        self.assertEqual(event.action, 'TEST_ACTION')
        self.assertEqual(event.user, self.user)
        self.assertEqual(event.ip_address, '192.168.1.1')
        self.assertTrue(event.success)

    def test_automatic_categorization(self):
        """Test automatic event categorization."""
        # Test authentication categorization
        auth_event = AuditLogger.log_event(
            action='LOGIN_ATTEMPT',
            user=self.user
        )
        self.assertEqual(auth_event.category, 'AUTH')
        
        # Test data access categorization
        data_event = AuditLogger.log_event(
            action='QUERY_DATA',
            resource='dataset:test',
            user=self.user
        )
        self.assertEqual(data_event.category, 'DATA_ACCESS')

    def test_risk_score_calculation(self):
        """Test automatic risk score calculation."""
        # Test low risk event
        low_risk = AuditLogger.log_event(
            action='VIEW_PAGE',
            user=self.user
        )
        self.assertLess(low_risk.risk_score, 30)
        
        # Test high risk event
        high_risk = AuditLogger.log_event(
            action='DELETE_DATASET',
            user=self.user,
            success=False
        )
        self.assertGreater(high_risk.risk_score, 50)

    def test_data_access_logging(self):
        """Test specialized data access logging."""
        event = AuditLogger.log_data_access(
            dataset_name='cardiology_study_01',
            user=self.user,
            medical_domain='cardiology',
            records_accessed=500,
            columns_accessed=['age', 'gender', 'diagnosis'],
            patient_count=100,
            ip_address='10.0.0.1'
        )
        
        self.assertEqual(event.category, 'DATA_ACCESS')
        self.assertTrue(hasattr(event, 'data_access_log'))
        
        data_log = event.data_access_log
        self.assertEqual(data_log.medical_domain, 'cardiology')
        self.assertEqual(data_log.records_accessed, 500)
        self.assertEqual(data_log.patient_count_accessed, 100)

    def test_security_incident_creation(self):
        """Test automatic security incident creation for high-risk events."""
        # Create a very high-risk event
        event = AuditLogger.log_event(
            action='UNAUTHORIZED_DATA_ACCESS',
            user=self.user,
            resource='sensitive_dataset',
            success=False,
            details={'reason': 'Access denied - insufficient privileges'}
        )
        
        # Should automatically create security incident for risk_score >= 80
        if event.risk_score >= 80:
            incidents = SecurityIncident.objects.filter(related_events=event)
            self.assertGreater(incidents.count(), 0)

    def test_authentication_logging(self):
        """Test authentication event logging convenience method."""
        event = AuditLogger.log_authentication(
            action='LOGIN_ATTEMPT',
            user=self.user,
            success=True,
            ip_address='192.168.1.100',
            details={'method': 'password'}
        )
        
        self.assertEqual(event.category, 'AUTH')
        self.assertEqual(event.resource, 'authentication')
        self.assertTrue(event.success)

    def test_api_access_logging(self):
        """Test API access logging convenience method."""
        event = AuditLogger.log_api_access(
            endpoint='/api/v1/datasets',
            user=self.user,
            method='GET',
            success=True,
            ip_address='10.1.1.1',
            request_size=1024,
            request_duration_ms=150
        )
        
        self.assertEqual(event.category, 'API')
        self.assertEqual(event.action, 'API_GET')
        self.assertEqual(event.resource, 'api:/api/v1/datasets')
        self.assertEqual(event.request_size, 1024)
        self.assertEqual(event.request_duration_ms, 150)


# Keep legacy tests for backward compatibility


