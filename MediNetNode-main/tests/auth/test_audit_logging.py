from django.contrib.auth import authenticate
from django.test import TestCase

from audit.models import AuditEvent
from users.models import CustomUser, Role


class AuditLoggingTests(TestCase):
    def setUp(self) -> None:
        role = Role.objects.get(name='ADMIN')
        self.user = CustomUser.objects.create_user(
            username='alice', password='StrongPass123!', role=role
        )

    def test_login_success_creates_audit_log(self) -> None:
        user = authenticate(username='alice', password='StrongPass123!')
        self.assertIsNotNone(user)
        # Check for authentication event in the new system
        exists = AuditEvent.objects.filter(
            action='LOGIN_SUCCESS', 
            category='AUTH',
            user=self.user
        ).exists()
        self.assertTrue(exists)

    def test_login_fail_creates_audit_log(self) -> None:
        user = authenticate(username='alice', password='WrongPass')
        self.assertIsNone(user)
        # Check for failed authentication event in the new system
        exists = AuditEvent.objects.filter(
            action='LOGIN_FAIL',
            category='AUTH', 
            user=self.user,
            success=False
        ).exists()
        self.assertTrue(exists)


