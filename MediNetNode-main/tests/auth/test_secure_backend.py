from django.contrib.auth import authenticate
from django.test import TestCase
from django.utils import timezone

from users.models import CustomUser, Role


class SecureBackendTests(TestCase):
    def setUp(self) -> None:
        self.role = Role.objects.get(name='ADMIN')
        self.user = CustomUser.objects.create_user(
            username='john', password='StrongPass123!', role=self.role
        )

    def test_login_success_creates_activity(self) -> None:
        user = authenticate(username='john', password='StrongPass123!')
        self.assertIsNotNone(user)
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_active_session)
        self.assertIsNotNone(self.user.last_activity)

    def test_account_locks_after_failed_attempts(self) -> None:
        for _ in range(5):
            self.assertIsNone(authenticate(username='john', password='WrongPass'))
        self.user.refresh_from_db()
        self.assertIsNotNone(self.user.account_locked_until)
        self.assertGreater(self.user.account_locked_until, timezone.now())



