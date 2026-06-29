from django.contrib.auth import get_user_model
from django.test import TestCase, Client, override_settings
from django.utils import timezone


User = get_user_model()


@override_settings(SESSION_IDLE_TIMEOUT=1)
class SessionTimeoutMiddlewareTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(
            username='idleuser', password='StrongPass123!'
        )

    def test_session_times_out(self) -> None:
        self.client.force_login(self.user)
        session = self.client.session
        session['last_activity_ts'] = int(timezone.now().timestamp()) - 5
        session.save()
        # Use the auth login URL instead of admin login
        self.client.get('/auth/login/')
        # The idle session must be terminated by the middleware: the user is
        # logged out (auth key removed from the session), regardless of how the
        # login URL itself responds (200/302/405 across configs).
        self.assertNotIn('_auth_user_id', self.client.session)

