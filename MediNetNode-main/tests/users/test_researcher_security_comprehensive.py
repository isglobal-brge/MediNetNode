"""
Comprehensive security tests for RESEARCHER web access blocking.
Tests all known attack vectors that malicious users might attempt.
"""
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from users.models import Role

User = get_user_model()


class ResearcherSecurityComprehensiveTests(TestCase):
    """Test comprehensive security blocking for RESEARCHER users."""

    def setUp(self):
        """Set up test users and client."""
        self.client = Client()

        # Get RESEARCHER role (already created in conftest.py)
        self.researcher_role = Role.objects.get(name='RESEARCHER')

        # Create RESEARCHER user
        self.researcher_user = User.objects.create_user(
            username='researcher_test',
            password='TestPass123!',
            email='researcher@test.com',
            role=self.researcher_role
        )

        # Login as researcher
        self.client.login(username='researcher_test', password='TestPass123!')

    def test_django_admin_blocked(self):
        """CRITICAL: Django admin must be completely blocked."""
        attack_urls = [
            '/django-admin/',
            '/django-admin/auth/',
            '/django-admin/users/',
            '/DJANGO-ADMIN/',  # Case variation
            '/django-admin/../info/researcher/',  # Path traversal attempt
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_admin_dashboard_blocked(self):
        """CRITICAL: Admin dashboard must be blocked."""
        attack_urls = [
            '/admin/',
            '/Admin/',  # Case variation
            '/admin/users/',
            '/admin/../info/researcher/',  # Path traversal
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_user_management_blocked(self):
        """CRITICAL: All user management endpoints blocked."""
        attack_urls = [
            '/users/',
            '/users/create/',
            '/users/1/',
            '/users/1/edit/',
            '/users/1/delete/',
            '/users/export/',  # CSV export
            '/USERS/',  # Case variation
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_system_logs_blocked(self):
        """CRITICAL: System logs must be blocked."""
        attack_urls = [
            '/system/logs/',
            '/System/Logs/',  # Case variation
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_path_traversal_attacks_blocked(self):
        """CRITICAL: Path traversal attacks must be blocked."""
        attack_urls = [
            '/info/researcher/../admin/',
            '/info/researcher/../../django-admin/',
            '/info/researcher/../users/',
            '/info/researcher/../system/logs/',
            '/api/v1/../../admin/',
            '/api/v1/../users/',
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_url_encoded_attacks_blocked(self):
        """CRITICAL: URL encoded path traversal blocked."""
        attack_urls = [
            '/info/researcher/%2e%2e/admin/',  # ../ encoded
            '/info/researcher/%2e%2e%2fadmin%2f',  # ../ and / encoded
            '/info%2fresearcher%2f%2e%2e%2fadmin%2f',  # Fully encoded
            '/%69%6e%66%6f/%72%65%73%65%61%72%63%68%65%72/../admin/',  # Mixed encoding
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_api_documentation_blocked(self):
        """CRITICAL: API documentation should be blocked."""
        attack_urls = [
            '/api/docs/',
            '/api/docs/swagger/',
            '/api/docs/redoc/',
            '/api/docs/swagger.json',
            '/API/DOCS/',  # Case variation
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_static_admin_files_blocked(self):
        """CRITICAL: Django admin static files blocked."""
        attack_urls = [
            '/static/admin/css/base.css',
            '/static/admin/js/admin.js',
            '/static/debug_toolbar/js/toolbar.js',
            '/STATIC/ADMIN/css/base.css',  # Case variation
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_case_sensitivity_attacks_blocked(self):
        """CRITICAL: Case variations must be blocked."""
        attack_urls = [
            '/ADMIN/',
            '/Admin/',
            '/USERS/',
            '/Users/',
            '/DJANGO-ADMIN/',
            '/Django-Admin/',
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_root_redirect_blocked(self):
        """CRITICAL: Root redirect to admin must be blocked."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_allowed_paths_work(self):
        """Verify legitimate paths are still accessible."""
        allowed_urls = [
            '/info/researcher/',
            '/auth/logout/',
        ]

        for url in allowed_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                # Should NOT redirect to researcher_info (would indicate blocking)
                if response.status_code == 302:
                    self.assertNotIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_api_endpoints_allowed(self):
        """Verify API endpoints are accessible."""
        # Note: These might return 404 if endpoints don't exist, but should NOT redirect
        api_urls = [
            '/api/v1/ping',
            '/api/v1/get-data-info',
        ]

        for url in api_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                # Should NOT redirect to researcher_info (blocking would cause redirect)
                if response.status_code == 302:
                    self.assertNotIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_double_slash_attacks_blocked(self):
        """CRITICAL: Double slash bypass attempts blocked."""
        attack_urls = [
            '//admin/',
            '//django-admin/',
            '//users/',
            '///admin/',
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))

    def test_query_parameter_bypasses_blocked(self):
        """CRITICAL: Query parameter bypass attempts blocked."""
        attack_urls = [
            '/admin/?next=/info/researcher/',
            '/users/?redirect=/info/researcher/',
            '/django-admin/?next=/api/v1/ping',
        ]

        for url in attack_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn('/info/researcher/', response.url or response.get('Location', ''))