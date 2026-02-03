"""
Tests for Server header removal (version disclosure prevention).

Validates that Django version is not exposed in HTTP headers.
"""
import pytest
from django.test import Client
from django.contrib.auth import get_user_model
from users.models import Role, APIKey

User = get_user_model()


@pytest.fixture
def admin_user(db):
    """Create ADMIN user."""
    admin_role = Role.objects.get(name='ADMIN')
    user = User.objects.create_user(
        username='admin_test',
        email='admin@test.com',
        password='TestPass123!',
        role=admin_role
    )
    return user


@pytest.fixture
def researcher_user(db):
    """Create RESEARCHER user."""
    researcher_role = Role.objects.get(name='RESEARCHER')
    user = User.objects.create_user(
        username='researcher_test',
        email='researcher@test.com',
        password='TestPass123!',
        role=researcher_role
    )
    return user


@pytest.fixture
def api_key(researcher_user):
    """Create API key for researcher."""
    raw_key = APIKey.generate_api_key()
    api_key_obj = APIKey(
        user=researcher_user,
        name='test_key',
        ip_whitelist=['127.0.0.1']
    )
    api_key_obj.set_key(raw_key)
    api_key_obj.save()
    return raw_key


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestServerHeaderRemoval:
    """Test that Server header is removed to prevent version disclosure."""

    def test_server_header_removed_from_web_pages(self, admin_user):
        """Test that Server header is not present on web pages."""
        client = Client()
        client.force_login(admin_user)

        response = client.get('/users/')

        # Server header should not be present
        assert 'Server' not in response, "Server header should be removed to prevent version disclosure"

    def test_server_header_removed_from_api_endpoints(self, api_key):
        """Test that Server header is not present on API endpoints."""
        client = Client()

        response = client.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        # Server header should not be present
        assert 'Server' not in response, "Server header should be removed from API responses"

    def test_server_header_removed_from_error_pages(self):
        """Test that Server header is not present on error pages."""
        client = Client()

        # Test 404 error
        response = client.get('/nonexistent-page/')
        assert 'Server' not in response, "Server header should be removed from 404 responses"

        # Test 403 error (unauthenticated access to protected page)
        response = client.get('/users/')
        assert 'Server' not in response, "Server header should be removed from 403 responses"

    def test_server_header_removed_from_static_files(self):
        """Test that Server header is not present on static file responses."""
        client = Client()

        # Test static file (if exists)
        response = client.get('/static/css/style.css')
        # May return 404 if static files not collected, but header should still be removed
        assert 'Server' not in response, "Server header should be removed from static file responses"

    def test_no_django_version_in_response_headers(self, admin_user):
        """Test that no response headers contain Django version information."""
        client = Client()
        client.force_login(admin_user)

        response = client.get('/users/')

        # Check all headers for Django version patterns
        django_patterns = ['django', 'wsgi', 'python']
        for header_name, header_value in response.items():
            header_value_lower = str(header_value).lower()
            for pattern in django_patterns:
                # Allow 'django' in CSRF token or session cookie names, but not in version strings
                if pattern in header_value_lower and any(version_indicator in header_value_lower for version_indicator in ['/', 'version', 'v3', 'v4', 'v5']):
                    pytest.fail(f"Header '{header_name}' contains potential version information: {header_value}")

    def test_server_header_removed_on_post_requests(self, admin_user):
        """Test that Server header is removed on POST requests."""
        client = Client()
        client.force_login(admin_user)

        # Test POST to user creation (will fail validation, but that's OK)
        response = client.post('/users/create/', {
            'username': 'test',
            'email': 'test@test.com'
        })

        assert 'Server' not in response, "Server header should be removed from POST responses"

    def test_server_header_removed_on_api_post_requests(self, api_key):
        """Test that Server header is removed on API POST requests."""
        client = Client()

        # Test POST to API endpoint
        response = client.post(
            '/api/v1/start-client',
            data='{"invalid": "json"}',
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        assert 'Server' not in response, "Server header should be removed from API POST responses"
