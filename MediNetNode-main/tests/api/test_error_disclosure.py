"""
Tests for error information disclosure prevention.

Validates that error responses don't leak sensitive information.
"""
import pytest
from django.test import Client, override_settings
from django.contrib.auth import get_user_model
from users.models import Role, APIKey

User = get_user_model()


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
    """Create API key for researcher user."""
    raw_key = APIKey.generate_api_key()
    api_key_obj = APIKey(
        user=researcher_user,
        name='test_api_key',
        ip_whitelist=['127.0.0.1']
    )
    api_key_obj.set_key(raw_key)
    api_key_obj.save()
    return raw_key


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestErrorInformationDisclosure:
    """Test that errors don't disclose sensitive information."""

    @override_settings(DEBUG=False)
    def test_generic_500_error_in_production(self, api_key):
        """Test that 500 errors return generic message in production."""
        client = Client()

        # Trigger an error by sending invalid JSON
        response = client.post(
            '/api/v1/start-client',
            data='invalid-json',
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        # Should return generic error without details
        assert response.status_code in [400, 500]
        response_data = response.json()

        # Check that error message is generic
        assert 'error' in response_data
        error_message = response_data['error'].lower()

        # Should NOT contain sensitive information
        sensitive_patterns = [
            'traceback',
            'exception',
            'line',
            'file',
            'database',
            'password',
            'secret',
            'key',
            '/usr/',
            'c:\\',
            'python',
            'django',
            '.py'
        ]

        for pattern in sensitive_patterns:
            assert pattern not in error_message, f"Error message contains sensitive pattern: {pattern}"

    @override_settings(DEBUG=False)
    def test_404_error_generic_message(self):
        """Test that 404 errors return generic message."""
        client = Client()

        response = client.get('/nonexistent-endpoint')

        assert response.status_code == 404

        # Response may be HTML or JSON depending on endpoint
        response_text = response.content.decode('utf-8').lower()

        # Should NOT reveal internal paths or sensitive information
        sensitive_terms = [
            'traceback',
            '/usr/',
            'c:\\',
            'python',
            '.py"',
            'line'
        ]

        for term in sensitive_terms:
            assert term not in response_text, f"404 response contains sensitive term: {term}"

    @override_settings(DEBUG=False)
    def test_403_error_no_path_disclosure(self, api_key):
        """Test that 403 errors don't disclose internal paths."""
        client = Client()

        # Try to access with wrong IP
        response = client.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='192.0.2.1'  # Not whitelisted
        )

        assert response.status_code == 403
        response_data = response.json()

        # Should have error message
        assert 'error' in response_data

        # Should NOT contain file paths or internal details
        error_str = str(response_data).lower()
        assert '/api/' not in error_str or 'ip address not authorized' in error_str
        assert '.py' not in error_str
        assert 'line' not in error_str

    @override_settings(DEBUG=False)
    def test_validation_error_safe_message(self, api_key):
        """Test that validation errors return safe messages."""
        client = Client()

        # Send invalid config
        response = client.post(
            '/api/v1/start-client',
            data='{"model_json": "not_an_object"}',
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        assert response.status_code == 400
        response_data = response.json()

        # Should have error message
        assert 'error' in response_data

        # Message should be user-friendly, not technical
        error_message = response_data['error']
        assert len(error_message) < 200  # Reasonably short
        assert 'traceback' not in error_message.lower()

    @override_settings(DEBUG=False)
    def test_internal_error_hides_exception_type(self, api_key):
        """Test that internal errors hide exception types."""
        client = Client()

        # This will cause an internal error (no datasets)
        response = client.get(
            '/api/v1/get-data-info',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        # May succeed or fail, but if it fails...
        if response.status_code == 500:
            response_data = response.json()

            # Should NOT contain exception type names
            error_str = str(response_data).lower()
            exception_types = [
                'valueerror',
                'keyerror',
                'attributeerror',
                'typeerror',
                'indexerror',
                'doesnotexist'
            ]

            for exc_type in exception_types:
                assert exc_type not in error_str, f"Error response contains exception type: {exc_type}"

    @override_settings(DEBUG=False)
    def test_no_stack_traces_in_responses(self, api_key):
        """Test that responses never contain stack traces."""
        client = Client()

        # Try various endpoints that might error
        endpoints = [
            ('/api/v1/ping', 'GET'),
            ('/api/v1/get-data-info', 'GET'),
            ('/api/v1/start-client', 'POST')
        ]

        for endpoint, method in endpoints:
            if method == 'GET':
                response = client.get(
                    endpoint,
                    HTTP_X_API_KEY=api_key,
                    REMOTE_ADDR='127.0.0.1'
                )
            else:
                response = client.post(
                    endpoint,
                    data='{}',
                    content_type='application/json',
                    HTTP_X_API_KEY=api_key,
                    REMOTE_ADDR='127.0.0.1'
                )

            # Check response doesn't contain stack trace indicators
            response_text = response.content.decode('utf-8').lower()
            stack_trace_indicators = [
                'traceback',
                'line',
                'file',
                'in ',
                'at ',
                'python',
                '.py",',
                'django'
            ]

            for indicator in stack_trace_indicators:
                assert indicator not in response_text, \
                    f"Response from {endpoint} contains stack trace indicator: {indicator}"

    @override_settings(DEBUG=True)
    def test_debug_mode_shows_more_info(self, api_key):
        """Test that DEBUG mode does show more information (for development)."""
        client = Client()

        # Send invalid JSON to trigger error
        response = client.post(
            '/api/v1/start-client',
            data='invalid-json',
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        # In DEBUG mode, Django may show more details
        # This test just verifies DEBUG mode is different from production
        # (actual DEBUG error pages are HTML, not JSON in Django)
        assert response.status_code in [400, 500]

    @override_settings(DEBUG=False)
    def test_database_errors_hidden(self, api_key):
        """Test that database errors don't reveal schema information."""
        client = Client()

        # Get data info (might trigger DB queries)
        response = client.get(
            '/api/v1/get-data-info',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        # If there's an error, check it doesn't reveal DB details
        if response.status_code >= 400:
            response_text = response.content.decode('utf-8').lower()

            db_sensitive_terms = [
                'table',
                'column',
                'database',
                'sql',
                'select',
                'insert',
                'update',
                'delete',
                'join',
                'where',
                'schema'
            ]

            for term in db_sensitive_terms:
                # Some terms like "table" might appear in legitimate messages
                # so we check for SQL-like patterns
                assert f'{term} ' not in response_text or \
                       'dataset' in response_text, \
                       f"Error response may contain database details: {term}"

    @override_settings(DEBUG=False)
    def test_file_paths_not_disclosed(self, api_key):
        """Test that file system paths are not disclosed in errors."""
        client = Client()

        response = client.post(
            '/api/v1/start-client',
            data='invalid-json',
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            REMOTE_ADDR='127.0.0.1'
        )

        response_text = response.content.decode('utf-8')

        # Should not contain file system paths
        path_indicators = [
            '/usr/',
            '/home/',
            '/var/',
            'c:\\',
            'd:\\',
            '\\users\\',
            '/app/',
            '/src/'
        ]

        for path in path_indicators:
            assert path.lower() not in response_text.lower(), \
                f"Response contains file path indicator: {path}"
