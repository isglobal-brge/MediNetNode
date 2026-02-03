"""
Tests for Session Fixation protection in authentication views.

Validates that session IDs are regenerated on login to prevent session fixation attacks.
"""
import pytest
from django.test import Client
from django.contrib.auth import get_user_model
from users.models import Role

User = get_user_model()


@pytest.fixture
def test_user(db):
    """Create a test user for authentication tests."""
    admin_role = Role.objects.get(name='ADMIN')
    user = User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='TestPass123!',
        role=admin_role
    )
    return user


@pytest.mark.django_db
class TestSessionFixationProtection:
    """Test session ID regeneration on login."""

    def test_login_view_regenerates_session_id(self, test_user):
        """Test that login_page (POST) regenerates session ID after successful authentication."""
        client = Client()

        # Get initial session ID by making a request
        client.get('/auth/login/')
        initial_session_key = client.session.session_key

        # Perform login via HTML form (which calls login_page with POST)
        response = client.post('/auth/login/', {
            'username': 'testuser',
            'password': 'TestPass123!'
        })

        # Should redirect after successful login
        assert response.status_code == 302

        # Session ID should be different after login
        new_session_key = client.session.session_key
        assert new_session_key is not None
        assert new_session_key != initial_session_key

    def test_login_page_regenerates_session_id(self, test_user):
        """Test that login_page regenerates session ID after successful authentication."""
        client = Client()

        # Get initial session ID
        client.get('/auth/login/')
        initial_session_key = client.session.session_key

        # Perform login via HTML form
        response = client.post('/auth/login/', {
            'username': 'testuser',
            'password': 'TestPass123!'
        })

        # Should redirect after successful login
        assert response.status_code == 302

        # Session ID should be different after login
        new_session_key = client.session.session_key
        assert new_session_key is not None
        assert new_session_key != initial_session_key

    def test_failed_login_keeps_session_id(self, test_user):
        """Test that failed login does not regenerate session ID."""
        client = Client()

        # Get initial session ID
        client.get('/auth/login/')
        initial_session_key = client.session.session_key

        # Attempt login with wrong password
        response = client.post('/auth/login/', {
            'username': 'testuser',
            'password': 'WrongPassword123!'
        })

        assert response.status_code == 400

        # Session ID should remain the same after failed login
        current_session_key = client.session.session_key
        assert current_session_key == initial_session_key

    def test_session_data_preserved_after_regeneration(self, test_user):
        """Test that session data is preserved when session ID is regenerated."""
        client = Client()

        # Set some session data before login
        session = client.session
        session['test_data'] = 'preserved_value'
        session.save()

        initial_session_key = client.session.session_key

        # Perform login
        response = client.post('/auth/login/', {
            'username': 'testuser',
            'password': 'TestPass123!'
        })

        assert response.status_code == 302

        # Session ID should change
        new_session_key = client.session.session_key
        assert new_session_key != initial_session_key

        # But session data should be preserved
        assert client.session.get('test_data') == 'preserved_value'

        # User should be authenticated
        assert client.session.get('_auth_user_id') == str(test_user.id)

    def test_login_activity_timestamp_set(self, test_user):
        """Test that last_activity_ts is set after successful login."""
        client = Client()

        response = client.post('/auth/login/', {
            'username': 'testuser',
            'password': 'TestPass123!'
        })

        assert response.status_code == 302

        # Verify last_activity_ts is set
        assert 'last_activity_ts' in client.session
        assert isinstance(client.session['last_activity_ts'], int)
        assert client.session['last_activity_ts'] > 0

