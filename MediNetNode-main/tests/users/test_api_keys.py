"""
Comprehensive tests for API key functionality.

Tests user creation with automatic API key generation for RESEARCHER users,
API key models, views, and security features.
"""
import json
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import datetime, timedelta
from users.models import Role, APIKey, APIRequest

User = get_user_model()


class APIKeyModelTests(TestCase):
    """Test APIKey model functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        self.admin_role = Role.objects.get(name='ADMIN')
        
        self.researcher_user = User.objects.create_user(
            username='researcher1',
            email='researcher@test.com',
            password='TestPass123!',
            role=self.researcher_role
        )
    
    def test_api_key_generation(self):
        """Test automatic API key generation."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name="Test API Key",
            ip_whitelist=['127.0.0.1', '192.168.1.0/24']
        )
        
        self.assertEqual(len(api_key.key), 64)
        self.assertTrue(api_key.is_active)
        self.assertIsNotNone(api_key.created_at)
        self.assertIsNone(api_key.expires_at)
    
    def test_api_key_uniqueness(self):
        """Test that API keys are unique."""
        api_key1 = APIKey.objects.create(
            user=self.researcher_user,
            name="Key 1"
        )
        api_key2 = APIKey.objects.create(
            user=self.researcher_user,
            name="Key 2"
        )
        
        self.assertNotEqual(api_key1.key, api_key2.key)
    
    def test_api_key_expiration(self):
        """Test API key expiration functionality."""
        # Create expired API key
        expired_key = APIKey.objects.create(
            user=self.researcher_user,
            name="Expired Key",
            expires_at=timezone.now() - timedelta(days=1)
        )
        
        # Create active API key
        active_key = APIKey.objects.create(
            user=self.researcher_user,
            name="Active Key",
            expires_at=timezone.now() + timedelta(days=30)
        )
        
        self.assertTrue(expired_key.is_expired())
        self.assertFalse(active_key.is_expired())
    
    def test_ip_whitelist_validation(self):
        """Test IP whitelist functionality."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name="IP Restricted Key",
            ip_whitelist=['192.168.1.100', '10.0.0.0/8']
        )
        
        # Test allowed IPs
        self.assertTrue(api_key.is_ip_allowed('192.168.1.100'))
        self.assertTrue(api_key.is_ip_allowed('10.1.2.3'))
        
        # Test blocked IPs
        self.assertFalse(api_key.is_ip_allowed('192.168.2.100'))
        self.assertFalse(api_key.is_ip_allowed('172.16.1.1'))
    
    def test_api_key_usage_tracking(self):
        """Test API key usage tracking."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name="Tracked Key"
        )
        
        self.assertIsNone(api_key.last_used_at)
        self.assertIsNone(api_key.last_used_ip)
        
        # Simulate usage
        api_key.update_last_used('192.168.1.100')
        
        self.assertIsNotNone(api_key.last_used_at)
        self.assertEqual(api_key.last_used_ip, '192.168.1.100')


class UserCreationWithAPIKeyTests(TestCase):
    """Test user creation with automatic API key generation."""
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        
        # Create roles
        self.admin_role = Role.objects.get(name='ADMIN')
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        self.auditor_role = Role.objects.get(name='AUDITOR')
        
        # Create admin user
        self.admin_user = User.objects.create_user(
            username='admin',
            email='admin@test.com',
            password='TestPass123!',
            role=self.admin_role,
            is_superuser=True
        )
        
        self.client.login(username='admin', password='TestPass123!')
    
    def test_researcher_user_creation_generates_api_key(self):
        """Test that RESEARCHER user creation automatically generates API key."""
        user_data = {
            'username': 'newresearcher',
            'email': 'newresearcher@test.com',
            'first_name': 'New',
            'last_name': 'Researcher',
            'password1': 'ComplexPass123!',
            'password2': 'ComplexPass123!',
            'role': self.researcher_role.id
        }
        
        response = self.client.post(reverse('create_user'), user_data)
        
        # Should redirect to success page
        self.assertEqual(response.status_code, 302)
        
        # Check user was created
        user = User.objects.get(username='newresearcher')
        self.assertEqual(user.role, self.researcher_role)
        
        # Check API key was generated
        api_keys = APIKey.objects.filter(user=user)
        self.assertEqual(api_keys.count(), 1)
        
        api_key = api_keys.first()
        self.assertEqual(api_key.name, "Auto-generated API Key")
        self.assertTrue(api_key.is_active)
        self.assertEqual(api_key.ip_whitelist, ['0.0.0.0/0'])
    
    def test_non_researcher_user_creation_no_api_key(self):
        """Test that non-RESEARCHER users don't get API keys."""
        user_data = {
            'username': 'newauditor',
            'email': 'newauditor@test.com',
            'first_name': 'New',
            'last_name': 'Auditor',
            'password1': 'ComplexPass123!',
            'password2': 'ComplexPass123!',
            'role': self.auditor_role.id
        }
        
        response = self.client.post(reverse('create_user'), user_data)
        
        # Should redirect to success page
        self.assertEqual(response.status_code, 302)
        
        # Check user was created
        user = User.objects.get(username='newauditor')
        self.assertEqual(user.role, self.auditor_role)
        
        # Check NO API key was generated
        api_keys = APIKey.objects.filter(user=user)
        self.assertEqual(api_keys.count(), 0)
    
    def test_user_created_success_view_with_api_key(self):
        """Test user creation success view displays API key for RESEARCHER."""
        # Create researcher user manually to simulate the process
        user = User.objects.create_user(
            username='testresearcher',
            email='test@example.com',
            password='TestPass123!',
            role=self.researcher_role
        )
        
        api_key = APIKey.objects.create(
            user=user,
            name="Auto-generated API Key",
            ip_whitelist=['0.0.0.0/0']
        )
        
        # Store user data in session (simulates create_user view)
        session = self.client.session
        session['new_user_data'] = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'role': user.role.name,
            'api_key': api_key.key,
            'api_key_created': api_key.created_at.isoformat()
        }
        session.save()
        
        response = self.client.get(reverse('user_created_success'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, user.username)
        self.assertContains(response, api_key.key)
        self.assertContains(response, 'API Key:')
        self.assertContains(response, 'Download Credentials')
    
    def test_user_created_success_view_without_api_key(self):
        """Test user creation success view for non-RESEARCHER users."""
        # Create auditor user manually
        user = User.objects.create_user(
            username='testauditor',
            email='auditor@example.com',
            password='TestPass123!',
            role=self.auditor_role
        )
        
        # Store user data in session (no API key)
        session = self.client.session
        session['new_user_data'] = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'role': user.role.name,
            'api_key': None,
            'api_key_created': None
        }
        session.save()
        
        response = self.client.get(reverse('user_created_success'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, user.username)
        self.assertNotContains(response, 'API Key:')
        self.assertNotContains(response, 'Download Credentials')
    
    def test_download_user_info_functionality(self):
        """Test download user info functionality."""
        # Create researcher with API key
        user = User.objects.create_user(
            username='downloadtest',
            email='download@test.com',
            password='TestPass123!',
            role=self.researcher_role
        )
        
        api_key = APIKey.objects.create(
            user=user,
            name="Auto-generated API Key"
        )
        
        # Store in session
        session = self.client.session
        session['new_user_data'] = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'role': user.role.name,
            'api_key': api_key.key,
            'api_key_created': api_key.created_at.isoformat()
        }
        session.save()
        
        response = self.client.get(reverse('download_user_info'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')
        self.assertEqual(
            response['Content-Disposition'],
            f'attachment; filename="user_{user.username}_credentials.json"'
        )
        
        # Check JSON content
        content = json.loads(response.content.decode())
        self.assertEqual(content['user_credentials']['username'], user.username)
        self.assertEqual(content['api_access']['api_key'], api_key.key)
        self.assertIn('created_at', content['user_credentials'])
        self.assertIn('created_at', content['api_access'])


class APIRequestAuditTests(TestCase):
    """Test API request auditing functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        self.researcher_user = User.objects.create_user(
            username='apiuser',
            email='api@test.com',
            password='TestPass123!',
            role=self.researcher_role
        )
        
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            name="Test API Key",
            ip_whitelist=['127.0.0.1']
        )
    
    def test_api_request_logging(self):
        """Test API request logging functionality."""
        # Create API request log
        api_request = APIRequest.objects.create(
            api_key=self.api_key,
            user=self.researcher_user,
            endpoint='/api/v1/ping',
            method='GET',
            ip_address='127.0.0.1',
            user_agent='TestAgent/1.0',
            status_code=200,
            response_time_ms=45,
            is_successful=True
        )
        
        self.assertEqual(api_request.endpoint, '/api/v1/ping')
        self.assertEqual(api_request.method, 'GET')
        self.assertEqual(api_request.status_code, 200)
        self.assertTrue(api_request.is_successful)
        self.assertIsNotNone(api_request.timestamp)
    
    def test_api_request_error_logging(self):
        """Test API request error logging."""
        # Create failed API request log
        api_request = APIRequest.objects.create(
            api_key=self.api_key,
            user=self.researcher_user,
            endpoint='/api/v1/invalid',
            method='POST',
            ip_address='127.0.0.1',
            user_agent='TestAgent/1.0',
            status_code=404,
            response_time_ms=120,
            is_successful=False,
            error_message='Endpoint not found'
        )
        
        self.assertEqual(api_request.status_code, 404)
        self.assertFalse(api_request.is_successful)
        self.assertEqual(api_request.error_message, 'Endpoint not found')
    
    def test_api_request_model_str(self):
        """Test APIRequest model string representation."""
        api_request = APIRequest.objects.create(
            api_key=self.api_key,
            user=self.researcher_user,
            endpoint='/api/v1/test',
            method='GET',
            ip_address='127.0.0.1',
            status_code=200
        )
        
        expected_str = f"{self.researcher_user.username} - GET /api/v1/test (200)"
        self.assertEqual(str(api_request), expected_str)


class SecurityTests(TestCase):
    """Test security aspects of API key functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        self.researcher_user = User.objects.create_user(
            username='securitytest',
            email='security@test.com',
            password='TestPass123!',
            role=self.researcher_role
        )
    
    def test_api_key_secure_generation(self):
        """Test that API keys are generated securely."""
        api_key1 = APIKey.objects.create(
            user=self.researcher_user,
            name="Key 1"
        )
        api_key2 = APIKey.objects.create(
            user=self.researcher_user,
            name="Key 2"
        )
        
        # Keys should be different
        self.assertNotEqual(api_key1.key, api_key2.key)
        
        # Keys should be 64 characters long
        self.assertEqual(len(api_key1.key), 64)
        self.assertEqual(len(api_key2.key), 64)
        
        # Keys should contain only alphanumeric characters
        self.assertTrue(api_key1.key.isalnum())
        self.assertTrue(api_key2.key.isalnum())
    
    def test_session_cleanup_after_success_view(self):
        """Test that sensitive session data is cleaned up after viewing."""
        client = Client()
        
        # Create admin for access
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = User.objects.create_user(
            username='admin',
            email='admin@test.com',
            password='TestPass123!',
            role=admin_role,
            is_superuser=True
        )
        client.login(username='admin', password='TestPass123!')
        
        # Create researcher with API key
        user = User.objects.create_user(
            username='sessiontest',
            email='session@test.com',
            password='TestPass123!',
            role=self.researcher_role
        )
        
        api_key = APIKey.objects.create(
            user=user,
            name="Auto-generated API Key"
        )
        
        # Store in session
        session = client.session
        session['new_user_data'] = {
            'user_id': user.id,
            'username': user.username,
            'api_key': api_key.key,
            'api_key_created': api_key.created_at.isoformat()
        }
        session.save()
        
        # Access success page
        response = client.get(reverse('user_created_success'))
        self.assertEqual(response.status_code, 200)
        
        # Session data should be cleaned up after first access
        # Access again to verify cleanup
        response = client.get(reverse('user_created_success'))
        self.assertEqual(response.status_code, 302)  # Should redirect since no session data