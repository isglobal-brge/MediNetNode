"""
Tests for API Key system - stateless authentication for RESEARCHER users.
Critical security tests for Phase 3 implementation.
"""
from django.test import TestCase
from django.utils import timezone
from datetime import timedelta
from users.models import CustomUser, Role, APIKey, APIRequest


class APIKeyModelTests(TestCase):
    """Test APIKey model functionality and security."""
    
    def setUp(self):
        """Set up test data."""
        # Create RESEARCHER role
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        # Create ADMIN role for negative testing
        self.admin_role = Role.objects.get(name='ADMIN')
        
        # Create users
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
        
        self.admin_user = CustomUser.objects.create_user(
            username='admin_test',
            email='admin@test.com', 
            password='AdminPass123!',
            role=self.admin_role
        )
    
    def test_api_key_generation(self):
        """Test that API keys are automatically generated."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Test API Key',
            ip_whitelist=['192.168.1.100', '10.0.0.50']
        )
        
        # API key should be generated automatically
        self.assertIsNotNone(api_key.key)
        self.assertEqual(len(api_key.key), 64)
        
        # Key should be unique
        api_key2 = APIKey.objects.create(
            user=self.researcher_user,
            name='Test API Key 2',
            ip_whitelist=['192.168.1.200']
        )
        self.assertNotEqual(api_key.key, api_key2.key)
    
    def test_api_key_generation_security(self):
        """Test API key generation produces secure random keys."""
        keys = set()
        for i in range(100):
            key = APIKey.generate_api_key()
            self.assertEqual(len(key), 64)
            self.assertNotIn(key, keys)  # Ensure uniqueness
            keys.add(key)
        
        # Check character set (alphanumeric only)
        test_key = APIKey.generate_api_key()
        self.assertTrue(test_key.isalnum())
    
    def test_ip_whitelist_validation(self):
        """Test IP whitelist functionality."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='IP Test Key',
            ip_whitelist=['192.168.1.100', '10.0.0.50']
        )
        
        # Allowed IPs
        self.assertTrue(api_key.is_ip_allowed('192.168.1.100'))
        self.assertTrue(api_key.is_ip_allowed('10.0.0.50'))
        
        # Disallowed IPs
        self.assertFalse(api_key.is_ip_allowed('192.168.1.101'))
        self.assertFalse(api_key.is_ip_allowed('8.8.8.8'))
        
        # Empty whitelist should deny all
        empty_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Empty Whitelist Key',
            ip_whitelist=[]
        )
        self.assertFalse(empty_key.is_ip_allowed('192.168.1.100'))
    
    def test_api_key_expiration(self):
        """Test API key expiration functionality."""
        # Non-expiring key
        no_expiry_key = APIKey.objects.create(
            user=self.researcher_user,
            name='No Expiry Key',
            ip_whitelist=['192.168.1.100']
        )
        self.assertFalse(no_expiry_key.is_expired())
        
        # Future expiry key
        future_expiry = timezone.now() + timedelta(days=30)
        future_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Future Expiry Key',
            ip_whitelist=['192.168.1.100'],
            expires_at=future_expiry
        )
        self.assertFalse(future_key.is_expired())
        
        # Past expiry key
        past_expiry = timezone.now() - timedelta(days=1)
        expired_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Expired Key',
            ip_whitelist=['192.168.1.100'],
            expires_at=past_expiry
        )
        self.assertTrue(expired_key.is_expired())
    
    def test_last_used_tracking(self):
        """Test that API key usage is tracked."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Usage Tracking Key',
            ip_whitelist=['192.168.1.100']
        )
        
        # Initially no usage
        self.assertIsNone(api_key.last_used_at)
        self.assertIsNone(api_key.last_used_ip)
        
        # Update usage
        test_ip = '192.168.1.100'
        api_key.update_last_used(test_ip)
        
        # Reload from database
        api_key.refresh_from_db()
        self.assertIsNotNone(api_key.last_used_at)
        self.assertEqual(api_key.last_used_ip, test_ip)
    
    def test_researcher_role_requirement(self):
        """Test that API keys can be created for RESEARCHER users."""
        # RESEARCHER should be able to have API keys
        researcher_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Researcher Key',
            ip_whitelist=['192.168.1.100']
        )
        self.assertEqual(researcher_key.user.role.name, 'RESEARCHER')
        
        # ADMIN should also be able to have API keys (for testing/admin purposes)
        admin_key = APIKey.objects.create(
            user=self.admin_user,
            name='Admin Key',
            ip_whitelist=['192.168.1.100']
        )
        self.assertEqual(admin_key.user.role.name, 'ADMIN')


class APIRequestModelTests(TestCase):
    """Test APIRequest audit logging model."""
    
    def setUp(self):
        """Set up test data."""
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
        
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Test Key',
            ip_whitelist=['192.168.1.100']
        )
    
    def test_api_request_logging(self):
        """Test that API requests are properly logged."""
        request_log = APIRequest.objects.create(
            api_key=self.api_key,
            user=self.researcher_user,
            endpoint='/api/v1/ping',
            method='GET',
            ip_address='192.168.1.100',
            user_agent='TestClient/1.0',
            status_code=200,
            response_time_ms=45,
            is_successful=True
        )
        
        self.assertEqual(request_log.api_key, self.api_key)
        self.assertEqual(request_log.user, self.researcher_user)
        self.assertEqual(request_log.endpoint, '/api/v1/ping')
        self.assertEqual(request_log.method, 'GET')
        self.assertEqual(request_log.status_code, 200)
        self.assertTrue(request_log.is_successful)
    
    def test_failed_request_logging(self):
        """Test logging of failed API requests."""
        failed_request = APIRequest.objects.create(
            api_key=None,  # No API key for failed auth
            user=None,     # No user for failed auth
            endpoint='/api/v1/get-data-info',
            method='GET',
            ip_address='192.168.1.200',  # Unauthorized IP
            user_agent='TestClient/1.0',
            status_code=403,
            is_successful=False,
            error_message='IP address not in whitelist'
        )
        
        self.assertIsNone(failed_request.api_key)
        self.assertIsNone(failed_request.user)
        self.assertEqual(failed_request.status_code, 403)
        self.assertFalse(failed_request.is_successful)
        self.assertIn('whitelist', failed_request.error_message)
    
    def test_api_request_indexing(self):
        """Test that database indexes work for efficient queries."""
        # Create multiple requests for testing
        for i in range(5):
            APIRequest.objects.create(
                api_key=self.api_key,
                user=self.researcher_user,
                endpoint=f'/api/v1/test-{i}',
                method='GET',
                ip_address='192.168.1.100',
                status_code=200,
                is_successful=True
            )
        
        # Test queries that should use indexes
        user_requests = APIRequest.objects.filter(user=self.researcher_user)
        self.assertEqual(user_requests.count(), 5)
        
        key_requests = APIRequest.objects.filter(api_key=self.api_key)
        self.assertEqual(key_requests.count(), 5)
        
        ip_requests = APIRequest.objects.filter(ip_address='192.168.1.100')
        self.assertEqual(ip_requests.count(), 5)


class APIKeySecurityTests(TestCase):
    """Security-focused tests for API key system."""
    
    def setUp(self):
        """Set up test data."""
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
    
    def test_api_key_uniqueness_constraint(self):
        """Test that API keys must be unique."""
        api_key1 = APIKey.objects.create(
            user=self.researcher_user,
            name='Key 1',
            ip_whitelist=['192.168.1.100']
        )
        
        # Try to create another key with same key value
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            APIKey.objects.create(
                user=self.researcher_user,
                name='Key 2',
                ip_whitelist=['192.168.1.100'],
                key=api_key1.key  # Same key should fail
            )
    
    def test_api_key_deactivation(self):
        """Test API key can be deactivated for security."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Deactivation Test Key',
            ip_whitelist=['192.168.1.100'],
            is_active=True
        )
        
        self.assertTrue(api_key.is_active)
        
        # Deactivate key
        api_key.is_active = False
        api_key.save()
        
        api_key.refresh_from_db()
        self.assertFalse(api_key.is_active)
    
    def test_multiple_keys_per_user(self):
        """Test that users can have multiple API keys."""
        key1 = APIKey.objects.create(
            user=self.researcher_user,
            name='Production Key',
            ip_whitelist=['192.168.1.100']
        )
        
        key2 = APIKey.objects.create(
            user=self.researcher_user,
            name='Development Key',
            ip_whitelist=['192.168.1.200']
        )
        
        user_keys = self.researcher_user.api_keys.all()
        self.assertEqual(user_keys.count(), 2)
        self.assertIn(key1, user_keys)
        self.assertIn(key2, user_keys)
    
    def test_cascade_deletion(self):
        """Test that API keys are deleted when user is deleted."""
        api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Cascade Test Key',
            ip_whitelist=['192.168.1.100']
        )
        
        # Create request log
        APIRequest.objects.create(
            api_key=api_key,
            user=self.researcher_user,
            endpoint='/api/v1/ping',
            method='GET',
            ip_address='192.168.1.100',
            status_code=200,
            is_successful=True
        )
        
        key_id = api_key.id
        user_id = self.researcher_user.id
        
        # Delete user should cascade delete API key and requests
        self.researcher_user.delete()
        
        self.assertFalse(APIKey.objects.filter(id=key_id).exists())
        self.assertFalse(APIRequest.objects.filter(user_id=user_id).exists())