"""
Tests for API authentication middleware - stateless authentication system.
Critical security tests for Phase 3 implementation.
"""
from django.test import TestCase, RequestFactory, override_settings
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from users.models import CustomUser, Role, APIKey, APIRequest
from users.middleware import APIAuthenticationMiddleware, RateLimitMiddleware
from unittest.mock import Mock, patch
import json


class APIAuthenticationMiddlewareTests(TestCase):
    """Test stateless API authentication middleware."""
    
    def get_json_response_data(self, response):
        """Helper to parse JsonResponse content."""
        return json.loads(response.content.decode('utf-8'))
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.middleware = APIAuthenticationMiddleware(Mock())
        
        # Create RESEARCHER role
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        # Create RESEARCHER user
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
        
        # Create API key
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Test API Key',
            ip_whitelist=['192.168.1.100', '10.0.0.50']
        )
    
    def test_non_api_requests_pass_through(self):
        """Test that non-API requests are not processed by middleware."""
        request = self.factory.get('/')
        
        # Mock get_response to return a simple response
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Should pass through without authentication
        self.assertEqual(response, mock_response)
        self.middleware.get_response.assert_called_once_with(request)
    
    def test_missing_api_key_returns_401(self):
        """Test that requests without X-API-Key header return 401."""
        request = self.factory.get('/api/v1/ping')
        
        response = self.middleware(request)
        
        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 401)
        
        # Check response content
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'Missing X-API-Key header')
    
    def test_invalid_api_key_returns_401(self):
        """Test that invalid API key returns 401."""
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY='invalid_key_12345',
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 401)
        
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'Invalid API key')
    
    def test_expired_api_key_returns_401(self):
        """Test that expired API key returns 401."""
        # Create expired API key
        expired_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Expired Key',
            ip_whitelist=['192.168.1.100'],
            expires_at=timezone.now() - timedelta(days=1)
        )
        
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=expired_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 401)
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'API key has expired')
    
    def test_unauthorized_ip_returns_403(self):
        """Test that unauthorized IP address returns 403."""
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.200'  # Not in whitelist
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 403)
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'IP address not authorized for this API key')
    
    def test_non_researcher_role_returns_403(self):
        """Test that non-RESEARCHER users cannot access API."""
        # Create ADMIN user and API key
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_test',
            email='admin@test.com',
            password='AdminPass123!',
            role=admin_role
        )
        admin_key = APIKey.objects.create(
            user=admin_user,
            name='Admin Key',
            ip_whitelist=['192.168.1.100']
        )
        
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=admin_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 403)
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'Only RESEARCHER users can access API endpoints')
    
    def test_inactive_user_returns_403(self):
        """Test that inactive user account returns 403."""
        self.researcher_user.is_active = False
        self.researcher_user.save()
        
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 403)
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'User account is inactive')
    
    def test_locked_account_returns_403(self):
        """Test that locked user account returns 403."""
        self.researcher_user.account_locked_until = timezone.now() + timedelta(minutes=5)
        self.researcher_user.save()
        
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 403)
        response_data = self.get_json_response_data(response)
        self.assertEqual(response_data['error'], 'User account is locked')
    
    def test_valid_authentication_sets_request_attributes(self):
        """Test that valid authentication sets api_key and api_user on request."""
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        # Mock get_response to return a simple response
        mock_response = Mock()
        mock_response.status_code = 200
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Check that request attributes are set
        self.assertEqual(request.api_key, self.api_key)
        self.assertEqual(request.api_user, self.researcher_user)
        self.assertTrue(hasattr(request, 'start_time'))
    
    def test_successful_request_logging(self):
        """Test that successful requests are logged."""
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.100',
            HTTP_USER_AGENT='TestClient/1.0'
        )
        
        # Mock get_response
        mock_response = Mock()
        mock_response.status_code = 200
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Check that request was logged
        log_entry = APIRequest.objects.get(
            api_key=self.api_key,
            endpoint='/api/v1/ping'
        )
        self.assertEqual(log_entry.method, 'GET')
        self.assertEqual(log_entry.ip_address, '192.168.1.100')
        self.assertEqual(log_entry.status_code, 200)
        self.assertTrue(log_entry.is_successful)
        self.assertEqual(log_entry.user_agent, 'TestClient/1.0')
    
    def test_failed_request_logging(self):
        """Test that failed authentication requests are logged."""
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY='invalid_key',
            REMOTE_ADDR='192.168.1.100'
        )
        
        response = self.middleware(request)
        
        # Check that failed request was logged
        log_entry = APIRequest.objects.get(endpoint='/api/v1/ping')
        self.assertIsNone(log_entry.api_key)
        self.assertIsNone(log_entry.user)
        self.assertEqual(log_entry.status_code, 401)
        self.assertFalse(log_entry.is_successful)
        self.assertEqual(log_entry.error_message, 'Invalid API key')
    
    def test_api_key_last_used_update(self):
        """Test that API key last_used fields are updated."""
        initial_last_used = self.api_key.last_used_at
        initial_last_ip = self.api_key.last_used_ip
        
        request = self.factory.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            REMOTE_ADDR='192.168.1.100'
        )
        
        # Mock get_response
        mock_response = Mock()
        mock_response.status_code = 200
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Reload API key from database
        self.api_key.refresh_from_db()
        
        # Check that last used fields were updated
        self.assertNotEqual(self.api_key.last_used_at, initial_last_used)
        self.assertEqual(self.api_key.last_used_ip, '192.168.1.100')
    
    def test_client_ip_extraction(self):
        """Test client IP extraction from various headers."""
        # Test X-Forwarded-For header
        request = self.factory.get('/', HTTP_X_FORWARDED_FOR='203.0.113.1, 192.168.1.100')
        ip = self.middleware.get_client_ip(request)
        self.assertEqual(ip, '203.0.113.1')
        
        # Test X-Client-IP header
        request = self.factory.get('/', HTTP_X_CLIENT_IP='10.0.0.100')
        ip = self.middleware.get_client_ip(request)
        self.assertEqual(ip, '10.0.0.100')
        
        # Test REMOTE_ADDR fallback
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = '127.0.0.1'
        ip = self.middleware.get_client_ip(request)
        self.assertEqual(ip, '127.0.0.1')


class RateLimitMiddlewareTests(TestCase):
    """Test rate limiting middleware for API endpoints."""
    
    def get_json_response_data(self, response):
        """Helper to parse JsonResponse content."""
        return json.loads(response.content.decode('utf-8'))
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.middleware = RateLimitMiddleware(Mock())
        
        # Create test user and API key
        self.researcher_role = Role.objects.get(name='RESEARCHER')
        
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
        
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Rate Limit Test Key',
            ip_whitelist=['192.168.1.100']
        )
    
    def test_non_api_requests_pass_through(self):
        """Test that non-API requests bypass rate limiting."""
        request = self.factory.get('/')
        
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        self.assertEqual(response, mock_response)
    
    def test_unauthenticated_api_requests_pass_through(self):
        """Test that unauthenticated API requests bypass rate limiting."""
        request = self.factory.get('/api/v1/ping')
        
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        self.assertEqual(response, mock_response)
    
    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced."""
        request = self.factory.get('/api/v1/test')
        request.api_user = self.researcher_user
        
        # Create 100 recent successful requests (at the limit)
        for i in range(100):
            APIRequest.objects.create(
                api_key=self.api_key,
                user=self.researcher_user,
                endpoint=f'/api/v1/test-{i}',
                method='GET',
                ip_address='192.168.1.100',
                status_code=200,
                is_successful=True,
                timestamp=timezone.now() - timedelta(minutes=30)  # Within the 1-hour window
            )
        
        response = self.middleware(request)
        
        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 429)
        
        response_data = self.get_json_response_data(response)
        self.assertIn('Rate limit exceeded', response_data['error'])
        self.assertEqual(response_data['retry_after'], 3600)
    
    def test_rate_limit_not_exceeded(self):
        """Test that requests under rate limit pass through."""
        request = self.factory.get('/api/v1/test')
        request.api_user = self.researcher_user
        
        # Create only 50 recent requests (under the limit)
        for i in range(50):
            APIRequest.objects.create(
                api_key=self.api_key,
                user=self.researcher_user,
                endpoint=f'/api/v1/test-{i}',
                method='GET',
                ip_address='192.168.1.100',
                status_code=200,
                is_successful=True,
                timestamp=timezone.now() - timedelta(minutes=30)
            )
        
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        self.assertEqual(response, mock_response)
    
    def test_ping_endpoint_higher_rate_limit(self):
        """Test that ping endpoint has higher rate limit."""
        request = self.factory.get('/api/v1/ping')
        request.api_user = self.researcher_user
        
        # Create 500 ping requests (should still be allowed)
        for i in range(500):
            APIRequest.objects.create(
                api_key=self.api_key,
                user=self.researcher_user,
                endpoint='/api/v1/ping',
                method='GET',
                ip_address='192.168.1.100',
                status_code=200,
                is_successful=True,
                timestamp=timezone.now() - timedelta(minutes=30)
            )
        
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Should pass through (ping has 1000 requests/hour limit)
        self.assertEqual(response, mock_response)
    
    def test_old_requests_not_counted(self):
        """Test that requests outside the time window are not counted."""
        request = self.factory.get('/api/v1/test')
        request.api_user = self.researcher_user
        
        # Create 100 old requests (outside 1-hour window) 
        # Use bulk_create and then update timestamps to bypass auto_now_add
        old_timestamp = timezone.now() - timedelta(hours=3)
        
        bulk_requests = []
        for i in range(100):
            bulk_requests.append(APIRequest(
                api_key=self.api_key,
                user=self.researcher_user,
                endpoint=f'/api/v1/test-{i}',
                method='GET',
                ip_address='192.168.1.100',
                status_code=200,
                is_successful=True
            ))
        
        created_requests = APIRequest.objects.bulk_create(bulk_requests)
        
        # Update all timestamps to the old timestamp
        APIRequest.objects.filter(
            user=self.researcher_user
        ).update(timestamp=old_timestamp)
        
        # Verify old requests exist but should not be counted
        total_requests = APIRequest.objects.filter(user=self.researcher_user).count()
        self.assertEqual(total_requests, 100)
        
        # Check rate limiting logic
        time_threshold = timezone.now() - timedelta(seconds=3600)  # 1 hour
        recent_requests = APIRequest.objects.filter(
            user=self.researcher_user,
            timestamp__gte=time_threshold,
            is_successful=True
        ).count()
        
        # Old requests should not be in recent count
        self.assertEqual(recent_requests, 0)
        
        mock_response = Mock()
        self.middleware.get_response = Mock(return_value=mock_response)
        
        response = self.middleware(request)
        
        # Should pass through since old requests don't count
        self.assertEqual(response, mock_response)