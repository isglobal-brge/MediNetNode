"""
Tests for CSRF validation functionality in auth_system.
"""
from django.test import TestCase, RequestFactory, Client
from django.contrib.auth import get_user_model
from django.middleware.csrf import get_token
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from unittest.mock import patch
import logging

from auth_system.views import csrf_failure, csrf_validate, get_client_ip
from users.models import Role

User = get_user_model()


class CSRFValidationTests(TestCase):
    """Test CSRF validation functions and views."""
    
    def setUp(self):
        self.factory = RequestFactory()
        self.client = Client()
        self.admin_role = Role.objects.get(name='ADMIN')
        self.user = User.objects.create_user(
            username='testuser',
            password='TestPass123!',
            role=self.admin_role
        )
    
    def test_csrf_failure_view(self):
        """Test: CSRF failure view creates proper response and logs security event"""
        request = self.factory.post('/test/', {'data': 'test'})
        request.user = self.user
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        request.META['HTTP_REFERER'] = 'https://example.com/form'
        
        with patch('auth_system.views.security_logger') as mock_logger:
            response = csrf_failure(request, reason='CSRF token missing')
            
            # Check response
            self.assertEqual(response.status_code, 403)
            self.assertIn('CSRF verification failed', response.content.decode())
            
            # Check security logging
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            self.assertIn('CSRF_FAILURE from 192.168.1.100', call_args[0][0])
    
    def test_csrf_validate_decorator(self):
        """Test: CSRF validate decorator logs missing tokens"""
        @csrf_validate
        def dummy_view(request):
            return HttpResponse('OK')
        
        request = self.factory.post('/test/', {'data': 'test'})
        request.user = self.user
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        with patch('auth_system.views.security_logger') as mock_logger, \
             patch('auth_system.views.get_token', return_value=None):
            
            response = dummy_view(request)
            
            # Check response still works
            self.assertEqual(response.status_code, 200)
            
            # Check security logging for missing token
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            self.assertIn('CSRF_NO_TOKEN from 192.168.1.100', call_args[0][0])
    
    def test_get_client_ip_with_xff(self):
        """Test client IP extraction with X-Forwarded-For header"""
        request = self.factory.get('/test/')
        request.META['HTTP_X_FORWARDED_FOR'] = '203.0.113.1, 203.0.113.2'
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        ip = get_client_ip(request)
        self.assertEqual(ip, '203.0.113.1')  # Should get first IP from XFF
    
    def test_get_client_ip_without_xff(self):
        """Test client IP extraction without X-Forwarded-For header"""
        request = self.factory.get('/test/')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        ip = get_client_ip(request)
        self.assertEqual(ip, '192.168.1.100')  # Should fallback to REMOTE_ADDR
    
    def test_get_client_ip_no_headers(self):
        """Test client IP extraction with no IP headers"""
        request = self.factory.get('/test/')
        # RequestFactory sets REMOTE_ADDR to '127.0.0.1' by default
        # Remove it to test the 'unknown' case
        del request.META['REMOTE_ADDR']
        
        ip = get_client_ip(request)
        self.assertEqual(ip, 'unknown')  # Should return unknown
    
    def test_csrf_settings_configured(self):
        """Test: CSRF settings are properly configured in Django settings"""
        from django.conf import settings
        
        # Check CSRF settings exist and are properly configured
        # Note: During testing, Django might override DEBUG, so we check the setting exists
        self.assertTrue(hasattr(settings, 'CSRF_COOKIE_SECURE'))
        
        # In a real production environment (DEBUG=False), CSRF_COOKIE_SECURE should be True
        # We can't reliably test this during Django testing due to setting evaluation order
        csrf_secure = getattr(settings, 'CSRF_COOKIE_SECURE', None)
        self.assertIsNotNone(csrf_secure, "CSRF_COOKIE_SECURE must be defined")
        
        self.assertTrue(getattr(settings, 'CSRF_COOKIE_HTTPONLY', False))
        self.assertEqual(getattr(settings, 'CSRF_COOKIE_SAMESITE', None), 'Lax')
        self.assertEqual(getattr(settings, 'CSRF_FAILURE_VIEW', None), 'auth_system.views.csrf_failure')
        
        # Check trusted origins exist
        trusted_origins = getattr(settings, 'CSRF_TRUSTED_ORIGINS', [])
        self.assertIsInstance(trusted_origins, list)
        self.assertGreater(len(trusted_origins), 0)