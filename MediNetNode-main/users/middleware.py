from __future__ import annotations

from django.conf import settings
from django.contrib.auth import logout
from django.utils import timezone
from django.shortcuts import redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from users.models import APIKey, APIRequest
import time
import logging

logger = logging.getLogger(__name__)


class SessionTimeoutMiddleware:
    """Auto-logout after idle timeout; updates last activity for authenticated users."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            idle_timeout = int(getattr(settings, 'SESSION_IDLE_TIMEOUT', 7200))
            last_activity_ts = request.session.get('last_activity_ts')
            now_ts = int(timezone.now().timestamp())

            # Initialize session activity timestamp if not present (new session)
            if last_activity_ts is None:
                last_activity_ts = now_ts
                request.session['last_activity_ts'] = now_ts

            # Check if session has expired
            if (now_ts - int(last_activity_ts)) > idle_timeout:
                logout(request)
                # For AJAX requests, return JSON error instead of redirect
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    from django.http import JsonResponse
                    return JsonResponse({
                        'error': 'Session expired', 
                        'redirect': '/auth/login/',
                        'message': 'Your session has expired. Please log in again.'
                    }, status=401)
                else:
                    # For regular requests, redirect to login
                    return redirect('login')

            # Update last activity markers for active sessions
            request.session['last_activity_ts'] = now_ts
            request.session.modified = True  # Ensure session is saved
            try:
                request.user.last_activity = timezone.now()
                request.user.is_active_session = True
                request.user.save(update_fields=['last_activity', 'is_active_session'])
            except Exception:
                pass

        # SECURITY: Comprehensive RESEARCHER web access blocking
        if self._is_researcher_user(request):
            security_result = self._enforce_researcher_security(request)
            if security_result:
                return security_result

        response = self.get_response(request)
        return response

    def _is_researcher_user(self, request):
        """Safely check if user is a RESEARCHER."""
        try:
            return (
                request.user.is_authenticated
                and hasattr(request.user, 'role')
                and request.user.role
                and request.user.role.name == 'RESEARCHER'
            )
        except Exception:
            return False

    def _normalize_path(self, path):
        """Normalize path to prevent traversal attacks."""
        import urllib.parse

        # URL decode the path
        decoded_path = urllib.parse.unquote(path)

        # Remove any backslashes (Windows path separators)
        normalized = decoded_path.replace('\\', '/')

        # Remove duplicate slashes
        while '//' in normalized:
            normalized = normalized.replace('//', '/')

        # Ensure it starts with /
        if not normalized.startswith('/'):
            normalized = '/' + normalized

        # Remove trailing slash for consistency (except for root /)
        if len(normalized) > 1 and normalized.endswith('/'):
            normalized = normalized[:-1]

        return normalized.lower()  # Case insensitive

    def _enforce_researcher_security(self, request):
        """Comprehensive security enforcement for RESEARCHER users."""
        original_path = request.path
        normalized_path = self._normalize_path(original_path)

        # WHITELIST: Only allow these specific paths for RESEARCHER users
        allowed_patterns = [
            '/api/v1/',           # Their legitimate API endpoints
            '/info/researcher',   # Their info page (with or without trailing slash)
            '/auth/logout',       # Logout functionality (with or without trailing slash)
        ]

        # Check if path starts with any allowed pattern
        is_allowed = any(normalized_path.startswith(pattern) for pattern in allowed_patterns)

        # Allow specific static files (CSS, JS, images) but block admin static files
        if normalized_path.startswith('/static/'):
            # Block Django admin static files
            blocked_static_patterns = [
                '/static/admin/',
                '/static/debug_toolbar/',
                '/static/swagger/',
                '/static/redoc/',
            ]
            if any(normalized_path.startswith(pattern) for pattern in blocked_static_patterns):
                self._log_security_violation(request, 'BLOCKED_ADMIN_STATIC', original_path)
                return redirect('researcher_info')
            else:
                is_allowed = True  # Allow non-admin static files

        # Block access if not explicitly allowed
        if not is_allowed:
            self._log_security_violation(request, 'BLOCKED_WEB_ACCESS', original_path)
            return redirect('researcher_info')

        return None  # Allow the request to proceed

    def _log_security_violation(self, request, violation_type, attempted_path):
        """Log security violations for monitoring."""
        try:
            user_info = f"User: {request.user.username}" if request.user.is_authenticated else "Anonymous"
            client_ip = request.META.get('REMOTE_ADDR', 'Unknown')
            user_agent = request.META.get('HTTP_USER_AGENT', 'Unknown')

            logger.warning(
                f"SECURITY VIOLATION - {violation_type}: {user_info} "
                f"attempted to access '{attempted_path}' from IP {client_ip} "
                f"(UA: {user_agent[:100]})"
            )
        except Exception:
            pass


class APIAuthenticationMiddleware:
    """Stateless API authentication using API keys and IP validation."""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Only process API requests, but exclude documentation URLs and setup endpoint
        if not request.path.startswith('/api/') or request.path.startswith('/api/docs/') or request.path.startswith('/api/setup/'):
            return self.get_response(request)
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Extract authentication headers
        api_key = request.headers.get('X-API-Key')
        client_ip = self.get_client_ip(request)
        
        # Log the API request attempt
        logger.info(f"API request: {request.method} {request.path} from IP {client_ip}")
        
        # Validate API authentication
        auth_result = self.authenticate_request(api_key, client_ip, request)
        
        if not auth_result['success']:
            # Log failed authentication
            self.log_api_request(
                api_key=None,
                user=None,
                request=request,
                status_code=auth_result['status_code'],
                response_time_ms=int((time.time() - start_time) * 1000),
                is_successful=False,
                error_message=auth_result['error']
            )
            
            return JsonResponse(
                {'error': auth_result['error']},
                status=auth_result['status_code']
            )
        
        # Set authenticated user and API key for the request
        request.api_key = auth_result['api_key']
        request.api_user = auth_result['user']
        request.start_time = start_time
        
        # Process request
        response = self.get_response(request)
        
        # Log successful request
        response_time_ms = int((time.time() - start_time) * 1000)
        self.log_api_request(
            api_key=auth_result['api_key'],
            user=auth_result['user'],
            request=request,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            is_successful=200 <= response.status_code < 400
        )
        
        # Update API key last used
        auth_result['api_key'].update_last_used(client_ip)
        
        return response
    
    def get_client_ip(self, request):
        """Extract client IP address from request."""
        # Check for forwarded IP first (for proxy/load balancer scenarios)
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        
        # Check X-Client-IP header (explicitly provided by client)
        x_client_ip = request.headers.get('X-Client-IP')
        if x_client_ip:
            return x_client_ip.strip()
        
        # Fall back to REMOTE_ADDR
        return request.META.get('REMOTE_ADDR', '0.0.0.0')
    
    def authenticate_request(self, api_key_value, client_ip, request):
        """Validate API key and IP address."""
        if not api_key_value:
            return {
                'success': False,
                'error': 'Missing X-API-Key header',
                'status_code': 401
            }
        
        if not client_ip:
            return {
                'success': False,
                'error': 'Unable to determine client IP address',
                'status_code': 400
            }
        
        try:
            # Get API key from database
            api_key = APIKey.objects.select_related('user', 'user__role').get(
                key=api_key_value,
                is_active=True
            )
        except APIKey.DoesNotExist:
            return {
                'success': False,
                'error': 'Invalid API key',
                'status_code': 401
            }
        
        # Check if API key is expired
        if api_key.is_expired():
            return {
                'success': False,
                'error': 'API key has expired',
                'status_code': 401
            }
        
        # Check IP whitelist
        if not api_key.is_ip_allowed(client_ip):
            return {
                'success': False,
                'error': 'IP address not authorized for this API key',
                'status_code': 403
            }
        
        # Validate user has RESEARCHER role
        if not api_key.user.role or api_key.user.role.name != 'RESEARCHER':
            return {
                'success': False,
                'error': 'Only RESEARCHER users can access API endpoints',
                'status_code': 403
            }
        
        # Check if user account is active
        if not api_key.user.is_active:
            return {
                'success': False,
                'error': 'User account is inactive',
                'status_code': 403
            }
        
        # Check if user account is locked
        if api_key.user.is_account_locked():
            return {
                'success': False,
                'error': 'User account is locked',
                'status_code': 403
            }
        
        return {
            'success': True,
            'api_key': api_key,
            'user': api_key.user
        }
    
    def log_api_request(self, api_key, user, request, status_code, response_time_ms, is_successful=True, error_message=''):
        """Log API request for audit purposes."""
        try:
            APIRequest.objects.create(
                api_key=api_key,
                user=user,
                endpoint=request.path,
                method=request.method,
                ip_address=self.get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                status_code=status_code,
                response_time_ms=response_time_ms,
                is_successful=is_successful,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Failed to log API request: {str(e)}")


class RateLimitMiddleware:
    """Rate limiting for API endpoints."""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'ping': {'requests': 1000, 'window': 3600},    # 1000 pings per hour
        }
    
    def __call__(self, request):
        # Only apply rate limiting to API requests
        if not request.path.startswith('/api/'):
            return self.get_response(request)
        
        # Skip rate limiting if no API user is authenticated
        if not hasattr(request, 'api_user'):
            return self.get_response(request)
        
        # Check rate limits
        if self.is_rate_limited(request):
            return JsonResponse(
                {
                    'error': 'Rate limit exceeded. Maximum 100 requests per hour.',
                    'retry_after': 3600
                },
                status=429
            )
        
        return self.get_response(request)
    
    def is_rate_limited(self, request):
        """Check if user has exceeded rate limits."""
        user = request.api_user
        endpoint_type = self.get_endpoint_type(request.path)
        
        # Get rate limit config
        limit_config = self.rate_limits.get(endpoint_type, self.rate_limits['default'])
        
        # Count recent requests within the time window
        from datetime import timedelta
        time_threshold = timezone.now() - timedelta(seconds=limit_config['window'])
        
        recent_requests = APIRequest.objects.filter(
            user=user,
            timestamp__gte=time_threshold,
            is_successful=True
        ).count()
        
        return recent_requests >= limit_config['requests']
    
    def get_endpoint_type(self, path):
        """Determine endpoint type for rate limiting."""
        if '/ping' in path:
            return 'ping'
        return 'default'



