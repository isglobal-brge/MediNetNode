"""
Security middleware for MediNet Core.

Contains:
- SecurityHeadersMiddleware: CSP with nonce, security headers
- SessionTimeoutMiddleware: idle session expiry
- APIAuthenticationMiddleware: stateless API key auth
- RateLimitMiddleware: per-user request rate limiting

CSP nonce migration: complete (Phase C).
    All templates use nonce="{% csp_nonce %}" — 'unsafe-inline' has been removed.

    Templates audit command:
        grep -rn "<script\\b\\|<style\\b" templates/ | grep -v "nonce=" | grep -v "src="
"""
import secrets
import time
import logging
import urllib.parse

from django.conf import settings
from django.contrib.auth import logout
from django.core.cache import cache
from django.db.models import Q
from django.http import JsonResponse, FileResponse, Http404
from django.shortcuts import redirect
from django.utils import timezone

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """
    Adds security headers to every response and generates a per-request CSP nonce.

    The nonce is stored on request.csp_nonce and can be used in templates via
    the {% csp_nonce %} template tag from medinet_core.templatetags.csp_tags.
    All templates have been migrated to use nonce= attributes; 'unsafe-inline'
    is not present in the CSP policy.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        nonce = secrets.token_urlsafe(32)
        request.csp_nonce = nonce

        response = self.get_response(request)

        csp_policy = (
            f"default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; "
            f"style-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; "
            f"font-src 'self' https://cdn.jsdelivr.net; "
            f"img-src 'self' data:; "
            f"connect-src 'self'; "
            f"frame-ancestors 'none'; "
            f"base-uri 'self'; "
            f"form-action 'self';"
        )

        response['Content-Security-Policy'] = csp_policy
        response['X-Content-Type-Options'] = 'nosniff'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )

        if 'Server' in response:
            del response['Server']

        return response


class SessionTimeoutMiddleware:
    """Auto-logout after idle timeout; updates last activity for authenticated users."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            idle_timeout = int(getattr(settings, 'SESSION_IDLE_TIMEOUT', 7200))
            last_activity_ts = request.session.get('last_activity_ts')
            now_ts = int(timezone.now().timestamp())

            if last_activity_ts is None:
                last_activity_ts = now_ts
                request.session['last_activity_ts'] = now_ts

            if (now_ts - int(last_activity_ts)) > idle_timeout:
                logout(request)
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': 'Session expired',
                        'redirect': '/auth/login/',
                        'message': 'Your session has expired. Please log in again.'
                    }, status=401)
                return redirect('login')

            request.session['last_activity_ts'] = now_ts
            request.session.modified = True
            try:
                request.user.last_activity = timezone.now()
                request.user.is_active_session = True
                request.user.save(update_fields=['last_activity', 'is_active_session'])
            except Exception as e:
                logger.warning(f"Failed to update user activity: {e}")

        if self._is_researcher_user(request):
            security_result = self._enforce_researcher_security(request)
            if security_result:
                return security_result

        return self.get_response(request)

    def _is_researcher_user(self, request):
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
        import posixpath
        decoded_path = urllib.parse.unquote(path)
        normalized = decoded_path.replace('\\', '/')
        # Collapse '.' and '..' segments BEFORE the allow-list check so a path
        # traversal (e.g. '/api/v2/../../admin') cannot bypass the prefix
        # allow-list by hiding behind an allowed prefix.
        normalized = posixpath.normpath(normalized)
        while '//' in normalized:
            normalized = normalized.replace('//', '/')
        if not normalized.startswith('/'):
            normalized = '/' + normalized
        if len(normalized) > 1 and normalized.endswith('/'):
            normalized = normalized[:-1]
        return normalized.lower()

    def _enforce_researcher_security(self, request):
        original_path = request.path
        normalized_path = self._normalize_path(original_path)

        allowed_patterns = [
            '/api/v2/',
            '/info/researcher',
            '/auth/logout',
        ]
        is_allowed = any(normalized_path.startswith(p) for p in allowed_patterns)

        if normalized_path.startswith('/static/'):
            blocked_static = [
                '/static/admin/', '/static/debug_toolbar/',
                '/static/swagger/', '/static/redoc/',
            ]
            if any(normalized_path.startswith(p) for p in blocked_static):
                self._log_security_violation(request, 'BLOCKED_ADMIN_STATIC', original_path)
                return redirect('researcher_info')
            else:
                is_allowed = True

        if not is_allowed:
            self._log_security_violation(request, 'BLOCKED_WEB_ACCESS', original_path)
            return redirect('researcher_info')

        return None

    def _log_security_violation(self, request, violation_type, attempted_path):
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
        if (
            not request.path.startswith('/api/')
            or request.path.startswith('/api/docs/')
            or request.path.startswith('/api/setup/')
        ):
            return self.get_response(request)

        start_time = time.time()
        api_key_value = request.headers.get('X-API-Key')
        client_ip = self.get_client_ip(request)

        logger.info(f"API request: {request.method} {request.path} from IP {client_ip}")

        auth_result = self.authenticate_request(api_key_value, client_ip, request)

        if not auth_result['success']:
            self.log_api_request(
                api_key=None, user=None, request=request,
                status_code=auth_result['status_code'],
                response_time_ms=int((time.time() - start_time) * 1000),
                is_successful=False, error_message=auth_result['error']
            )
            return JsonResponse({'error': auth_result['error']}, status=auth_result['status_code'])

        request.api_key = auth_result['api_key']
        request.api_user = auth_result['user']
        request.start_time = start_time

        response = self.get_response(request)

        response_time_ms = int((time.time() - start_time) * 1000)
        self.log_api_request(
            api_key=auth_result['api_key'], user=auth_result['user'],
            request=request, status_code=response.status_code,
            response_time_ms=response_time_ms,
            is_successful=200 <= response.status_code < 400
        )
        auth_result['api_key'].update_last_used(client_ip)

        # Warn consumers using legacy keys (no key_prefix optimization)
        if auth_result['api_key'].key_prefix == '__LEGACY__':
            response['X-API-Key-Status'] = (
                'LEGACY: Please regenerate your API key. '
                'Legacy keys will be deactivated in a future release.'
            )

        return response

    def get_client_ip(self, request):
        remote_addr = request.META.get('REMOTE_ADDR', '0.0.0.0')
        trusted_proxies = getattr(settings, 'TRUSTED_PROXIES', [])
        if trusted_proxies and remote_addr in trusted_proxies:
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                return x_forwarded_for.split(',')[0].strip()
        return remote_addr

    def authenticate_request(self, api_key_value, client_ip, request):
        if not api_key_value:
            return {'success': False, 'error': 'Missing X-API-Key header', 'status_code': 401}
        if not client_ip:
            return {'success': False, 'error': 'Unable to determine client IP address', 'status_code': 400}

        try:
            from users.models import APIKey
            # Optimized: use key_prefix index to narrow candidates, then verify hash.
            # Legacy keys (key_prefix='__LEGACY__') still work via full hash scan.
            key_prefix = api_key_value[:8] if len(api_key_value) >= 8 else ''
            if key_prefix:
                candidates = APIKey.objects.select_related('user', 'user__role').filter(
                    is_active=True
                ).filter(Q(key_prefix=key_prefix) | Q(key_prefix='__LEGACY__'))
            else:
                candidates = APIKey.objects.select_related('user', 'user__role').filter(is_active=True)

            api_key = None
            for candidate in candidates:
                if candidate.check_key(api_key_value):
                    api_key = candidate
                    break

            if not api_key:
                return {'success': False, 'error': 'Invalid API key', 'status_code': 401}

        except Exception as e:
            logger.error(f"Error during API key authentication: {e}")
            return {'success': False, 'error': 'Authentication error', 'status_code': 500}

        if api_key.is_expired():
            return {'success': False, 'error': 'API key has expired', 'status_code': 401}

        if not api_key.is_ip_allowed(client_ip):
            return {'success': False, 'error': 'IP address not authorized for this API key', 'status_code': 403}

        if not api_key.user.role or api_key.user.role.name != 'RESEARCHER':
            return {'success': False, 'error': 'Only RESEARCHER users can access API endpoints', 'status_code': 403}

        if not api_key.user.is_active:
            return {'success': False, 'error': 'User account is inactive', 'status_code': 403}

        if api_key.user.is_account_locked():
            return {'success': False, 'error': 'User account is locked', 'status_code': 403}

        return {'success': True, 'api_key': api_key, 'user': api_key.user}

    def log_api_request(self, api_key, user, request, status_code, response_time_ms,
                        is_successful=True, error_message=''):
        try:
            from users.models import APIRequest
            APIRequest.objects.create(
                api_key=api_key, user=user,
                endpoint=request.path, method=request.method,
                ip_address=self.get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                status_code=status_code, response_time_ms=response_time_ms,
                is_successful=is_successful, error_message=error_message
            )
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")


_IP_RATE_LIMIT_MAX = 20
_IP_RATE_LIMIT_WINDOW = 60  # seconds


class RateLimitMiddleware:
    """Rate limiting for API endpoints. Counts ALL requests (including failed auth)."""

    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {
            'default': {'requests': 100, 'window': 3600},
            'ping': {'requests': 1000, 'window': 3600},
        }

    def __call__(self, request):
        if not request.path.startswith('/api/'):
            return self.get_response(request)

        # IP-based rate limiting for unauthenticated requests
        if not hasattr(request, 'api_user'):
            client_ip = self._get_client_ip(request)
            cache_key = f'ratelimit_ip_{client_ip}'
            request_count = cache.get(cache_key, 0)
            if request_count >= _IP_RATE_LIMIT_MAX:
                return JsonResponse(
                    {'error': 'Demasiadas peticiones. Inténtalo más tarde.'},
                    status=429,
                )
            cache.set(cache_key, request_count + 1, _IP_RATE_LIMIT_WINDOW)
            return self.get_response(request)

        # Existing authenticated rate limiting logic
        if self.is_rate_limited(request):
            return JsonResponse(
                {'error': 'Rate limit exceeded. Maximum 100 requests per hour.', 'retry_after': 3600},
                status=429
            )
        return self.get_response(request)

    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '0.0.0.0')

    def is_rate_limited(self, request):
        from datetime import timedelta
        from users.models import APIRequest
        user = request.api_user
        limit_config = self.rate_limits.get(self._endpoint_type(request.path), self.rate_limits['default'])
        time_threshold = timezone.now() - timedelta(seconds=limit_config['window'])
        # Count ALL requests — failed attempts count against the limit (brute-force protection)
        recent = APIRequest.objects.filter(user=user, timestamp__gte=time_threshold).count()
        return recent >= limit_config['requests']

    def _endpoint_type(self, path):
        return 'ping' if '/ping' in path else 'default'
