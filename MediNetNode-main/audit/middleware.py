from __future__ import annotations

import time
from typing import Optional, Dict, Any
from urllib.parse import parse_qs

from django.utils.deprecation import MiddlewareMixin
from django.conf import settings

# All audit logging now uses the new AuditEvent system
from .audit_logger import AuditLogger


class AuditMiddleware(MiddlewareMixin):
    """Advanced request auditing with risk scoring and anomaly detection.

    Features:
    - Automatic categorization and risk scoring
    - Real-time anomaly detection
    - Comprehensive context capture
    - Automatic security incident creation
    """

    # Expanded sensitive prefixes for comprehensive monitoring
    SENSITIVE_PREFIXES = (
        "/admin", "/auth", "/accounts", "/api", "/datasets", 
        "/training", "/users", "/audit", "/dashboard"
    )
    
    # Paths that require data access logging
    DATA_ACCESS_PATTERNS = [
        "/datasets/", "/api/datasets/", "/download/", "/export/", 
        "/query/", "/data/", "/training/data/"
    ]
    
    # High-risk paths that need immediate attention
    HIGH_RISK_PATHS = [
        "/admin/delete/", "/api/delete/", "/datasets/delete/",
        "/users/delete/", "/admin/users/", "/export/all/"
    ]

    def process_request(self, request):  # type: ignore[override]
        """Capture request start time and context."""
        request._audit_start_time = time.time()
        
        # Capture request context for detailed auditing
        request._audit_context = self._extract_request_context(request)

    def process_response(self, request, response):  # type: ignore[override]
        """Process response and create comprehensive audit log."""
        try:
            path: str = getattr(request, 'path', '') or ''
            method: str = getattr(request, 'method', 'GET')
            
            # Determine if this request should be logged
            should_log = self._should_log_request(path, method)
            
            if should_log:
                # Extract comprehensive request information
                audit_data = self._build_audit_data(request, response)
                
                # Use advanced audit logger
                self._create_audit_event(audit_data, path)
                
        except Exception as e:
            # Log the auditing error but don't break the response
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Audit middleware error: {e}")
            
        return response

    def _should_log_request(self, path: str, method: str) -> bool:
        """Determine if request should be logged based on path and method."""
        # Always log sensitive paths
        if path.startswith(self.SENSITIVE_PREFIXES):
            return True
            
        # Always log destructive operations
        if method in ['DELETE', 'POST', 'PUT', 'PATCH']:
            return True
            
        # Log API calls
        if '/api/' in path:
            return True
            
        return False

    def _extract_request_context(self, request) -> Dict[str, Any]:
        """Extract detailed context from request."""
        context = {}
        
        # Basic request info
        context['method'] = getattr(request, 'method', 'GET')
        context['path'] = getattr(request, 'path', '')
        context['query_params'] = dict(request.GET) if hasattr(request, 'GET') else {}
        
        # User context
        if hasattr(request, 'user') and request.user.is_authenticated:
            context['user_id'] = request.user.id
            context['username'] = request.user.username
            context['is_staff'] = request.user.is_staff
            context['is_superuser'] = request.user.is_superuser
        
        # Session context
        if hasattr(request, 'session'):
            context['session_key'] = request.session.session_key
            context['session_data'] = dict(request.session)
        
        return context

    def _build_audit_data(self, request, response) -> Dict[str, Any]:
        """Build comprehensive audit data from request and response."""
        # Get IP address
        ip_address = self._get_client_ip(request)
        
        # Calculate request duration
        duration_ms = self._calculate_duration(request)
        
        # Get request size
        request_size = self._get_request_size(request)
        
        # Extract user agent
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        
        # Get session ID
        session_id = getattr(request.session, 'session_key', '') if hasattr(request, 'session') else ''
        session_id = session_id or ''  # Ensure not None
        
        # Build details
        details = {
            'status_code': getattr(response, 'status_code', None),
            'duration_ms': duration_ms,
            'request_size': request_size,
            'query_params': dict(request.GET) if hasattr(request, 'GET') else {},
            'content_type': request.META.get('CONTENT_TYPE', ''),
            'referer': request.META.get('HTTP_REFERER', ''),
        }
        
        # Add POST data for audit trails (excluding sensitive fields)
        if hasattr(request, 'POST') and request.POST:
            safe_post_data = self._sanitize_post_data(dict(request.POST))
            details['post_data'] = safe_post_data
        
        return {
            'user': getattr(request, 'user', None) if hasattr(request, 'user') and request.user.is_authenticated else None,
            'ip_address': ip_address,
            'session_id': session_id,
            'user_agent': user_agent,
            'request_size': request_size,
            'request_duration_ms': duration_ms,
            'success': 200 <= getattr(response, 'status_code', 0) < 400,
            'details': details,
        }

    def _create_audit_event(self, audit_data: Dict[str, Any], path: str):
        """Create audit event using advanced audit logger."""
        # Determine action based on path and method
        method = audit_data['details'].get('method', 'GET')
        action = self._determine_action(method, path, audit_data)
        
        # Determine if this is a data access event
        is_data_access = any(pattern in path for pattern in self.DATA_ACCESS_PATTERNS)
        
        # Extract data access context if applicable
        medical_domain = ''
        records_accessed = 0
        columns_accessed = []
        patient_count = 0
        
        if is_data_access:
            # Extract medical context from request details
            query_params = audit_data['details'].get('query_params', {})
            post_data = audit_data['details'].get('post_data', {})
            
            medical_domain = query_params.get('domain', post_data.get('domain', ['']))[0] if isinstance(query_params.get('domain', post_data.get('domain', [''])), list) else query_params.get('domain', post_data.get('domain', ''))
            
            # Try to extract record count from response or parameters
            try:
                records_accessed = int(query_params.get('limit', post_data.get('limit', ['0']))[0] if isinstance(query_params.get('limit', post_data.get('limit', ['0'])), list) else query_params.get('limit', post_data.get('limit', '0')))
            except (ValueError, TypeError):
                records_accessed = 0
        
        # Use AuditLogger to create event with automatic categorization and scoring
        AuditLogger.log_event(
            action=action,
            resource=f"{method} {path}",
            medical_domain=medical_domain,
            patient_count=patient_count,
            records_accessed=records_accessed,
            columns_accessed=columns_accessed,
            **audit_data
        )

    def _determine_action(self, method: str, path: str, audit_data: Dict[str, Any]) -> str:
        """Determine action name based on method, path, and context."""
        # Authentication actions
        if 'login' in path:
            return 'LOGIN_ATTEMPT' if audit_data['success'] else 'FAILED_LOGIN'
        elif 'logout' in path:
            return 'LOGOUT'
        elif 'password' in path:
            return 'PASSWORD_CHANGE'
        
        # Admin actions
        elif '/admin/' in path:
            if method == 'DELETE':
                return 'ADMIN_DELETE'
            elif method == 'POST':
                return 'ADMIN_CREATE'
            elif method in ['PUT', 'PATCH']:
                return 'ADMIN_UPDATE'
            else:
                return 'ADMIN_ACCESS'
        
        # Data access actions
        elif any(pattern in path for pattern in self.DATA_ACCESS_PATTERNS):
            if method == 'DELETE':
                return 'DATA_DELETE'
            elif 'export' in path or 'download' in path:
                return 'DATA_EXPORT'
            elif method == 'POST':
                return 'DATA_CREATE'
            elif method in ['PUT', 'PATCH']:
                return 'DATA_UPDATE'
            else:
                return 'DATA_ACCESS'
        
        # API actions
        elif '/api/' in path:
            if method == 'DELETE':
                return 'API_DELETE'
            elif method == 'POST':
                return 'API_CREATE'
            elif method in ['PUT', 'PATCH']:
                return 'API_UPDATE'
            else:
                return 'API_ACCESS'
        
        # User management actions
        elif '/users/' in path:
            if method == 'DELETE':
                return 'USER_DELETE'
            elif method == 'POST':
                return 'USER_CREATE'
            elif method in ['PUT', 'PATCH']:
                return 'USER_UPDATE'
            else:
                return 'USER_ACCESS'
        
        # Default HTTP action
        return f'HTTP_{method}'

    def _get_client_ip(self, request) -> str:
        """Extract real client IP address."""
        # Check X-Forwarded-For header first
        xff = request.META.get('HTTP_X_FORWARDED_FOR')
        if xff:
            return xff.split(',')[0].strip()
        
        # Check X-Real-IP header
        x_real_ip = request.META.get('HTTP_X_REAL_IP')
        if x_real_ip:
            return x_real_ip.strip()
        
        # Fall back to REMOTE_ADDR
        return request.META.get('REMOTE_ADDR', '')

    def _calculate_duration(self, request) -> Optional[int]:
        """Calculate request duration in milliseconds."""
        start = getattr(request, '_audit_start_time', None)
        if isinstance(start, float):
            return int((time.time() - start) * 1000)
        return None

    def _get_request_size(self, request) -> Optional[int]:
        """Calculate request size in bytes."""
        try:
            content_length = request.META.get('CONTENT_LENGTH')
            if content_length:
                return int(content_length)
        except (ValueError, TypeError):
            pass
        
        # Fallback: estimate from body if available
        if hasattr(request, 'body'):
            return len(request.body)
        
        return None

    def _sanitize_post_data(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from POST data before logging."""
        sensitive_fields = [
            'password', 'passwd', 'pwd', 'secret', 'token', 'key',
            'csrf', 'csrfmiddlewaretoken', 'api_key', 'auth_token'
        ]
        
        sanitized = {}
        for field, value in post_data.items():
            if field.lower() in sensitive_fields:
                sanitized[field] = '[REDACTED]'
            else:
                sanitized[field] = value
        
        return sanitized





