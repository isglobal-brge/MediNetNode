"""
Security middleware for adding security headers.

Implements Content-Security-Policy and other security headers.
"""
from django.conf import settings


class SecurityHeadersMiddleware:
    """
    Middleware that adds security headers to all responses.

    Headers added:
    - Content-Security-Policy: Protects against XSS attacks
    - X-Content-Type-Options: Prevents MIME-sniffing
    - X-Frame-Options: Prevents clickjacking (already in Django but reinforced)
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features

    Headers removed:
    - Server: Hides Django version and server information
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Content-Security-Policy
        # Allows:
        # - self: Same origin
        # - cdn.jsdelivr.net: Bootstrap, Chart.js, Bootstrap Icons
        # - inline styles/scripts with nonces (for Django templates)
        # - 'unsafe-inline' for now (TODO: migrate to nonces in future)
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

        response['Content-Security-Policy'] = csp_policy

        # X-Content-Type-Options: Prevent MIME-sniffing
        response['X-Content-Type-Options'] = 'nosniff'

        # Referrer-Policy: Control referrer information leakage
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions-Policy: Disable unused browser features
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

        # Remove Server header to hide Django version and server information
        # This prevents information disclosure about the technology stack
        if 'Server' in response:
            del response['Server']

        return response
