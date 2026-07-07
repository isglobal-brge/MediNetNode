"""Client IP resolution that resists header spoofing.

A single source of truth for extracting the client IP from a request. X-Forwarded-For
(and X-Real-IP) are only honored when the direct peer (``REMOTE_ADDR``) is a
configured trusted proxy; otherwise those headers are client-controlled and would
let an attacker rotate rate-limit buckets or falsify forensic/audit trails.
"""
from django.conf import settings


def get_trusted_client_ip(request):
    """Return the client IP, trusting proxy headers only from trusted proxies."""
    remote_addr = request.META.get('REMOTE_ADDR', '0.0.0.0')
    trusted_proxies = getattr(settings, 'TRUSTED_PROXIES', [])
    if trusted_proxies and remote_addr in trusted_proxies:
        xff = request.META.get('HTTP_X_FORWARDED_FOR')
        if xff:
            return xff.split(',')[0].strip()
        x_real_ip = request.META.get('HTTP_X_REAL_IP')
        if x_real_ip:
            return x_real_ip.strip()
    return remote_addr
