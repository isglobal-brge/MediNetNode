from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth import authenticate, login
from django_ratelimit.decorators import ratelimit


@require_http_methods(["POST"])
@csrf_protect
@ratelimit(key="ip", rate="5/5m", block=True)
@ratelimit(key="post:username", rate="10/h", block=True)
def login_view(request):
    username = request.POST.get("username", "")
    password = request.POST.get("password", "")
    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({"ok": False, "error": "invalid_credentials"}, status=400)
    login(request, user)
    
    # Initialize session activity timestamp
    from django.utils import timezone
    request.session['last_activity_ts'] = int(timezone.now().timestamp())
    request.session.modified = True
    
    return JsonResponse({"ok": True})

from django.shortcuts import render, redirect
from django.http import HttpResponseForbidden
from django.views.decorators.csrf import requires_csrf_token
from django.views.decorators.cache import never_cache
from functools import wraps
from django.middleware.csrf import get_token
import logging

security_logger = logging.getLogger('security')
@require_http_methods(["GET", "POST"])
@csrf_protect
@ratelimit(key="ip", rate="5/5m", method=['POST'], block=True)
@ratelimit(key="post:username", rate="10/h", method=['POST'], block=True)
def login_page(request):
    """HTML login page with POST handling and role-based redirect support."""
    next_url = request.GET.get('next') or request.POST.get('next')

    if request.method == 'GET':
        # For GET requests, only pass next_url if explicitly provided
        return render(request, 'auth/login.html', {
            'next': next_url,  # Don't set default here - let role-based redirect handle it
            'error': None,
        })

    # POST branch
    username = request.POST.get("username", "")
    password = request.POST.get("password", "")
    user = authenticate(request, username=username, password=password)
    if user is None:
        return render(request, 'auth/login.html', {
            'next': next_url,  # Don't set default here - let role-based redirect handle it
            'error': 'Invalid username or password.'
        }, status=400)

    login(request, user)
    
    # Initialize session activity timestamp
    from django.utils import timezone
    request.session['last_activity_ts'] = int(timezone.now().timestamp())
    request.session.modified = True
    
    # If there's a specific next URL, use it
    if next_url and next_url.strip():
        return redirect(next_url)
    
    # Default redirect based on user role
    try:
        if user.role and hasattr(user.role, 'name'):
            role_name = user.role.name
            if role_name == 'ADMIN':
                return redirect('admin_dashboard')
            elif role_name == 'AUDITOR':
                return redirect('audit:auditor_dashboard')
            elif role_name == 'RESEARCHER':
                return redirect('researcher_info')
    except AttributeError:
        pass
    
    # Fallback for users without roles or any errors
    return redirect('admin_dashboard')



@never_cache
@requires_csrf_token
def csrf_failure(request, reason=""):
    """Handle CSRF failures with security logging."""
    ip = get_client_ip(request)
    security_logger.warning(
        f'CSRF_FAILURE from {ip}: {reason}',
        extra={
            'user': getattr(request, 'user', None),
            'ip': ip,
            'reason': reason,
            'path': request.path,
            'referer': request.META.get('HTTP_REFERER', '')
        }
    )
    
    return HttpResponseForbidden(
        '<h1>403 Forbidden</h1><p>CSRF verification failed. Request aborted.</p>',
        content_type='text/html'
    )


def csrf_validate(view_func):
    """Custom CSRF validation decorator with enhanced security logging."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Additional CSRF validations can be added here
        if request.method == 'POST':
            token = get_token(request)
            if not token:
                security_logger.warning(
                    f'CSRF_NO_TOKEN from {get_client_ip(request)}',
                    extra={'user': getattr(request, 'user', None)}
                )
        
        return view_func(request, *args, **kwargs)
    
    return wrapper


def get_client_ip(request):
    """Get the real client IP address."""
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def logout_view(request):
    """Custom logout view with message and redirect."""
    from django.contrib.auth import logout
    from django.shortcuts import redirect
    from django.contrib import messages
    
    if request.user.is_authenticated:
        username = request.user.username
        logout(request)
        messages.success(request, f"You have been successfully logged out. Thank you, {username}!")
    
    return redirect('login')
