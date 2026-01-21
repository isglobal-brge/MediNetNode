from functools import wraps
from django.http import HttpResponseForbidden
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.contrib import messages


def require_role(*allowed_roles):
    """
    Decorator that restricts access to users with specific roles.
    
    Usage:
        @require_role('ADMIN')
        @require_role('ADMIN', 'INVESTIGADOR')
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)
            if not request.user.role:
                messages.error(request, "You do not have permission to access this section.")
                return HttpResponseForbidden("Access denied: No role assigned")
            
            if request.user.role.name not in allowed_roles:
                messages.error(request, "You do not have permission to access this section.")
                return HttpResponseForbidden("Access denied: Insufficient permissions")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_permission(*permissions):
    """
    Decorator that restricts access to users with specific permissions.
    
    Usage:
        @require_permission('user.create')
        @require_permission('user.create', 'user.view')
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)
            if not request.user.role:
                messages.error(request, "You do not have permission to perform this action.")
                return HttpResponseForbidden("Access denied: No role assigned")
            
            # Check if user has any of the required permissions
            has_permission = any(
                request.user.has_permission(perm) for perm in permissions
            )
            
            if not has_permission:
                messages.error(request, "You do not have permission to perform this action.")
                return HttpResponseForbidden("Access denied: Insufficient permissions")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def admin_required(view_func):
    """
    Decorator that restricts access to ADMIN users only.
    """
    return require_role('ADMIN')(view_func)