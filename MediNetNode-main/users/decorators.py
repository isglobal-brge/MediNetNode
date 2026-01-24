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


def require_permission(*permissions, domain=None):
    """
    Decorator that restricts access to users with specific permissions.

    Supports both simple boolean permissions and scope-based permissions.

    Usage:
        @require_permission('user.create')
        @require_permission('user.create', 'user.view')
        @require_permission('inference.execute', domain='cardiology')

    Args:
        *permissions: One or more permission keys to check
        domain: Optional domain to check against scope-based permissions
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
                request.user.has_permission(perm, domain=domain) for perm in permissions
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


def check_model_access(permission_key):
    """
    Decorator that checks if user has permission to access a specific model's domain.

    Usage:
        @check_model_access('inference.execute')
        def predict_view(request, model_id):
            # model_id will be used to check domain access
            pass

    The decorator expects a 'model_id' parameter in the view and will check
    if the user has permission for that model's domain.
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

            # Get model_id from kwargs or request
            model_id = kwargs.get('model_id') or request.GET.get('model_id') or request.POST.get('model_id')

            if not model_id:
                # If no model_id, just check basic permission
                if not request.user.has_permission(permission_key):
                    messages.error(request, "You do not have permission to perform this action.")
                    return HttpResponseForbidden("Access denied: Insufficient permissions")
                return view_func(request, *args, **kwargs)

            # Import here to avoid circular dependency
            try:
                from inference.models import DeployedModel
                model = DeployedModel.objects.get(id=model_id)

                # Check permission with domain
                if not request.user.has_permission(permission_key, domain=model.domain):
                    messages.error(request, f"You do not have permission to access models in the {model.domain} domain.")
                    return HttpResponseForbidden("Access denied: Domain not in scope")

            except ImportError:
                # inference app not installed yet, fallback to basic permission check
                if not request.user.has_permission(permission_key):
                    messages.error(request, "You do not have permission to perform this action.")
                    return HttpResponseForbidden("Access denied: Insufficient permissions")
            except Exception as e:
                messages.error(request, "Error checking model access.")
                return HttpResponseForbidden(f"Access error: {str(e)}")

            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator