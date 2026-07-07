from functools import wraps
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import logging

_security_logger = logging.getLogger('security')


def _render_access_denied(request):
    """
    Render the 403 access denied page and emit a structured security log entry.

    Used by all access-control decorators so that every denial produces:
    - A consistent HTTP 403 response (renders access_denied.html)
    - A security.WARNING log with user, path, method, and client IP
    """
    _security_logger.warning(
        f"ACCESS_DENIED: user={getattr(request.user, 'username', 'anonymous')} "
        f"path={request.path} method={request.method} "
        f"ip={request.META.get('REMOTE_ADDR', 'unknown')}"
    )
    return render(request, 'access_denied.html', status=403)


def require_role(*allowed_roles):
    """
    Restrict view access to users with one of the specified roles.

    Usage:
        @require_role('ADMIN')
        @require_role('ADMIN', 'MEMBER')
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            # H9: Django superusers do NOT bypass RBAC. The 4-role model
            # (ADMIN/MEMBER/RESEARCHER/AUDITOR) is the single authorization
            # boundary. The platform admin is created with the ADMIN role
            # (see core/views/setup.py), so it still passes on that basis.
            if not request.user.role:
                return _render_access_denied(request)
            if request.user.role.name not in allowed_roles:
                return _render_access_denied(request)
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_permission(*permissions, domain=None):
    """
    Restrict view access to users with one or more specific permissions.

    Supports both simple boolean permissions and scope-based permissions.

    Usage:
        @require_permission('user.create')
        @require_permission('user.create', 'user.view')
        @require_permission('inference.execute', domain='cardiology')

    Args:
        *permissions: One or more permission keys (any match grants access)
        domain: Optional domain for scope-based permission checks
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            # H9: no superuser bypass — permissions come only from the role.
            if not request.user.role:
                return _render_access_denied(request)
            has_permission = any(
                request.user.has_permission(perm, domain=domain) for perm in permissions
            )
            if not has_permission:
                return _render_access_denied(request)
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def admin_required(view_func):
    """Restrict view access to ADMIN users only."""
    return require_role('ADMIN')(view_func)


def check_model_access(permission_key):
    """
    Check domain-scoped permission for a specific deployed model.

    Reads model_id from URL kwargs ONLY (never from GET/POST params) to
    prevent parameter injection attacks that could bypass domain scope checks.

    Usage:
        @check_model_access('inference.execute')
        def predict_view(request, model_id):
            ...
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            # H9: no superuser bypass — access comes only from the role/permission.
            if not request.user.role:
                return _render_access_denied(request)

            # URL kwargs ONLY — GET/POST params are attacker-controlled
            model_id = kwargs.get('model_id')

            if not model_id:
                if not request.user.has_permission(permission_key):
                    return _render_access_denied(request)
                return view_func(request, *args, **kwargs)

            try:
                from inference.models import DeployedModel
                model = DeployedModel.objects.get(id=model_id)
                if not request.user.has_permission(permission_key, domain=model.domain):
                    return _render_access_denied(request)
            except ImportError:
                if not request.user.has_permission(permission_key):
                    return _render_access_denied(request)
            except Exception:
                return _render_access_denied(request)

            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator
