from typing import Dict


def nav_permissions(request) -> Dict[str, bool]:
    user = getattr(request, 'user', None)
    role_name = None
    if user and getattr(user, 'is_authenticated', False):
        role = getattr(user, 'role', None)
        role_name = getattr(role, 'name', None)
    # Treat Django superusers as admin for UI controls as well
    is_admin_role = (role_name == 'ADMIN') or getattr(user, 'is_superuser', False)
    return {
        'is_admin_role': is_admin_role,
    }


