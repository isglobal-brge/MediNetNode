from typing import Dict


def nav_permissions(request) -> Dict[str, bool]:
    user = getattr(request, 'user', None)
    role_name = None
    if user and getattr(user, 'is_authenticated', False):
        role = getattr(user, 'role', None)
        role_name = getattr(role, 'name', None)
    # Treat Django superusers as admin for UI controls as well
    is_admin_role = (role_name == 'ADMIN') or getattr(user, 'is_superuser', False)
    is_member_role = (role_name == 'MEMBER')
    is_researcher_role = (role_name == 'RESEARCHER')
    is_auditor_role = (role_name == 'AUDITOR')
    return {
        'is_admin_role': is_admin_role,
        'is_member_role': is_member_role,
        'is_researcher_role': is_researcher_role,
        'is_auditor_role': is_auditor_role,
        'user_role_name': role_name,
    }


