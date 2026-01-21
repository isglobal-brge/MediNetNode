"""
Test utility functions to help with common test setup patterns.
"""
from users.models import Role


def get_or_create_role(name, permissions=None):
    """
    Get or create a role with the given name and permissions.
    This prevents unique constraint violations when roles are created
    automatically via post_migrate signals.

    Args:
        name: Role name (e.g., 'ADMIN', 'RESEARCHER', 'AUDITOR')
        permissions: Dict of permissions (defaults to empty dict)

    Returns:
        Role instance
    """
    if permissions is None:
        permissions = {}

    role, created = Role.objects.get_or_create(
        name=name,
        defaults={'permissions': permissions}
    )
    return role