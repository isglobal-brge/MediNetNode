"""
Base role definitions for the core MediNet platform.

Imported by ``MediNetCoreConfig.ready()`` to register ADMIN, MEMBER,
RESEARCHER, and AUDITOR before any request is served.

These definitions are the single source of truth for default permissions.
The ``setup_roles`` management command reads from ``role_registry.all()``
so adding a new base role here is sufficient — no command changes needed.
"""
from .registry import RoleDefinition, role_registry

_BASE_ROLES = [
    RoleDefinition(
        name='ADMIN',
        display_name='Administrator',
        description='Full platform access including user management and system settings.',
        permissions={
            'api.access': True,
            # Datasets
            'dataset.view': True,
            'dataset.train': True,
            'dataset.create': True,
            'dataset.edit': True,
            'dataset.delete': True,
            # Users
            'user.view': True,
            'user.create': True,
            'user.edit': True,
            'user.delete': True,
            # Audit
            'audit.view': True,
            # Training
            'training.view': True,
            'training.manage': True,
            # System
            'system.admin': True,
            # Inference
            'inference.execute': {'scope': 'ALL'},
            'inference.upload': {'scope': 'ALL'},
            'inference.approve': True,
            'inference.admin': True,
        },
    ),
    RoleDefinition(
        name='MEMBER',
        display_name='Member',
        description='Clinical staff with dataset and inference access.',
        permissions={
            'api.access': True,
            'dataset.view': {'scope': 'ALL'},
            'dataset.create': True,
            'dataset.train': {'scope': 'ALL'},
            'training.view': True,
            'inference.execute': {'scope': 'ALL'},
            'inference.upload': True,
        },
    ),
    RoleDefinition(
        name='RESEARCHER',
        display_name='Researcher',
        description='External researcher with stateless API access only.',
        permissions={
            'api.access': True,
            'dataset.view': {'scope': 'ALL'},
            'dataset.train': {'scope': 'ALL'},
            'inference.execute': {'scope': 'ALL'},
        },
    ),
    RoleDefinition(
        name='AUDITOR',
        display_name='Auditor',
        description='Read-only access to audit logs, datasets, trainings, and users.',
        permissions={
            'dataset.view': True,
            'audit.view': True,
            'training.view': True,
            'user.view': True,
            'inference.view': True,
        },
    ),
]


def register_base_roles() -> None:
    """Register all base roles.  Called from MediNetCoreConfig.ready()."""
    for role_def in _BASE_ROLES:
        role_registry.register(role_def)
