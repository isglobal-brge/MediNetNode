from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission


def ensure_admin_superuser_setup(sender, **kwargs):
    # Ensure base roles exist with appropriate permissions
    # These permissions must match setup_roles.py for consistency
    from .models import Role

    default_permissions = {
        'ADMIN': {
            'api.access': True,
            'dataset.view': True,
            'dataset.train': True,
            'dataset.create': True,
            'dataset.edit': True,
            'dataset.delete': True,
            'user.view': True,
            'user.create': True,
            'user.edit': True,
            'user.delete': True,
            'audit.view': True,
            'training.view': True,
            'training.manage': True,
            'system.admin': True,
        },
        'RESEARCHER': {
            'api.access': True,
            'dataset.view': True,
            'dataset.train': True,
        },
        'AUDITOR': {
            'dataset.view': True,
            'audit.view': True,
            'training.view': True,
            'user.view': True,
        }
    }

    for role_name in ['ADMIN', 'RESEARCHER', 'AUDITOR']:
        Role.objects.get_or_create(
            name=role_name,
            defaults={'permissions': default_permissions.get(role_name, {})}
        )

    # Ensure a Django 'Admin' group exists with all permissions
    admin_group, _ = Group.objects.get_or_create(name='Admin')
    admin_group.permissions.set(Permission.objects.all())
    admin_group.save()

    # Add all existing superusers to the Admin group and mark them as staff
    User = get_user_model()
    for su in User.objects.filter(is_superuser=True):
        su.groups.add(admin_group)
        if not su.is_staff:
            su.is_staff = True
            su.save(update_fields=['is_staff'])


