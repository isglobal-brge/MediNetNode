"""
Management command to set up default role permissions.
Ensures RESEARCHER, ADMIN, and AUDITOR roles have correct permissions.
"""
from django.core.management.base import BaseCommand
from users.models import Role


class Command(BaseCommand):
    help = 'Set up default role permissions for RESEARCHER, ADMIN, and AUDITOR roles'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update existing role permissions',
        )

    def handle(self, *args, **options):
        """Set up default permissions for each role."""
        force_update = options.get('force', False)
        
        # Define default permissions for each role
        role_permissions = {
            'RESEARCHER': {
                # API Access
                'api.access': True,
                # Datasets
                'dataset.view': {'scope': 'ALL'},
                'dataset.train': {'scope': 'ALL'},
                # Inference (NEW)
                'inference.execute': {'scope': 'ALL'},
            },
            'MEMBER': {
                # API Access
                'api.access': True,
                # Datasets
                'dataset.view': {'scope': 'ALL'},
                'dataset.create': True,
                'dataset.train': {'scope': 'ALL'},
                # Training
                'training.view': True,
                # Inference (NEW)
                'inference.execute': {'scope': 'ALL'},
                'inference.upload': True,
            },
            'ADMIN': {
                # API Access
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
                # Inference (NEW)
                'inference.execute': {'scope': 'ALL'},
                'inference.upload': {'scope': 'ALL'},
                'inference.approve': True,
                'inference.admin': True,
            },
            'AUDITOR': {
                # Datasets
                'dataset.view': True,
                # Audit
                'audit.view': True,
                # Training
                'training.view': True,
                # Users
                'user.view': True,
                # Inference (NEW)
                'inference.view': True,
            }
        }

        created_count = 0
        updated_count = 0
        
        for role_name, permissions in role_permissions.items():
            role, created = Role.objects.get_or_create(
                name=role_name,
                defaults={'permissions': permissions}
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'[OK] Created {role_name} role with permissions')
                )
            elif not role.permissions or force_update:
                # Update empty permissions or force update
                old_permissions = role.permissions.copy() if role.permissions else {}
                role.permissions = permissions
                role.save()
                updated_count += 1
                
                self.stdout.write(
                    self.style.WARNING(f'[OK] Updated {role_name} role permissions')
                )
                self.stdout.write(f'  Old: {old_permissions}')
                self.stdout.write(f'  New: {permissions}')
            else:
                self.stdout.write(f'- {role_name} role already has permissions (use --force to update)')

        # Summary
        if created_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'\n[OK] Created {created_count} roles with permissions')
            )
        
        if updated_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'[OK] Updated {updated_count} roles with permissions')
            )
            
        if created_count == 0 and updated_count == 0:
            self.stdout.write(
                self.style.WARNING('No roles were created or updated')
            )

        # Show current role status
        self.stdout.write('\n--- Current Role Status ---')
        for role in Role.objects.all().order_by('name'):
            permission_count = len(role.permissions) if role.permissions else 0
            self.stdout.write(f'{role.name}: {permission_count} permissions')
            if role.permissions:
                for perm, value in role.permissions.items():
                    self.stdout.write(f'  - {perm}: {value}')
            else:
                self.stdout.write('  (no permissions)')