"""
Management command to create a test MEMBER user for development/testing.
Creates user: member / password: test123
"""
from django.core.management.base import BaseCommand
from users.models import CustomUser, Role


class Command(BaseCommand):
    help = 'Create a test MEMBER user (username: member, password: test123)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            default='member',
            help='Username for test MEMBER user (default: member)',
        )
        parser.add_argument(
            '--password',
            type=str,
            default='test123',
            help='Password for test MEMBER user (default: test123)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete existing user if exists',
        )

    def handle(self, *args, **options):
        """Create test MEMBER user."""
        username = options['username']
        password = options['password']
        force = options.get('force', False)

        # Get or create MEMBER role
        try:
            member_role = Role.objects.get(name='MEMBER')
            self.stdout.write(f'[OK] Found MEMBER role')
        except Role.DoesNotExist:
            self.stdout.write(
                self.style.ERROR('[ERROR] MEMBER role not found. Run: python manage.py setup_roles')
            )
            return

        # Check if user exists
        if CustomUser.objects.filter(username=username).exists():
            if force:
                CustomUser.objects.filter(username=username).delete()
                self.stdout.write(
                    self.style.WARNING(f'[OK] Deleted existing user: {username}')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'[ERROR] User "{username}" already exists. Use --force to recreate.')
                )
                return

        # Create test MEMBER user
        user = CustomUser.objects.create_user(
            username=username,
            password=password,
            email=f'{username}@test.com',
            role=member_role,
            is_active=True
        )

        self.stdout.write(
            self.style.SUCCESS(f'\n[OK] Test MEMBER user created successfully!')
        )
        self.stdout.write(f'  Username: {username}')
        self.stdout.write(f'  Password: {password}')
        self.stdout.write(f'  Email: {user.email}')
        self.stdout.write(f'  Role: {user.role.name}')
        self.stdout.write(f'  Active: {user.is_active}')
        self.stdout.write(f'\n  Login at: http://localhost:5001/auth/login/')

        # Show permissions
        if member_role.permissions:
            self.stdout.write(f'\n  Permissions ({len(member_role.permissions)}):')
            for perm, value in member_role.permissions.items():
                self.stdout.write(f'    - {perm}: {value}')
