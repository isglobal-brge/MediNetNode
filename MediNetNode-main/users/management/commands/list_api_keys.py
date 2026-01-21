"""
Django management command to list API keys for users.
Usage: python manage.py list_api_keys [--user <username>] [--show-keys]
"""
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import APIKey
from django.utils import timezone

User = get_user_model()


class Command(BaseCommand):
    help = 'List API keys with their status and usage information'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='Show API keys only for specific user'
        )
        parser.add_argument(
            '--show-keys',
            action='store_true',
            help='Show actual API key values (SECURITY RISK - use carefully)'
        )
        parser.add_argument(
            '--active-only',
            action='store_true',
            help='Show only active (non-expired) API keys'
        )

    def handle(self, *args, **options):
        # Build query
        queryset = APIKey.objects.select_related('user', 'user__role')
        
        if options['user']:
            try:
                user = User.objects.get(username=options['user'])
                queryset = queryset.filter(user=user)
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'User "{options["user"]}" does not exist')
                )
                return

        if options['active_only']:
            queryset = queryset.filter(is_active=True)

        # Get API keys
        api_keys = queryset.order_by('user__username', '-created_at')

        if not api_keys.exists():
            self.stdout.write(
                self.style.WARNING('No API keys found')
            )
            return

        # Display header
        self.stdout.write(
            self.style.SUCCESS('API Keys List:')
        )
        self.stdout.write('=' * 80)

        current_user = None
        for api_key in api_keys:
            # Group by user
            if current_user != api_key.user.username:
                current_user = api_key.user.username
                self.stdout.write(
                    f'\nUser: {current_user} (Role: {api_key.user.role.name if api_key.user.role else "None"})'
                )
                self.stdout.write('-' * 50)

            # Key info
            status_indicators = []
            if not api_key.is_active:
                status_indicators.append('INACTIVE')
            if api_key.is_expired():
                status_indicators.append('EXPIRED')
            
            status = ' '.join(status_indicators) if status_indicators else 'ACTIVE'
            
            self.stdout.write(f'  Name: {api_key.name}')
            
            if options['show_keys']:
                self.stdout.write(
                    self.style.ERROR(f'  Key: {api_key.key}')
                )
            else:
                self.stdout.write(f'  Key: ***{api_key.key[-8:]}')
            
            self.stdout.write(f'  Status: {status}')
            self.stdout.write(f'  Created: {api_key.created_at}')
            
            if api_key.expires_at:
                self.stdout.write(f'  Expires: {api_key.expires_at}')
            else:
                self.stdout.write('  Expires: Never')
            
            if api_key.last_used_at:
                self.stdout.write(f'  Last Used: {api_key.last_used_at}')
                self.stdout.write(f'  Last IP: {api_key.last_used_ip}')
            else:
                self.stdout.write('  Last Used: Never')
            
            self.stdout.write(f'  Allowed IPs: {", ".join(api_key.ip_whitelist)}')
            self.stdout.write('')

        if options['show_keys']:
            self.stdout.write(
                self.style.ERROR(
                    'WARNING: API keys were displayed in plain text. '
                    'Ensure your terminal/logs are secure.'
                )
            )