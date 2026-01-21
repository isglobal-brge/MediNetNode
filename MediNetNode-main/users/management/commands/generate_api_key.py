"""
Django management command to generate API keys for RESEARCHER users.
Usage: python manage.py generate_api_key <username> --ips <ip1,ip2,...> --name <key_name>
"""
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from users.models import APIKey
from datetime import datetime, timedelta
from django.utils import timezone

User = get_user_model()


class Command(BaseCommand):
    help = 'Generate API key for RESEARCHER user with IP whitelist'

    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            type=str,
            help='Username of the RESEARCHER user'
        )
        parser.add_argument(
            '--ips',
            type=str,
            required=True,
            help='Comma-separated list of allowed IP addresses'
        )
        parser.add_argument(
            '--name',
            type=str,
            default='API Key',
            help='Descriptive name for the API key'
        )
        parser.add_argument(
            '--expires-days',
            type=int,
            default=None,
            help='Number of days until expiration (default: no expiration)'
        )

    def handle(self, *args, **options):
        username = options['username']
        ip_list = [ip.strip() for ip in options['ips'].split(',')]
        key_name = options['name']
        expires_days = options['expires_days']

        try:
            # Get user
            user = User.objects.get(username=username)
            
            # Validate user has RESEARCHER role
            if not user.role or user.role.name != 'RESEARCHER':
                raise CommandError(
                    f'User {username} does not have RESEARCHER role. '
                    f'Current role: {user.role.name if user.role else "None"}'
                )

            # Set expiration if specified
            expires_at = None
            if expires_days:
                expires_at = timezone.now() + timedelta(days=expires_days)

            # Create API key
            api_key = APIKey.objects.create(
                user=user,
                name=key_name,
                ip_whitelist=ip_list,
                expires_at=expires_at
            )

            # Output results
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created API key for {username}'
                )
            )
            self.stdout.write(f'API Key: {api_key.key}')
            self.stdout.write(f'Name: {key_name}')
            self.stdout.write(f'Allowed IPs: {", ".join(ip_list)}')
            
            if expires_at:
                self.stdout.write(f'Expires: {expires_at}')
            else:
                self.stdout.write('Expires: Never')

            self.stdout.write(
                self.style.WARNING(
                    '\nSECURITY WARNING: Store this API key securely. '
                    'It will not be shown again.'
                )
            )

        except User.DoesNotExist:
            raise CommandError(f'User "{username}" does not exist')
        except Exception as e:
            raise CommandError(f'Error creating API key: {str(e)}')