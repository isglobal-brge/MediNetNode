"""
Django management command to generate API keys for RESEARCHER users.

Usage examples:
  # Single IP
  python manage.py generate_api_key researcher1 --ips 203.0.113.10 --name "Lab PC"

  # Multiple IPs
  python manage.py generate_api_key researcher1 --ips 203.0.113.10,203.0.113.20 --name "Lab Network"

  # CIDR range
  python manage.py generate_api_key researcher1 --ips 203.0.113.0/24 --name "University Network"

  # Mixed (single IPs + CIDR)
  python manage.py generate_api_key researcher1 --ips 203.0.113.10,192.168.1.0/24 --name "Mixed Access"
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
            help='Comma-separated list of allowed IP addresses or CIDR ranges (e.g., 203.0.113.10,192.168.1.0/24)'
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
            user = User.objects.get(username=username)

            if not user.role or user.role.name != 'RESEARCHER':
                raise CommandError(
                    f'User {username} does not have RESEARCHER role. '
                    f'Current role: {user.role.name if user.role else "None"}'
                )

            expires_at = None
            if expires_days:
                expires_at = timezone.now() + timedelta(days=expires_days)

            raw_key = APIKey.generate_api_key()

            api_key = APIKey(
                user=user,
                name=key_name,
                ip_whitelist=ip_list,
                expires_at=expires_at
            )
            api_key.set_key(raw_key)
            api_key.save()

            # Show raw key ONLY THIS ONE TIME — it is stored hashed and cannot be retrieved later
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nSuccessfully created API key for {username}'
                )
            )
            self.stdout.write(self.style.WARNING('\n' + '='*60))
            self.stdout.write(self.style.WARNING('  API KEY (SAVE THIS - WILL NOT BE SHOWN AGAIN)'))
            self.stdout.write(self.style.WARNING('='*60))
            self.stdout.write(f'\n  {raw_key}\n')
            self.stdout.write(self.style.WARNING('='*60))
            self.stdout.write(f'\nName: {key_name}')
            self.stdout.write(f'Allowed IPs: {", ".join(ip_list)}')

            if expires_at:
                self.stdout.write(f'Expires: {expires_at.strftime("%Y-%m-%d %H:%M:%S")}')
            else:
                self.stdout.write('Expires: Never')

            self.stdout.write(
                self.style.WARNING(
                    '\nWARNING: SECURITY WARNING: The API key above is hashed in the database.\n'
                    '   It cannot be retrieved later. Store it securely now!\n'
                )
            )

        except User.DoesNotExist:
            raise CommandError(f'User "{username}" does not exist')
        except Exception as e:
            raise CommandError(f'Error creating API key: {str(e)}')