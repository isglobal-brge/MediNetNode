"""
Management command to audit and manage legacy API keys.

'Legacy' keys were created before migration 0010 added the key_prefix field.
They are marked with key_prefix='__LEGACY__' because their first-8-char prefix
cannot be recovered without the original plaintext key.

Usage:
    python manage.py audit_legacy_api_keys           # report only
    python manage.py audit_legacy_api_keys --deactivate   # deactivate all legacy keys
    python manage.py audit_legacy_api_keys --user alice   # filter to one user
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from users.models import APIKey

LEGACY_SENTINEL = '__LEGACY__'


class Command(BaseCommand):
    help = (
        "Report legacy API keys (key_prefix='__LEGACY__') that pre-date prefix indexing. "
        "Optionally deactivate them to force re-issuance of new indexed keys."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--deactivate',
            action='store_true',
            help='Deactivate all active legacy keys (requires owner to generate new key)',
        )
        parser.add_argument(
            '--user',
            type=str,
            metavar='USERNAME',
            help='Limit audit to a single user',
        )
        parser.add_argument(
            '--active-only',
            action='store_true',
            default=False,
            help='Report only active legacy keys (skip already-inactive ones)',
        )

    def handle(self, *args, **options):
        qs = APIKey.objects.filter(key_prefix=LEGACY_SENTINEL).select_related('user', 'user__role')

        if options['user']:
            qs = qs.filter(user__username=options['user'])
            if not qs.exists():
                self.stdout.write(
                    self.style.WARNING(
                        f"No legacy keys found for user '{options['user']}'."
                    )
                )
                return

        if options['active_only']:
            qs = qs.filter(is_active=True)

        total = qs.count()
        active_count = qs.filter(is_active=True).count()

        if total == 0:
            self.stdout.write(self.style.SUCCESS("No legacy API keys found. All keys are indexed."))
            return

        # ── Report ────────────────────────────────────────────────────────────
        self.stdout.write(self.style.WARNING(
            f"\nLegacy API keys: {total} total, {active_count} active"
        ))
        self.stdout.write('=' * 70)

        for key in qs.order_by('user__username', '-created_at'):
            status = 'ACTIVE' if key.is_active else 'inactive'
            expired = ''
            if key.expires_at and key.expires_at < timezone.now():
                expired = ' [EXPIRED]'

            self.stdout.write(
                f"  user={key.user.username:<20} "
                f"name={key.name:<25} "
                f"status={status}{expired}"
            )
            self.stdout.write(
                f"    created={key.created_at.strftime('%Y-%m-%d')}  "
                f"last_used={key.last_used_at.strftime('%Y-%m-%d') if key.last_used_at else 'never'}"
            )

        self.stdout.write('=' * 70)

        # ── Optional deactivation ─────────────────────────────────────────────
        if options['deactivate']:
            active_qs = qs.filter(is_active=True)
            deactivated = active_qs.update(is_active=False)
            self.stdout.write(self.style.SUCCESS(
                f"\nDeactivated {deactivated} legacy key(s). "
                "Affected users must generate new API keys via the researcher portal."
            ))
        else:
            if active_count > 0:
                self.stdout.write(self.style.WARNING(
                    f"\n{active_count} active legacy key(s) are still in use. "
                    "These lack prefix indexing, causing full-table scans on auth. "
                    "Run with --deactivate to force re-issuance."
                ))
