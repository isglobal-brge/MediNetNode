"""
Migration: add APIKey.key_prefix field and mark existing keys as '__LEGACY__'.

key_prefix stores the first 8 chars of the raw API key for an indexed pre-filter
before bcrypt hash verification, reducing key lookup from O(n·bcrypt) to O(1·index).

Existing keys cannot have their prefix recovered (raw key was never stored), so they
are marked with the sentinel '__LEGACY__'. The authentication middleware handles
legacy keys transparently — they still work but trigger a deprecation response header.
"""
from django.db import migrations, models


def mark_existing_keys_as_legacy(apps, schema_editor):
    """Set key_prefix='__LEGACY__' on all keys that exist before this migration."""
    APIKey = apps.get_model('users', 'APIKey')
    APIKey.objects.filter(key_prefix='').update(key_prefix='__LEGACY__')


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0009_add_api_key_hashing'),
    ]

    operations = [
        migrations.AddField(
            model_name='apikey',
            name='key_prefix',
            field=models.CharField(
                blank=True,
                db_index=True,
                default='',
                help_text=(
                    "First 8 chars of raw key for indexed pre-filter before hash verification. "
                    "Non-secret. '__LEGACY__' marks keys created before this field was added."
                ),
                max_length=8,
            ),
        ),
        migrations.RunPython(
            mark_existing_keys_as_legacy,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
