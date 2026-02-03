# Generated migration for MD5 to SHA-256 security upgrade
#
# SECURITY FIX: Migrating from MD5 (vulnerable to collision attacks) to SHA-256
#
# Migration Strategy:
# 1. Add new checksum_sha256 field (64 chars)
# 2. Keep old checksum_md5 field temporarily (for rollback capability)
# 3. Data migration handled at application level (checksums recalculated on next access)
# 4. Future migration (0003) will remove checksum_md5 after validation period
#
# IMPORTANT: All existing datasets will need checksum recalculation
# This is handled automatically when datasets are accessed/validated

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0001_initial'),
    ]

    operations = [
        # Step 1: Add new SHA-256 checksum field (allows NULL initially for existing records)
        migrations.AddField(
            model_name='dataset',
            name='checksum_sha256',
            field=models.CharField(
                editable=False,
                max_length=64,
                null=True,
                blank=True,
                help_text='SHA-256 checksum for file integrity verification (replaces MD5)'
            ),
        ),

        # Step 2: Rename old field to make it clear it's deprecated
        migrations.RenameField(
            model_name='dataset',
            old_name='checksum_md5',
            new_name='checksum_md5_deprecated',
        ),

        # Step 3: Alter the deprecated field to allow NULL (existing records keep MD5)
        migrations.AlterField(
            model_name='dataset',
            name='checksum_md5_deprecated',
            field=models.CharField(
                editable=False,
                max_length=32,
                null=True,
                blank=True,
                help_text='DEPRECATED: MD5 checksum (vulnerable to collisions). Use checksum_sha256 instead.'
            ),
        ),
    ]
