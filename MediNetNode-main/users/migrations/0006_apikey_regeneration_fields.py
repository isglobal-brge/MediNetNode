# Generated manually for APIKey regeneration fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0005_apikey_apirequest_apikey_users_apike_key_fe45f4_idx_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='apikey',
            name='regenerated_count',
            field=models.IntegerField(default=0, help_text='Number of times this key has been regenerated'),
        ),
        migrations.AddField(
            model_name='apikey',
            name='last_regenerated_at',
            field=models.DateTimeField(blank=True, help_text='Last time the key was regenerated', null=True),
        ),
    ]