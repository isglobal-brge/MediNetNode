from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0002_migrate_checksum_to_sha256'),
    ]

    operations = [
        migrations.CreateModel(
            name='DatasetPrivacyPolicy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sensitivity', models.CharField(
                    choices=[
                        ('high', 'Alta — diagnóstico, salud mental, genómica'),
                        ('medium', 'Media — riesgo cardiovascular, general'),
                        ('low', 'Baja — estadísticas agregadas'),
                    ],
                    default='medium',
                    max_length=10,
                )),
                ('max_epsilon_per_job', models.FloatField(
                    help_text='Maximum ε allowed per training job (Node-enforced).',
                )),
                ('lifetime_budget', models.FloatField(
                    help_text='Total ε allowed across all jobs for this dataset.',
                )),
                ('spent_epsilon', models.FloatField(
                    default=0.0,
                    help_text='Cumulative ε spent across all completed jobs.',
                )),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('dataset', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='privacy_policy',
                    to='dataset.dataset',
                )),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
