from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trainings', '0003_trainingsession_use_experiment'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='reserved_epsilon',
            field=models.FloatField(
                default=0.0,
                help_text='Epsilon reserved against the researcher budget at accept time (reconciled at end).',
            ),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='budget_reconciled',
            field=models.BooleanField(
                default=False,
                help_text='True once the reserved epsilon has been reconciled to actual spend (exactly-once guard).',
            ),
        ),
    ]
