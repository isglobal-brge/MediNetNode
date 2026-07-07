from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trainings', '0002_add_budget_reset_request'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='use_experiment',
            field=models.BooleanField(
                default=False,
                help_text='True when the job runs on the experimental subset — epsilon spend is not recorded.',
            ),
        ),
    ]
