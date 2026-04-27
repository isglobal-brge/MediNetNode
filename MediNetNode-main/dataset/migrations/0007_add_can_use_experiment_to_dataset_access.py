from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0006_add_dataset_experiment_split'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasetaccess',
            name='can_use_experiment',
            field=models.BooleanField(default=False),
        ),
    ]
