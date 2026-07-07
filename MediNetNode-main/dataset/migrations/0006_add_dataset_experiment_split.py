from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0005_add_researcher_epsilon_budget'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='experiment_file_path',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='experiment_row_count',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='experiment_split_ratio',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
