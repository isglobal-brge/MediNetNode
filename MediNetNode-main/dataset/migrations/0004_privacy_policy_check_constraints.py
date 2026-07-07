from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0003_add_dataset_privacy_policy'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='datasetprivacypolicy',
            options={
                'ordering': ['-created_at'],
                'verbose_name': 'Dataset Privacy Policy',
                'verbose_name_plural': 'Dataset Privacy Policies',
            },
        ),
        migrations.AddConstraint(
            model_name='datasetprivacypolicy',
            constraint=models.CheckConstraint(
                check=models.Q(max_epsilon_per_job__gt=0.0),
                name='privacy_policy_max_eps_positive',
            ),
        ),
        migrations.AddConstraint(
            model_name='datasetprivacypolicy',
            constraint=models.CheckConstraint(
                check=models.Q(lifetime_budget__gt=0.0),
                name='privacy_policy_lifetime_positive',
            ),
        ),
        migrations.AddConstraint(
            model_name='datasetprivacypolicy',
            constraint=models.CheckConstraint(
                check=models.Q(spent_epsilon__gte=0.0),
                name='privacy_policy_spent_nonneg',
            ),
        ),
    ]
