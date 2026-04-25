import sys
sys.modules.setdefault('magic', None)

import pytest
from unittest.mock import MagicMock, patch
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _setup_fixtures(username="rps_r1", ds_name="rps_ds1"):
    role = Role.objects.get(name='RESEARCHER')
    researcher = User.objects.create_user(username=username, password="x")
    researcher.role = role
    researcher.save()

    ds = Dataset(
        name=ds_name, description="t", file_path=f"/f/{ds_name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    ds = Dataset.objects.get(name=ds_name)

    policy = DatasetPrivacyPolicy.objects.create(
        dataset=ds, sensitivity='high',
        max_epsilon_per_job=0.5, lifetime_budget=2.0,
    )
    budget, _ = ResearcherEpsilonBudget.get_or_create_for(
        dataset=ds, researcher_id=researcher.id, policy=policy,
    )
    return researcher, ds, policy, budget


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestRecordPrivacySpendUpdatesResearcherBudget:

    def setup_method(self):
        self.researcher, self.ds, self.policy, self.budget = _setup_fixtures()

    def test_researcher_budget_updated_after_job(self):
        from api.federated.utils import _record_privacy_spend

        session = MagicMock()
        session.user_id = self.researcher.id

        round_mock = MagicMock()
        round_mock.metrics = {'privacy_epsilon': 0.4}
        session.rounds.order_by.return_value.first.return_value = round_mock

        with patch('api.federated.utils.DatasetPrivacyPolicy.objects') as mock_policy_qs, \
             patch('api.federated.utils.ResearcherEpsilonBudget.objects') as mock_budget_qs:

            mock_policy_qs.get.return_value = self.policy
            mock_budget_qs.get.return_value = self.budget
            mock_budget_qs.filter.return_value.update.return_value = 1

            session.model_config = {
                'model': {'dataset': {'selected_datasets': [{'dataset_id': self.ds.id}]}}
            }

            _record_privacy_spend(session)

            mock_budget_qs.filter.assert_called()
