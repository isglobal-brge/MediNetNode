import sys
sys.modules.setdefault('magic', None)

import pytest
import json
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()

MODEL_JSON = {
    "model": {
        "metadata": {"model_type": "dl"},
        "dataset": {
            "selected_datasets": [{"dataset_id": 1}]
        },
        "training": {
            "optimizer": {"type": "Adam", "learning_rate": 0.001},
            "dp": {"noise_multiplier": 1.1, "max_grad_norm": 1.0},
        }
    },
    "train": {"rounds": 3, "epochs": 1, "batch_size": 32},
    "federated": {
        "name": "FedAvg",
        "parameters": {"fraction_fit": 1.0, "min_fit_clients": 1, "min_available_clients": 1}
    },
}


def _make_researcher(username):
    role = Role.objects.get(name='RESEARCHER')
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_dataset(name):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherBudgetInStartClient:

    def setup_method(self):
        from dataset.models import DatasetAccess
        self.researcher = _make_researcher("rbv_r1")
        self.dataset = _make_dataset("rbv_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset,
            sensitivity='high',
            max_epsilon_per_job=0.5,
            lifetime_budget=2.0,
        )
        DatasetAccess.objects.create(
            dataset=self.dataset,
            user_id=self.researcher.id,
            assigned_by_id=self.researcher.id,
            can_train=True,
            can_view_metadata=True,
        )

    def _call_validate(self, researcher, model_json=None):
        from api.views import validate_training_permissions
        mj = model_json or {
            **MODEL_JSON,
            "model": {
                **MODEL_JSON["model"],
                "dataset": {"selected_datasets": [{"dataset_id": self.dataset.id}]}
            }
        }
        return validate_training_permissions(researcher, mj)

    def test_creates_researcher_budget_on_first_call(self):
        with patch('api.views.estimate_job_epsilon', return_value=0.3):
            result = self._call_validate(self.researcher)
        assert result is None  # None = ok, no error
        assert ResearcherEpsilonBudget.objects.filter(
            dataset=self.dataset, researcher_id=self.researcher.id
        ).exists()

    def test_rejects_when_researcher_budget_exhausted(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        budget.spent_epsilon = 1.9  # only 0.1 remaining
        budget.save()

        with patch('api.views.estimate_job_epsilon', return_value=0.5):
            result = self._call_validate(self.researcher)
        assert result is not None  # JsonResponse error
        assert result.status_code == 403

    def test_allows_when_researcher_budget_sufficient(self):
        with patch('api.views.estimate_job_epsilon', return_value=0.3):
            result = self._call_validate(self.researcher)
        assert result is None

    def test_allows_when_dataset_aggregate_full_but_researcher_fresh(self):
        # H1: the dataset-level counter is audit-only and must NOT block. A
        # researcher with a fresh personal quota is allowed even when the
        # dataset aggregate is "full" (e.g. spent by other researchers).
        self.policy.spent_epsilon = 99.0  # aggregate well past the template
        self.policy.save()

        with patch('api.views.estimate_job_epsilon', return_value=0.5):
            result = self._call_validate(self.researcher)
        assert result is None  # allowed — researcher quota governs

    def test_second_job_blocked_after_budget_consumed(self):
        # End-to-end exhaustion: a first job is allowed, the spend is recorded,
        # and an identical second job is then blocked at the gate.
        self.policy.lifetime_budget = 0.5  # one 0.5-job fills it
        self.policy.save()
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,  # inherits lifetime_budget=0.5
        )

        with patch('api.views.estimate_job_epsilon', return_value=0.5):
            first = self._call_validate(self.researcher)
            assert first is None  # allowed: exactly at budget

            # Consume the budget at both levels (what _record_privacy_spend does).
            self.policy.record_spent(0.5)
            budget.record_spent(0.5)

            second = self._call_validate(self.researcher)
        assert second is not None
        assert second.status_code == 403
