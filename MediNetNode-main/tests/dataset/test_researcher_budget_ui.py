import sys
sys.modules.setdefault('magic', None)

import pytest
from django.test import Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_user(username, role_name):
    role = Role.objects.get(name=role_name)
    u = User.objects.create_user(username=username, password="testpass")
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
class TestResearcherBudgetUIInDatasetDetail:

    def setup_method(self):
        self.admin = _make_user("ui_admin", "ADMIN")
        self.researcher = _make_user("ui_researcher", "RESEARCHER")
        self.dataset = _make_dataset("ui_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='high',
            max_epsilon_per_job=0.5, lifetime_budget=2.0,
        )
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        self.budget.spent_epsilon = 0.5
        self.budget.save()

    def test_admin_sees_researcher_budgets_section(self):
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert resp.status_code == 200
        content = resp.content.decode()
        assert 'Presupuesto' in content

    def test_context_contains_researcher_budgets(self):
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert 'researcher_budgets' in resp.context
        budgets = list(resp.context['researcher_budgets'])
        assert len(budgets) == 1
        assert budgets[0].researcher_id == self.researcher.id

    def test_context_contains_pending_reset_requests(self):
        from trainings.models import BudgetResetRequest
        BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Nuevo proyecto.',
        )
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert 'pending_reset_requests' in resp.context
        assert len(resp.context['pending_reset_requests']) == 1
