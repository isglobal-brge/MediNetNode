import sys
sys.modules.setdefault('magic', None)

import pytest
import json
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from trainings.models import BudgetResetRequest
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_user(username, role_name):
    role = Role.objects.get(name=role_name)
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
class TestBudgetResetEndpoints:

    def setup_method(self):
        self.researcher = _make_user("be_r1", "RESEARCHER")
        self.admin = _make_user("be_a1", "ADMIN")
        self.dataset = _make_dataset("be_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='high',
            max_epsilon_per_job=0.5, lifetime_budget=2.0,
        )
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy,
        )
        self.budget.spent_epsilon = 1.5
        self.budget.save()
        self.factory = RequestFactory()

    def _api_request(self, method, path, user, data=None):
        req = getattr(self.factory, method)(
            path,
            data=json.dumps(data or {}),
            content_type='application/json',
        )
        req.api_user = user
        return req

    def test_researcher_can_request_reset(self):
        from api.budget_views import request_budget_reset
        req = self._api_request('post', '/api/v1/budget-reset/', self.researcher, {
            'dataset_id': self.dataset.id,
            'reason': 'Nuevo proyecto aprobado por comité.',
        })
        resp = request_budget_reset(req)
        assert resp.status_code == 201
        assert BudgetResetRequest.objects.filter(
            researcher_id=self.researcher.id, dataset_id=self.dataset.id, status='pending'
        ).exists()

    def test_admin_can_approve_reset(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Motivo válido.',
        )
        from api.budget_views import approve_budget_reset
        req = self._api_request('post', f'/api/v1/budget-reset/{reset_req.id}/approve/', self.admin, {
            'notes': 'Aprobado por revisión ética.',
        })
        resp = approve_budget_reset(req, reset_req.id)
        assert resp.status_code == 200

        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0

        reset_req.refresh_from_db()
        assert reset_req.status == 'approved'

    def test_admin_can_reject_reset(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Motivo.',
        )
        from api.budget_views import reject_budget_reset
        req = self._api_request('post', f'/api/v1/budget-reset/{reset_req.id}/reject/', self.admin, {
            'notes': 'No procede.',
        })
        resp = reject_budget_reset(req, reset_req.id)
        assert resp.status_code == 200

        reset_req.refresh_from_db()
        assert reset_req.status == 'rejected'
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(1.5)  # unchanged

    def test_researcher_cannot_approve(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='x',
        )
        from api.budget_views import approve_budget_reset
        req = self._api_request('post', f'/api/v1/budget-reset/{reset_req.id}/approve/', self.researcher, {})
        resp = approve_budget_reset(req, reset_req.id)
        assert resp.status_code == 403
