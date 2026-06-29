import sys
sys.modules.setdefault('magic', None)

import pytest
from unittest.mock import MagicMock
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


def _make_session(*, dataset_id, user_id, epsilon, use_experiment=False):
    """Minimal stand-in for a TrainingSession that _record_privacy_spend reads.

    Only the attributes the function touches are stubbed; the DB managers are
    NOT mocked, so the real policy / researcher-budget rows get debited.
    """
    session = MagicMock()
    session.use_experiment = use_experiment
    session.dataset_id = dataset_id
    session.user_id = user_id
    session.session_id = "sess-1"
    round_mock = MagicMock()
    round_mock.metrics = {'privacy_epsilon': epsilon}
    round_mock.round_number = 1
    session.rounds.order_by.return_value.first.return_value = round_mock
    return session


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestRecordPrivacySpendUpdatesResearcherBudget:

    def setup_method(self):
        self.researcher, self.ds, self.policy, self.budget = _setup_fixtures()

    def test_researcher_budget_debited_after_job(self):
        from api.federated.utils import _record_privacy_spend

        session = _make_session(
            dataset_id=self.ds.id, user_id=self.researcher.id, epsilon=0.4,
        )
        _record_privacy_spend(session)

        self.budget.refresh_from_db()
        self.policy.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.4)
        assert self.policy.spent_epsilon == pytest.approx(0.4)

    def test_experimental_session_is_not_debited(self):
        from api.federated.utils import _record_privacy_spend

        session = _make_session(
            dataset_id=self.ds.id, user_id=self.researcher.id, epsilon=0.4,
            use_experiment=True,
        )
        _record_privacy_spend(session)

        self.budget.refresh_from_db()
        self.policy.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0
        assert self.policy.spent_epsilon == 0.0

    def test_researcher_budget_records_real_spend_even_past_budget(self):
        """A job that already ran leaked real ε; recording must reflect that
        truthfully even if it crosses the budget. Enforcement (the gate) blocks
        the NEXT job — the spend is never dropped (DP accounting soundness)."""
        from api.federated.utils import _record_privacy_spend

        self.budget.lifetime_budget = 0.5
        self.budget.spent_epsilon = 0.3
        self.budget.save()

        session = _make_session(
            dataset_id=self.ds.id, user_id=self.researcher.id, epsilon=0.4,
        )
        _record_privacy_spend(session)

        self.budget.refresh_from_db()
        # 0.3 + 0.4 = 0.7: recorded truthfully; remaining clamps to 0.
        assert self.budget.spent_epsilon == pytest.approx(0.7)
        assert self.budget.remaining_budget == 0.0
