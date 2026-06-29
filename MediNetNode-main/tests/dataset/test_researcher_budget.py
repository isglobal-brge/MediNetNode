import sys
sys.modules.setdefault('magic', None)

import pytest
import math
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_researcher(username="r1"):
    role = Role.objects.get(name='RESEARCHER')
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_dataset(name="ds1"):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


def _make_policy(dataset, sensitivity="high"):
    return DatasetPrivacyPolicy.objects.create(
        dataset=dataset,
        sensitivity=sensitivity,
        max_epsilon_per_job=DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS[sensitivity]['max_epsilon_per_job'],
        lifetime_budget=DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS[sensitivity]['lifetime_budget'],
    )


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetCreation:

    def setup_method(self):
        self.researcher = _make_researcher("rb_r1")
        self.dataset = _make_dataset("rb_ds1")
        self.policy = _make_policy(self.dataset, "high")

    def test_get_or_create_initialises_from_policy(self):
        budget, created = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        assert created is True
        assert budget.lifetime_budget == self.policy.lifetime_budget
        assert budget.spent_epsilon == 0.0
        assert budget.researcher_id == self.researcher.id

    def test_get_or_create_idempotent(self):
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        _, created = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        assert created is False
        assert ResearcherEpsilonBudget.objects.filter(
            dataset=self.dataset, researcher_id=self.researcher.id
        ).count() == 1

    def test_unique_per_dataset_and_researcher(self):
        r2 = _make_researcher("rb_r2")
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=r2.id, policy=self.policy
        )
        assert ResearcherEpsilonBudget.objects.filter(dataset=self.dataset).count() == 2


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetCanAccept:

    def setup_method(self):
        self.researcher = _make_researcher("ca_r1")
        self.dataset = _make_dataset("ca_ds1")
        self.policy = _make_policy(self.dataset, "high")
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )

    def test_accepts_within_budget(self):
        ok, reason = self.budget.can_accept_job(0.5)
        assert ok is True
        assert reason == ""

    def test_rejects_exceeds_per_job_limit(self):
        ok, reason = self.budget.can_accept_job(0.9)  # max per job = 0.5
        assert ok is False
        assert "máximo por job" in reason

    def test_rejects_exceeds_remaining_budget(self):
        self.budget.spent_epsilon = 1.8
        self.budget.save()
        ok, reason = self.budget.can_accept_job(0.5)  # remaining = 0.2
        assert ok is False
        assert "presupuesto" in reason

    def test_rejects_nan_epsilon(self):
        ok, reason = self.budget.can_accept_job(float('nan'))
        assert ok is False

    def test_rejects_zero_epsilon(self):
        ok, reason = self.budget.can_accept_job(0.0)
        assert ok is False

    def test_rejects_negative_epsilon(self):
        ok, reason = self.budget.can_accept_job(-1.0)
        assert ok is False


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetRecordSpent:

    def setup_method(self):
        self.researcher = _make_researcher("rs_r1")
        self.dataset = _make_dataset("rs_ds1")
        self.policy = _make_policy(self.dataset, "medium")
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )

    def test_records_valid_epsilon(self):
        self.budget.record_spent(0.5)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.5)

    def test_accumulates_across_calls(self):
        self.budget.record_spent(0.5)
        self.budget.record_spent(0.3)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.8)

    def test_ignores_nan(self):
        self.budget.record_spent(float('nan'))
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0

    def test_ignores_negative(self):
        self.budget.record_spent(-0.5)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0

    def test_refreshes_in_memory_value(self):
        """After record_spent the in-memory instance must reflect the DB value
        without a manual refresh_from_db (parity with DatasetPrivacyPolicy)."""
        self.budget.record_spent(0.3)
        assert self.budget.spent_epsilon == pytest.approx(0.3)

    def test_records_truthfully_past_budget(self):
        """A spend that crosses lifetime_budget must STILL be recorded — DP
        accounting must reflect real privacy leakage, never drop it. The
        enforcement gate (can_accept_job) is what blocks the NEXT job."""
        self.budget.lifetime_budget = 0.5
        self.budget.spent_epsilon = 0.0
        self.budget.save()

        self.budget.record_spent(0.4)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.4)

        # Second 0.4 totals 0.8 > 0.5 → recorded truthfully (not dropped).
        self.budget.record_spent(0.4)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.8)

        # remaining clamps to 0 and the next job is blocked by the gate.
        assert self.budget.remaining_budget == 0.0
        ok, _ = self.budget.can_accept_job(0.1)
        assert ok is False

    def test_concurrent_record_spent_accumulates_no_lost_write(self):
        """Two stale instances racing record_spent must both apply: the atomic
        F() increment accumulates the real total with no lost write."""
        self.budget.lifetime_budget = 0.5
        self.budget.spent_epsilon = 0.0
        self.budget.save()

        instance_a = ResearcherEpsilonBudget.objects.get(pk=self.budget.pk)
        instance_b = ResearcherEpsilonBudget.objects.get(pk=self.budget.pk)

        instance_a.record_spent(0.4)   # DB → 0.4
        instance_b.record_spent(0.4)   # stale in memory, F() adds at DB → 0.8

        final = ResearcherEpsilonBudget.objects.get(pk=self.budget.pk)
        assert final.spent_epsilon == pytest.approx(0.8)


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetPeriodReset:

    def setup_method(self):
        self.researcher = _make_researcher("pr_r1")
        self.dataset = _make_dataset("pr_ds1")
        self.policy = _make_policy(self.dataset, "low")

    def test_annual_period_not_expired(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        assert budget.is_period_expired() is False

    def test_reset_zeroes_spent_epsilon(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        budget.spent_epsilon = 2.0
        budget.save()
        budget.reset_period()
        budget.refresh_from_db()
        assert budget.spent_epsilon == 0.0

    def test_never_period_never_expires(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='never',
        )
        assert budget.is_period_expired() is False

    def test_annual_period_expired_after_boundary(self):
        from django.utils import timezone
        from dateutil.relativedelta import relativedelta
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        budget.period_start = timezone.now() - relativedelta(years=1, days=1)
        budget.save()
        assert budget.is_period_expired() is True

    def test_monthly_period_expired_after_boundary(self):
        from django.utils import timezone
        from dateutil.relativedelta import relativedelta
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='monthly',
        )
        budget.period_start = timezone.now() - relativedelta(months=1, days=1)
        budget.save()
        assert budget.is_period_expired() is True

    def test_monthly_period_not_expired_before_boundary(self):
        from django.utils import timezone
        from dateutil.relativedelta import relativedelta
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='monthly',
        )
        budget.period_start = timezone.now() - relativedelta(days=20)
        budget.save()
        assert budget.is_period_expired() is False

    def test_reset_period_updates_period_start_and_last_reset(self):
        from django.utils import timezone
        from dateutil.relativedelta import relativedelta
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        budget.spent_epsilon = 1.0
        budget.period_start = timezone.now() - relativedelta(years=2)
        budget.last_reset = None
        budget.save()
        assert budget.is_period_expired() is True

        budget.reset_period()
        budget.refresh_from_db()
        assert budget.spent_epsilon == 0.0
        assert budget.last_reset is not None
        # period_start moved to ~now → no longer expired
        assert budget.is_period_expired() is False
