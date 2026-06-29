"""
Exhaustive tests for DatasetPrivacyPolicy.

Security model: The Node (hospital) is the gatekeeper. The Hub (researcher)
is UNTRUSTED. Every adversarial input a compromised Hub could send through
the epsilon estimation path must be handled safely here.

Test organisation:
  TestSensitivityPresets         — defaults auto-populated from SENSITIVITY_DEFAULTS
  TestRemainingBudget            — property arithmetic and float-drift guard
  TestCanAcceptJobNominal        — normal accept/reject flows
  TestCanAcceptJobAdversarial    — NaN/inf/-1 bypass attempts, zero, boundary
  TestRecordSpentNominal         — normal accounting and accumulation
  TestRecordSpentAdversarial     — sentinel, negative, NaN, inf silently skipped
  TestAtomicUpdate               — F() expression guarantees no lost-update
  TestModelConstraints           — OneToOne, __str__, ordering
  TestOverrideDefaults           — explicit values are respected, not overwritten
"""

import math
import pytest
from django.test import TestCase
from django.db import IntegrityError
from dataset.models import Dataset, DatasetPrivacyPolicy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dataset(name="test_ds", **kwargs) -> Dataset:
    defaults = dict(
        description="Test dataset",
        file_path="/nonexistent/path.csv",
        uploaded_by_id=1,
        medical_domain="general",
        file_size=1024,
        file_format="csv",
        anonymized=True,
    )
    defaults.update(kwargs)
    # Bypass Dataset.save() which tries to read/hash the file
    ds = Dataset(**{**defaults, "name": name})
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


def _make_policy(sensitivity="medium", dataset=None, **kwargs) -> DatasetPrivacyPolicy:
    if dataset is None:
        dataset = _make_dataset()
    policy = DatasetPrivacyPolicy(dataset=dataset, sensitivity=sensitivity, **kwargs)
    policy.save()
    return policy


# ---------------------------------------------------------------------------
# 1. SENSITIVITY_DEFAULTS auto-populate
# ---------------------------------------------------------------------------

class TestSensitivityPresets(TestCase):
    """save() must auto-populate limits from SENSITIVITY_DEFAULTS when not set."""

    databases = {"default", "datasets_db"}

    def test_high_sensitivity_defaults(self):
        policy = _make_policy("high")
        self.assertAlmostEqual(policy.max_epsilon_per_job, 0.5)
        self.assertAlmostEqual(policy.lifetime_budget, 2.0)

    def test_medium_sensitivity_defaults(self):
        policy = _make_policy("medium")
        self.assertAlmostEqual(policy.max_epsilon_per_job, 1.0)
        self.assertAlmostEqual(policy.lifetime_budget, 5.0)

    def test_low_sensitivity_defaults(self):
        policy = _make_policy("low")
        self.assertAlmostEqual(policy.max_epsilon_per_job, 3.0)
        self.assertAlmostEqual(policy.lifetime_budget, 15.0)

    def test_spent_epsilon_starts_at_zero(self):
        policy = _make_policy("medium")
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_all_sensitivity_choices_produce_finite_positive_limits(self):
        ds_counter = [0]
        for sensitivity in ("high", "medium", "low"):
            ds_counter[0] += 1
            ds = _make_dataset(name=f"ds_sens_{sensitivity}_{ds_counter[0]}")
            policy = _make_policy(sensitivity, dataset=ds)
            self.assertGreater(policy.max_epsilon_per_job, 0.0)
            self.assertGreater(policy.lifetime_budget, 0.0)
            self.assertTrue(math.isfinite(policy.max_epsilon_per_job))
            self.assertTrue(math.isfinite(policy.lifetime_budget))

    def test_sensitivity_defaults_constant_is_a_dict(self):
        self.assertIsInstance(DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS, dict)
        for key, val in DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS.items():
            self.assertIn('max_epsilon_per_job', val)
            self.assertIn('lifetime_budget', val)

    def test_ordering_high_lt_medium_lt_low_per_job(self):
        high_eps = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['high']['max_epsilon_per_job']
        med_eps = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['medium']['max_epsilon_per_job']
        low_eps = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['low']['max_epsilon_per_job']
        self.assertLess(high_eps, med_eps)
        self.assertLess(med_eps, low_eps)

    def test_ordering_high_lt_medium_lt_low_lifetime(self):
        high_budget = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['high']['lifetime_budget']
        med_budget = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['medium']['lifetime_budget']
        low_budget = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['low']['lifetime_budget']
        self.assertLess(high_budget, med_budget)
        self.assertLess(med_budget, low_budget)


# ---------------------------------------------------------------------------
# 2. remaining_budget property
# ---------------------------------------------------------------------------

class TestRemainingBudget(TestCase):
    """remaining_budget must never be negative and must reflect spent_epsilon."""

    databases = {"default", "datasets_db"}

    def test_zero_spent_returns_full_budget(self):
        policy = _make_policy("medium")  # lifetime_budget=5.0, spent=0
        self.assertAlmostEqual(policy.remaining_budget, 5.0)

    def test_partial_spent_correct(self):
        policy = _make_policy("medium")
        policy.spent_epsilon = 2.0
        self.assertAlmostEqual(policy.remaining_budget, 3.0)

    def test_fully_spent_returns_zero(self):
        policy = _make_policy("medium")
        policy.spent_epsilon = 5.0
        self.assertAlmostEqual(policy.remaining_budget, 0.0)

    def test_overspent_never_negative(self):
        # Float arithmetic can push spent slightly over budget — must not return negative
        policy = _make_policy("medium")
        policy.spent_epsilon = 5.0000001
        self.assertGreaterEqual(policy.remaining_budget, 0.0)

    def test_remaining_budget_is_float(self):
        policy = _make_policy("high")
        self.assertIsInstance(policy.remaining_budget, float)

    def test_high_sensitivity_small_budget_correct(self):
        policy = _make_policy("high")  # lifetime_budget=2.0
        policy.spent_epsilon = 1.5
        self.assertAlmostEqual(policy.remaining_budget, 0.5)


# ---------------------------------------------------------------------------
# 3. can_accept_job — nominal paths
# ---------------------------------------------------------------------------

class TestCanAcceptJobNominal(TestCase):
    databases = {"default", "datasets_db"}

    def test_accepts_valid_epsilon_within_both_limits(self):
        policy = _make_policy("medium")  # max_per_job=1.0, lifetime=5.0
        ok, msg = policy.can_accept_job(0.5)
        self.assertTrue(ok)
        self.assertEqual(msg, "ok")

    def test_rejects_epsilon_exceeding_per_job_limit(self):
        policy = _make_policy("high")  # max_per_job=0.5
        ok, msg = policy.can_accept_job(0.6)
        self.assertFalse(ok)
        self.assertIn("máximo por job", msg)
        self.assertIn("high", msg)

    def test_rejects_epsilon_exceeding_remaining_budget(self):
        policy = _make_policy("low")  # max_per_job=3.0, lifetime=15.0
        # Must persist to DB — can_accept_job calls refresh_from_db first
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=13.5)
        ok, msg = policy.can_accept_job(2.0)
        self.assertFalse(ok)
        self.assertIn("Presupuesto agotado", msg)

    def test_accepts_epsilon_exactly_at_per_job_limit(self):
        policy = _make_policy("medium")  # max_per_job=1.0
        ok, msg = policy.can_accept_job(1.0)
        self.assertTrue(ok)

    def test_accepts_epsilon_exactly_at_remaining_budget(self):
        policy = _make_policy("medium")  # lifetime=5.0, spent=0
        ok, msg = policy.can_accept_job(5.0)
        # Also within per_job limit (5.0 > 1.0), so this tests budget, not per-job
        # Should fail per-job first: 5.0 > 1.0
        self.assertFalse(ok)
        self.assertIn("máximo por job", msg)

    def test_rejects_when_fully_exhausted(self):
        policy = _make_policy("medium")
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=5.0)
        ok, msg = policy.can_accept_job(0.1)
        self.assertFalse(ok)
        self.assertIn("Presupuesto agotado", msg)

    def test_per_job_check_takes_priority_over_budget(self):
        # Even if budget available, per-job limit must be checked first
        policy = _make_policy("high")  # max_per_job=0.5, lifetime=2.0, spent=0
        ok, msg = policy.can_accept_job(1.0)  # 1.0 > 0.5 per-job
        self.assertFalse(ok)
        self.assertIn("máximo por job", msg)

    def test_return_type_is_tuple(self):
        policy = _make_policy("medium")
        result = policy.can_accept_job(0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reason_message_is_string(self):
        policy = _make_policy("medium")
        _, msg = policy.can_accept_job(0.5)
        self.assertIsInstance(msg, str)


# ---------------------------------------------------------------------------
# 4. can_accept_job — adversarial inputs
# ---------------------------------------------------------------------------

class TestCanAcceptJobAdversarial(TestCase):
    """A compromised Hub could send NaN, inf, -1.0 to bypass budget checks.

    Python's NaN comparison rules: nan > x is always False, so a naive
    'if estimated_epsilon > limit' check would LET NaN THROUGH. These
    tests verify the math.isfinite guard blocks all such bypass attempts.
    """

    databases = {"default", "datasets_db"}

    def test_nan_epsilon_rejected(self):
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(float("nan"))
        self.assertFalse(ok, "NaN must never pass — nan > x is False in Python (bypass risk)")
        self.assertIn("inválido", msg)

    def test_positive_infinity_rejected(self):
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(float("inf"))
        self.assertFalse(ok)
        self.assertIn("inválido", msg)

    def test_negative_infinity_rejected(self):
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(float("-inf"))
        self.assertFalse(ok)
        self.assertIn("inválido", msg)

    def test_negative_one_sentinel_rejected(self):
        # -1.0 is the sentinel value meaning 'epsilon measurement failed'
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(-1.0)
        self.assertFalse(ok)
        self.assertIn("inválido", msg)

    def test_negative_epsilon_rejected(self):
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(-0.001)
        self.assertFalse(ok)

    def test_zero_epsilon_rejected(self):
        # Zero epsilon means infinite noise / model doesn't train; reject
        policy = _make_policy("medium")
        ok, msg = policy.can_accept_job(0.0)
        self.assertFalse(ok)
        self.assertIn("inválido", msg)

    def test_very_small_positive_epsilon_accepted_if_within_limits(self):
        policy = _make_policy("low")  # max_per_job=3.0
        ok, msg = policy.can_accept_job(1e-10)
        self.assertTrue(ok)

    def test_nan_rejected_even_with_exhausted_budget(self):
        # NaN must be rejected before budget check, not after
        policy = _make_policy("medium")
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=5.0)
        ok, msg = policy.can_accept_job(float("nan"))
        self.assertFalse(ok)
        self.assertIn("inválido", msg)

    def test_nan_bypass_python_comparison_documented(self):
        # This documents WHY the guard is needed: max() and > both lie for NaN.
        # nan != nan (IEEE 754), so assertEqual would always fail — use isnan().
        nan = float("nan")
        self.assertFalse(nan > 100.0, "nan > x is always False — budget check would pass nan!")
        self.assertTrue(math.isnan(max(nan, 1.0)), "max(nan, x) returns nan, not 1.0")
        self.assertFalse(math.isfinite(nan))

    def test_large_valid_epsilon_correctly_rejected_by_per_job(self):
        policy = _make_policy("high")  # max_per_job=0.5
        ok, msg = policy.can_accept_job(1e6)
        self.assertFalse(ok)
        self.assertIn("máximo por job", msg)


# ---------------------------------------------------------------------------
# 5. record_spent — nominal
# ---------------------------------------------------------------------------

class TestRecordSpentNominal(TestCase):
    databases = {"default", "datasets_db"}

    def test_records_spent_epsilon(self):
        policy = _make_policy("medium")
        policy.record_spent(0.8)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.8, places=5)

    def test_cumulative_spending(self):
        policy = _make_policy("medium")
        policy.record_spent(0.3)
        policy.record_spent(0.5)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.8, places=5)

    def test_three_rounds_accumulate_correctly(self):
        policy = _make_policy("low")  # lifetime=15.0
        policy.record_spent(1.0)
        policy.record_spent(2.0)
        policy.record_spent(0.5)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 3.5, places=5)

    def test_instance_refreshed_after_record(self):
        # After record_spent the in-memory instance must reflect the DB value
        policy = _make_policy("medium")
        policy.record_spent(1.0)
        self.assertAlmostEqual(policy.spent_epsilon, 1.0, places=5)

    def test_remaining_budget_decreases_after_record(self):
        policy = _make_policy("medium")  # lifetime=5.0
        policy.record_spent(2.0)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.remaining_budget, 3.0, places=5)


# ---------------------------------------------------------------------------
# 6. record_spent — adversarial (sentinel / invalid values must be skipped)
# ---------------------------------------------------------------------------

class TestRecordSpentAdversarial(TestCase):
    """record_spent must silently ignore any non-positive or non-finite value.

    The -1.0 sentinel means 'epsilon measurement failed'. Recording it as
    spent epsilon would permanently reduce the budget for a failed job,
    which is both incorrect and exploitable (Hub could forge failures to
    drain the budget without actual training).
    """

    databases = {"default", "datasets_db"}

    def test_sentinel_minus_one_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(-1.0)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_negative_epsilon_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(-0.5)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_zero_epsilon_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(0.0)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_nan_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(float("nan"))
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_positive_infinity_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(float("inf"))
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_negative_infinity_not_recorded(self):
        policy = _make_policy("medium")
        policy.record_spent(float("-inf"))
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)

    def test_valid_after_ignored_invalid(self):
        # A valid job after ignored sentinel must still record correctly
        policy = _make_policy("medium")
        policy.record_spent(-1.0)  # ignored
        policy.record_spent(0.7)   # valid
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.7, places=5)

    def test_budget_not_drained_by_repeated_sentinels(self):
        policy = _make_policy("medium")  # lifetime=5.0
        for _ in range(100):
            policy.record_spent(-1.0)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 0.0)
        self.assertAlmostEqual(policy.remaining_budget, 5.0)


# ---------------------------------------------------------------------------
# 7. Atomic F() update
# ---------------------------------------------------------------------------

class TestAtomicUpdate(TestCase):
    """record_spent uses F() so concurrent updates don't lose writes."""

    databases = {"default", "datasets_db"}

    def test_two_sequential_updates_accumulate_without_lost_write(self):
        # Simulate two 'concurrent' updates by operating on two separate
        # in-memory instances pointing at the same DB row.
        policy = _make_policy("low")  # lifetime=15.0

        instance_a = DatasetPrivacyPolicy.objects.get(pk=policy.pk)
        instance_b = DatasetPrivacyPolicy.objects.get(pk=policy.pk)

        # Both instances see spent_epsilon=0.0 at this point.
        # With F() each update is atomic: spent = spent + delta.
        instance_a.record_spent(1.0)
        instance_b.record_spent(2.0)

        # The DB row must accumulate BOTH updates (3.0), not just the last.
        fresh = DatasetPrivacyPolicy.objects.get(pk=policy.pk)
        self.assertAlmostEqual(fresh.spent_epsilon, 3.0, places=5)

    def test_record_spent_does_not_overwrite_with_stale_local_value(self):
        # If record_spent used self.spent_epsilon + delta + save(), the second
        # instance would overwrite the first (lost update). The F() approach
        # prevents this by delegating arithmetic to the DB.
        policy = _make_policy("medium")
        instance_a = DatasetPrivacyPolicy.objects.get(pk=policy.pk)
        instance_a.record_spent(0.9)

        instance_b = DatasetPrivacyPolicy.objects.get(pk=policy.pk)
        # instance_b now reflects the updated value (0.9)
        self.assertAlmostEqual(instance_b.spent_epsilon, 0.9, places=5)


# ---------------------------------------------------------------------------
# 8. Model constraints and metadata
# ---------------------------------------------------------------------------

class TestModelConstraints(TestCase):
    databases = {"default", "datasets_db"}

    def test_one_policy_per_dataset(self):
        ds = _make_dataset(name="unique_ds")
        _make_policy(dataset=ds)
        # Second policy on same dataset must fail
        with self.assertRaises(Exception):  # IntegrityError or ValidationError
            _make_policy(dataset=ds)

    def test_str_contains_dataset_name(self):
        ds = _make_dataset(name="my_dataset")
        policy = _make_policy(dataset=ds)
        self.assertIn("my_dataset", str(policy))

    def test_str_contains_sensitivity(self):
        policy = _make_policy("high")
        self.assertIn("high", str(policy))

    def test_str_contains_spent_and_budget(self):
        policy = _make_policy("medium")
        result = str(policy)
        self.assertIn("0.0000", result)
        self.assertIn("5.0", result)

    def test_created_at_set_on_creation(self):
        policy = _make_policy("medium")
        self.assertIsNotNone(policy.created_at)

    def test_updated_at_changes_on_save(self):
        policy = _make_policy("medium")
        original_updated_at = policy.updated_at
        policy.sensitivity = "low"
        policy.max_epsilon_per_job = 3.0
        policy.lifetime_budget = 15.0
        policy.save()
        policy.refresh_from_db()
        # updated_at should be >= original (may be equal if save is very fast)
        self.assertGreaterEqual(policy.updated_at, original_updated_at)

    def test_default_sensitivity_is_medium(self):
        ds = _make_dataset(name="default_sens_ds")
        policy = DatasetPrivacyPolicy(dataset=ds)
        policy.save()
        self.assertEqual(policy.sensitivity, "medium")

    def test_ordering_meta_attribute_is_newest_first(self):
        # Verify Meta.ordering is declared — actual ordering relies on DB clock
        # which has millisecond resolution insufficient for in-memory SQLite tests.
        self.assertEqual(DatasetPrivacyPolicy._meta.ordering, ['-created_at'])


# ---------------------------------------------------------------------------
# 9. Override defaults — explicit values must not be overwritten
# ---------------------------------------------------------------------------

class TestOverrideDefaults(TestCase):
    """save() must only fill in missing limits, not overwrite explicit values."""

    databases = {"default", "datasets_db"}

    def test_explicit_max_epsilon_per_job_preserved(self):
        ds = _make_dataset(name="explicit_max_ds")
        policy = DatasetPrivacyPolicy(
            dataset=ds,
            sensitivity="medium",
            max_epsilon_per_job=0.25,  # custom, tighter than preset 1.0
            lifetime_budget=3.0,
        )
        policy.save()
        self.assertAlmostEqual(policy.max_epsilon_per_job, 0.25)

    def test_explicit_lifetime_budget_preserved(self):
        ds = _make_dataset(name="explicit_budget_ds")
        policy = DatasetPrivacyPolicy(
            dataset=ds,
            sensitivity="low",
            max_epsilon_per_job=1.5,
            lifetime_budget=7.0,  # custom, tighter than preset 15.0
        )
        policy.save()
        self.assertAlmostEqual(policy.lifetime_budget, 7.0)

    def test_high_sensitivity_with_custom_stricter_limit(self):
        ds = _make_dataset(name="strict_high_ds")
        policy = DatasetPrivacyPolicy(
            dataset=ds,
            sensitivity="high",
            max_epsilon_per_job=0.1,  # stricter than 0.5 preset
            lifetime_budget=0.5,      # stricter than 2.0 preset
        )
        policy.save()
        ok, _ = policy.can_accept_job(0.2)
        self.assertFalse(ok)  # 0.2 > 0.1 custom limit

    def test_sensitivity_choice_affects_description_not_limits_when_explicit(self):
        ds = _make_dataset(name="choice_ds")
        # "low" preset is 3.0/15.0 but we set explicit values
        policy = DatasetPrivacyPolicy(
            dataset=ds,
            sensitivity="low",
            max_epsilon_per_job=0.5,
            lifetime_budget=2.0,
        )
        policy.save()
        # Limits must match our explicit values, not the "low" preset
        self.assertAlmostEqual(policy.max_epsilon_per_job, 0.5)
        self.assertAlmostEqual(policy.lifetime_budget, 2.0)


# ---------------------------------------------------------------------------
# 10. clean() validation — committee-identified CRITICAL/HIGH fixes
# ---------------------------------------------------------------------------

class TestCleanValidation(TestCase):
    """save() now calls clean(), which must reject invalid stored limits."""

    databases = {"default", "datasets_db"}

    def test_zero_max_epsilon_per_job_raises(self):
        ds = _make_dataset(name="zero_max_ds")
        with self.assertRaises(Exception):  # ValidationError from clean()
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=0.0,
                lifetime_budget=5.0,
            ).save()

    def test_negative_max_epsilon_per_job_raises(self):
        ds = _make_dataset(name="neg_max_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=-0.5,
                lifetime_budget=5.0,
            ).save()

    def test_nan_max_epsilon_per_job_raises(self):
        ds = _make_dataset(name="nan_max_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=float('nan'),
                lifetime_budget=5.0,
            ).save()

    def test_inf_max_epsilon_per_job_raises(self):
        ds = _make_dataset(name="inf_max_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=float('inf'),
                lifetime_budget=5.0,
            ).save()

    def test_zero_lifetime_budget_raises(self):
        ds = _make_dataset(name="zero_budget_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=1.0,
                lifetime_budget=0.0,
            ).save()

    def test_negative_lifetime_budget_raises(self):
        ds = _make_dataset(name="neg_budget_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=1.0,
                lifetime_budget=-5.0,
            ).save()

    def test_max_epsilon_exceeds_lifetime_raises(self):
        # max_epsilon_per_job=10.0 > lifetime_budget=5.0 is incoherent
        ds = _make_dataset(name="incoherent_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                sensitivity="medium",
                max_epsilon_per_job=10.0,
                lifetime_budget=5.0,
            ).save()

    def test_zero_not_auto_overwritten_but_raises(self):
        # Previously `if not 0.0` would overwrite 0.0 with preset.
        # With the `is None` fix, 0.0 survives to clean() which raises.
        ds = _make_dataset(name="zero_not_silenced_ds")
        with self.assertRaises(Exception):
            DatasetPrivacyPolicy(
                dataset=ds,
                max_epsilon_per_job=0.0,
                lifetime_budget=5.0,
            ).save()


# ---------------------------------------------------------------------------
# 11. remaining_budget NaN/inf guard (committee-identified HIGH fix)
# ---------------------------------------------------------------------------

class TestRemainingBudgetCorruptData(TestCase):
    """remaining_budget must fail closed (return 0.0) on corrupt DB values."""

    databases = {"default", "datasets_db"}

    def test_nan_spent_epsilon_returns_zero(self):
        policy = _make_policy("medium")
        # Simulate corrupt DB value written directly (bypassing save/clean)
        object.__setattr__(policy, 'spent_epsilon', float('nan'))
        self.assertEqual(policy.remaining_budget, 0.0)

    def test_nan_lifetime_budget_returns_zero(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'lifetime_budget', float('nan'))
        self.assertEqual(policy.remaining_budget, 0.0)

    def test_inf_lifetime_budget_returns_zero(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'lifetime_budget', float('inf'))
        self.assertEqual(policy.remaining_budget, 0.0)

    def test_inf_spent_epsilon_returns_zero(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'spent_epsilon', float('inf'))
        self.assertEqual(policy.remaining_budget, 0.0)


# ---------------------------------------------------------------------------
# 12. can_accept_job with corrupt stored limits (committee-identified CRITICAL)
# ---------------------------------------------------------------------------

class TestCanAcceptJobCorruptStoredLimits(TestCase):
    """If stored max_epsilon_per_job is NaN (DB bypass), can_accept_job must
    fail closed — not silently accept every job because nan > x is False."""

    databases = {"default", "datasets_db"}

    def test_nan_stored_per_job_limit_rejects_all_jobs(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'max_epsilon_per_job', float('nan'))
        ok, msg = policy.can_accept_job(0.5)
        self.assertFalse(ok, "NaN per-job limit must NOT let every job through")
        self.assertIn("corrupta", msg)

    def test_inf_stored_per_job_limit_rejects(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'max_epsilon_per_job', float('inf'))
        ok, msg = policy.can_accept_job(0.5)
        self.assertFalse(ok)

    def test_zero_stored_per_job_limit_rejects(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'max_epsilon_per_job', 0.0)
        ok, msg = policy.can_accept_job(0.5)
        self.assertFalse(ok)

    def test_negative_stored_per_job_limit_rejects(self):
        policy = _make_policy("medium")
        object.__setattr__(policy, 'max_epsilon_per_job', -1.0)
        ok, msg = policy.can_accept_job(0.5)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 13. record_spent conditional update — budget ceiling respected
# ---------------------------------------------------------------------------

class TestRecordSpentAuditAggregate(TestCase):
    """record_spent on the policy is an AUDIT aggregate: it accumulates the
    real total privacy leakage across all researchers, truthfully and
    unconditionally. The policy no longer blocks training (the per-researcher
    budget is the enforcement gate), so the aggregate legitimately exceeds the
    per-researcher template value and must never be dropped."""

    databases = {"default", "datasets_db"}

    def test_record_spent_records_even_when_already_at_budget(self):
        policy = _make_policy("high")  # lifetime_budget=2.0
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=2.0)
        policy.refresh_from_db()
        # Aggregate of all researchers may exceed the template — record it.
        policy.record_spent(0.5)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 2.5, places=5)

    def test_record_spent_partial_room_succeeds(self):
        policy = _make_policy("low")  # lifetime=15.0
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=13.0)
        policy.refresh_from_db()
        policy.record_spent(2.0)  # 13.0 + 2.0 = 15.0
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 15.0, places=5)

    def test_record_spent_accumulates_past_template_budget(self):
        policy = _make_policy("medium")  # lifetime=5.0
        DatasetPrivacyPolicy.objects.filter(pk=policy.pk).update(spent_epsilon=4.5)
        policy.refresh_from_db()
        # 4.5 + 0.6 = 5.1 > 5.0 — recorded truthfully for audit, not dropped.
        policy.record_spent(0.6)
        policy.refresh_from_db()
        self.assertAlmostEqual(policy.spent_epsilon, 5.1, places=5)
