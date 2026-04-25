"""
Tests for Tarea 6: Dataset detail view — privacy policy selector.

These tests focus on the view's handling of the new action='update_privacy_policy'
POST path and the context variables it provides. We use pytest-django's
@pytest.mark.django_db approach (same as the existing dataset tests) and test
both the model-level behaviour and the view function directly via RequestFactory.
"""
import sys
import pytest
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.backends.db import SessionStore

from users.models import Role
from dataset.models import Dataset, DatasetPrivacyPolicy

# python-magic's libmagic.dll hangs on this Windows environment.  Block the
# import so dataset.uploader's try/except ImportError falls through to the
# MAGIC_AVAILABLE = False branch instead.
sys.modules.setdefault('magic', None)

# Lazy import: dataset.views pulls in the uploader (pandas etc.) which is slow to
# load on Windows when multiple pytest workers collect in parallel.  Deferring
# until the first test call keeps collection fast.
def _dataset_detail_view():
    from dataset.views import dataset_detail  # noqa: PLC0415
    return dataset_detail

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(name="dp_view_ds") -> Dataset:
    """Create a Dataset bypassing file-hashing in Dataset.save()."""
    ds = Dataset(
        name=name,
        description="test",
        file_path=f"/fake/{name}.csv",
        file_size=1000,
        file_format="csv",
        uploaded_by_id=1,
        patient_count=500,
        columns_count=10,
        target_column="label",
        medical_domain="cardiology",
        data_type="tabular",
        anonymized=True,
        is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


def _make_admin_user(username="dpview_admin"):
    role = Role.objects.get(name='ADMIN')
    user = User.objects.create_user(username=username, password="pass")
    user.role = role
    user.save()
    return user


def _request(method, user, data=None):
    """Build a fake request with messages middleware support."""
    factory = RequestFactory()
    if method == 'GET':
        req = factory.get('/')
    else:
        req = factory.post('/', data or {})
    req.user = user
    # Attach messages framework (needed by view's messages.success/error calls)
    req.session = SessionStore()
    req._messages = FallbackStorage(req)
    return req


# ---------------------------------------------------------------------------
# Model-level: DatasetPrivacyPolicy SENSITIVITY_DEFAULTS
# ---------------------------------------------------------------------------

class TestSensitivityDefaults:
    """DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS covers all choice values."""

    def test_all_choices_have_defaults(self):
        for value, _ in DatasetPrivacyPolicy.SENSITIVITY_CHOICES:
            assert value in DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS, \
                f"Missing default for sensitivity={value!r}"

    def test_high_sensitivity_limits(self):
        d = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['high']
        assert d['max_epsilon_per_job'] == 0.5
        assert d['lifetime_budget'] == 2.0

    def test_medium_sensitivity_limits(self):
        d = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['medium']
        assert d['max_epsilon_per_job'] == 1.0
        assert d['lifetime_budget'] == 5.0

    def test_low_sensitivity_limits(self):
        d = DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS['low']
        assert d['max_epsilon_per_job'] == 3.0
        assert d['lifetime_budget'] == 15.0


# ---------------------------------------------------------------------------
# View tests: RequestFactory (avoids full test-client migration overhead)
# ---------------------------------------------------------------------------

@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestDatasetDetailPrivacyPolicyContext:
    """dataset_detail view passes correct privacy_policy context."""

    def setup_method(self):
        self.admin = _make_admin_user("ctxadmin")
        self.dataset = _make_dataset("ctx_ds")

    def test_no_policy_passes_none_in_context(self):
        req = _request('GET', self.admin)
        resp = _dataset_detail_view()(req, self.dataset.id)
        # The view returns a TemplateResponse — context is available before render
        assert resp.context_data['privacy_policy'] is None

    def test_existing_policy_passes_object_in_context(self):
        policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='high',
            max_epsilon_per_job=0.5, lifetime_budget=2.0,
        )
        req = _request('GET', self.admin)
        resp = _dataset_detail_view()(req, self.dataset.id)
        ctx_policy = resp.context_data['privacy_policy']
        assert ctx_policy is not None
        assert ctx_policy.pk == policy.pk

    def test_sensitivity_choices_always_in_context(self):
        req = _request('GET', self.admin)
        resp = _dataset_detail_view()(req, self.dataset.id)
        choices = resp.context_data['sensitivity_choices']
        values = [c[0] for c in choices]
        assert 'high' in values
        assert 'medium' in values
        assert 'low' in values

    def test_can_edit_true_for_admin(self):
        req = _request('GET', self.admin)
        resp = _dataset_detail_view()(req, self.dataset.id)
        assert resp.context_data['can_edit'] is True


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestDatasetDetailPrivacyPolicyPost:
    """dataset_detail POST action=update_privacy_policy creates/updates policy."""

    def setup_method(self):
        self.admin = _make_admin_user("postadmin")
        self.dataset = _make_dataset("post_ds")

    def _post_sensitivity(self, sensitivity):
        req = _request('POST', self.admin, {
            'action': 'update_privacy_policy',
            'sensitivity': sensitivity,
        })
        # View returns redirect — we care about DB side-effects
        _dataset_detail_view()(req, self.dataset.id)

    def test_post_creates_high_policy(self):
        self._post_sensitivity('high')
        policy = DatasetPrivacyPolicy.objects.get(dataset=self.dataset)
        assert policy.sensitivity == 'high'
        assert policy.max_epsilon_per_job == pytest.approx(0.5)
        assert policy.lifetime_budget == pytest.approx(2.0)

    def test_post_creates_medium_policy(self):
        self._post_sensitivity('medium')
        policy = DatasetPrivacyPolicy.objects.get(dataset=self.dataset)
        assert policy.sensitivity == 'medium'
        assert policy.max_epsilon_per_job == pytest.approx(1.0)
        assert policy.lifetime_budget == pytest.approx(5.0)

    def test_post_creates_low_policy(self):
        self._post_sensitivity('low')
        policy = DatasetPrivacyPolicy.objects.get(dataset=self.dataset)
        assert policy.sensitivity == 'low'
        assert policy.max_epsilon_per_job == pytest.approx(3.0)
        assert policy.lifetime_budget == pytest.approx(15.0)

    def test_post_updates_existing_sensitivity(self):
        DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='low',
            max_epsilon_per_job=3.0, lifetime_budget=15.0,
        )
        self._post_sensitivity('high')
        policy = DatasetPrivacyPolicy.objects.get(dataset=self.dataset)
        assert policy.sensitivity == 'high'
        assert policy.max_epsilon_per_job == pytest.approx(0.5)

    def test_post_preserves_spent_epsilon_on_update(self):
        DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='medium',
            max_epsilon_per_job=1.0, lifetime_budget=5.0,
            spent_epsilon=0.75,
        )
        self._post_sensitivity('high')
        policy = DatasetPrivacyPolicy.objects.get(dataset=self.dataset)
        assert policy.spent_epsilon == pytest.approx(0.75)

    def test_post_invalid_sensitivity_does_not_create_policy(self):
        self._post_sensitivity('ultra_secret')
        assert not DatasetPrivacyPolicy.objects.filter(dataset=self.dataset).exists()

    def test_post_only_one_policy_created(self):
        self._post_sensitivity('medium')
        self._post_sensitivity('high')
        assert DatasetPrivacyPolicy.objects.filter(dataset=self.dataset).count() == 1

    def test_post_no_action_does_not_create_policy(self):
        req = _request('POST', self.admin, {})  # no action field
        _dataset_detail_view()(req, self.dataset.id)
        assert not DatasetPrivacyPolicy.objects.filter(dataset=self.dataset).exists()
