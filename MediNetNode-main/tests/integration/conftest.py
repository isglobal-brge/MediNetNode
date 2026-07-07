"""
Shared pytest fixtures for Node integration tests.

Provides researcher user, API key, auth headers, and dataset fixtures
that are reused across all integration test modules.
"""
import shutil
from pathlib import Path

import pytest

# Suppress flush errors for in-memory datasets_db.
#
# pytest-django calls `call_command('flush', database='datasets_db', ...)`
# between transactional tests to reset state.  The shared-cache in-memory
# SQLite URI ('file:memorydb_datasets_db?mode=memory&cache=shared') used by
# pytest-django sometimes causes the flush management command to fail with
# CommandError.  The integration fixtures clean up their own data in finally
# blocks, so skipping an unsuccessful flush for this alias is safe.
import django.core.management as _mgmt  # noqa: E402

_orig_call_command = _mgmt.call_command


def _safe_call_command(command, *args, **kwargs):
    if command == "flush" and kwargs.get("database") == "datasets_db":
        try:
            return _orig_call_command(command, *args, **kwargs)
        except Exception:
            pass
        return
    return _orig_call_command(command, *args, **kwargs)


_mgmt.call_command = _safe_call_command

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_TEST_CLIENT_IP = "127.0.0.1"


@pytest.fixture(autouse=True)
def _fl_env_vars(monkeypatch) -> None:
    """Ensure FL env vars are set for every integration test in this directory."""
    monkeypatch.setenv("ALLOW_PRIVATE_FL_SERVERS", "True")
    monkeypatch.setenv("FLOWER_SSL_ENABLED", "false")


@pytest.fixture
def _datasets_db(db, django_db_blocker) -> None:
    """Unblock the datasets_db database alias for fixtures that write to it."""
    with django_db_blocker.unblock():
        yield


@pytest.fixture
def integration_researcher_user(db):
    """Create a RESEARCHER user for integration tests.

    Uses a distinct username from the root conftest's researcher_user
    to avoid UniqueConstraint conflicts when both fixtures are active.
    Depends on RESEARCHER role already existing (created by root conftest.py
    session fixture).
    """
    from django.contrib.auth import get_user_model
    from users.models import Role

    User = get_user_model()
    role = Role.objects.get(name="RESEARCHER")
    return User.objects.create_user(
        username="integration_researcher",
        password="IntPass123!",
        email="integration@test.com",
        role=role,
    )


@pytest.fixture
def api_key(integration_researcher_user):
    """Create an API key for the integration researcher user.

    After save(), the plaintext key is available on the instance as
    ``api_key._raw_key``.  It is stored only as a hash in the DB, so
    ``_raw_key`` must be used when constructing request headers.
    """
    from users.models import APIKey

    return APIKey.objects.create(
        user=integration_researcher_user,
        name="Integration Test Key",
        ip_whitelist=[_TEST_CLIENT_IP],
    )


@pytest.fixture
def auth_headers(api_key):
    """Return HTTP META dict for authenticated Django test-client requests.

    The APIAuthenticationMiddleware reads the raw key from the
    ``X-API-Key`` header and compares it against stored hashes.
    The client IP is taken from ``REMOTE_ADDR`` (not a custom header),
    so we set that META key directly.
    """
    return {
        "HTTP_X_API_KEY": api_key._raw_key,
        "REMOTE_ADDR": _TEST_CLIENT_IP,
    }


@pytest.fixture
def heart_attack_dataset(db, _datasets_db, integration_researcher_user, tmp_path):
    """Register the heart attack CSV in the Node DB.

    Copies the fixture CSV to *tmp_path* so the Dataset record points to a
    real file on disk (the model's ``save()`` calculates the SHA-256 checksum
    from the file).  The DB record is deleted at teardown; the temp file is
    cleaned up automatically by pytest.
    """
    from dataset.models import Dataset, DatasetAccess

    src = FIXTURES_DIR / "heart_attack_prediction_dataset_preprocessed.csv"
    dest = tmp_path / "heart_attack_prediction_dataset_preprocessed.csv"
    shutil.copy(src, dest)

    dataset = Dataset.objects.using("datasets_db").create(
        name="Heart Attack Risk Integration Test",
        description="Heart attack prediction dataset for integration tests",
        file_path=str(dest),
        uploaded_by_id=integration_researcher_user.id,
        medical_domain="cardiology",
        patient_count=8762,
        data_type="tabular",
        file_size=dest.stat().st_size,
        file_format="csv",
        target_column="Heart Attack Risk",
    )

    try:
        DatasetAccess.objects.using("datasets_db").create(
            dataset=dataset,
            user_id=integration_researcher_user.id,
            assigned_by_id=integration_researcher_user.id,
            can_train=True,
            can_view_metadata=True,
        )

        yield dataset

    finally:
        DatasetAccess.objects.using("datasets_db").filter(dataset=dataset).delete()
        Dataset.objects.using("datasets_db").filter(pk=dataset.pk).delete()


@pytest.fixture
def tabular_dataset(db, _datasets_db, integration_researcher_user, tmp_path):
    """Register the small synthetic tabular CSV in the Node DB.

    The CSV has 200 rows, 4 numeric features (f1–f4) and a binary ``label``
    column.  Used for SVM and DP Random Forest integration tests.
    """
    from dataset.models import Dataset, DatasetAccess

    src = FIXTURES_DIR / "tabular_test.csv"
    dest = tmp_path / "tabular_test.csv"
    shutil.copy(src, dest)

    dataset = Dataset.objects.using("datasets_db").create(
        name="Tabular Integration Test",
        description="Synthetic tabular dataset for SVM/DPRF integration tests",
        file_path=str(dest),
        uploaded_by_id=integration_researcher_user.id,
        medical_domain="other",
        patient_count=200,
        data_type="tabular",
        file_size=dest.stat().st_size,
        file_format="csv",
        target_column="label",
    )

    try:
        DatasetAccess.objects.using("datasets_db").create(
            dataset=dataset,
            user_id=integration_researcher_user.id,
            assigned_by_id=integration_researcher_user.id,
            can_train=True,
            can_view_metadata=True,
        )

        yield dataset

    finally:
        DatasetAccess.objects.using("datasets_db").filter(dataset=dataset).delete()
        Dataset.objects.using("datasets_db").filter(pk=dataset.pk).delete()
