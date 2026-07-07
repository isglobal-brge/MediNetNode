"""
Integration tests for the Dataset Experimental Split feature.

Covers:
  - Uploader creates experiment CSV on disk when split_ratio is provided
  - Dataset model fields are populated correctly
  - Rollback cleans up the experiment file
  - View rejects out-of-range split_ratio values
  - budget check is skipped when use_experiment=True and experiment file exists
  - load_data_from_django routes to experiment file when use_experiment=True
"""
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _make_dataset_record(tmp_path, user_id, split_ratio=None):
    """Create a Dataset DB record with an optional experiment split already set."""
    from dataset.models import Dataset

    src = FIXTURES_DIR / "tabular_test.csv"
    dest = tmp_path / "tabular_test.csv"
    shutil.copy(src, dest)

    kwargs = dict(
        name="Exp Split Test Dataset",
        description="test",
        file_path=str(dest),
        uploaded_by_id=user_id,
        medical_domain="other",
        patient_count=200,
        data_type="tabular",
        file_size=dest.stat().st_size,
        file_format="csv",
        target_column="label",
    )

    if split_ratio is not None:
        exp_dest = tmp_path / "tabular_test_experiment.csv"
        shutil.copy(src, exp_dest)
        kwargs.update(
            experiment_file_path=str(exp_dest),
            experiment_row_count=40,
            experiment_split_ratio=split_ratio,
        )

    return Dataset.objects.using("datasets_db").create(**kwargs)


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestUploaderSplitCreation:
    """Tests for _maybe_create_experiment_split inside SecureDatasetUploader."""

    def test_no_split_when_ratio_is_none(self, tmp_path):
        """When split_ratio=None, no experiment file is created."""
        from dataset.uploader import SecureDatasetUploader

        uploader = SecureDatasetUploader.__new__(SecureDatasetUploader)
        exp_path, exp_rows = uploader._maybe_create_experiment_split(
            str(FIXTURES_DIR / "tabular_test.csv"), None
        )

        assert exp_path is None
        assert exp_rows is None

    def test_split_creates_file_on_disk(self, tmp_path):
        """split_ratio=0.2 creates a smaller CSV file on disk."""
        from dataset.uploader import SecureDatasetUploader

        src = FIXTURES_DIR / "tabular_test.csv"
        dest = tmp_path / "tabular_test.csv"
        shutil.copy(src, dest)

        uploader = SecureDatasetUploader.__new__(SecureDatasetUploader)
        exp_path, exp_rows = uploader._maybe_create_experiment_split(str(dest), 0.2)

        assert exp_path is not None, "Expected experiment file path to be returned"
        assert os.path.exists(exp_path), "Experiment file must exist on disk"
        assert exp_rows is not None and exp_rows > 0

    def test_split_row_count_is_fraction_of_production(self, tmp_path):
        """Experiment row count ≈ split_ratio × total rows."""
        import pandas as pd
        from dataset.uploader import SecureDatasetUploader

        src = FIXTURES_DIR / "tabular_test.csv"
        dest = tmp_path / "tabular_test.csv"
        shutil.copy(src, dest)

        total_rows = len(pd.read_csv(dest))
        split_ratio = 0.2

        uploader = SecureDatasetUploader.__new__(SecureDatasetUploader)
        exp_path, exp_rows = uploader._maybe_create_experiment_split(str(dest), split_ratio)

        expected = int(total_rows * split_ratio)
        assert abs(exp_rows - expected) <= 2, (
            f"Expected ~{expected} experiment rows, got {exp_rows}"
        )

    def test_split_invalid_ratio_returns_none(self, tmp_path):
        """split_ratio=0 returns (None, None) gracefully."""
        from dataset.uploader import SecureDatasetUploader

        uploader = SecureDatasetUploader.__new__(SecureDatasetUploader)
        exp_path, exp_rows = uploader._maybe_create_experiment_split(
            str(FIXTURES_DIR / "tabular_test.csv"), 0
        )

        assert exp_path is None
        assert exp_rows is None


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestUploadViewSplitRatioValidation:
    """Tests that the dataset_upload view rejects bad split_ratio values."""

    def _upload_client(self, admin_user):
        from django.test import Client

        c = Client()
        c.force_login(admin_user)
        return c

    def test_split_ratio_above_max_rejected(self, admin_user, tmp_path):
        """split_ratio > 0.5 must return HTTP 400."""
        from django.test import RequestFactory
        from dataset.views import dataset_upload

        factory = RequestFactory()

        csv_content = b"f1,f2,label\n1,2,0\n3,4,1\n"
        with open(tmp_path / "test.csv", "wb") as f:
            f.write(csv_content)

        with open(tmp_path / "test.csv", "rb") as f:
            from django.core.files.uploadedfile import SimpleUploadedFile

            uploaded = SimpleUploadedFile("test.csv", csv_content, content_type="text/csv")
            request = factory.post(
                "/dataset/upload/",
                {
                    "name": "Bad ratio dataset",
                    "description": "test",
                    "medical_domain": "other",
                    "data_type": "tabular",
                    "anonymized": "on",
                    "file": uploaded,
                    "split_ratio": "0.9",
                },
            )
            request.user = admin_user

        import json
        from dataset.views import dataset_upload as view_fn

        response = view_fn(request)
        assert response.status_code == 400
        body = json.loads(response.content)
        assert not body["success"]

    def test_split_ratio_below_min_rejected(self, admin_user, tmp_path):
        """split_ratio < 0.1 must return HTTP 400."""
        import json
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import RequestFactory
        from dataset.views import dataset_upload as view_fn

        csv_content = b"f1,f2,label\n1,2,0\n3,4,1\n"
        uploaded = SimpleUploadedFile("test.csv", csv_content, content_type="text/csv")
        factory = RequestFactory()
        request = factory.post(
            "/dataset/upload/",
            {
                "name": "Low ratio dataset",
                "description": "test",
                "medical_domain": "other",
                "data_type": "tabular",
                "anonymized": "on",
                "file": uploaded,
                "split_ratio": "0.05",
            },
        )
        request.user = admin_user

        response = view_fn(request)
        assert response.status_code == 400
        body = json.loads(response.content)
        assert not body["success"]

    def test_non_numeric_split_ratio_rejected(self, admin_user):
        """Non-numeric split_ratio must return HTTP 400."""
        import json
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import RequestFactory
        from dataset.views import dataset_upload as view_fn

        csv_content = b"f1,f2,label\n1,2,0\n"
        uploaded = SimpleUploadedFile("test.csv", csv_content, content_type="text/csv")
        factory = RequestFactory()
        request = factory.post(
            "/dataset/upload/",
            {
                "name": "NaN ratio dataset",
                "description": "test",
                "medical_domain": "other",
                "data_type": "tabular",
                "anonymized": "on",
                "file": uploaded,
                "split_ratio": "not-a-number",
            },
        )
        request.user = admin_user

        response = view_fn(request)
        assert response.status_code == 400
        body = json.loads(response.content)
        assert not body["success"]


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestBudgetSkipForExperimentalJobs:
    """validate_training_permissions should return None (pass) when use_experiment=True."""

    def _make_model_json(self, dataset_id, use_experiment=True):
        return {
            "use_experiment": use_experiment,
            "model": {
                "metadata": {"model_type": "ml"},
                "dataset": {
                    "selected_datasets": [{"dataset_id": str(dataset_id)}]
                },
                "training": {},
            },
        }

    def test_skips_budget_when_experiment_file_exists(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """Returns None (skips budget) for a GENUINE experiment job: use_experiment=True,
        the researcher has can_use_experiment granted, and the experiment file exists.
        (Tightened by H2 — the flag + file alone is no longer sufficient.)"""
        from api.views import validate_training_permissions
        from dataset.models import DatasetAccess

        dataset = _make_dataset_record(
            tmp_path, integration_researcher_user.id, split_ratio=0.2
        )
        access = DatasetAccess.objects.using("datasets_db").create(
            dataset=dataset,
            user_id=integration_researcher_user.id,
            assigned_by_id=integration_researcher_user.id,
            can_train=True,
            can_use_experiment=True,
        )

        model_json = self._make_model_json(dataset.id, use_experiment=True)

        try:
            result = validate_training_permissions(integration_researcher_user, model_json)
            assert result is None, f"Expected None (pass), got: {result}"
        finally:
            access.delete()
            dataset.delete()

    def test_experiment_flag_without_permission_enforces_budget(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """H2: use_experiment=True + experiment file present but WITHOUT can_use_experiment
        must NOT skip the budget — it falls through to normal enforcement."""
        from api.views import validate_training_permissions
        from dataset.models import DatasetAccess

        dataset = _make_dataset_record(
            tmp_path, integration_researcher_user.id, split_ratio=0.2
        )
        access = DatasetAccess.objects.using("datasets_db").create(
            dataset=dataset,
            user_id=integration_researcher_user.id,
            assigned_by_id=integration_researcher_user.id,
            can_train=True,
            can_use_experiment=False,  # permission NOT granted
        )

        model_json = self._make_model_json(dataset.id, use_experiment=True)

        try:
            result = validate_training_permissions(integration_researcher_user, model_json)
            # No experiment bypass → hits the budget step (no policy configured) → non-None.
            assert result is not None
        finally:
            access.delete()
            dataset.delete()

    def test_falls_through_budget_when_no_experiment_file(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """Falls through to normal budget checks when dataset has no experiment file."""
        from api.views import validate_training_permissions

        dataset = _make_dataset_record(
            tmp_path, integration_researcher_user.id, split_ratio=None
        )

        model_json = self._make_model_json(dataset.id, use_experiment=True)

        try:
            result = validate_training_permissions(integration_researcher_user, model_json)
            # No budget configured → validation should fail (non-None error response)
            # We just verify it does NOT return None (i.e. didn't early-exit as experiment)
            # Could be a JsonResponse or None depending on whether budget exists.
            # Since no policy is configured, it will fail at some budget check step.
            # We assert it's not the clean early-return path.
            # (If it returns None, the dataset had no experiment path and still passed — that
            # would be a bug where use_experiment=True with no file silently passes.)
            # In practice the researcher has no DatasetAccess record here, so it will fail
            # at the access check, returning a non-None response.
            assert result is not None
        finally:
            dataset.delete()


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestDataLoaderExperimentRouting:
    """load_data_from_django routes to experiment_file_path when use_experiment=True."""

    def test_production_path_used_by_default(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """Without use_experiment, the production file is loaded."""
        from api.federated.data_loaders import load_data_from_django

        dataset = _make_dataset_record(
            tmp_path, integration_researcher_user.id, split_ratio=0.2
        )

        try:
            data_df, target_col = load_data_from_django(dataset.id, use_experiment=False)
            assert data_df is not None
            # Full file has 200 rows
            assert len(data_df) == 200
        finally:
            dataset.delete()

    def test_experiment_path_used_when_flagged(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """With use_experiment=True, the experiment file is loaded (fewer rows)."""
        import pandas as pd
        from api.federated.data_loaders import load_data_from_django

        src = FIXTURES_DIR / "tabular_test.csv"
        dest = tmp_path / "tabular_test.csv"
        shutil.copy(src, dest)

        full_df = pd.read_csv(dest)
        exp_df = full_df.sample(frac=0.2, random_state=42)
        exp_dest = tmp_path / "tabular_test_experiment.csv"
        exp_df.to_csv(exp_dest, index=False)

        from dataset.models import Dataset

        dataset = Dataset.objects.using("datasets_db").create(
            name="Exp Routing Test",
            description="test",
            file_path=str(dest),
            uploaded_by_id=integration_researcher_user.id,
            medical_domain="other",
            patient_count=200,
            data_type="tabular",
            file_size=dest.stat().st_size,
            file_format="csv",
            target_column="label",
            experiment_file_path=str(exp_dest),
            experiment_row_count=len(exp_df),
            experiment_split_ratio=0.2,
        )

        try:
            data_df, target_col = load_data_from_django(dataset.id, use_experiment=True)
            assert data_df is not None
            assert len(data_df) == len(exp_df), (
                f"Expected {len(exp_df)} rows from experiment file, got {len(data_df)}"
            )
        finally:
            dataset.delete()

    def test_falls_back_to_production_when_experiment_file_missing(
        self, db, _datasets_db, integration_researcher_user, tmp_path
    ):
        """use_experiment=True falls back to production file if experiment path is missing."""
        from api.federated.data_loaders import load_data_from_django
        from dataset.models import Dataset

        src = FIXTURES_DIR / "tabular_test.csv"
        dest = tmp_path / "tabular_test.csv"
        shutil.copy(src, dest)

        dataset = Dataset.objects.using("datasets_db").create(
            name="Fallback Routing Test",
            description="test",
            file_path=str(dest),
            uploaded_by_id=integration_researcher_user.id,
            medical_domain="other",
            patient_count=200,
            data_type="tabular",
            file_size=dest.stat().st_size,
            file_format="csv",
            target_column="label",
            experiment_file_path="/nonexistent/path/experiment.csv",
            experiment_row_count=40,
            experiment_split_ratio=0.2,
        )

        try:
            data_df, target_col = load_data_from_django(dataset.id, use_experiment=True)
            assert data_df is not None
            assert len(data_df) == 200
        finally:
            dataset.delete()
