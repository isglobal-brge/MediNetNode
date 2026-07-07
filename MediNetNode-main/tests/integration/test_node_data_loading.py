"""
Integration tests for Node data-loading functions.

Validates that create_train_val_loaders() and load_ml_data() work correctly
with real CSV fixture data registered via the Django dataset ORM.
"""
import numpy as np
import pytest
import torch
import torch.utils.data


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestDLDataLoading:
    """Tests for create_train_val_loaders()."""

    def test_returns_two_dataloaders(self, heart_attack_dataset) -> None:
        """Both return values must be DataLoader instances."""
        from api.federated.data_loaders import create_train_val_loaders

        train_loader, val_loader = create_train_val_loaders(
            dataset_id=heart_attack_dataset.id
        )

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_correct_split_sizes(self, heart_attack_dataset) -> None:
        """80/20 split of 8 762 rows → ~7 009 train / ~1 753 val."""
        from api.federated.data_loaders import create_train_val_loaders

        # Use a batch size large enough to consume all samples in one pass.
        total_rows = 8762
        # expected counts from sklearn with random_state=42: ceil(8762*0.2)=1753 val
        train_loader, val_loader = create_train_val_loaders(
            dataset_id=heart_attack_dataset.id,
            val_size=0.2,
            batch_size=total_rows,
            random_state=42,
        )

        train_count = sum(features.shape[0] for features, _ in train_loader)
        val_count = sum(features.shape[0] for features, _ in val_loader)

        # Allow ±5 rows for rounding differences in train_test_split.
        assert abs(train_count - 7009) <= 5, (
            f"Expected ~7009 train samples, got {train_count}"
        )
        assert abs(val_count - 1753) <= 5, (
            f"Expected ~1753 val samples, got {val_count}"
        )

    def test_feature_shape_matches_heart_attack_dataset(
        self, heart_attack_dataset
    ) -> None:
        """Each batch must have 52 feature columns.

        Column count breakdown: the CSV has 53 columns total (52 named feature
        columns + 1 target column 'Heart Attack Risk'). There is no unnamed
        index column — pd.read_csv is called without index_col, and the raw CSV
        header starts directly with 'Age'. After dropping the target column,
        prepare_dataset produces 52 feature columns.
        """
        from api.federated.data_loaders import create_train_val_loaders

        train_loader, _ = create_train_val_loaders(
            dataset_id=heart_attack_dataset.id
        )

        features, _ = next(iter(train_loader))
        assert features.shape[1] == 52, (
            f"Expected 52 features, got {features.shape[1]}"
        )

    def test_raises_on_invalid_dataset_id(self) -> None:
        """A non-existent (but valid-format) dataset ID must raise RuntimeError.

        dataset_id=999999 passes the positive-integer validation, so the loader
        proceeds to the DB lookup, which returns (None, None) for a missing
        record. The loader then raises RuntimeError with a descriptive message.
        """
        from api.federated.data_loaders import create_train_val_loaders

        with pytest.raises(RuntimeError):
            create_train_val_loaders(dataset_id=999999)

    def test_data_is_float_tensor(self, heart_attack_dataset) -> None:
        """Features must be float tensors; targets must be numeric tensors."""
        from api.federated.data_loaders import create_train_val_loaders

        train_loader, _ = create_train_val_loaders(
            dataset_id=heart_attack_dataset.id
        )

        features, targets = next(iter(train_loader))

        assert features.dtype == torch.float32, (
            f"Features must be float32 for FL compatibility, got {features.dtype}"
        )
        assert targets.dtype in (
            torch.float32, torch.float64, torch.int32, torch.int64, torch.long
        ), f"Expected numeric tensor for targets, got {targets.dtype}"


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestMLDataLoading:
    """Tests for load_ml_data()."""

    def test_returns_train_val_arrays(self, tabular_dataset) -> None:
        """Return value must be ((X_train, y_train), (X_val, y_val)) of ndarrays."""
        from api.federated.data_loaders import load_ml_data

        (X_train, y_train), (X_val, y_val) = load_ml_data(
            dataset_id=tabular_dataset.id
        )

        for arr, name in [
            (X_train, "X_train"),
            (y_train, "y_train"),
            (X_val, "X_val"),
            (y_val, "y_val"),
        ]:
            assert isinstance(arr, np.ndarray), (
                f"Expected ndarray for {name}, got {type(arr)}"
            )

    def test_correct_feature_count(self, tabular_dataset) -> None:
        """tabular_test.csv has 4 feature columns (f1-f4)."""
        from api.federated.data_loaders import load_ml_data

        (X_train, _), _ = load_ml_data(dataset_id=tabular_dataset.id)

        assert X_train.shape[1] == 4, (
            f"Expected 4 features, got {X_train.shape[1]}"
        )

    def test_correct_split_sizes(self, tabular_dataset) -> None:
        """80/20 split of 200 rows → 160 train / 40 val."""
        from api.federated.data_loaders import load_ml_data

        (X_train, y_train), (X_val, y_val) = load_ml_data(
            dataset_id=tabular_dataset.id,
            val_size=0.2,
        )

        assert X_train.shape[0] == 160, (
            f"Expected 160 train samples, got {X_train.shape[0]}"
        )
        assert X_val.shape[0] == 40, (
            f"Expected 40 val samples, got {X_val.shape[0]}"
        )
        assert y_train.shape[0] == 160
        assert y_val.shape[0] == 40

    def test_labels_are_integer_encoded(self, tabular_dataset) -> None:
        """y_train must be integer dtype after load_ml_data encodes the target.

        load_ml_data always casts non-string targets to np.int64 (or applies
        LabelEncoder for string targets). We only assert integer dtype here —
        the specific label values are a fixture property, not a function contract.
        """
        from api.federated.data_loaders import load_ml_data

        (_, y_train), _ = load_ml_data(dataset_id=tabular_dataset.id)

        assert np.issubdtype(y_train.dtype, np.integer), (
            f"Expected integer dtype for y_train, got {y_train.dtype}"
        )

    def test_raises_on_missing_dataset(self) -> None:
        """A non-existent (but valid-format) dataset ID must raise RuntimeError.

        dataset_id=999999 passes the positive-integer validation, so the loader
        proceeds to the DB lookup, which returns (None, None) for a missing
        record. The loader then raises RuntimeError with a descriptive message.
        """
        from api.federated.data_loaders import load_ml_data

        with pytest.raises(RuntimeError):
            load_ml_data(dataset_id=999999)
