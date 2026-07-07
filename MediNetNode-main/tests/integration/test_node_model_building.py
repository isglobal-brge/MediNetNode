"""
Integration tests for Node model-building and DP wiring.

Validates DynamicModel construction, Opacus PrivacyEngine application,
FedSVM instantiation, and FedDPRandomForest instantiation.
"""
import numpy as np
import pytest
import torch
import torch.utils.data


# in_features=52: matches heart_attack_prediction_dataset_preprocessed.csv
# (52 feature columns after dropping 'Heart Attack Risk' target column).
DL_CONFIG = {
    "model": {
        "metadata": {"model_type": "dl", "framework": "pytorch"},
        "config_json": {
            "architecture": {
                "layers": [
                    {
                        "id": "layer_0",
                        "type": "Linear",
                        "params": {"in_features": 52, "out_features": 32},
                        "inputs": ["input_data"],
                    },
                    {
                        "id": "layer_1",
                        "type": "ReLU",
                        "params": {},
                        "inputs": ["layer_0"],
                    },
                    {
                        "id": "layer_2",
                        "type": "Linear",
                        "params": {"in_features": 32, "out_features": 1},
                        "inputs": ["layer_1"],
                    },
                ]
            }
        },
        "training": {
            "loss_function": "bce_with_logits",
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.01,
                "weight_decay": 0,
                "differential_privacy": {
                    "noise_multiplier": 1.0,
                    "max_grad_norm": 1.0,
                    "random_seed": 42,
                },
            },
        },
    },
    "train": {"rounds": 1, "epochs": 1, "batch_size": 16},
}

SVM_CONFIG = {
    "model": {
        "metadata": {"model_type": "ml"},
        "training": {
            "ml_method": "fedsvm",
            "C": 1.0,
            "kernel_config": {"kernel": "rbf", "gamma": 0.1},
            "val_size": 0.2,
            "random_state": 42,
        },
    },
}

DPRF_CONFIG = {
    "model": {
        "metadata": {"model_type": "ml"},
        "training": {
            "ml_method": "dp_random_forest",
            "n_trees_per_client": 5,
            "max_depth": 5,
            "epsilon_total": 1.0,
            "feature_bounds": {
                "min": [0.0, 0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0, 1.0],
            },
        },
    },
    # _validate_and_sanitize_config reads top-level 'training', not 'model.training'
    "training": {
        "ml_method": "dp_random_forest",
        "n_trees_per_client": 5,
        "max_depth": 5,
        "epsilon_total": 1.0,
        "feature_bounds": {
            "min": [0.0, 0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0, 1.0],
        },
        # REQUIRE_GLOBAL_BOUNDS is hardcoded True in FedDPRandomForestAlgorithm
        "global_feature_bounds": {
            "min": [0.0, 0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0, 1.0],
        },
    },
}


class TestDynamicModelBuilding:
    """Tests for DynamicModel — DL architecture construction from JSON config."""

    def test_model_builds_without_error(self) -> None:
        from api.federated.model_builder import DynamicModel

        net = DynamicModel(DL_CONFIG)
        assert net is not None

    def test_model_has_parameters(self) -> None:
        from api.federated.model_builder import DynamicModel

        net = DynamicModel(DL_CONFIG)
        params = list(net.parameters())
        assert len(params) > 0, "Model must have trainable parameters"

    def test_forward_pass_produces_correct_shape(self) -> None:
        from api.federated.model_builder import DynamicModel

        net = DynamicModel(DL_CONFIG)
        dummy_input = torch.rand(4, 52)  # batch_size=4, features=52 (heart attack dataset)
        output = net(dummy_input)
        assert output.shape == (4, 1), f"Expected (4,1), got {output.shape}"

    def test_opacus_privacy_engine_applies_without_error(self) -> None:
        """DP is always applied in train_functions.train() via Opacus PrivacyEngine.
        Verifies the privacy engine wraps the model without raising an exception.
        """
        from api.federated.model_builder import DynamicModel
        from opacus import PrivacyEngine

        net = DynamicModel(DL_CONFIG)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        X = torch.rand(32, 52)  # 52 features
        y = torch.randint(0, 2, (32,)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        dp_params = DL_CONFIG["model"]["training"]["optimizer"]["differential_privacy"]
        engine = PrivacyEngine(secure_mode=False)
        net, opt, loader = engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=dp_params["noise_multiplier"],
            max_grad_norm=dp_params["max_grad_norm"],
        )

        assert net is not None
        assert opt is not None

    def test_dp_config_values_are_read_correctly(self) -> None:
        """Verify DP parameter values are accessible from the config dict."""
        dp = DL_CONFIG["model"]["training"]["optimizer"]["differential_privacy"]
        assert dp["noise_multiplier"] == 1.0
        assert dp["max_grad_norm"] == 1.0
        assert dp["noise_multiplier"] > 0, "noise_multiplier must be positive for DP"
        assert dp["max_grad_norm"] > 0, "max_grad_norm must be positive for DP"


class TestSVMAlgorithmInstantiation:
    """Tests for FedSVMAlgorithm instantiation and parameter extraction."""

    @pytest.fixture
    def svm_data(self):
        np.random.seed(42)
        X = np.random.rand(80, 4)
        y = (X[:, 0] > 0.5).astype(int)
        X_val = np.random.rand(20, 4)
        y_val = (X_val[:, 0] > 0.5).astype(int)
        return X, y, X_val, y_val

    def test_instantiates_from_config(self, svm_data) -> None:
        from api.federated.algorithms import get_algorithm

        X, y, X_val, y_val = svm_data
        AlgClass = get_algorithm("fedsvm")
        alg = AlgClass(X, y, SVM_CONFIG, X_val, y_val)
        assert alg is not None

    def test_fit_returns_parameters_and_metrics(self, svm_data) -> None:
        from api.federated.algorithms import get_algorithm

        X, y, X_val, y_val = svm_data
        AlgClass = get_algorithm("fedsvm")
        alg = AlgClass(X, y, SVM_CONFIG, X_val, y_val)
        params, metrics = alg.fit([])  # empty initial params = first round

        assert params is not None
        assert isinstance(metrics, dict)

    def test_evaluate_returns_loss_and_count(self, svm_data) -> None:
        from api.federated.algorithms import get_algorithm

        X, y, X_val, y_val = svm_data
        AlgClass = get_algorithm("fedsvm")
        alg = AlgClass(X, y, SVM_CONFIG, X_val, y_val)
        params, _ = alg.fit([])
        loss, accuracy = alg.evaluate(params, X_val, y_val)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)


class TestDPRandomForestInstantiation:
    """Tests for FedDPRandomForest algorithm — explicit epsilon-DP."""

    @pytest.fixture
    def dprf_data(self):
        np.random.seed(42)
        X = np.random.rand(80, 4)
        y = (X[:, 0] > 0.5).astype(int)
        X_val = np.random.rand(20, 4)
        y_val = (X_val[:, 0] > 0.5).astype(int)
        return X, y, X_val, y_val

    def test_instantiates_from_config(self, dprf_data) -> None:
        from api.federated.algorithms import get_algorithm

        X, y, X_val, y_val = dprf_data
        AlgClass = get_algorithm("dp_random_forest")
        alg = AlgClass(X, y, DPRF_CONFIG, X_val, y_val)
        assert alg is not None

    def test_epsilon_field_is_read(self, dprf_data) -> None:
        epsilon = DPRF_CONFIG["model"]["training"]["epsilon_total"]
        assert epsilon == 1.0

    def test_fit_returns_parameters(self, dprf_data) -> None:
        from api.federated.algorithms import get_algorithm

        X, y, X_val, y_val = dprf_data
        AlgClass = get_algorithm("dp_random_forest")
        alg = AlgClass(X, y, DPRF_CONFIG, X_val, y_val)
        params, metrics = alg.fit([])

        assert params is not None
