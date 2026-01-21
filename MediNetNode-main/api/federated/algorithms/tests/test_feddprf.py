"""
Tests for FedDP Random Forest Algorithm.

Validates:
- Algorithm initialization and configuration
- Federated training workflow (fit/evaluate)
- Tree serialization and aggregation
- Integration with MLFlowerClient
- Validation data metrics
"""

import pytest
import numpy as np
from ..feddprf import FedDPRandomForestAlgorithm
from ..base import FederatedMLAlgorithm


class TestFedDPRandomForestInitialization:
    """Test algorithm initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        # Compute bounds (simulating server-side global bounds)
        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 5,
                'max_depth': 8,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds  # Required by client security
            }
        }

        return X_train, y_train, X_val, y_val, config

    def test_algorithm_inherits_from_base(self):
        """Test FedDPRandomForestAlgorithm inherits from FederatedMLAlgorithm."""
        assert issubclass(FedDPRandomForestAlgorithm, FederatedMLAlgorithm)

    def test_initialization_with_valid_config(self, sample_data):
        """Test algorithm initializes correctly with valid config."""
        X_train, y_train, X_val, y_val, config = sample_data

        algorithm = FedDPRandomForestAlgorithm(
            X_train, y_train, config, X_val, y_val
        )

        assert algorithm.n_trees_per_client == 5
        assert algorithm.max_depth == 8
        assert algorithm.epsilon_total == 1.0
        assert algorithm.epsilon_per_tree == 0.2
        assert len(algorithm.trees) == 0

    def test_initialization_requires_feature_bounds(self, sample_data):
        """Test algorithm raises error without feature_bounds."""
        X_train, y_train, X_val, y_val, config = sample_data

        config['training'].pop('feature_bounds')

        with pytest.raises(ValueError, match="SECURITY.*feature_bounds"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_initialization_validates_bounds_structure(self, sample_data):
        """Test algorithm validates feature_bounds structure."""
        X_train, y_train, X_val, y_val, config = sample_data

        config['training']['feature_bounds'] = {'min': [0, 0]}  # Missing 'max'
        config['training']['global_feature_bounds'] = {'min': [0, 0]}

        with pytest.raises(ValueError, match="must have 'min' and 'max'"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_epsilon_per_tree_calculation(self, sample_data):
        """Test epsilon is correctly divided among trees."""
        X_train, y_train, X_val, y_val, config = sample_data

        config['training']['n_trees_per_client'] = 10
        config['training']['epsilon_total'] = 2.0
        config['training']['global_feature_bounds'] = config['training']['feature_bounds']

        algorithm = FedDPRandomForestAlgorithm(
            X_train, y_train, config, X_val, y_val
        )

        assert algorithm.epsilon_per_tree == 0.2


class TestClientSideSecurityValidation:
    """Test client-side security validation (hardcoded limits)."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for security tests."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 10,
                'max_depth': 10,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        return X_train, y_train, X_val, y_val, config

    def test_rejects_epsilon_too_high(self, base_config):
        """Test client rejects epsilon > MAX_EPSILON_ALLOWED."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training']['epsilon_total'] = 10.0  # Above MAX_EPSILON_ALLOWED=5.0

        with pytest.raises(ValueError, match="SECURITY: epsilon_total.*exceeds maximum allowed"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_rejects_epsilon_too_low(self, base_config):
        """Test client rejects epsilon < MIN_EPSILON_ALLOWED."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training']['epsilon_total'] = 0.05  # Below MIN_EPSILON_ALLOWED=0.1

        with pytest.raises(ValueError, match="SECURITY: epsilon_total.*below minimum allowed"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_uses_default_epsilon_if_missing(self, base_config):
        """Test client uses DEFAULT_EPSILON if server doesn't provide one."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training'].pop('epsilon_total')

        with pytest.warns(UserWarning, match="Using secure default"):
            algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

        assert algorithm.epsilon_total == 1.0  # DEFAULT_EPSILON

    def test_rejects_too_many_trees(self, base_config):
        """Test client rejects n_trees > MAX_TREES."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training']['n_trees_per_client'] = 100  # Above MAX_TREES=50

        with pytest.raises(ValueError, match="SECURITY: n_trees_per_client.*outside allowed range"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_rejects_too_deep_trees(self, base_config):
        """Test client rejects max_depth > MAX_DEPTH."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training']['max_depth'] = 30  # Above MAX_DEPTH=20

        with pytest.raises(ValueError, match="SECURITY: max_depth.*outside allowed range"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_requires_global_feature_bounds(self, base_config):
        """Test client REQUIRES global_feature_bounds."""
        X_train, y_train, X_val, y_val, config = base_config

        config['training'].pop('global_feature_bounds')

        with pytest.raises(ValueError, match="SECURITY: Server did not provide global_feature_bounds"):
            FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)

    def test_accepts_valid_epsilon_range(self, base_config):
        """Test client accepts epsilon in valid range."""
        X_train, y_train, X_val, y_val, config = base_config

        # Test boundary values
        for epsilon in [0.1, 1.0, 3.0, 5.0]:
            config['training']['epsilon_total'] = epsilon
            algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config, X_val, y_val)
            assert algorithm.epsilon_total == epsilon


class TestFedDPRandomForestTraining:
    """Test training workflow."""

    @pytest.fixture
    def trained_algorithm(self):
        """Create and train algorithm."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        # Compute bounds (simulating server-side global bounds)
        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 3,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds  # Required by client security
            }
        }

        algorithm = FedDPRandomForestAlgorithm(
            X_train, y_train, config, X_val, y_val
        )

        return algorithm, X_val, y_val

    def test_fit_trains_n_trees(self, trained_algorithm):
        """Test fit() trains N trees locally."""
        algorithm, X_val, y_val = trained_algorithm

        params, metrics = algorithm.fit([])

        assert len(algorithm.trees) == 3
        assert metrics['n_trees_local'] == 3
        assert metrics['n_trees_sent'] == 3

    def test_fit_returns_serialized_trees(self, trained_algorithm):
        """Test fit() returns serialized trees."""
        algorithm, X_val, y_val = trained_algorithm

        params, metrics = algorithm.fit([])

        assert len(params) == 1
        assert params[0].size > 0

    def test_fit_computes_validation_metrics(self, trained_algorithm):
        """Test fit() computes metrics on validation data."""
        algorithm, X_val, y_val = trained_algorithm

        params, metrics = algorithm.fit([])

        assert 'accuracy' in metrics
        assert 'loss' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_fit_aggregates_received_trees(self):
        """Test fit() aggregates trees from server."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 3,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(
            X_train, y_train, config, X_val, y_val
        )

        # First round: train 3 trees
        params1, metrics1 = algorithm.fit([])
        assert len(algorithm.trees) == 3

        # Second round: receive 3 trees + train 3 new = 6 total
        params2, metrics2 = algorithm.fit(params1)
        assert len(algorithm.trees) == 6
        assert metrics2['n_trees_local'] == 6
        assert metrics2['n_trees_sent'] == 3


class TestFedDPRandomForestEvaluation:
    """Test evaluation workflow."""

    @pytest.fixture
    def algorithm_with_trees(self):
        """Create algorithm with trained trees."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 5,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(
            X_train, y_train, config, X_val, y_val
        )
        algorithm.fit([])

        return algorithm, X_val, y_val

    def test_evaluate_returns_loss_and_accuracy(self, algorithm_with_trees):
        """Test evaluate() returns loss and accuracy."""
        algorithm, X_val, y_val = algorithm_with_trees

        loss, accuracy = algorithm.evaluate([], X_val, y_val)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert 0 <= loss <= 1

    def test_predict_uses_majority_voting(self, algorithm_with_trees):
        """Test predictions use majority voting across trees."""
        algorithm, X_val, y_val = algorithm_with_trees

        predictions = algorithm.predict(X_val)

        assert len(predictions) == len(X_val)
        assert all(pred in [0, 1] for pred in predictions)


class TestFedDPRandomForestSerialization:
    """Test tree serialization/deserialization."""

    @pytest.fixture
    def algorithm_with_trees(self):
        """Create algorithm with trained trees."""
        np.random.seed(42)
        X_train = np.random.randn(80, 4)
        y_train = np.random.randint(0, 2, 80)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 3,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)
        algorithm.fit([])

        return algorithm, X_train

    def test_get_parameters_returns_serialized_trees(self, algorithm_with_trees):
        """Test get_parameters() returns serialized trees."""
        algorithm, X_train = algorithm_with_trees

        params = algorithm.get_parameters()

        assert len(params) == 1
        assert params[0].size > 0

    def test_set_parameters_deserializes_trees(self, algorithm_with_trees):
        """Test set_parameters() deserializes trees."""
        algorithm, X_train = algorithm_with_trees

        params = algorithm.get_parameters()
        original_predictions = algorithm.predict(X_train)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        # Create new algorithm and set parameters
        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 3,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }
        new_algorithm = FedDPRandomForestAlgorithm(X_train, np.zeros(len(X_train)), config)
        new_algorithm.set_parameters(params)

        # Predictions should be identical
        new_predictions = new_algorithm.predict(X_train)
        np.testing.assert_array_equal(original_predictions, new_predictions)

    def test_serialization_roundtrip(self, algorithm_with_trees):
        """Test serialization → deserialization preserves predictions."""
        algorithm, X_train = algorithm_with_trees

        original_predictions = algorithm.predict(X_train)

        # Serialize
        serialized = algorithm._serialize_trees(algorithm.trees)

        # Deserialize
        deserialized_trees = algorithm._deserialize_trees(serialized[0])

        # Reconstruct algorithm with deserialized trees
        algorithm.trees = deserialized_trees
        new_predictions = algorithm.predict(X_train)

        np.testing.assert_array_equal(original_predictions, new_predictions)


class TestFedDPRandomForestIntegration:
    """Test integration with federated learning workflow."""

    def test_federated_workflow_simulation(self):
        """Simulate federated workflow: multiple clients training and aggregating."""
        np.random.seed(42)

        # Shared config
        X_global = np.random.randn(300, 4)
        y_global = np.random.randint(0, 3, 300)

        bounds = {
            'min': X_global.min(axis=0).tolist(),
            'max': X_global.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 5,
                'max_depth': 6,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        # Client 1: Train 5 trees
        X1 = X_global[:100]
        y1 = y_global[:100]
        client1 = FedDPRandomForestAlgorithm(X1, y1, config)
        params1, metrics1 = client1.fit([])

        assert metrics1['n_trees_sent'] == 5

        # Client 2: Receive 5 trees from server, train 5 more, send only new 5
        X2 = X_global[100:200]
        y2 = y_global[100:200]
        client2 = FedDPRandomForestAlgorithm(X2, y2, config)
        params2, metrics2 = client2.fit(params1)

        assert metrics2['n_trees_local'] == 10  # 5 received + 5 new (locally)
        assert metrics2['n_trees_sent'] == 5  # Only sends new 5 to server

        # Simulate server aggregation (10 trees total)
        # In real federated learning, server would aggregate params1 + params2
        # For test simplicity, we use params2 which client2 sent (5 trees)

        # Client 3: Receives aggregated forest, trains 5 more
        X3 = X_global[200:300]
        y3 = y_global[200:300]
        client3 = FedDPRandomForestAlgorithm(X3, y3, config)
        params3, metrics3 = client3.fit(params2)

        # Client3 receives 5 trees (from params2), trains 5 new = 10 local
        assert metrics3['n_trees_local'] == 10
        assert metrics3['n_trees_sent'] == 5

    def test_multiclass_classification(self):
        """Test algorithm handles multiclass classification."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 5, 100)  # 5 classes

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 5,
                'max_depth': 6,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)
        params, metrics = algorithm.fit([])

        predictions = algorithm.predict(X_train)

        assert all(0 <= pred < 5 for pred in predictions)


class TestFedDPRandomForestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info_returns_complete_info(self):
        """Test get_model_info() returns all relevant information."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 7,
                'max_depth': 8,
                'epsilon_total': 2.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)
        algorithm.fit([])

        info = algorithm.get_model_info()

        assert info['algorithm'] == 'FedDPRandomForestAlgorithm'
        assert info['n_trees_per_client'] == 7
        assert info['n_trees_total'] == 7
        assert info['max_depth'] == 8
        assert info['epsilon_total'] == 2.0
        assert info['epsilon_per_tree'] == 2.0 / 7
        assert info['n_classes'] == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_without_trees_raises_error(self):
        """Test predict() raises error when no trees trained."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 5,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)

        with pytest.raises(ValueError, match="No trees in forest"):
            algorithm.predict(X_train)

    def test_empty_parameters_handling(self):
        """Test algorithm handles empty parameters gracefully."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        bounds = {
            'min': X_train.min(axis=0).tolist(),
            'max': X_train.max(axis=0).tolist()
        }

        config = {
            'training': {
                'ml_method': 'dp_random_forest',
                'n_trees_per_client': 3,
                'max_depth': 5,
                'epsilon_total': 1.0,
                'feature_bounds': bounds,
                'global_feature_bounds': bounds
            }
        }

        algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)

        # Empty parameters should not crash
        params, metrics = algorithm.fit([np.array([])])

        assert len(algorithm.trees) == 3
