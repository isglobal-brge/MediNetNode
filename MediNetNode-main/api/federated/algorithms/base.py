"""
Base abstract class for all federated ML algorithms.

This module defines the interface that all federated machine learning algorithms
must implement to be compatible with the MLFlowerClient.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class FederatedMLAlgorithm(ABC):
    """
    Abstract base class for federated machine learning algorithms.

    All federated ML algorithms must inherit from this class and implement
    the required abstract methods to work with Flower's federated learning framework.

    The lifecycle of an algorithm is:
        1. __init__(): Initialize with training data and configuration
        2. fit(): Execute local training round with parameters from server
        3. get_parameters(): Serialize model state for aggregation
        4. evaluate(): Evaluate model on validation data
        5. set_parameters(): Optionally receive aggregated parameters

    Attributes:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        config: Algorithm-specific configuration dictionary
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any],
                 X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Initialize algorithm with training data and configuration.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            config: Algorithm-specific configuration from model JSON
                   Expected to contain 'training' key with algorithm parameters
            X_val: Validation features (optional), shape (n_val_samples, n_features)
            y_val: Validation labels (optional), shape (n_val_samples,)

        Example config structure:
            {
                'training': {
                    'ml_method': 'fedsvm',
                    'C': 2.0,
                    'kernel_config': {'kernel': 'rbf', 'gamma': 0.1},
                    'client_eps': 1e-4
                },
                'metadata': {...}
            }
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config

        # Validate input data
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples. "
                f"Got X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"
            )

        # Validate validation data if provided
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"X_val and y_val must have same number of samples. "
                    f"Got X_val: {X_val.shape[0]}, y_val: {y_val.shape[0]}"
                )

    @abstractmethod
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Execute one local training round.

        This method should:
        1. Deserialize parameters received from server (aggregated from other clients)
        2. Update local model state if needed
        3. Perform local training
        4. Serialize updated model state
        5. Compute training metrics

        Args:
            parameters: Model parameters from server as list of numpy arrays.
                       Format and structure depend on the specific algorithm.
                       Can be empty on first round.

        Returns:
            Tuple containing:
                - updated_parameters: List of numpy arrays representing updated model state
                - metrics: Dictionary with training metrics including:
                    * 'loss': float (training loss)
                    * 'accuracy': float (training accuracy)
                    * 'precision': float (optional, training precision)
                    * 'recall': float (optional, training recall)
                    * 'f1': float (optional, training F1 score)

        Example:
            >>> parameters, metrics = algorithm.fit([np.array([...]), np.array([...])])
            >>> print(metrics)
            {'loss': 0.23, 'accuracy': 0.89, 'precision': 0.87, 'recall': 0.91, 'f1': 0.89}
        """
        pass

    @abstractmethod
    def evaluate(self, parameters: List[np.ndarray], X_val: np.ndarray,
                 y_val: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.

        This method should:
        1. Deserialize parameters from server if provided
        2. Update model state if needed
        3. Make predictions on validation data
        4. Compute evaluation metrics

        Args:
            parameters: Model parameters from server as list of numpy arrays.
                       Can be same format as fit() parameters.
            X_val: Validation features, shape (n_samples_val, n_features)
            y_val: Validation labels, shape (n_samples_val,)

        Returns:
            Tuple containing:
                - loss: float (validation loss)
                - accuracy: float (validation accuracy, 0.0-1.0)

        Example:
            >>> loss, accuracy = algorithm.evaluate(parameters, X_val, y_val)
            >>> print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            Loss: 0.2156, Accuracy: 0.9123
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """
        Serialize current model state to list of numpy arrays.

        This method should extract the current state of the model and convert
        it to a list of numpy arrays that can be sent to the server for aggregation.

        The serialization format should match what fit() expects to receive
        as parameters.

        Returns:
            List of numpy arrays representing current model state.
            Can be empty if model hasn't been trained yet.

        Example:
            For SVM: [support_vectors_array, labels_array]
            For Neural Network: [weights_layer1, biases_layer1, weights_layer2, ...]
            For Random Forest: [tree_structures, feature_importances]

        Example:
            >>> params = algorithm.get_parameters()
            >>> print([p.shape for p in params])
            [(100, 30), (100,)]  # 100 support vectors with 30 features + labels
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Deserialize model state from list of numpy arrays.

        This method receives aggregated parameters from the server and should
        update the model's internal state accordingly.

        Args:
            parameters: Model state as list of numpy arrays (same format as get_parameters())

        Note:
            This method is optional in the Flower workflow. Some algorithms may not
            need to explicitly set parameters if fit() already handles parameter updates.

        Example:
            >>> algorithm.set_parameters([support_vectors, labels])
            >>> # Model state is now updated with new support vectors
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.

        This is an optional method that provides a convenient interface for making
        predictions. Not required by Flower but useful for evaluation and testing.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)

        Raises:
            NotImplementedError: If algorithm doesn't implement prediction

        Example:
            >>> y_pred = algorithm.predict(X_test)
            >>> print(y_pred.shape)
            (150,)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement predict() method"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get algorithm and model information.

        Returns dictionary with algorithm metadata useful for logging and debugging.

        Returns:
            Dictionary with model information

        Example:
            >>> info = algorithm.get_model_info()
            >>> print(info)
            {
                'algorithm': 'FedSVM',
                'n_samples': 1000,
                'n_features': 30,
                'config': {...}
            }
        """
        return {
            'algorithm': self.__class__.__name__,
            'n_train_samples': len(self.X_train),
            'n_features': self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1,
            'config': self.config
        }
