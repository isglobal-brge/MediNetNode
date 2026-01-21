"""
FedSVM Algorithm Implementation.

Implementation of Federated Support Vector Machine (FedSVM) with optimized
multiple deltas for privacy-preserving federated learning.

Reference: Support Vector Federation paper
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from .base import FederatedMLAlgorithm

# Import FedSVM core implementation (copied from FSV project)
try:
    from .fedsvm_core import FedSVMClientOptMD, evaluate_fedsvm
    FEDSVM_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] FedSVM core module not available: {e}")
    FEDSVM_AVAILABLE = False
    FedSVMClientOptMD = None
    evaluate_fedsvm = None


class FedSVMAlgorithm(FederatedMLAlgorithm):
    """
    Federated Support Vector Machine with Optimized Multiple Deltas.

    This implementation uses the FedSVMClientOptMD class which provides:
    - Privacy preservation through delta perturbations
    - Support for RBF and polynomial kernels
    - Optimized delta computation for better accuracy

    Configuration:
        training:
            ml_method: 'fedsvm'
            C: float (regularization parameter, default: 2.0)
            kernel_config:
                kernel: 'linear' | 'rbf' | 'poly'
                gamma: float (for RBF/poly, default: 0.1)
                degree: int (for poly, default: 3)
            client_eps: float (convergence threshold, default: 1e-4)
            device: 'cpu' | 'cuda' (default: 'cpu')

    Example:
        >>> config = {
        ...     'training': {
        ...         'ml_method': 'fedsvm',
        ...         'C': 2.0,
        ...         'kernel_config': {'kernel': 'rbf', 'gamma': 0.1},
        ...         'client_eps': 1e-4
        ...     }
        ... }
        >>> algorithm = FedSVMAlgorithm(X_train, y_train, config)
        >>> params, metrics = algorithm.fit([])
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any],
                 X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Initialize FedSVM algorithm.

        Args:
            X_train: Training features
            y_train: Training labels (binary: -1 or 1, or 0 or 1)
            config: Model configuration dictionary
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Raises:
            ImportError: If FedSVM module is not available
            ValueError: If configuration is invalid
        """
        super().__init__(X_train, y_train, config, X_val, y_val)

        if not FEDSVM_AVAILABLE:
            raise ImportError(
                "FedSVM module not available. Please ensure the 'fsv' directory "
                "is accessible and contains the required FedSVM implementation."
            )

        # Extract FedSVM-specific configuration
        training_config = config.get('training', {})

        # Try to get dataset characteristics from metadata
        model_config = config.get('model', config)
        metadata = model_config.get('metadata', {})
        target_info = metadata.get('target_info', {})
        dataset_chars = target_info.get('dataset_characteristics', {})
        ml_considerations = target_info.get('ml_considerations', {})

        # Intelligent defaults based on dataset characteristics
        n_samples = dataset_chars.get('n_samples', 1000)
        n_features = dataset_chars.get('n_features', 10)
        is_imbalanced = dataset_chars.get('is_imbalanced', False)

        # SVM-specific heuristics based on dataset
        if n_features > 1000:
            # High-dimensional: linear kernel recommended
            default_kernel = 'linear'
            default_C = 1.0
            default_gamma = 'scale'
        elif n_samples < 1000:
            # Small dataset: RBF with moderate C
            default_kernel = 'rbf'
            default_C = 1.0
            default_gamma = 'scale'
        elif n_samples > 100000:
            # Large dataset: linear kernel for speed
            default_kernel = 'linear'
            default_C = 1.0
            default_gamma = 'scale'
        else:
            # Medium dataset: RBF kernel works well
            default_kernel = 'rbf'
            default_C = 2.0
            default_gamma = 0.1

        # Use explicit config if provided, otherwise use intelligent defaults
        self.kernel_config = training_config.get('kernel_config', {
            'kernel': default_kernel,
            'gamma': default_gamma if default_gamma == 'scale' else default_gamma
        })

        self.C = training_config.get('C', default_C)
        self.client_eps = training_config.get('client_eps', 1e-4)
        self.device = training_config.get('device', 'cpu')

        # Log if using automatic configuration
        if dataset_chars and not training_config.get('kernel_config'):
            print(f"[INFO] Auto-configured FedSVM from dataset characteristics:")
            print(f"   Dataset: {n_samples} samples, {n_features} features")
            print(f"   Kernel: {self.kernel_config['kernel']}")
            print(f"   C: {self.C}")
            print(f"   Gamma: {self.kernel_config.get('gamma', 'N/A')}")
            if is_imbalanced:
                print(f"   [WARNING] Class imbalance detected: Consider class weighting")

        # Show ML recommendations if available
        ml_recommendations = target_info.get('ml_recommendations', [])
        if ml_recommendations:
            print(f"\n[TIP] Dataset recommendations:")
            for rec in ml_recommendations:
                print(f"   - {rec}")

        # Validate kernel configuration
        valid_kernels = ['linear', 'rbf', 'poly']
        kernel_type = self.kernel_config.get('kernel', 'rbf')

        if kernel_type not in valid_kernels:
            raise ValueError(
                f"Invalid kernel '{kernel_type}'. Must be one of: {valid_kernels}"
            )

        print(f"[INIT] Initializing FedSVM algorithm:")
        print(f"   Kernel: {kernel_type}")
        print(f"   C (regularization): {self.C}")
        print(f"   Client epsilon: {self.client_eps}")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(X_train)}")

        # Initialize FedSVM client
        # Note: client_no will be set by MLFlowerClient
        self.client = FedSVMClientOptMD(
            client_no=0,
            X=X_train,
            y=y_train,
            rff=None,  # Random Fourier Features not used for now
            client_eps=self.client_eps,
            C=self.C,
            kernel=self.kernel_config,
            device=self.device
        )

        print(f"[OK] FedSVM client initialized successfully")

    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Execute one FedSVM training round.

        Process:
        1. Deserialize support vectors from server (if provided)
        2. Update local SVM with received support vectors
        3. Train local SVM and generate perturbed support vectors
        4. Serialize perturbed support vectors for server
        5. Compute training metrics

        Args:
            parameters: [svs_array, labels_array] from server
                       svs_array: shape (n_svs, n_features)
                       labels_array: shape (n_svs,)

        Returns:
            Tuple of (serialized_svs, metrics_dict)
        """
        print(f"\n{'─'*60}")
        print(f"[FEDSVM] fit() started")
        print(f"[FEDSVM] Parameters type: {type(parameters)}")
        print(f"[FEDSVM] Parameters length: {len(parameters) if parameters else 0}")
        print(f"{'─'*60}")

        # 1. Deserialize and receive support vectors from server
        print(f"[FEDSVM] Step 1: Checking for parameters from server...")
        if parameters and len(parameters) >= 2 and parameters[0].size > 0:
            print(f"[FEDSVM] Deserializing support vectors...")
            svs_clients, labels_clients = self._deserialize_svs(parameters)
            num_received_svs = sum(len(svs) for svs in svs_clients)
            print(f"[RECV] Received {num_received_svs} support vectors from server")

            # Update local SVM with received support vectors
            print(f"[FEDSVM] Updating local SVM with received SVs...")
            self.client.receive_svs(svs_clients, labels_clients)
            print(f"[FEDSVM] Local SVM updated successfully")
        else:
            print(f"[FEDSVM] No support vectors received (first round)")

        # 2. Generate perturbed support vectors with privacy-preserving deltas
        print(f"[FEDSVM] Step 2: Generating perturbed support vectors...")
        svs_delta, labels = self.client.send_svs()
        num_svs_to_send = len(svs_delta)

        print(f"[FEDSVM] Generated {num_svs_to_send} perturbed support vectors")

        # 3. Serialize support vectors for transmission
        print(f"[FEDSVM] Step 3: Serializing support vectors for transmission...")
        parameters_out = self._serialize_svs(svs_delta, labels)
        print(f"[FEDSVM] Serialization complete, returning {len(parameters_out)} arrays")

        # 4. Compute metrics on VALIDATION data (if available)
        print(f"[FEDSVM] Step 4: Computing metrics...")
        if self.X_val is not None and self.y_val is not None:
            # Use validation data for metrics (correct approach)
            print(f"[FEDSVM] Computing metrics on VALIDATION data ({len(self.X_val)} samples)...")
            y_pred = self.client.predict(self.X_val, rff=None)
            metrics = self._compute_metrics(self.y_val, y_pred)
            print(f"[FEDSVM] Validation metrics - Acc: {metrics['accuracy']:.4f} | "
                  f"Loss: {metrics['loss']:.4f} | F1: {metrics['f1']:.4f}")
        else:
            # Fallback: no validation data, return basic info
            print(f"[FEDSVM] No validation data available, returning basic info")
            metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        # Add algorithm-specific info to metrics
        metrics.update({
            'n_support_vectors': len(self.client.svs) if self.client.svs is not None else 0,
            'n_svs_sent': num_svs_to_send,
            'round_type': 'training'
        })
        print(f"[FEDSVM] Training info: {num_svs_to_send} SVs sent, "
              f"{metrics['n_support_vectors']} total SVs")

        print(f"{'─'*60}")
        print(f"[FEDSVM] fit() completed successfully")
        print(f"{'─'*60}\n")

        return parameters_out, metrics

    def evaluate(self, parameters: List[np.ndarray], X_val: np.ndarray,
                 y_val: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate FedSVM model on validation data.

        Args:
            parameters: [svs_array, labels_array] from server
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (loss, accuracy)
        """
        # Receive support vectors if provided
        if parameters and len(parameters) >= 2 and parameters[0].size > 0:
            svs_clients, labels_clients = self._deserialize_svs(parameters)
            self.client.receive_svs(svs_clients, labels_clients)

        # Predict on validation data
        y_pred = self.client.predict(X_val, rff=None)

        # Compute comprehensive metrics
        metrics = self._compute_metrics(y_val, y_pred)

        # Return loss and accuracy (Flower interface requirement)
        # Full metrics are logged by _compute_metrics
        return metrics['loss'], metrics['accuracy']

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get current SVM parameters (support vectors).

        Returns:
            List containing [support_vectors_array, labels_array]
        """
        if self.client.svs is None or len(self.client.svs) == 0:
            return [np.array([]), np.array([])]

        svs_array = self.client.svs.cpu().numpy()
        labels_array = self.client.svs_labels

        return [svs_array, labels_array]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set SVM parameters (support vectors).

        Args:
            parameters: [svs_array, labels_array]
        """
        if not parameters or len(parameters) < 2:
            return

        svs_clients, labels_clients = self._deserialize_svs(parameters)
        if svs_clients:
            self.client.receive_svs(svs_clients, labels_clients)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using trained FedSVM model.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        return self.client.predict(X, rff=None)

    # ==================== Helper Methods ====================

    def _serialize_svs(self, svs_deltas: List[Tuple[str, torch.Tensor]],
                       labels: np.ndarray) -> List[np.ndarray]:
        """
        Convert support vectors with deltas to numpy arrays.

        Format: [svs_array, labels_array, shas_array]
        Compatible with server's aggregate_fit().

        Args:
            svs_deltas: List of (sha_key, sv_tensor) tuples
            labels: Support vector labels

        Returns:
            [svs_array, labels_array, shas_array]
        """
        if not svs_deltas:
            return [np.array([]), np.array([]), np.array([])]

        # Extract tensors, SHAs and convert to numpy
        svs_list = []
        shas_list = []

        for sha_key, sv_tensor in svs_deltas:
            # Convert tensor to numpy
            if torch.is_tensor(sv_tensor):
                sv_numpy = sv_tensor.cpu().numpy()
            else:
                sv_numpy = sv_tensor
            svs_list.append(sv_numpy)

            # Convert SHA string to numeric hash for serialization
            shas_list.append(hash(sha_key) % (2**31))

        # Stack into arrays
        svs_array = np.stack(svs_list, axis=0).astype(np.float32) if svs_list else np.array([])
        labels_array = np.array(labels, dtype=np.float32) if len(labels) > 0 else np.array([])
        shas_array = np.array(shas_list, dtype=np.float32) if shas_list else np.array([])

        return [svs_array, labels_array, shas_array]

    def _deserialize_svs(self, parameters: List[np.ndarray]) -> Tuple[List, List]:
        """
        Convert numpy arrays back to support vector format.

        Server sends: [svs_array, labels_array, shas_count]

        Args:
            parameters: [svs_array, labels_array, shas_count]

        Returns:
            (svs_clients, labels_clients) in FedSVM format
        """
        if len(parameters) < 2:
            return [], []

        svs_array = parameters[0]
        labels_array = parameters[1]

        # Handle empty arrays
        if svs_array.size == 0 or labels_array.size == 0:
            return [], []

        # Convert to FedSVM format: List of [(sha_key, tensor), ...]
        client_svs = []
        for i, sv in enumerate(svs_array):
            sha_key = f"received_sv_{i}"
            sv_tensor = torch.tensor(sv, dtype=torch.float32, device=self.device)
            client_svs.append((sha_key, sv_tensor))

        # FedSVM expects list of clients, each with their SVs
        svs_clients = [client_svs]
        labels_clients = [torch.tensor(labels_array, dtype=torch.float32, device=self.device)]

        return svs_clients, labels_clients

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with accuracy, precision, recall, f1, loss
        """
        # Use FedSVM's evaluation function
        metrics = evaluate_fedsvm(y_true, y_pred)

        # Add loss (SVM doesn't have traditional loss, use 1 - accuracy as proxy)
        metrics['loss'] = 1.0 - metrics['accuracy']

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """Get FedSVM model information."""
        info = super().get_model_info()
        info.update({
            'kernel': self.kernel_config.get('kernel', 'unknown'),
            'C': self.C,
            'n_support_vectors': len(self.client.svs) if self.client.svs is not None else 0,
            'client_epsilon': self.client_eps
        })
        return info
