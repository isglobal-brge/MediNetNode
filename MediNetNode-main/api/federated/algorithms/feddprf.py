"""
FedDP Random Forest Algorithm Implementation.

Implementation of Federated Differentially Private Random Forest for
privacy-preserving federated learning.

Reference: DP Random Forest paper
"""

import numpy as np
import pickle
import base64
from typing import List, Tuple, Dict, Any, Optional
from .base import FederatedMLAlgorithm
from .dp_tree_core import SecureDPTree, check_random_state_secure
from .privacy_guards import (
    FeatureBoundsValidator,
    EpsilonValidator,
    compute_data_hash
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
SKLEARN_AVAILABLE = True



class FedDPRandomForestAlgorithm(FederatedMLAlgorithm):
    """
    Federated Differentially Private Random Forest.

    Each client trains N DP trees locally using random structure and
    PermuteAndFlip mechanism. Trees are serialized and sent to server
    for aggregation into global forest.

    Configuration:
        training:
            ml_method: 'dp_random_forest'
            n_trees_per_client: int (default: 10)
            max_depth: int (default: 10)
            min_samples_split: int (default: 2)
            epsilon_total: float (default: 1.0)
            feature_bounds:
                min: List[float] (per-feature minimums)
                max: List[float] (per-feature maximums)

    Example:
        >>> config = {
        ...     'training': {
        ...         'ml_method': 'dp_random_forest',
        ...         'n_trees_per_client': 10,
        ...         'max_depth': 10,
        ...         'epsilon_total': 1.0,
        ...         'feature_bounds': {
        ...             'min': [0.0, 0.0, 0.0],
        ...             'max': [1.0, 1.0, 1.0]
        ...         }
        ...     }
        ... }
        >>> algorithm = FedDPRandomForestAlgorithm(X_train, y_train, config)
        >>> params, metrics = algorithm.fit([])
    """

    # ===== CLIENT-SIDE SECURITY CONSTRAINTS =====
    # These limits are HARDCODED in client to prevent server tampering
    # Hospital code is protected - server code could be modified

    # Privacy constraints (strictest limits)
    MIN_EPSILON_ALLOWED = 0.1    # Below this: unusable model
    MAX_EPSILON_ALLOWED = 5.0    # Above this: weak privacy (stricter than EpsilonValidator)
    DEFAULT_EPSILON = 1.0         # Safe default if missing/invalid

    # Model complexity constraints
    MIN_TREES = 1
    MAX_TREES = 50               # Limit to prevent resource exhaustion
    DEFAULT_TREES = 10

    MIN_DEPTH = 1
    MAX_DEPTH = 20               # Limit to prevent overfitting and resource issues
    DEFAULT_DEPTH = 10

    # Feature bounds requirement
    REQUIRE_GLOBAL_BOUNDS = True  # ALWAYS validate against global bounds

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """
        Initialize FedDP Random Forest algorithm with privacy safeguards.

        Args:
            X_train: Training features
            y_train: Training labels (integer class labels)
            config: Model configuration dictionary
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Raises:
            ValueError: If configuration is invalid or feature_bounds not provided
        """
        super().__init__(X_train, y_train, config, X_val, y_val)

        # ===== DEFENSIVE CONFIG VALIDATION (CLIENT-SIDE SECURITY) =====
        # Validate and sanitize config BEFORE using any values
        config = self._validate_and_sanitize_config(config)

        # Extract configuration
        training_config = config.get('training', {})
        privacy_config = config.get('privacy', {})

        # ===== ALGORITHM CONFIGURATION =====
        self.n_trees_per_client = training_config.get('n_trees_per_client', self.DEFAULT_TREES)
        self.max_depth = training_config.get('max_depth', self.DEFAULT_DEPTH)
        self.min_samples_split = training_config.get('min_samples_split', 2)
        self.epsilon_total = training_config.get('epsilon_total', self.DEFAULT_EPSILON)

        # ===== EPSILON VALIDATION =====
        EpsilonValidator.validate(
            self.epsilon_total,
            allow_high_risk=privacy_config.get('allow_high_risk_epsilon', False)
        )

        # ===== FEATURE BOUNDS VALIDATION =====
        self.feature_bounds = training_config.get('feature_bounds')
        if self.feature_bounds is None:
            raise ValueError(
                "feature_bounds must be provided in config.training. "
                "Use server-provided global bounds or DP-protected bounds. "
                "Example: {'min': [0, 0, 0], 'max': [1, 1, 1]}"
            )

        # Validate bounds structure
        if 'min' not in self.feature_bounds or 'max' not in self.feature_bounds:
            raise ValueError("feature_bounds must have 'min' and 'max' keys")

        # Validate global bounds (if server-provided global bounds exist)
        global_bounds = training_config.get('global_feature_bounds')
        require_global = privacy_config.get('require_global_bounds', True)

        if global_bounds and require_global:
            print("Validating feature bounds against server global bounds...")
            FeatureBoundsValidator.validate_global_bounds(
                local_bounds=self.feature_bounds,
                global_bounds=global_bounds
            )
            print("Feature bounds validated successfully")

        if len(self.feature_bounds['min']) != X_train.shape[1]:
            raise ValueError(
                f"feature_bounds['min'] length ({len(self.feature_bounds['min'])}) "
                f"must match n_features ({X_train.shape[1]})"
            )

        # Privacy budget per tree
        self.epsilon_per_tree = self.epsilon_total / self.n_trees_per_client

        # Initialize secure random state
        self.random_state = check_random_state_secure()

        # Store trained trees
        self.trees: List[SecureDPTree] = []

        # Class information
        self.n_classes = len(np.unique(y_train))

        # Store data hash for integrity verification
        self.data_hash = compute_data_hash(X_train)

        print(f"[INIT] Initializing FedDP Random Forest:")
        print(f"   Trees per client: {self.n_trees_per_client}")
        print(f"   Max depth: {self.max_depth}")
        print(f"   Epsilon total: {self.epsilon_total}")
        print(f"   Epsilon per tree: {self.epsilon_per_tree:.4f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {self.n_classes}")
        print(f"   Data hash: {self.data_hash[:16]}...")

    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Train N DP trees locally.

        Process:
        1. Receive trees from server (if provided) and add to forest
        2. Train N new DP trees using epsilon_per_tree each
        3. Serialize all trees (existing + new) for transmission
        4. Compute validation metrics

        Args:
            parameters: [serialized_trees] from server (base64 encoded pickle)

        Returns:
            Tuple of (serialized_trees, metrics_dict)
        """
        print(f"\n{'─'*60}")
        print(f"[FEDDPRF] fit() started")
        print(f"{'─'*60}")

        # 1. Receive trees from server (if provided)
        if parameters and len(parameters) > 0 and parameters[0].size > 0:
            print(f"[FEDDPRF] Deserializing trees from server...")
            received_trees = self._deserialize_trees(parameters[0])
            # Replace local forest with received global forest
            self.trees = received_trees
            print(f"[RECV] Received {len(received_trees)} trees from server")
            print(f"   Replaced local forest with global forest")

        # 2. Train N new DP trees
        print(f"[FEDDPRF] Training {self.n_trees_per_client} new DP trees...")
        new_trees = []

        for i in range(self.n_trees_per_client):
            # Create and train tree
            tree = SecureDPTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_bounds=self.feature_bounds,
                epsilon=self.epsilon_per_tree,
                random_state=self.random_state
            )

            tree.fit(self.X_train, self.y_train)
            new_trees.append(tree)

            if (i + 1) % 5 == 0 or (i + 1) == self.n_trees_per_client:
                print(f"   Trained {i + 1}/{self.n_trees_per_client} trees")

        # Add new trees to forest
        self.trees.extend(new_trees)
        print(f"[OK] Training complete: {len(new_trees)} new trees trained")
        print(f"   Total trees in forest: {len(self.trees)}")

        # 3. Serialize trees for transmission
        print(f"[FEDDPRF] Serializing {len(new_trees)} new trees...")
        parameters_out = self._serialize_trees(new_trees)

        # 4. Compute metrics on VALIDATION data (if available)
        print(f"[FEDDPRF] Computing metrics...")
        if self.X_val is not None and self.y_val is not None:
            print(f"[FEDDPRF] Using VALIDATION data ({len(self.X_val)} samples)")
            y_pred = self.predict(self.X_val)
            metrics = self._compute_metrics(self.y_val, y_pred)
            print(f"[FEDDPRF] Validation metrics - Acc: {metrics['accuracy']:.4f} | "
                  f"Loss: {metrics['loss']:.4f} | F1: {metrics['f1']:.4f}")
        else:
            print(f"[FEDDPRF] No validation data available")
            metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        # Add algorithm-specific info
        metrics.update({
            'n_trees_local': len(self.trees),
            'n_trees_sent': len(new_trees),
            'epsilon_per_tree': self.epsilon_per_tree,
            'round_type': 'training'
        })

        print(f"{'─'*60}")
        print(f"[FEDDPRF] fit() completed")
        print(f"{'─'*60}\n")

        return parameters_out, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate forest on validation data.

        Args:
            parameters: [serialized_trees] from server
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (loss, accuracy)
        """
        # Receive trees if provided
        if parameters and len(parameters) > 0 and parameters[0].size > 0:
            received_trees = self._deserialize_trees(parameters[0])
            self.trees = received_trees

        # Predict on validation data
        y_pred = self.predict(X_val)

        # Compute metrics
        metrics = self._compute_metrics(y_val, y_pred)

        return metrics['loss'], metrics['accuracy']

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get current forest parameters (all trees).

        Returns:
            List containing [serialized_trees]
        """
        if not self.trees:
            return [np.array([])]

        return self._serialize_trees(self.trees)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set forest parameters (deserialize trees).

        Args:
            parameters: [serialized_trees]
        """
        if not parameters or len(parameters) == 0 or parameters[0].size == 0:
            return

        self.trees = self._deserialize_trees(parameters[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using majority voting across all trees.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if not self.trees:
            raise ValueError("No trees in forest. Train or receive trees first.")

        # Collect predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority voting (axis=0 votes across trees for each sample)
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=all_predictions
        )

        return predictions

    # ==================== Security & Validation Methods ====================

    def _validate_and_sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize configuration from server.

        CLIENT-SIDE SECURITY: This method enforces hardcoded limits to prevent
        a compromised server from sending dangerous configurations.

        Args:
            config: Configuration dict from server

        Returns:
            Sanitized configuration dict

        Raises:
            ValueError: If configuration violates security constraints
        """
        import warnings

        sanitized_config = config.copy()
        training_config = sanitized_config.get('training', {})

        # ===== EPSILON VALIDATION (STRICTEST CHECK) =====
        epsilon = training_config.get('epsilon_total')

        if epsilon is None:
            warnings.warn(
                f"Server did not provide epsilon_total. "
                f"Using secure default: {self.DEFAULT_EPSILON}",
                category=UserWarning
            )
            training_config['epsilon_total'] = self.DEFAULT_EPSILON
        elif epsilon < self.MIN_EPSILON_ALLOWED:
            raise ValueError(
                f"SECURITY: epsilon_total={epsilon} is below minimum allowed "
                f"({self.MIN_EPSILON_ALLOWED}). Model would be unusable. "
                f"Rejecting configuration from server."
            )
        elif epsilon > self.MAX_EPSILON_ALLOWED:
            raise ValueError(
                f"SECURITY: epsilon_total={epsilon} exceeds maximum allowed "
                f"({self.MAX_EPSILON_ALLOWED}). Privacy guarantees too weak. "
                f"Client rejects this configuration. "
                f"This limit is HARDCODED in client code for security."
            )

        # ===== N_TREES VALIDATION =====
        n_trees = training_config.get('n_trees_per_client')

        if n_trees is None:
            warnings.warn(
                f"Server did not provide n_trees_per_client. "
                f"Using default: {self.DEFAULT_TREES}",
                category=UserWarning
            )
            training_config['n_trees_per_client'] = self.DEFAULT_TREES
        elif n_trees < self.MIN_TREES or n_trees > self.MAX_TREES:
            raise ValueError(
                f"SECURITY: n_trees_per_client={n_trees} outside allowed range "
                f"[{self.MIN_TREES}, {self.MAX_TREES}]. "
                f"Rejecting configuration from server."
            )

        # ===== MAX_DEPTH VALIDATION =====
        max_depth = training_config.get('max_depth')

        if max_depth is None:
            warnings.warn(
                f"Server did not provide max_depth. "
                f"Using default: {self.DEFAULT_DEPTH}",
                category=UserWarning
            )
            training_config['max_depth'] = self.DEFAULT_DEPTH
        elif max_depth < self.MIN_DEPTH or max_depth > self.MAX_DEPTH:
            raise ValueError(
                f"SECURITY: max_depth={max_depth} outside allowed range "
                f"[{self.MIN_DEPTH}, {self.MAX_DEPTH}]. "
                f"Rejecting configuration from server."
            )

        # ===== FEATURE BOUNDS VALIDATION =====
        feature_bounds = training_config.get('feature_bounds')

        if feature_bounds is None:
            raise ValueError(
                "SECURITY: Server did not provide feature_bounds. "
                "Feature bounds are REQUIRED for DP Random Forest. "
                "Rejecting configuration."
            )

        # ===== GLOBAL BOUNDS REQUIREMENT (HARDCODED) =====
        global_bounds = training_config.get('global_feature_bounds')

        if self.REQUIRE_GLOBAL_BOUNDS and global_bounds is None:
            raise ValueError(
                "SECURITY: Server did not provide global_feature_bounds. "
                "Client REQUIRES global bounds validation to prevent information leakage. "
                "This requirement is HARDCODED in client for security. "
                "Rejecting configuration."
            )

        # Force require_global_bounds to True (override server config)
        if 'privacy' not in sanitized_config:
            sanitized_config['privacy'] = {}
        sanitized_config['privacy']['require_global_bounds'] = True

        print("[OK] Configuration validated by client security checks")
        print(f"   Epsilon: {training_config['epsilon_total']} (range: [{self.MIN_EPSILON_ALLOWED}, {self.MAX_EPSILON_ALLOWED}])")
        print(f"   Trees: {training_config['n_trees_per_client']} (range: [{self.MIN_TREES}, {self.MAX_TREES}])")
        print(f"   Max Depth: {training_config['max_depth']} (range: [{self.MIN_DEPTH}, {self.MAX_DEPTH}])")
        print(f"   Global Bounds: Required and validated")

        return sanitized_config

    # ==================== Helper Methods ====================

    def _serialize_trees(self, trees: List[SecureDPTree]) -> List[np.ndarray]:
        """
        Serialize trees for transmission.

        Format: pickle → base64 → numpy array of bytes

        Args:
            trees: List of SecureDPTree instances

        Returns:
            [serialized_trees_array]
        """
        if not trees:
            return [np.array([])]

        # Extract tree states
        tree_states = [tree.get_state() for tree in trees]

        # Pickle and base64 encode
        pickled = pickle.dumps(tree_states)
        encoded = base64.b64encode(pickled)

        # Convert to numpy array of bytes
        serialized_array = np.frombuffer(encoded, dtype=np.uint8)

        return [serialized_array]

    def _deserialize_trees(self, serialized_array: np.ndarray) -> List[SecureDPTree]:
        """
        Deserialize trees from transmission format.

        Args:
            serialized_array: Numpy array of bytes

        Returns:
            List of SecureDPTree instances
        """
        if serialized_array.size == 0:
            return []

        # Convert back to bytes and decode
        encoded_bytes = serialized_array.tobytes()
        pickled = base64.b64decode(encoded_bytes)
        tree_states = pickle.loads(pickled)

        # Reconstruct trees
        trees = []
        for state in tree_states:
            tree = SecureDPTree()
            tree.set_state(state)
            trees.append(tree)

        return trees

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with accuracy, precision, recall, f1, loss
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: compute only accuracy
            accuracy = np.mean(y_true == y_pred)
            return {
                'accuracy': float(accuracy),
                'loss': float(1.0 - accuracy),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        # Compute comprehensive metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Handle binary and multiclass
        average_method = 'binary' if self.n_classes == 2 else 'weighted'
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average_method,
            zero_division=0
        )

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'loss': float(1.0 - accuracy)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get FedDP Random Forest model information."""
        info = super().get_model_info()
        info.update({
            'n_trees_per_client': self.n_trees_per_client,
            'n_trees_total': len(self.trees),
            'max_depth': self.max_depth,
            'epsilon_total': self.epsilon_total,
            'epsilon_per_tree': self.epsilon_per_tree,
            'n_classes': self.n_classes,
            'data_hash': self.data_hash[:16] if hasattr(self, 'data_hash') else None
        })
        return info

    def get_privacy_audit_info(self) -> Dict[str, Any]:
        """
        Get privacy-related information for audit logging.

        This method provides comprehensive privacy metrics that can be
        logged using MediNet's AuditLogger system for compliance tracking.

        Returns:
            Dictionary containing:
                - epsilon_total: Total privacy budget
                - epsilon_per_tree: Budget per tree
                - n_trees_trained: Number of trees trained
                - privacy_budget_consumed: Total epsilon consumed
                - data_hash: Training data integrity hash (truncated)
                - feature_bounds_validated: Whether bounds were validated
                - algorithm: Algorithm name

        Example:
            >>> audit_info = algorithm.get_privacy_audit_info()
            >>> # Can be logged with MediNet's AuditLogger:
            >>> # AuditLogger.log_event(
            >>> #     action='DP_TRAINING_ROUND',
            >>> #     user=user,
            >>> #     resource='dp_random_forest',
            >>> #     details=audit_info,
            >>> #     category='TRAINING'
            >>> # )
        """
        return {
            'algorithm': 'FedDP Random Forest',
            'epsilon_total': self.epsilon_total,
            'epsilon_per_tree': self.epsilon_per_tree,
            'n_trees_trained': len(self.trees),
            'privacy_budget_consumed': self.epsilon_total,
            'data_hash': self.data_hash[:16] if hasattr(self, 'data_hash') else None,
            'feature_bounds_validated': True,
            'max_depth': self.max_depth,
            'n_classes': self.n_classes,
            'training_samples': len(self.X_train) if hasattr(self, 'X_train') else 0
        }
