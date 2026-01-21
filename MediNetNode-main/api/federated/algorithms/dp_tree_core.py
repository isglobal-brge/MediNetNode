"""
Differentially Private Decision Tree Core Implementation.

Implements the DP Decision Tree algorithm using the PermuteAndFlip mechanism
for privacy-preserving federated learning with random forests.

Reference: Differentially Private Random Forest paper
"""

import numpy as np
import secrets
from typing import List, Tuple, Optional, Dict, Any


def check_random_state_secure():
    """
    Get cryptographically secure random number generator.

    Uses secrets.SystemRandom() instead of numpy.random for security.

    Returns:
        SystemRandom instance
    """
    return secrets.SystemRandom()


class SecureDPTree:
    """
    Differentially Private Decision Tree using PermuteAndFlip mechanism.

    Two-phase construction:
    1. Phase 1 (0-DP): Random tree structure without looking at data
    2. Phase 2 (ε-DP): Private label assignment using PermuteAndFlip

    Attributes:
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        feature_bounds: Dict with 'min' and 'max' bounds per feature
        epsilon: Privacy budget for label assignment
        random_state: Secure random number generator
        tree_structure: Dictionary representing tree structure
        n_features: Number of input features
        n_classes: Number of unique classes
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        feature_bounds: Optional[Dict[str, List[float]]] = None,
        epsilon: float = 1.0,
        random_state: Optional[Any] = None
    ):
        """
        Initialize DP Decision Tree.

        Args:
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples required to split
            feature_bounds: Dict with 'min' and 'max' arrays for feature bounds
            epsilon: Privacy budget for label assignment
            random_state: Secure random state (uses SystemRandom if None)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_bounds = feature_bounds
        self.epsilon = epsilon
        self.random_state = random_state if random_state else check_random_state_secure()
        self.tree_structure = None
        self.n_features = None
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SecureDPTree':
        """
        Train DP decision tree with two-phase approach.

        Phase 1: Build random tree structure (0-DP)
        Phase 2: Assign labels privately using PermuteAndFlip (ε-DP)

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self (fitted tree)
        """
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))

        # Validate feature bounds
        if self.feature_bounds is None:
            raise ValueError(
                "feature_bounds must be provided for DP tree. "
                "Calculate bounds from aggregated statistics, not raw data."
            )

        # Phase 1: Build random tree structure (0-DP)
        self.tree_structure = self._build_random_tree(depth=0)

        # Phase 2: Assign labels privately (ε-DP)
        self._assign_labels_dp(self.tree_structure, X, y)

        return self

    def _build_random_tree(self, depth: int) -> Dict[str, Any]:
        """
        Phase 1: Build random tree structure without looking at data (0-DP).

        Randomly selects features and split points based only on feature bounds.

        Args:
            depth: Current depth in tree

        Returns:
            Tree node dictionary with structure
        """
        # Stop conditions: max depth reached
        if depth >= self.max_depth:
            return {'type': 'leaf', 'label': None}

        # Randomly decide: split or leaf (50% probability)
        if self.random_state.random() < 0.5:
            return {'type': 'leaf', 'label': None}

        # Random split: choose feature and threshold
        feature_idx = self.random_state.randint(0, self.n_features - 1)

        # Split point from feature bounds (uniform random)
        min_val = self.feature_bounds['min'][feature_idx]
        max_val = self.feature_bounds['max'][feature_idx]
        threshold = self.random_state.uniform(min_val, max_val)

        # Recursively build left and right subtrees
        left_child = self._build_random_tree(depth + 1)
        right_child = self._build_random_tree(depth + 1)

        return {
            'type': 'split',
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_child,
            'right': right_child
        }

    def _assign_labels_dp(
        self,
        node: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Phase 2: Assign labels to leaf nodes using PermuteAndFlip (ε-DP).

        Args:
            node: Current tree node
            X: Training features for samples reaching this node
            y: Training labels for samples reaching this node
        """
        if node['type'] == 'leaf':
            # Leaf node: assign label using PermuteAndFlip
            if len(y) > 0:
                node['label'] = self._permute_and_flip(y)
            else:
                # Empty leaf: random label
                node['label'] = self.random_state.randint(0, self.n_classes - 1)
            return

        # Split node: route samples and recurse
        feature_idx = node['feature_idx']
        threshold = node['threshold']

        # Split samples based on threshold
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively assign labels to children
        self._assign_labels_dp(node['left'], X[left_mask], y[left_mask])
        self._assign_labels_dp(node['right'], X[right_mask], y[right_mask])

    def _permute_and_flip(self, y: np.ndarray) -> int:
        """
        PermuteAndFlip mechanism for differentially private label selection.

        Assigns label with probability proportional to exp(epsilon * count / 2).

        Args:
            y: Labels of samples reaching this leaf

        Returns:
            Selected label (int)
        """
        # Count samples per class
        class_counts = np.zeros(self.n_classes)
        for label in y:
            class_counts[int(label)] += 1

        # Compute probabilities: exp(epsilon * count / 2)
        scores = np.exp(self.epsilon * class_counts / 2.0)
        probabilities = scores / scores.sum()

        # Sample label according to probabilities
        # Use numpy for weighted sampling (secrets doesn't support this)
        return np.random.choice(self.n_classes, p=probabilities)

    def predict_sample(self, x: np.ndarray) -> int:
        """
        Predict label for single sample by traversing tree.

        Args:
            x: Feature vector (n_features,)

        Returns:
            Predicted label (int)
        """
        node = self.tree_structure

        while node['type'] == 'split':
            if x[node['feature_idx']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']

        return node['label']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.tree_structure is None:
            raise ValueError("Tree not fitted. Call fit() first.")

        predictions = np.array([self.predict_sample(x) for x in X])
        return predictions

    def get_state(self) -> Dict[str, Any]:
        """
        Get tree state for serialization.

        Returns:
            Dictionary with tree parameters and structure
        """
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'feature_bounds': self.feature_bounds,
            'epsilon': self.epsilon,
            'tree_structure': self.tree_structure,
            'n_features': self.n_features,
            'n_classes': self.n_classes
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore tree from serialized state.

        Args:
            state: Dictionary with tree parameters and structure
        """
        self.max_depth = state['max_depth']
        self.min_samples_split = state['min_samples_split']
        self.feature_bounds = state['feature_bounds']
        self.epsilon = state['epsilon']
        self.tree_structure = state['tree_structure']
        self.n_features = state['n_features']
        self.n_classes = state['n_classes']


def permute_and_flip(
    counts: np.ndarray,
    epsilon: float,
    random_state: Optional[Any] = None
) -> int:
    """
    Standalone PermuteAndFlip mechanism for DP label selection.

    Selects label with probability proportional to exp(epsilon * count / 2).

    Args:
        counts: Count of samples per class
        epsilon: Privacy budget
        random_state: Random state (uses SystemRandom if None)

    Returns:
        Selected label index
    """
    if random_state is None:
        random_state = check_random_state_secure()

    # Compute probabilities
    scores = np.exp(epsilon * counts / 2.0)
    probabilities = scores / scores.sum()

    # Sample according to probabilities
    return np.random.choice(len(counts), p=probabilities)
