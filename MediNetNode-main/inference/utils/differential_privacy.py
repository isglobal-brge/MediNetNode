"""
Differential Privacy for model predictions.

Adds calibrated noise to predictions to prevent reverse engineering
while maintaining acceptable accuracy.

Epsilon values are configurable per model via DeployedModel.dp_epsilon.
Lower epsilon = stronger privacy but more noise.
"""
import numpy as np
from typing import Dict, Any, Optional


class DifferentialPrivacy:
    """
    Differential Privacy noise addition for model outputs.

    Uses Laplace mechanism for adding calibrated noise to predictions.
    """

    # Default epsilon if not specified (medium privacy)
    DEFAULT_EPSILON = 1.0

    # Recommended epsilon values by domain (configurable via admin)
    RECOMMENDED_EPSILON = {
        'oncology': 1.0,      # High privacy
        'cardiology': 2.0,    # Medium-high privacy
        'neurology': 1.5,     # High privacy
        'diabetes': 3.0,      # Medium privacy
        'general': 3.0,       # Medium privacy
        'research': 8.0,      # Low privacy (more utility)
        'public': 5.0,        # Medium privacy
    }

    def __init__(self, epsilon: float = None):
        """
        Initialize Differential Privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more privacy, more noise)
                    If None, uses DEFAULT_EPSILON
        """
        self.epsilon = epsilon if epsilon is not None else self.DEFAULT_EPSILON
        self.scale = 1.0 / self.epsilon  # Laplace scale parameter

    def add_noise_to_probabilities(
        self,
        probabilities: np.ndarray,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add Laplace noise to probability predictions and re-normalize.

        Args:
            probabilities: Numpy array of probabilities (must sum to ~1.0)
            sensitivity: Sensitivity of the query (default 1.0)

        Returns:
            Noisy probabilities (re-normalized to sum to 1.0)
        """
        # Generate Laplace noise
        noise = np.random.laplace(
            loc=0.0,
            scale=sensitivity * self.scale,
            size=probabilities.shape
        )

        # Add noise to probabilities
        noisy_probs = probabilities + noise

        # Clip to [0, 1] range
        noisy_probs = np.clip(noisy_probs, 0.0, 1.0)

        # Re-normalize to sum to 1.0
        prob_sum = np.sum(noisy_probs)
        if prob_sum > 0:
            noisy_probs = noisy_probs / prob_sum
        else:
            # If all probabilities became 0, return uniform distribution
            noisy_probs = np.ones_like(probabilities) / len(probabilities)

        return noisy_probs

    def add_noise_to_scalar(
        self,
        value: float,
        sensitivity: float = 1.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """
        Add Laplace noise to a scalar value (for regression).

        Args:
            value: Scalar value to add noise to
            sensitivity: Sensitivity of the query (default 1.0)
            min_val: Minimum allowed value (optional)
            max_val: Maximum allowed value (optional)

        Returns:
            Noisy value
        """
        noise = np.random.laplace(loc=0.0, scale=sensitivity * self.scale)
        noisy_value = value + noise

        # Clip to range if specified
        if min_val is not None:
            noisy_value = max(noisy_value, min_val)
        if max_val is not None:
            noisy_value = min(noisy_value, max_val)

        return noisy_value

    def add_noise_to_logits(
        self,
        logits: np.ndarray,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add noise to logits (before softmax).

        Args:
            logits: Raw model outputs (pre-softmax)
            sensitivity: Sensitivity of the query

        Returns:
            Noisy logits
        """
        noise = np.random.laplace(
            loc=0.0,
            scale=sensitivity * self.scale,
            size=logits.shape
        )

        return logits + noise

    def calculate_privacy_budget_spent(self, num_queries: int) -> float:
        """
        Calculate total privacy budget spent for multiple queries.

        Args:
            num_queries: Number of queries made

        Returns:
            Total epsilon spent (privacy budget)
        """
        # Under basic composition, privacy degrades linearly
        return self.epsilon * num_queries

    def get_recommended_epsilon(self, domain: str) -> float:
        """
        Get recommended epsilon for a domain.

        Args:
            domain: Model domain (e.g., 'oncology', 'cardiology')

        Returns:
            Recommended epsilon value
        """
        domain_lower = domain.lower()
        return self.RECOMMENDED_EPSILON.get(domain_lower, self.DEFAULT_EPSILON)

    @staticmethod
    def estimate_accuracy_impact(epsilon: float) -> Dict[str, Any]:
        """
        Estimate accuracy impact for a given epsilon.

        Args:
            epsilon: Privacy budget

        Returns:
            Dict with estimated accuracy impact metrics
        """
        # These are rough estimates - actual impact depends on model and data
        if epsilon >= 8.0:
            impact = 'minimal'
            accuracy_drop = '<1%'
        elif epsilon >= 5.0:
            impact = 'low'
            accuracy_drop = '1-2%'
        elif epsilon >= 3.0:
            impact = 'moderate'
            accuracy_drop = '2-5%'
        elif epsilon >= 1.5:
            impact = 'significant'
            accuracy_drop = '5-10%'
        else:
            impact = 'high'
            accuracy_drop = '>10%'

        return {
            'impact_level': impact,
            'estimated_accuracy_drop': accuracy_drop,
            'privacy_level': 'high' if epsilon <= 2.0 else 'medium' if epsilon <= 5.0 else 'low'
        }


class OutputSanitizer:
    """
    Sanitizes model outputs to prevent information leakage.

    Features:
    - Probability rounding (reduce precision)
    - Low confidence hiding (mask uncertain predictions)
    - Confidence discretization (bucket probabilities)
    """

    def __init__(
        self,
        precision: int = 2,
        min_confidence: float = 0.05,
        discretize: bool = False
    ):
        """
        Initialize output sanitizer.

        Args:
            precision: Decimal places for probability rounding (default 2)
            min_confidence: Minimum confidence to show (default 0.05 = 5%)
            discretize: Whether to discretize probabilities into buckets
        """
        self.precision = precision
        self.min_confidence = min_confidence
        self.discretize = discretize

    def sanitize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Sanitize probability predictions.

        Args:
            probabilities: Raw probability predictions

        Returns:
            Sanitized probabilities
        """
        # Round to reduce precision
        sanitized = np.round(probabilities, decimals=self.precision)

        # Hide low confidence predictions
        sanitized[sanitized < self.min_confidence] = 0.0

        # Discretize if enabled
        if self.discretize:
            sanitized = self._discretize_probabilities(sanitized)

        # Re-normalize
        prob_sum = np.sum(sanitized)
        if prob_sum > 0:
            sanitized = sanitized / prob_sum
        else:
            # If all became 0, return uniform
            sanitized = np.ones_like(probabilities) / len(probabilities)

        return sanitized

    def _discretize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Discretize probabilities into buckets.

        Buckets: 0.0, 0.1, 0.2, ..., 1.0

        Args:
            probabilities: Continuous probabilities

        Returns:
            Discretized probabilities
        """
        # Round to nearest 0.1
        discretized = np.round(probabilities * 10) / 10
        return discretized

    def sanitize_scalar(
        self,
        value: float,
        round_to: Optional[int] = None
    ) -> float:
        """
        Sanitize scalar output (regression).

        Args:
            value: Scalar value
            round_to: Decimal places to round to (default uses self.precision)

        Returns:
            Sanitized value
        """
        decimal_places = round_to if round_to is not None else self.precision
        return round(value, decimal_places)
