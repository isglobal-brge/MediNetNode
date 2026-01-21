"""
Privacy Guards and Safeguards for Federated Learning.

Implements technical safeguards for privacy-preserving machine learning:
- Privacy budget management and tracking
- Feature bounds validation to prevent information leakage
- Epsilon parameter validation based on industry standards
- Data integrity verification
"""

import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import warnings


# ============================================================================
# PRIVACY BUDGET MANAGEMENT
# ============================================================================

class PrivacyBudgetExhausted(Exception):
    """Raised when privacy budget is exhausted."""
    pass


class PrivacyBudgetTracker:
    """
    Tracks and enforces privacy budget limits.

    Prevents excessive privacy budget consumption that could lead to
    re-identification attacks in federated learning scenarios.
    """

    def __init__(self, max_epsilon: float, max_delta: float = 1e-6):
        """
        Initialize privacy budget tracker.

        Args:
            max_epsilon: Maximum total privacy budget
            max_delta: Maximum failure probability
        """
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.operations = []

    def consume(self, epsilon: float, delta: float = 0, operation: str = ""):
        """
        Consume privacy budget for an operation.

        Args:
            epsilon: Privacy budget to consume
            delta: Failure probability to consume
            operation: Description of operation

        Raises:
            PrivacyBudgetExhausted: If budget would be exceeded
        """
        if self.spent_epsilon + epsilon > self.max_epsilon:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted. "
                f"Requested: {epsilon}, Available: {self.remaining_epsilon()}, "
                f"Total spent: {self.spent_epsilon}/{self.max_epsilon}"
            )

        if self.spent_delta + delta > self.max_delta:
            raise PrivacyBudgetExhausted(
                f"Delta budget exhausted. "
                f"Requested: {delta}, Available: {self.remaining_delta()}, "
                f"Total spent: {self.spent_delta}/{self.max_delta}"
            )

        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.operations.append({
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'delta': delta,
            'operation': operation
        })

    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return self.max_epsilon - self.spent_epsilon

    def remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return self.max_delta - self.spent_delta

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """
        Get log of all privacy budget operations.

        Returns:
            List of operations with timestamps and budgets
        """
        return self.operations.copy()


# ============================================================================
# FEATURE BOUNDS VALIDATION
# ============================================================================

class FeatureBoundsValidator:
    """
    Validates feature bounds to prevent information leakage.

    Prevents re-identification attacks through feature bounds correlation.
    Feature bounds can leak information about local dataset distribution
    (e.g., age ranges revealing hospital demographics).
    """

    @staticmethod
    def validate_global_bounds(
        local_bounds: Dict[str, List[float]],
        global_bounds: Dict[str, List[float]],
        tolerance: float = 1e-6
    ) -> None:
        """
        Validate that local bounds match server-provided global bounds.

        This prevents clients from sending data-dependent bounds that could
        leak information about their local dataset distribution.

        Args:
            local_bounds: Bounds provided by client
            global_bounds: Canonical bounds from server
            tolerance: Numerical tolerance for comparison

        Raises:
            ValueError: If bounds don't match global bounds
        """
        if local_bounds.keys() != global_bounds.keys():
            raise ValueError(
                f"Bounds keys mismatch. Local: {local_bounds.keys()}, "
                f"Global: {global_bounds.keys()}"
            )

        for key in ['min', 'max']:
            local_vals = np.array(local_bounds[key])
            global_vals = np.array(global_bounds[key])

            if not np.allclose(local_vals, global_vals, atol=tolerance):
                raise ValueError(
                    f"Feature bounds must use server-provided global bounds. "
                    f"Local bounds differ from global bounds for '{key}'. "
                    f"Using data-dependent bounds can leak information about "
                    f"local dataset distribution. "
                    f"Local: {local_vals}, Global: {global_vals}"
                )

    @staticmethod
    def compute_dp_bounds(
        X: np.ndarray,
        epsilon: float = 0.1,
        sensitivity_multiplier: float = 2.0
    ) -> Dict[str, List[float]]:
        """
        Compute differentially private feature bounds using Laplace mechanism.

        Alternative to global bounds when server can't provide them.
        Adds calibrated noise to bounds to prevent exact leakage.

        Args:
            X: Training data
            epsilon: Privacy budget for bounds computation
            sensitivity_multiplier: Multiplier for sensitivity (default: 2.0)

        Returns:
            DP-protected bounds dictionary
        """
        # Sensitivity: changing 1 record can change min/max by at most range
        data_range = X.max(axis=0) - X.min(axis=0)
        sensitivity = sensitivity_multiplier * data_range

        # Laplace mechanism: add noise ~ Lap(sensitivity / epsilon)
        scale = sensitivity / epsilon
        noise_min = np.random.laplace(0, scale, size=X.shape[1])
        noise_max = np.random.laplace(0, scale, size=X.shape[1])

        dp_min = X.min(axis=0) + noise_min
        dp_max = X.max(axis=0) + noise_max

        # Ensure min < max after noise
        dp_min = np.minimum(dp_min, dp_max - 1e-6)

        return {
            'min': dp_min.tolist(),
            'max': dp_max.tolist(),
            'epsilon_used': epsilon,
            'method': 'laplace_mechanism'
        }


# ============================================================================
# EPSILON VALIDATION
# ============================================================================

class EpsilonValidator:
    """
    Validates epsilon values according to industry standards.

    References:
    - Google: ε ∈ [0.1, 10] for production systems
    - Apple: ε ∈ [1, 8] for telemetry
    - NIST: ε < 0.01 is very high risk
    """

    # Industry-standard limits
    MIN_EPSILON = 0.1   # Lower bound for practical utility
    MAX_EPSILON = 10.0  # Upper bound for reasonable privacy
    RECOMMENDED_EPSILON = 1.0  # Safe default
    HIGH_RISK_THRESHOLD = 5.0  # Requires justification

    @classmethod
    def validate(cls, epsilon: float, allow_high_risk: bool = False) -> None:
        """
        Validate epsilon value against industry standards.

        Args:
            epsilon: Privacy budget to validate
            allow_high_risk: Whether to allow epsilon > HIGH_RISK_THRESHOLD

        Raises:
            ValueError: If epsilon is outside acceptable range
        """
        if epsilon <= 0:
            raise ValueError(
                f"Epsilon must be positive. Got: {epsilon}"
            )

        if epsilon < cls.MIN_EPSILON:
            warnings.warn(
                f"Epsilon={epsilon} is very small (< {cls.MIN_EPSILON}). "
                f"This will result in extremely poor model utility. "
                f"Consider increasing epsilon or using more data.",
                category=UserWarning
            )

        if epsilon > cls.MAX_EPSILON:
            raise ValueError(
                f"Epsilon={epsilon} exceeds maximum safe value ({cls.MAX_EPSILON}). "
                f"This violates industry best practices. "
                f"Privacy guarantees are too weak for production use."
            )

        if epsilon > cls.HIGH_RISK_THRESHOLD and not allow_high_risk:
            warnings.warn(
                f"Epsilon={epsilon} exceeds recommended threshold ({cls.HIGH_RISK_THRESHOLD}). "
                f"Privacy guarantees are weak. Use only with explicit approval. "
                f"Consider reducing epsilon to {cls.RECOMMENDED_EPSILON}.",
                category=UserWarning
            )


# ============================================================================
# DATA INTEGRITY
# ============================================================================

def compute_data_hash(X: np.ndarray) -> str:
    """
    Compute SHA256 hash of training data for integrity verification.

    Useful for detecting unauthorized modifications to training data
    and ensuring data consistency across federated learning rounds.

    Args:
        X: Training data array

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(X.tobytes()).hexdigest()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PrivacyBudgetTracker',
    'PrivacyBudgetExhausted',
    'FeatureBoundsValidator',
    'EpsilonValidator',
    'compute_data_hash'
]
