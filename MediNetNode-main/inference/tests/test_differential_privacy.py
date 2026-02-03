"""
Tests for Differential Privacy and Output Sanitization.
"""
import pytest
import numpy as np
from inference.utils.differential_privacy import DifferentialPrivacy, OutputSanitizer


class TestDifferentialPrivacy:
    """Test DifferentialPrivacy class."""

    def test_initialization_default(self):
        """Test default initialization."""
        dp = DifferentialPrivacy()
        assert dp.epsilon == DifferentialPrivacy.DEFAULT_EPSILON
        assert dp.scale == 1.0 / DifferentialPrivacy.DEFAULT_EPSILON

    def test_initialization_custom_epsilon(self):
        """Test initialization with custom epsilon."""
        dp = DifferentialPrivacy(epsilon=2.0)
        assert dp.epsilon == 2.0
        assert dp.scale == 0.5

    def test_add_noise_to_probabilities(self):
        """Test adding noise to probabilities."""
        dp = DifferentialPrivacy(epsilon=1.0)

        # Original probabilities
        probs = np.array([0.7, 0.2, 0.1])

        # Add noise
        noisy_probs = dp.add_noise_to_probabilities(probs)

        # Check properties
        assert noisy_probs.shape == probs.shape
        assert np.all(noisy_probs >= 0.0)
        assert np.all(noisy_probs <= 1.0)
        assert np.isclose(np.sum(noisy_probs), 1.0, atol=1e-6)

        # Noise should make it different from original
        assert not np.allclose(noisy_probs, probs)

    def test_add_noise_to_probabilities_deterministic_seed(self):
        """Test noise addition is different each time."""
        dp = DifferentialPrivacy(epsilon=1.0)
        probs = np.array([0.5, 0.5])

        # Two calls should produce different results
        noisy1 = dp.add_noise_to_probabilities(probs)
        noisy2 = dp.add_noise_to_probabilities(probs)

        assert not np.allclose(noisy1, noisy2)

    def test_add_noise_probabilities_renormalize(self):
        """Test probabilities are re-normalized after noise."""
        dp = DifferentialPrivacy(epsilon=0.5)  # High noise
        probs = np.array([0.8, 0.1, 0.1])

        noisy_probs = dp.add_noise_to_probabilities(probs)

        # Must sum to 1.0
        assert np.isclose(np.sum(noisy_probs), 1.0, atol=1e-6)

    def test_add_noise_to_scalar(self):
        """Test adding noise to scalar value."""
        dp = DifferentialPrivacy(epsilon=1.0)

        value = 50.0
        noisy_value = dp.add_noise_to_scalar(value)

        # Should be different
        assert noisy_value != value

        # Should be reasonable (within several standard deviations)
        assert abs(noisy_value - value) < 10.0  # 10 * scale

    def test_add_noise_to_scalar_with_bounds(self):
        """Test scalar noise with min/max bounds."""
        dp = DifferentialPrivacy(epsilon=0.1)  # Very high noise

        value = 50.0
        noisy_value = dp.add_noise_to_scalar(value, min_val=0.0, max_val=100.0)

        # Must be within bounds
        assert noisy_value >= 0.0
        assert noisy_value <= 100.0

    def test_add_noise_to_logits(self):
        """Test adding noise to logits."""
        dp = DifferentialPrivacy(epsilon=1.0)

        logits = np.array([2.0, 1.0, 0.5])
        noisy_logits = dp.add_noise_to_logits(logits)

        assert noisy_logits.shape == logits.shape
        assert not np.allclose(noisy_logits, logits)

    def test_calculate_privacy_budget_spent(self):
        """Test privacy budget calculation."""
        dp = DifferentialPrivacy(epsilon=1.0)

        # Single query
        assert dp.calculate_privacy_budget_spent(1) == 1.0

        # Multiple queries
        assert dp.calculate_privacy_budget_spent(10) == 10.0

        # High epsilon
        dp2 = DifferentialPrivacy(epsilon=5.0)
        assert dp2.calculate_privacy_budget_spent(10) == 50.0

    def test_get_recommended_epsilon(self):
        """Test getting recommended epsilon by domain."""
        dp = DifferentialPrivacy()

        # Test specific domains
        assert dp.get_recommended_epsilon('oncology') == 1.0
        assert dp.get_recommended_epsilon('Oncology') == 1.0  # Case insensitive
        assert dp.get_recommended_epsilon('cardiology') == 2.0
        assert dp.get_recommended_epsilon('neurology') == 1.5
        assert dp.get_recommended_epsilon('diabetes') == 3.0
        assert dp.get_recommended_epsilon('research') == 8.0
        assert dp.get_recommended_epsilon('public') == 5.0

        # Unknown domain should return default
        assert dp.get_recommended_epsilon('unknown') == DifferentialPrivacy.DEFAULT_EPSILON

    def test_estimate_accuracy_impact(self):
        """Test accuracy impact estimation."""
        # High epsilon (low privacy, low impact)
        impact = DifferentialPrivacy.estimate_accuracy_impact(8.0)
        assert impact['impact_level'] == 'minimal'
        assert impact['privacy_level'] == 'low'

        # Medium epsilon
        impact = DifferentialPrivacy.estimate_accuracy_impact(3.0)
        assert impact['impact_level'] == 'moderate'
        assert impact['privacy_level'] == 'medium'

        # Medium-low epsilon
        impact = DifferentialPrivacy.estimate_accuracy_impact(1.5)
        assert impact['impact_level'] == 'significant'
        assert impact['privacy_level'] == 'high'

        # Low epsilon (high privacy, high impact)
        impact = DifferentialPrivacy.estimate_accuracy_impact(1.0)
        assert impact['impact_level'] == 'high'
        assert impact['privacy_level'] == 'high'

    def test_epsilon_affects_noise_scale(self):
        """Test that lower epsilon produces more noise."""
        probs = np.array([0.6, 0.3, 0.1])

        # High privacy (low epsilon) - more noise
        dp_high_privacy = DifferentialPrivacy(epsilon=0.5)
        noisy_high = dp_high_privacy.add_noise_to_probabilities(probs)

        # Low privacy (high epsilon) - less noise
        dp_low_privacy = DifferentialPrivacy(epsilon=10.0)
        noisy_low = dp_low_privacy.add_noise_to_probabilities(probs)

        # High privacy should deviate more from original
        deviation_high = np.sum(np.abs(noisy_high - probs))
        deviation_low = np.sum(np.abs(noisy_low - probs))

        # This should hold statistically (may occasionally fail due to randomness)
        # Run multiple times to be more confident
        high_deviations = []
        low_deviations = []

        for _ in range(10):
            noisy_high = dp_high_privacy.add_noise_to_probabilities(probs)
            noisy_low = dp_low_privacy.add_noise_to_probabilities(probs)

            high_deviations.append(np.sum(np.abs(noisy_high - probs)))
            low_deviations.append(np.sum(np.abs(noisy_low - probs)))

        # Average deviation should be higher for high privacy
        assert np.mean(high_deviations) > np.mean(low_deviations)


class TestOutputSanitizer:
    """Test OutputSanitizer class."""

    def test_initialization(self):
        """Test sanitizer initialization."""
        sanitizer = OutputSanitizer(precision=3, min_confidence=0.1)
        assert sanitizer.precision == 3
        assert sanitizer.min_confidence == 0.1
        assert sanitizer.discretize is False

    def test_sanitize_probabilities_rounding(self):
        """Test probability rounding."""
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.0)

        probs = np.array([0.12345, 0.67891, 0.19764])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # Check rounding
        assert sanitized[0] == pytest.approx(0.12, abs=0.01)
        assert sanitized[1] == pytest.approx(0.68, abs=0.01)
        assert sanitized[2] == pytest.approx(0.20, abs=0.01)

        # Should still sum to 1.0
        assert np.isclose(np.sum(sanitized), 1.0, atol=1e-6)

    def test_sanitize_probabilities_min_confidence(self):
        """Test hiding low confidence predictions."""
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.1)

        probs = np.array([0.8, 0.15, 0.05])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # Low confidence (0.05) should be hidden
        assert sanitized[2] == 0.0

        # Should still sum to 1.0 after re-normalization
        assert np.isclose(np.sum(sanitized), 1.0, atol=1e-6)

    def test_sanitize_probabilities_discretization(self):
        """Test probability discretization."""
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.0, discretize=True)

        probs = np.array([0.73, 0.18, 0.09])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # Should be discretized to 0.1 buckets
        assert sanitized[0] == pytest.approx(0.7, abs=0.01)
        assert sanitized[1] == pytest.approx(0.2, abs=0.01)
        assert sanitized[2] == pytest.approx(0.1, abs=0.01)

        # Should sum to 1.0
        assert np.isclose(np.sum(sanitized), 1.0, atol=1e-6)

    def test_sanitize_scalar(self):
        """Test scalar sanitization."""
        sanitizer = OutputSanitizer(precision=2)

        value = 123.456789
        sanitized = sanitizer.sanitize_scalar(value)

        assert sanitized == 123.46

    def test_sanitize_scalar_custom_precision(self):
        """Test scalar with custom precision."""
        sanitizer = OutputSanitizer(precision=2)

        value = 123.456789
        sanitized = sanitizer.sanitize_scalar(value, round_to=1)

        assert sanitized == 123.5

    def test_all_low_confidence(self):
        """Test behavior when all probabilities are below threshold."""
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.5)

        probs = np.array([0.4, 0.35, 0.25])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # All below threshold, should return uniform distribution
        expected = np.array([1.0/3, 1.0/3, 1.0/3])
        assert np.allclose(sanitized, expected, atol=0.01)

    def test_combined_rounding_and_hiding(self):
        """Test combination of rounding and min confidence."""
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.1)

        probs = np.array([0.755, 0.189, 0.056])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # 0.056 rounded to 0.06 should be hidden (below 0.1)
        assert sanitized[2] == 0.0

        # Others should be rounded and re-normalized
        assert sanitized[0] == pytest.approx(0.80, abs=0.02)
        assert sanitized[1] == pytest.approx(0.20, abs=0.02)

        # Should sum to 1.0
        assert np.isclose(np.sum(sanitized), 1.0, atol=1e-6)

    def test_discretize_buckets(self):
        """Test discretization creates correct buckets."""
        sanitizer = OutputSanitizer(discretize=True, min_confidence=0.0, precision=2)

        # Test with values that sum to exactly 1.0 after discretization
        probs = np.array([0.65, 0.25, 0.10])
        sanitized = sanitizer.sanitize_probabilities(probs)

        # After discretization to 0.1 buckets and re-normalization
        # Values should be approximately 0.7, 0.2, 0.1 (which sum to 1.0)
        assert np.isclose(sanitized[0], 0.7, atol=0.05)
        assert np.isclose(sanitized[1], 0.2, atol=0.05)
        assert np.isclose(sanitized[2], 0.1, atol=0.05)

        # Should sum to 1.0
        assert np.isclose(np.sum(sanitized), 1.0, atol=1e-6)


class TestIntegration:
    """Integration tests combining DP and sanitization."""

    def test_dp_then_sanitize(self):
        """Test applying DP noise then sanitizing."""
        # Original probabilities
        probs = np.array([0.65, 0.25, 0.10])

        # Apply DP noise
        dp = DifferentialPrivacy(epsilon=2.0)
        noisy_probs = dp.add_noise_to_probabilities(probs)

        # Sanitize
        sanitizer = OutputSanitizer(precision=2, min_confidence=0.05)
        final_probs = sanitizer.sanitize_probabilities(noisy_probs)

        # Check final properties
        assert final_probs.shape == probs.shape
        assert np.all(final_probs >= 0.0)
        assert np.all(final_probs <= 1.0)
        assert np.isclose(np.sum(final_probs), 1.0, atol=1e-6)

        # Should be different from original (due to DP noise)
        assert not np.allclose(final_probs, probs)

    def test_recommended_epsilon_workflow(self):
        """Test workflow with recommended epsilon for domain."""
        dp = DifferentialPrivacy()

        # Get recommended epsilon for oncology
        epsilon = dp.get_recommended_epsilon('oncology')
        assert epsilon == 1.0

        # Create new DP instance with recommended epsilon
        dp_oncology = DifferentialPrivacy(epsilon=epsilon)

        # Add noise to probabilities
        probs = np.array([0.8, 0.15, 0.05])
        noisy_probs = dp_oncology.add_noise_to_probabilities(probs)

        # Check impact estimate
        impact = DifferentialPrivacy.estimate_accuracy_impact(epsilon)
        assert impact['privacy_level'] == 'high'

        # Verify noisy probs are valid
        assert np.isclose(np.sum(noisy_probs), 1.0, atol=1e-6)
