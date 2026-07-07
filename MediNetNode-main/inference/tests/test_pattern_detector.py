"""
Tests for QueryPatternDetector.
"""
import pytest
import numpy as np
from django.core.cache import cache
from django.utils import timezone
from datetime import timedelta
from inference.utils.pattern_detector import QueryPatternDetector


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before and after each test."""
    cache.clear()
    yield
    cache.clear()


@pytest.mark.django_db
class TestQueryPatternDetector:
    """Test QueryPatternDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = QueryPatternDetector()
        assert detector.GRID_SEARCH_SIMILARITY_THRESHOLD == 0.95
        assert detector.GRID_SEARCH_COUNT_THRESHOLD == 10
        assert detector.BOUNDARY_EXTREME_THRESHOLD == 0.1

    def test_normal_query_no_patterns(self):
        """Test normal query returns no suspicious patterns."""
        detector = QueryPatternDetector()
        features = np.array([[0.5, 0.6, 0.4, 0.7]])
        input_hash = "abc123"

        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash=input_hash
        )

        assert result['suspicious'] is False
        assert len(result['patterns']) == 0
        assert result['risk_score'] == 0.0
        assert result['action'] == 'allow'

    def test_grid_search_detection(self):
        """Test detection of grid search attack."""
        detector = QueryPatternDetector()

        # Simulate grid search: queries that differ only slightly
        base_features = np.array([[0.5, 0.5, 0.5, 0.5]])

        # Make 10 very similar queries
        for i in range(detector.GRID_SEARCH_COUNT_THRESHOLD):
            # Vary only one feature slightly
            features = base_features.copy()
            features[0, 0] = 0.5 + (i * 0.01)  # Slight variation

            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # Next similar query should trigger grid search detection
        features = base_features.copy()
        features[0, 0] = 0.6

        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash="hash_final"
        )

        assert 'grid_search' in result['patterns']
        assert result['suspicious'] is True
        assert result['risk_score'] >= detector.WEIGHT_GRID_SEARCH

    def test_boundary_probing_detection(self):
        """Test detection of boundary probing."""
        detector = QueryPatternDetector()

        # Create features with extreme values (all close to 0 or 1)
        features = np.array([[0.01, 0.99, 0.02, 0.98, 0.01]])

        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash="boundary_test"
        )

        assert 'boundary_probing' in result['patterns']
        assert result['suspicious'] is True
        assert result['risk_score'] >= detector.WEIGHT_BOUNDARY

    def test_high_volume_detection(self):
        """Test detection of high volume queries."""
        detector = QueryPatternDetector()

        # Make queries approaching hourly limit
        num_queries = int(detector.HIGH_VOLUME_HOURLY * 0.85)

        for i in range(num_queries):
            features = np.random.rand(1, 5)
            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # Next query should detect high volume
        features = np.random.rand(1, 5)
        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash="final"
        )

        assert 'high_volume' in result['patterns']
        assert result['suspicious'] is True

    def test_membership_inference_detection(self):
        """Test detection of membership inference attacks."""
        detector = QueryPatternDetector()

        # Query with same input hash multiple times
        features = np.array([[0.5, 0.5, 0.5]])
        same_hash = "repeated_input_hash"

        # Make queries below threshold
        for i in range(detector.MEMBERSHIP_DUPLICATE_THRESHOLD - 1):
            result = detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=same_hash
            )
            assert 'membership_inference' not in result['patterns']

        # Next identical query should trigger detection
        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash=same_hash
        )

        assert 'membership_inference' in result['patterns']
        assert result['suspicious'] is True

    def test_risk_score_calculation(self):
        """Test risk score is calculated correctly."""
        detector = QueryPatternDetector()

        # Create query that triggers multiple patterns
        # 1. Boundary probing (extreme values)
        features = np.array([[0.01, 0.99, 0.02, 0.98]])

        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=features,
            input_hash="test"
        )

        # Should have at least boundary pattern
        assert result['risk_score'] > 0.0
        assert result['risk_score'] <= 1.0

    def test_action_determination(self):
        """Test action is determined correctly based on risk."""
        detector = QueryPatternDetector()

        # Test each action threshold
        assert detector._determine_action(0.0) == 'allow'
        assert detector._determine_action(0.3) == 'allow'
        assert detector._determine_action(0.5) == 'throttle'
        assert detector._determine_action(0.7) == 'alert'
        assert detector._determine_action(0.9) == 'block'
        assert detector._determine_action(1.0) == 'block'

    def test_different_users_independent(self):
        """Test different users have independent pattern tracking."""
        detector = QueryPatternDetector()

        # User 1 makes suspicious queries
        features = np.array([[0.01, 0.99, 0.01, 0.99]])

        for i in range(5):
            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # User 2 makes normal query
        normal_features = np.array([[0.5, 0.5, 0.5, 0.5]])
        result = detector.analyze_query(
            user_id=2,
            model_id=1,
            features=normal_features,
            input_hash="user2_hash"
        )

        # User 2 should not be affected by User 1's patterns
        assert result['risk_score'] == 0.0 or result['risk_score'] < 0.4

    def test_different_models_independent(self):
        """Test different models have independent pattern tracking."""
        detector = QueryPatternDetector()

        # Make suspicious queries to model 1
        features = np.array([[0.01, 0.99]])
        same_hash = "repeated"

        for i in range(detector.MEMBERSHIP_DUPLICATE_THRESHOLD):
            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=same_hash
            )

        # Query to model 2 with same hash should not trigger
        result = detector.analyze_query(
            user_id=1,
            model_id=2,
            features=features,
            input_hash=same_hash
        )

        assert 'membership_inference' not in result['patterns']

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        detector = QueryPatternDetector()

        # Identical vectors
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert detector._cosine_similarity(a, b) == pytest.approx(1.0)

        # Orthogonal vectors
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert detector._cosine_similarity(a, b) == pytest.approx(0.0)

        # Similar vectors
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        similarity = detector._cosine_similarity(a, b)
        assert similarity > 0.99

    def test_query_history_cleanup(self):
        """Test old queries are removed from history."""
        detector = QueryPatternDetector()

        features = np.array([[0.5, 0.5]])

        for i in range(10):
            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # Manually modify history to include old entries
        history = detector._get_query_history(1, 1)
        old_entry = {
            'timestamp': timezone.now() - timedelta(hours=25),
            'features': features.tolist(),
            'input_hash': 'old_hash'
        }
        history.insert(0, old_entry)

        key = detector._make_key(detector.PREFIX_QUERY_HISTORY, 1, 1)
        cache.set(key, history, detector.TTL_HISTORY)

        # Get history again - old entry should be filtered
        fresh_history = detector._get_query_history(1, 1)
        assert len(fresh_history) == 10  # Old entry removed
        assert all(entry['timestamp'] > timezone.now() - timedelta(hours=24) for entry in fresh_history)

    def test_history_size_limit(self):
        """Test history is limited to prevent memory issues."""
        detector = QueryPatternDetector()

        features = np.array([[0.5]])

        # Add more than 1000 queries
        for i in range(1100):
            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # History should be limited to 1000
        history = detector._get_query_history(1, 1)
        assert len(history) <= 1000

    def test_combined_patterns(self):
        """Test multiple patterns detected simultaneously."""
        detector = QueryPatternDetector()

        # Create scenario with multiple patterns:
        # 1. Boundary probing (extreme values)
        # 2. Grid search (similar queries)

        base_features = np.array([[0.01, 0.99, 0.01, 0.99]])

        # Make 10 similar boundary-probing queries
        for i in range(detector.GRID_SEARCH_COUNT_THRESHOLD):
            features = base_features.copy()
            features[0, 0] = 0.01 + (i * 0.001)  # Very slight variation

            detector.analyze_query(
                user_id=1,
                model_id=1,
                features=features,
                input_hash=f"hash_{i}"
            )

        # Final query
        result = detector.analyze_query(
            user_id=1,
            model_id=1,
            features=base_features,
            input_hash="final"
        )

        # Should detect both patterns
        assert 'boundary_probing' in result['patterns']
        assert 'grid_search' in result['patterns']
        assert result['risk_score'] >= (detector.WEIGHT_BOUNDARY + detector.WEIGHT_GRID_SEARCH)
        assert result['action'] in ['alert', 'block']
