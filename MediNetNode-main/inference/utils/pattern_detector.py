"""
Pattern detection for anti-reverse engineering.

Detects suspicious query patterns that may indicate attempts to:
- Extract model through systematic querying (grid search)
- Probe model boundaries (boundary probing)
- Perform membership inference attacks
- High volume abuse
"""
from typing import Dict, List, Optional, Any
from django.core.cache import cache
from django.utils import timezone
import numpy as np
from datetime import timedelta
import hashlib


class QueryPatternDetector:
    """
    Detects suspicious query patterns for anti-reverse engineering.

    Detection layers:
    1. Grid search attack - systematic feature variation
    2. Boundary probing - extreme values testing
    3. High volume detection - excessive queries
    4. Membership inference - similar query repetition
    """

    # Thresholds
    GRID_SEARCH_SIMILARITY_THRESHOLD = 0.95  # 95% similarity between consecutive queries
    GRID_SEARCH_COUNT_THRESHOLD = 10  # 10 similar queries in sequence

    BOUNDARY_EXTREME_THRESHOLD = 0.1  # Within 10% of min/max
    BOUNDARY_COUNT_THRESHOLD = 5  # 5 boundary probes in 24h

    HIGH_VOLUME_HOURLY = 100  # queries per hour (matches rate limit)
    HIGH_VOLUME_DAILY = 500  # queries per day

    MEMBERSHIP_DUPLICATE_THRESHOLD = 5  # Same input hash 5+ times

    # Risk scoring weights
    WEIGHT_GRID_SEARCH = 0.4
    WEIGHT_BOUNDARY = 0.3
    WEIGHT_HIGH_VOLUME = 0.2
    WEIGHT_MEMBERSHIP = 0.1

    # Cache TTL
    TTL_HISTORY = 86400  # 24 hours
    TTL_PATTERNS = 3600  # 1 hour

    # Cache prefixes
    PREFIX_QUERY_HISTORY = 'pattern:history'
    PREFIX_INPUT_HASH = 'pattern:hash'
    PREFIX_BOUNDARY_COUNT = 'pattern:boundary'

    def __init__(self):
        """Initialize pattern detector."""
        pass

    def analyze_query(
        self,
        user_id: int,
        model_id: int,
        features: np.ndarray,
        input_hash: str
    ) -> Dict[str, Any]:
        """
        Analyze a query for suspicious patterns.

        Args:
            user_id: User making the query
            model_id: Model being queried
            features: Input features (numpy array)
            input_hash: SHA256 hash of input

        Returns:
            Dict with keys:
                - suspicious (bool): Whether patterns detected
                - patterns (list): List of detected pattern names
                - risk_score (float): Overall risk score 0.0-1.0
                - action (str): Recommended action (allow/throttle/alert/block)
        """
        patterns_detected = []
        risk_scores = {}

        history = self._get_query_history(user_id, model_id)

        # Detection 1: Grid search attack
        if self._detect_grid_search(history, features):
            patterns_detected.append('grid_search')
            risk_scores['grid_search'] = self.WEIGHT_GRID_SEARCH

        # Detection 2: Boundary probing
        if self._detect_boundary_probing(features):
            patterns_detected.append('boundary_probing')
            risk_scores['boundary'] = self.WEIGHT_BOUNDARY
            self._increment_boundary_count(user_id, model_id)

        # Detection 3: High volume
        volume_risk = self._detect_high_volume(history)
        if volume_risk > 0:
            patterns_detected.append('high_volume')
            risk_scores['high_volume'] = self.WEIGHT_HIGH_VOLUME * volume_risk

        # Detection 4: Membership inference (duplicate inputs)
        if self._detect_membership_inference(user_id, model_id, input_hash):
            patterns_detected.append('membership_inference')
            risk_scores['membership'] = self.WEIGHT_MEMBERSHIP

        risk_score = sum(risk_scores.values())
        risk_score = min(risk_score, 1.0)  # Cap at 1.0

        action = self._determine_action(risk_score)

        self._add_to_history(user_id, model_id, features, input_hash)

        return {
            'suspicious': len(patterns_detected) > 0,
            'patterns': patterns_detected,
            'risk_score': risk_score,
            'action': action
        }

    def _detect_grid_search(self, history: List[Dict], features: np.ndarray) -> bool:
        """
        Detect grid search attacks - systematic feature variation.

        Grid search is characterized by:
        - Queries that differ in only 1-2 features
        - High similarity between consecutive queries
        - Multiple queries in sequence with this pattern

        Args:
            history: Recent query history
            features: Current query features

        Returns:
            bool: True if grid search detected
        """
        if len(history) < self.GRID_SEARCH_COUNT_THRESHOLD:
            return False

        # Look at last N queries
        recent = history[-self.GRID_SEARCH_COUNT_THRESHOLD:]

        # Count how many are highly similar to current query
        similar_count = 0
        for entry in recent:
            prev_features = np.array(entry['features'])

            # Skip if shapes don't match
            if prev_features.shape != features.shape:
                continue

            # Calculate similarity (cosine similarity)
            similarity = self._cosine_similarity(prev_features.flatten(), features.flatten())

            if similarity >= self.GRID_SEARCH_SIMILARITY_THRESHOLD:
                similar_count += 1

        # If most recent queries are highly similar, likely grid search
        return similar_count >= self.GRID_SEARCH_COUNT_THRESHOLD - 2

    def _detect_boundary_probing(self, features: np.ndarray) -> bool:
        """
        Detect boundary probing - testing with extreme values.

        Boundary probing uses values at the extremes (very high/low)
        to understand model behavior at decision boundaries.

        Args:
            features: Input features

        Returns:
            bool: True if boundary probing detected
        """
        features_flat = features.flatten()

        # Count how many features are at extremes
        # Assume normalized features in [0, 1] range
        extreme_count = 0
        for val in features_flat:
            # Check if very close to 0 or 1
            if val <= self.BOUNDARY_EXTREME_THRESHOLD or val >= (1.0 - self.BOUNDARY_EXTREME_THRESHOLD):
                extreme_count += 1

        # If >50% of features are extreme values, likely boundary probing
        extreme_ratio = extreme_count / len(features_flat)
        return extreme_ratio > 0.5

    def _detect_high_volume(self, history: List[Dict]) -> float:
        """
        Detect high volume queries.

        Args:
            history: Query history

        Returns:
            float: Risk multiplier 0.0-1.0
        """
        if len(history) == 0:
            return 0.0

        # Count queries in last hour
        now = timezone.now()
        one_hour_ago = now - timedelta(hours=1)

        hourly_count = sum(1 for entry in history if entry['timestamp'] >= one_hour_ago)

        # Calculate risk based on hourly volume
        if hourly_count >= self.HIGH_VOLUME_HOURLY:
            return 1.0
        elif hourly_count >= self.HIGH_VOLUME_HOURLY * 0.8:
            return 0.8
        elif hourly_count >= self.HIGH_VOLUME_HOURLY * 0.6:
            return 0.5

        return 0.0

    def _detect_membership_inference(self, user_id: int, model_id: int, input_hash: str) -> bool:
        """
        Detect membership inference attacks - repeated identical queries.

        Args:
            user_id: User ID
            model_id: Model ID
            input_hash: Hash of input data

        Returns:
            bool: True if membership inference detected
        """
        # Increment hash count
        hash_key = self._make_key(self.PREFIX_INPUT_HASH, user_id, model_id, input_hash)
        count = cache.get(hash_key, 0)
        count += 1
        cache.set(hash_key, count, self.TTL_HISTORY)

        # If same input queried multiple times, suspicious
        return count >= self.MEMBERSHIP_DUPLICATE_THRESHOLD

    def _determine_action(self, risk_score: float) -> str:
        """
        Determine action based on risk score.

        Args:
            risk_score: Overall risk score 0.0-1.0

        Returns:
            str: Action (allow/throttle/alert/block)
        """
        if risk_score >= 0.8:
            return 'block'
        elif risk_score >= 0.6:
            return 'alert'
        elif risk_score >= 0.4:
            return 'throttle'
        else:
            return 'allow'

    def _get_query_history(self, user_id: int, model_id: int) -> List[Dict]:
        """
        Get recent query history for user+model.

        Args:
            user_id: User ID
            model_id: Model ID

        Returns:
            List of query entries
        """
        key = self._make_key(self.PREFIX_QUERY_HISTORY, user_id, model_id)
        history = cache.get(key, [])

        # Filter out old entries
        now = timezone.now()
        cutoff = now - timedelta(hours=24)
        history = [entry for entry in history if entry['timestamp'] >= cutoff]

        return history

    def _add_to_history(self, user_id: int, model_id: int, features: np.ndarray, input_hash: str):
        """
        Add query to history.

        Args:
            user_id: User ID
            model_id: Model ID
            features: Input features
            input_hash: Input hash
        """
        key = self._make_key(self.PREFIX_QUERY_HISTORY, user_id, model_id)
        history = self._get_query_history(user_id, model_id)

        entry = {
            'timestamp': timezone.now(),
            'features': features.tolist(),  # Convert to list for JSON serialization
            'input_hash': input_hash
        }
        history.append(entry)

        # Keep only last 1000 entries to prevent memory issues
        if len(history) > 1000:
            history = history[-1000:]

        cache.set(key, history, self.TTL_HISTORY)

    def _increment_boundary_count(self, user_id: int, model_id: int):
        """Increment boundary probe count for user+model."""
        key = self._make_key(self.PREFIX_BOUNDARY_COUNT, user_id, model_id)
        count = cache.get(key, 0)
        cache.set(key, count + 1, self.TTL_HISTORY)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            float: Cosine similarity (0.0 to 1.0)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Ensure in [0, 1] range (handle floating point errors)
        return max(0.0, min(1.0, similarity))

    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key."""
        return f"{prefix}:{':'.join(map(str, args))}"
