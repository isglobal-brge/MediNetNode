"""
Rate limiter for inference API.

Multi-layer rate limiting system to prevent abuse and reverse engineering:
- Layer 1: Global API limits per user
- Layer 2: Per-model prediction limits
- Layer 3: Batch size limits
- Layer 4: Concurrent request limits
"""
from typing import Dict, Optional, Tuple
from django.core.cache import cache
from django.utils import timezone
import time


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int = 0):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class RateLimiter:
    """
    Multi-layer rate limiter for inference API.

    Uses Django cache backend (Redis/LocMem) for distributed rate limiting.
    """

    # Layer 1: Global API limits
    GLOBAL_HOURLY_LIMIT = 100  # requests per hour per user
    GLOBAL_DAILY_LIMIT = 500   # requests per day per user

    # Layer 2: Per-model limits (can be overridden by model settings)
    MODEL_HOURLY_DEFAULT = 60   # predictions per hour per model
    MODEL_DAILY_DEFAULT = 200   # predictions per day per model

    # Layer 3: Batch size limit (can be overridden by model settings)
    MAX_BATCH_SIZE_DEFAULT = 1000

    # Layer 4: Concurrent request limit
    MAX_CONCURRENT_REQUESTS = 5

    # Cache key prefixes
    PREFIX_GLOBAL_HOURLY = 'ratelimit:global:hourly'
    PREFIX_GLOBAL_DAILY = 'ratelimit:global:daily'
    PREFIX_MODEL_HOURLY = 'ratelimit:model:hourly'
    PREFIX_MODEL_DAILY = 'ratelimit:model:daily'
    PREFIX_CONCURRENT = 'ratelimit:concurrent'

    # TTL (Time To Live) for cache keys
    TTL_HOURLY = 3600       # 1 hour
    TTL_DAILY = 86400       # 24 hours
    TTL_CONCURRENT = 300    # 5 minutes (safety cleanup)

    def __init__(self):
        """Initialize rate limiter."""
        pass

    def check_limits(
        self,
        user_id: int,
        model_id: int,
        batch_size: int,
        model_max_batch_size: Optional[int] = None,
        model_hourly_limit: Optional[int] = None,
        model_daily_limit: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Check all rate limits for a prediction request.

        Args:
            user_id: User ID making the request
            model_id: Model ID being used
            batch_size: Number of records in batch
            model_max_batch_size: Model-specific max batch size (overrides default)
            model_hourly_limit: Model-specific hourly limit (overrides default)
            model_daily_limit: Model-specific daily limit (overrides default)

        Returns:
            Dict with keys:
                - allowed (bool): Whether request is allowed
                - reason (str): Reason if not allowed
                - retry_after (int): Seconds to wait before retry
                - remaining (dict): Remaining quotas for each layer

        Raises:
            RateLimitExceeded: If any rate limit is exceeded
        """
        # Use model-specific limits or defaults
        max_batch_size = model_max_batch_size or self.MAX_BATCH_SIZE_DEFAULT
        model_hourly = model_hourly_limit or self.MODEL_HOURLY_DEFAULT
        model_daily = model_daily_limit or self.MODEL_DAILY_DEFAULT

        # Layer 3: Check batch size
        if batch_size > max_batch_size:
            return {
                'allowed': False,
                'reason': f'Batch size ({batch_size}) exceeds maximum allowed ({max_batch_size})',
                'retry_after': 0,
                'remaining': {}
            }

        # Layer 4: Check concurrent requests
        concurrent_count = self._get_concurrent_count(user_id, model_id)
        if concurrent_count >= self.MAX_CONCURRENT_REQUESTS:
            return {
                'allowed': False,
                'reason': f'Too many concurrent requests ({concurrent_count}/{self.MAX_CONCURRENT_REQUESTS})',
                'retry_after': 60,  # Try again in 1 minute
                'remaining': {}
            }

        # Layer 1: Check global limits
        global_hourly = self._get_count(self.PREFIX_GLOBAL_HOURLY, user_id)
        global_daily = self._get_count(self.PREFIX_GLOBAL_DAILY, user_id)

        if global_hourly >= self.GLOBAL_HOURLY_LIMIT:
            retry_after = self._get_ttl(self.PREFIX_GLOBAL_HOURLY, user_id)
            return {
                'allowed': False,
                'reason': f'Global hourly limit exceeded ({global_hourly}/{self.GLOBAL_HOURLY_LIMIT})',
                'retry_after': retry_after,
                'remaining': {}
            }

        if global_daily >= self.GLOBAL_DAILY_LIMIT:
            retry_after = self._get_ttl(self.PREFIX_GLOBAL_DAILY, user_id)
            return {
                'allowed': False,
                'reason': f'Global daily limit exceeded ({global_daily}/{self.GLOBAL_DAILY_LIMIT})',
                'retry_after': retry_after,
                'remaining': {}
            }

        # Layer 2: Check per-model limits
        model_key = f"{user_id}:{model_id}"
        model_hourly_count = self._get_count(self.PREFIX_MODEL_HOURLY, model_key)
        model_daily_count = self._get_count(self.PREFIX_MODEL_DAILY, model_key)

        if model_hourly_count >= model_hourly:
            retry_after = self._get_ttl(self.PREFIX_MODEL_HOURLY, model_key)
            return {
                'allowed': False,
                'reason': f'Model hourly limit exceeded ({model_hourly_count}/{model_hourly})',
                'retry_after': retry_after,
                'remaining': {}
            }

        if model_daily_count >= model_daily:
            retry_after = self._get_ttl(self.PREFIX_MODEL_DAILY, model_key)
            return {
                'allowed': False,
                'reason': f'Model daily limit exceeded ({model_daily_count}/{model_daily})',
                'retry_after': retry_after,
                'remaining': {}
            }

        # All checks passed
        remaining = {
            'global_hourly': self.GLOBAL_HOURLY_LIMIT - global_hourly,
            'global_daily': self.GLOBAL_DAILY_LIMIT - global_daily,
            'model_hourly': model_hourly - model_hourly_count,
            'model_daily': model_daily - model_daily_count,
            'concurrent': self.MAX_CONCURRENT_REQUESTS - concurrent_count,
        }

        return {
            'allowed': True,
            'reason': '',
            'retry_after': 0,
            'remaining': remaining
        }

    def increment_counters(
        self,
        user_id: int,
        model_id: int
    ):
        """
        Increment all rate limit counters after a successful request.

        Args:
            user_id: User ID
            model_id: Model ID
        """
        # Increment global counters
        self._increment(self.PREFIX_GLOBAL_HOURLY, user_id, self.TTL_HOURLY)
        self._increment(self.PREFIX_GLOBAL_DAILY, user_id, self.TTL_DAILY)

        # Increment model counters
        model_key = f"{user_id}:{model_id}"
        self._increment(self.PREFIX_MODEL_HOURLY, model_key, self.TTL_HOURLY)
        self._increment(self.PREFIX_MODEL_DAILY, model_key, self.TTL_DAILY)

    def acquire_concurrent_slot(self, user_id: int, model_id: int) -> str:
        """
        Acquire a concurrent request slot.

        Args:
            user_id: User ID
            model_id: Model ID

        Returns:
            str: Slot ID (use this to release the slot later)

        Raises:
            RateLimitExceeded: If max concurrent requests reached
        """
        slot_id = f"{user_id}:{model_id}:{time.time()}"
        key = self._make_key(self.PREFIX_CONCURRENT, user_id, model_id)

        slots = cache.get(key, [])

        # Clean up expired slots (older than TTL)
        current_time = time.time()
        slots = [s for s in slots if current_time - float(s.split(':')[-1]) < self.TTL_CONCURRENT]

        if len(slots) >= self.MAX_CONCURRENT_REQUESTS:
            raise RateLimitExceeded(
                f"Maximum concurrent requests ({self.MAX_CONCURRENT_REQUESTS}) exceeded",
                retry_after=60
            )

        slots.append(slot_id)
        cache.set(key, slots, self.TTL_CONCURRENT)

        return slot_id

    def release_concurrent_slot(self, user_id: int, model_id: int, slot_id: str):
        """
        Release a concurrent request slot.

        Args:
            user_id: User ID
            model_id: Model ID
            slot_id: Slot ID returned by acquire_concurrent_slot()
        """
        key = self._make_key(self.PREFIX_CONCURRENT, user_id, model_id)

        slots = cache.get(key, [])

        if slot_id in slots:
            slots.remove(slot_id)
            cache.set(key, slots, self.TTL_CONCURRENT)

    def _get_concurrent_count(self, user_id: int, model_id: int) -> int:
        """Get current concurrent request count."""
        key = self._make_key(self.PREFIX_CONCURRENT, user_id, model_id)
        slots = cache.get(key, [])

        # Clean up expired slots
        current_time = time.time()
        active_slots = [s for s in slots if current_time - float(s.split(':')[-1]) < self.TTL_CONCURRENT]

        # Update cache if we cleaned up slots
        if len(active_slots) != len(slots):
            cache.set(key, active_slots, self.TTL_CONCURRENT)

        return len(active_slots)

    def get_remaining_quota(
        self,
        user_id: int,
        model_id: int,
        model_hourly_limit: Optional[int] = None,
        model_daily_limit: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get remaining quota for all rate limit layers.

        Args:
            user_id: User ID
            model_id: Model ID
            model_hourly_limit: Model-specific hourly limit
            model_daily_limit: Model-specific daily limit

        Returns:
            Dict with remaining quotas for each layer
        """
        model_hourly = model_hourly_limit or self.MODEL_HOURLY_DEFAULT
        model_daily = model_daily_limit or self.MODEL_DAILY_DEFAULT

        global_hourly = self._get_count(self.PREFIX_GLOBAL_HOURLY, user_id)
        global_daily = self._get_count(self.PREFIX_GLOBAL_DAILY, user_id)

        model_key = f"{user_id}:{model_id}"
        model_hourly_count = self._get_count(self.PREFIX_MODEL_HOURLY, model_key)
        model_daily_count = self._get_count(self.PREFIX_MODEL_DAILY, model_key)

        concurrent_count = self._get_concurrent_count(user_id, model_id)

        return {
            'global_hourly': max(0, self.GLOBAL_HOURLY_LIMIT - global_hourly),
            'global_daily': max(0, self.GLOBAL_DAILY_LIMIT - global_daily),
            'model_hourly': max(0, model_hourly - model_hourly_count),
            'model_daily': max(0, model_daily - model_daily_count),
            'concurrent': max(0, self.MAX_CONCURRENT_REQUESTS - concurrent_count),
        }

    def reset_limits(self, user_id: int, model_id: Optional[int] = None):
        """
        Reset rate limits for testing purposes.

        Args:
            user_id: User ID
            model_id: Model ID (if None, reset all model limits for user)
        """
        # Reset global limits
        cache.delete(self._make_key(self.PREFIX_GLOBAL_HOURLY, user_id))
        cache.delete(self._make_key(self.PREFIX_GLOBAL_DAILY, user_id))

        if model_id is not None:
            # Reset specific model limits
            model_key = f"{user_id}:{model_id}"
            cache.delete(self._make_key(self.PREFIX_MODEL_HOURLY, model_key))
            cache.delete(self._make_key(self.PREFIX_MODEL_DAILY, model_key))
            cache.delete(self._make_key(self.PREFIX_CONCURRENT, user_id, model_id))

    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key."""
        return f"{prefix}:{':'.join(map(str, args))}"

    def _get_count(self, prefix: str, identifier) -> int:
        """Get current count from cache."""
        key = self._make_key(prefix, identifier)
        return cache.get(key, 0)

    def _increment(self, prefix: str, identifier, ttl: int):
        """Increment counter in cache."""
        key = self._make_key(prefix, identifier)
        current = cache.get(key, 0)

        if current == 0:
            cache.set(key, 1, ttl)
        else:
            cache.set(key, current + 1, ttl)

    def _get_ttl(self, prefix: str, identifier) -> int:
        """
        Get remaining TTL for a cache key.

        Note: Django cache doesn't provide native TTL retrieval,
        so we estimate based on the limit type.
        """
        # Return estimated TTL based on prefix
        if 'hourly' in prefix:
            return self.TTL_HOURLY
        elif 'daily' in prefix:
            return self.TTL_DAILY
        else:
            return self.TTL_CONCURRENT
