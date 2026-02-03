"""
Tests for RateLimiter.
"""
import pytest
import time
from django.core.cache import cache
from inference.utils.rate_limiter import RateLimiter, RateLimitExceeded


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before and after each test."""
    cache.clear()
    yield
    cache.clear()


@pytest.mark.django_db
class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized."""
        limiter = RateLimiter()
        assert limiter.GLOBAL_HOURLY_LIMIT == 100
        assert limiter.GLOBAL_DAILY_LIMIT == 500
        assert limiter.MODEL_HOURLY_DEFAULT == 60
        assert limiter.MODEL_DAILY_DEFAULT == 200
        assert limiter.MAX_BATCH_SIZE_DEFAULT == 1000
        assert limiter.MAX_CONCURRENT_REQUESTS == 5

    def test_batch_size_limit(self):
        """Test batch size limit enforcement."""
        limiter = RateLimiter()

        # Within limit
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=500)
        assert result['allowed'] is True

        # Exceeds default limit
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=1500)
        assert result['allowed'] is False
        assert 'Batch size' in result['reason']
        assert result['retry_after'] == 0

    def test_batch_size_model_specific_limit(self):
        """Test model-specific batch size limit."""
        limiter = RateLimiter()

        # Model has custom limit of 100
        result = limiter.check_limits(
            user_id=1,
            model_id=1,
            batch_size=150,
            model_max_batch_size=100
        )
        assert result['allowed'] is False
        assert 'exceeds maximum allowed (100)' in result['reason']

        # Within custom limit
        result = limiter.check_limits(
            user_id=1,
            model_id=1,
            batch_size=50,
            model_max_batch_size=100
        )
        assert result['allowed'] is True

    def test_global_hourly_limit(self):
        """Test global hourly rate limit."""
        limiter = RateLimiter()

        # Make requests up to limit using different model IDs to avoid per-model limits
        for i in range(limiter.GLOBAL_HOURLY_LIMIT):
            model_id = (i % 10) + 1  # Rotate through model IDs 1-10
            result = limiter.check_limits(user_id=1, model_id=model_id, batch_size=10)
            assert result['allowed'] is True
            limiter.increment_counters(user_id=1, model_id=model_id)

        # Next request should be blocked regardless of model ID
        result = limiter.check_limits(user_id=1, model_id=99, batch_size=10)
        assert result['allowed'] is False
        assert 'Global hourly limit exceeded' in result['reason']
        assert result['retry_after'] > 0

    def test_global_daily_limit(self):
        """Test global daily rate limit."""
        limiter = RateLimiter()

        # Simulate multiple hours to reach daily limit without hitting hourly limit
        requests_per_hour = limiter.GLOBAL_HOURLY_LIMIT
        total_requests = limiter.GLOBAL_DAILY_LIMIT

        requests_made = 0
        while requests_made < total_requests:
            # Make requests up to hourly limit for this hour
            for i in range(min(requests_per_hour, total_requests - requests_made)):
                model_id = ((requests_made + i) % 10) + 1
                result = limiter.check_limits(user_id=1, model_id=model_id, batch_size=10)
                assert result['allowed'] is True
                limiter.increment_counters(user_id=1, model_id=model_id)

            requests_made += min(requests_per_hour, total_requests - requests_made)

            # Reset hourly counters to simulate hour passing (if not at daily limit yet)
            if requests_made < total_requests:
                cache.delete(limiter._make_key(limiter.PREFIX_GLOBAL_HOURLY, 1))
                for model_id in range(1, 11):
                    cache.delete(limiter._make_key(limiter.PREFIX_MODEL_HOURLY, f"1:{model_id}"))

        # Reset hourly counter one more time to test daily limit specifically
        cache.delete(limiter._make_key(limiter.PREFIX_GLOBAL_HOURLY, 1))
        for model_id in range(1, 11):
            cache.delete(limiter._make_key(limiter.PREFIX_MODEL_HOURLY, f"1:{model_id}"))

        # Should hit daily limit regardless of model ID (hourly is reset)
        result = limiter.check_limits(user_id=1, model_id=99, batch_size=10)
        assert result['allowed'] is False
        assert 'Global daily limit exceeded' in result['reason']

    def test_model_hourly_limit(self):
        """Test per-model hourly rate limit."""
        limiter = RateLimiter()

        # Make requests to model 1 up to its limit
        for i in range(limiter.MODEL_HOURLY_DEFAULT):
            result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
            assert result['allowed'] is True
            limiter.increment_counters(user_id=1, model_id=1)

        # Next request to model 1 should be blocked
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
        assert result['allowed'] is False
        assert 'Model hourly limit exceeded' in result['reason']

        # But model 2 should still work
        result = limiter.check_limits(user_id=1, model_id=2, batch_size=10)
        assert result['allowed'] is True

    def test_model_daily_limit(self):
        """Test per-model daily rate limit."""
        limiter = RateLimiter()

        # Simulate multiple hours to reach model daily limit
        model_hourly = limiter.MODEL_HOURLY_DEFAULT
        model_daily = limiter.MODEL_DAILY_DEFAULT

        requests_made = 0
        while requests_made < model_daily:
            # Make requests up to model hourly limit for this hour
            for i in range(min(model_hourly, model_daily - requests_made)):
                result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
                assert result['allowed'] is True
                limiter.increment_counters(user_id=1, model_id=1)

            requests_made += min(model_hourly, model_daily - requests_made)

            # Reset hourly counters to simulate hour passing (if not at limit yet)
            if requests_made < model_daily:
                cache.delete(limiter._make_key(limiter.PREFIX_GLOBAL_HOURLY, 1))
                cache.delete(limiter._make_key(limiter.PREFIX_MODEL_HOURLY, "1:1"))

        # Reset hourly counters one more time before final check
        cache.delete(limiter._make_key(limiter.PREFIX_GLOBAL_HOURLY, 1))
        cache.delete(limiter._make_key(limiter.PREFIX_MODEL_HOURLY, "1:1"))

        # Should hit model daily limit
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
        assert result['allowed'] is False
        assert 'Model daily limit exceeded' in result['reason']

    def test_model_specific_limits(self):
        """Test model-specific hourly/daily limits."""
        limiter = RateLimiter()

        # Model with custom limits: 10 hourly, 20 daily
        for i in range(10):
            result = limiter.check_limits(
                user_id=1,
                model_id=1,
                batch_size=10,
                model_hourly_limit=10,
                model_daily_limit=20
            )
            assert result['allowed'] is True
            limiter.increment_counters(user_id=1, model_id=1)

        # Should hit custom hourly limit
        result = limiter.check_limits(
            user_id=1,
            model_id=1,
            batch_size=10,
            model_hourly_limit=10,
            model_daily_limit=20
        )
        assert result['allowed'] is False
        assert '10/10' in result['reason']

    def test_concurrent_requests(self):
        """Test concurrent request limit."""
        limiter = RateLimiter()

        # Acquire maximum concurrent slots
        slots = []
        for i in range(limiter.MAX_CONCURRENT_REQUESTS):
            slot = limiter.acquire_concurrent_slot(user_id=1, model_id=1)
            slots.append(slot)

        # Next acquisition should fail
        with pytest.raises(RateLimitExceeded, match="Maximum concurrent requests"):
            limiter.acquire_concurrent_slot(user_id=1, model_id=1)

        # Release one slot
        limiter.release_concurrent_slot(1, 1, slots[0])

        # Should be able to acquire again
        new_slot = limiter.acquire_concurrent_slot(user_id=1, model_id=1)
        assert new_slot is not None

        # Cleanup
        for slot in slots[1:]:
            limiter.release_concurrent_slot(1, 1, slot)
        limiter.release_concurrent_slot(1, 1, new_slot)

    def test_concurrent_count_check(self):
        """Test concurrent count in check_limits."""
        limiter = RateLimiter()

        # Acquire slots
        slots = []
        for i in range(limiter.MAX_CONCURRENT_REQUESTS):
            slot = limiter.acquire_concurrent_slot(user_id=1, model_id=1)
            slots.append(slot)

        # check_limits should detect concurrent limit
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
        assert result['allowed'] is False
        assert 'Too many concurrent requests' in result['reason']
        assert result['retry_after'] == 60

        # Cleanup
        for slot in slots:
            limiter.release_concurrent_slot(1, 1, slot)

    def test_get_remaining_quota(self):
        """Test getting remaining quota."""
        limiter = RateLimiter()

        # Initial quota (nothing used)
        quota = limiter.get_remaining_quota(user_id=1, model_id=1)
        assert quota['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT
        assert quota['global_daily'] == limiter.GLOBAL_DAILY_LIMIT
        assert quota['model_hourly'] == limiter.MODEL_HOURLY_DEFAULT
        assert quota['model_daily'] == limiter.MODEL_DAILY_DEFAULT
        assert quota['concurrent'] == limiter.MAX_CONCURRENT_REQUESTS

        # Use some quota
        for i in range(10):
            limiter.increment_counters(user_id=1, model_id=1)

        # Check remaining
        quota = limiter.get_remaining_quota(user_id=1, model_id=1)
        assert quota['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT - 10
        assert quota['global_daily'] == limiter.GLOBAL_DAILY_LIMIT - 10
        assert quota['model_hourly'] == limiter.MODEL_HOURLY_DEFAULT - 10
        assert quota['model_daily'] == limiter.MODEL_DAILY_DEFAULT - 10

    def test_remaining_quota_in_check_result(self):
        """Test that check_limits returns remaining quota."""
        limiter = RateLimiter()

        # Make some requests
        for i in range(5):
            limiter.increment_counters(user_id=1, model_id=1)

        result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
        assert result['allowed'] is True
        assert 'remaining' in result

        remaining = result['remaining']
        assert remaining['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT - 5
        assert remaining['global_daily'] == limiter.GLOBAL_DAILY_LIMIT - 5
        assert remaining['model_hourly'] == limiter.MODEL_HOURLY_DEFAULT - 5
        assert remaining['model_daily'] == limiter.MODEL_DAILY_DEFAULT - 5

    def test_reset_limits(self):
        """Test resetting rate limits."""
        limiter = RateLimiter()

        # Use some quota
        for i in range(50):
            limiter.increment_counters(user_id=1, model_id=1)

        # Check that quota is used
        quota = limiter.get_remaining_quota(user_id=1, model_id=1)
        assert quota['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT - 50

        # Reset limits
        limiter.reset_limits(user_id=1, model_id=1)

        # Quota should be back to full
        quota = limiter.get_remaining_quota(user_id=1, model_id=1)
        assert quota['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT
        assert quota['model_hourly'] == limiter.MODEL_HOURLY_DEFAULT

    def test_different_users_independent_limits(self):
        """Test that different users have independent rate limits."""
        limiter = RateLimiter()

        # User 1 uses their quota
        for i in range(limiter.GLOBAL_HOURLY_LIMIT):
            limiter.increment_counters(user_id=1, model_id=1)

        # User 1 should be blocked
        result = limiter.check_limits(user_id=1, model_id=1, batch_size=10)
        assert result['allowed'] is False

        # User 2 should still have full quota
        result = limiter.check_limits(user_id=2, model_id=1, batch_size=10)
        assert result['allowed'] is True

        quota = limiter.get_remaining_quota(user_id=2, model_id=1)
        assert quota['global_hourly'] == limiter.GLOBAL_HOURLY_LIMIT

    def test_increment_counters_creates_keys_with_ttl(self):
        """Test that increment_counters creates cache keys with TTL."""
        limiter = RateLimiter()

        # Increment counters
        limiter.increment_counters(user_id=1, model_id=1)

        # Check that keys exist
        global_hourly_key = limiter._make_key(limiter.PREFIX_GLOBAL_HOURLY, 1)
        global_daily_key = limiter._make_key(limiter.PREFIX_GLOBAL_DAILY, 1)

        assert cache.get(global_hourly_key) == 1
        assert cache.get(global_daily_key) == 1

        # Increment again
        limiter.increment_counters(user_id=1, model_id=1)

        assert cache.get(global_hourly_key) == 2
        assert cache.get(global_daily_key) == 2

    def test_cache_key_format(self):
        """Test cache key format."""
        limiter = RateLimiter()

        key = limiter._make_key('prefix', 'arg1', 'arg2', 123)
        assert key == 'prefix:arg1:arg2:123'

    def test_concurrent_slot_cleanup(self):
        """Test that expired concurrent slots are cleaned up."""
        limiter = RateLimiter()

        # Manually create an old slot (simulate expired)
        key = limiter._make_key(limiter.PREFIX_CONCURRENT, 1, 1)
        old_time = time.time() - (limiter.TTL_CONCURRENT + 100)  # Expired
        old_slot = f"1:1:{old_time}"
        cache.set(key, [old_slot], limiter.TTL_CONCURRENT)

        # Get concurrent count should clean up expired slot
        count = limiter._get_concurrent_count(user_id=1, model_id=1)
        assert count == 0  # Old slot should be removed

    def test_rate_limit_exception(self):
        """Test RateLimitExceeded exception."""
        exc = RateLimitExceeded("Test message", retry_after=120)
        assert str(exc) == "Test message"
        assert exc.retry_after == 120
