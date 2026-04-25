import pytest
from unittest.mock import MagicMock, patch
from django.test import RequestFactory
from medinet_core.security.middleware import RateLimitMiddleware


def _make_unauth_request(ip='10.0.0.1', path='/api/v1/ping/'):
    factory = RequestFactory()
    req = factory.get(path, REMOTE_ADDR=ip)
    # No api_user — simulates unauthenticated request
    return req


class TestIPRateLimitForUnauthenticated:

    def _make_middleware(self):
        get_response = MagicMock(return_value=MagicMock(status_code=401))
        return RateLimitMiddleware(get_response)

    def test_first_request_passes(self):
        mw = self._make_middleware()
        req = _make_unauth_request()
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            mock_cache.get.return_value = 0
            mock_cache.set.return_value = None
            resp = mw(req)
        assert resp.status_code != 429

    def test_request_over_ip_limit_returns_429(self):
        mw = self._make_middleware()
        req = _make_unauth_request(ip='10.0.0.2')
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            mock_cache.get.return_value = 21  # already at 21 (limit=20)
            resp = mw(req)
        assert resp.status_code == 429

    def test_different_ips_have_independent_limits(self):
        mw = self._make_middleware()
        req_a = _make_unauth_request(ip='10.0.0.3')
        req_b = _make_unauth_request(ip='10.0.0.4')
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            def cache_get(key, default=0):
                if '10.0.0.3' in key:
                    return 21
                return 0
            mock_cache.get.side_effect = cache_get
            mock_cache.set.return_value = None
            resp_a = mw(req_a)
            resp_b = mw(req_b)
        assert resp_a.status_code == 429
        assert resp_b.status_code != 429
