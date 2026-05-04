"""
Tests for IP whitelist security vulnerabilities.

Validates that IP spoofing is prevented and CIDR ranges work correctly.
"""
import pytest
from django.test import Client
from django.contrib.auth import get_user_model
from users.models import Role, APIKey

User = get_user_model()


@pytest.fixture
def researcher_user(db):
    """Create RESEARCHER user."""
    researcher_role = Role.objects.get(name='RESEARCHER')
    user = User.objects.create_user(
        username='researcher_test',
        email='researcher@test.com',
        password='TestPass123!',
        role=researcher_role
    )
    return user


@pytest.fixture
def api_key_single_ip(researcher_user):
    """Create API key with single IP whitelist."""
    raw_key = APIKey.generate_api_key()
    api_key_obj = APIKey(
        user=researcher_user,
        name='test_single_ip',
        ip_whitelist=['203.0.113.10']  # TEST-NET-3 (RFC 5737)
    )
    api_key_obj.set_key(raw_key)
    api_key_obj.save()
    return raw_key


@pytest.fixture
def api_key_cidr_range(researcher_user):
    """Create API key with CIDR range whitelist."""
    raw_key = APIKey.generate_api_key()
    api_key_obj = APIKey(
        user=researcher_user,
        name='test_cidr',
        ip_whitelist=['203.0.113.0/24']  # TEST-NET-3 subnet
    )
    api_key_obj.set_key(raw_key)
    api_key_obj.save()
    return raw_key


@pytest.mark.django_db(databases=['default'])
class TestIPWhitelistSecurity:
    """Test IP whitelist security."""

    def test_spoofed_x_client_ip_rejected(self, api_key_single_ip):
        """Test that spoofed X-Client-IP header is rejected."""
        client = Client()

        # Attacker tries to spoof IP via X-Client-IP header
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=api_key_single_ip,
            HTTP_X_CLIENT_IP='203.0.113.10',  # Spoofed IP
            REMOTE_ADDR='192.0.2.1'  # Actual IP (different)
        )

        # Should be rejected because we now ignore X-Client-IP
        assert response.status_code == 403
        assert 'IP address not authorized' in response.json()['error']

    def test_x_forwarded_for_from_trusted_proxy_accepted(self, api_key_single_ip, settings):
        """Test that X-Forwarded-For from trusted proxy is accepted."""
        client = Client()

        # Configure trusted proxies
        settings.TRUSTED_PROXIES = ['10.0.0.1']

        # Request from load balancer with X-Forwarded-For
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=api_key_single_ip,
            HTTP_X_FORWARDED_FOR='203.0.113.10',  # Real client IP
            REMOTE_ADDR='10.0.0.1'  # Trusted proxy IP
        )

        # Should be accepted (real IP is whitelisted)
        assert response.status_code == 200

    def test_x_forwarded_for_from_untrusted_source_rejected(self, api_key_single_ip, settings):
        """Test that X-Forwarded-For from untrusted source is rejected."""
        client = Client()

        # No trusted proxies configured (or different IP)
        settings.TRUSTED_PROXIES = []

        # Attacker tries to spoof IP via X-Forwarded-For
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=api_key_single_ip,
            HTTP_X_FORWARDED_FOR='203.0.113.10',  # Spoofed IP
            REMOTE_ADDR='192.0.2.1'  # Actual attacker IP
        )

        # Should be rejected (uses REMOTE_ADDR, not spoofed header)
        assert response.status_code == 403
        assert 'IP address not authorized' in response.json()['error']

    def test_cidr_range_validation_works(self, api_key_cidr_range):
        """Test that CIDR range validation works correctly."""
        client = Client()

        # IP within CIDR range
        valid_ips = ['203.0.113.1', '203.0.113.100', '203.0.113.254']

        for ip_addr in valid_ips:
            response = client.get(
                '/api/v2/ping',
                HTTP_X_API_KEY=api_key_cidr_range,
                REMOTE_ADDR=ip_addr
            )
            assert response.status_code == 200, f"IP {ip_addr} should be allowed"

    def test_cidr_range_excludes_outside_ips(self, api_key_cidr_range):
        """Test that IPs outside CIDR range are rejected."""
        client = Client()

        # IPs outside CIDR range
        invalid_ips = ['203.0.112.10', '203.0.114.10', '192.0.2.1']

        for ip_addr in invalid_ips:
            response = client.get(
                '/api/v2/ping',
                HTTP_X_API_KEY=api_key_cidr_range,
                REMOTE_ADDR=ip_addr
            )
            assert response.status_code == 403, f"IP {ip_addr} should be rejected"
            assert 'IP address not authorized' in response.json()['error']

    def test_multiple_x_forwarded_for_ips_uses_first(self, api_key_single_ip, settings):
        """Test that with multiple X-Forwarded-For IPs, only first is used."""
        client = Client()

        settings.TRUSTED_PROXIES = ['10.0.0.1']

        # X-Forwarded-For with multiple IPs (client, proxy1, proxy2)
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=api_key_single_ip,
            HTTP_X_FORWARDED_FOR='203.0.113.10, 192.0.2.5, 10.0.0.2',
            REMOTE_ADDR='10.0.0.1'  # Trusted proxy
        )

        # Should use first IP (203.0.113.10) which is whitelisted
        assert response.status_code == 200

    def test_malformed_cidr_handled_gracefully(self, researcher_user):
        """Test that malformed CIDR notation is handled gracefully."""
        raw_key = APIKey.generate_api_key()
        api_key_obj = APIKey(
            user=researcher_user,
            name='test_malformed',
            ip_whitelist=['invalid-cidr/32', '203.0.113.10']
        )
        api_key_obj.set_key(raw_key)
        api_key_obj.save()

        client = Client()

        # Should fall back to string comparison and allow exact match
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=raw_key,
            REMOTE_ADDR='203.0.113.10'
        )

        assert response.status_code == 200

    def test_ipv6_addresses_supported(self, researcher_user):
        """Test that IPv6 addresses are properly validated."""
        raw_key = APIKey.generate_api_key()
        api_key_obj = APIKey(
            user=researcher_user,
            name='test_ipv6',
            ip_whitelist=['2001:db8::1']
        )
        api_key_obj.set_key(raw_key)
        api_key_obj.save()

        client = Client()

        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=raw_key,
            REMOTE_ADDR='2001:db8::1'
        )

        assert response.status_code == 200

    def test_ipv6_cidr_ranges_supported(self, researcher_user):
        """Test that IPv6 CIDR ranges work correctly."""
        raw_key = APIKey.generate_api_key()
        api_key_obj = APIKey(
            user=researcher_user,
            name='test_ipv6_cidr',
            ip_whitelist=['2001:db8::/32']
        )
        api_key_obj.set_key(raw_key)
        api_key_obj.save()

        client = Client()

        # IP within range
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=raw_key,
            REMOTE_ADDR='2001:db8:0:0:0:0:0:1'
        )
        assert response.status_code == 200

        # IP outside range
        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=raw_key,
            REMOTE_ADDR='2001:db9::1'
        )
        assert response.status_code == 403

    def test_empty_whitelist_rejects_all(self, researcher_user):
        """Test that empty whitelist rejects all requests."""
        raw_key = APIKey.generate_api_key()
        api_key_obj = APIKey(
            user=researcher_user,
            name='test_empty',
            ip_whitelist=[]
        )
        api_key_obj.set_key(raw_key)
        api_key_obj.save()

        client = Client()

        response = client.get(
            '/api/v2/ping',
            HTTP_X_API_KEY=raw_key,
            REMOTE_ADDR='203.0.113.10'
        )

        assert response.status_code == 403
        assert 'IP address not authorized' in response.json()['error']
