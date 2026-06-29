"""
Tests for training configuration validation in API endpoints.

Validates JSON schema enforcement, size limits, and server whitelist.
"""
import pytest
import json
from django.test import Client
from django.contrib.auth import get_user_model
from users.models import Role, APIKey
from dataset.models import Dataset, DatasetAccess

User = get_user_model()


@pytest.fixture
def researcher_user(db):
    """Create a RESEARCHER user with API access."""
    researcher_role = Role.objects.get(name='RESEARCHER')
    user = User.objects.create_user(
        username='researcher_test',
        email='researcher@test.com',
        password='TestPass123!',
        role=researcher_role
    )
    return user


@pytest.fixture
def api_key(researcher_user):
    """Create API key for researcher user."""
    raw_key = APIKey.generate_api_key()
    api_key_obj = APIKey(
        user=researcher_user,
        name='test_api_key',
        ip_whitelist=['127.0.0.1']
    )
    api_key_obj.set_key(raw_key)
    api_key_obj.save()
    return raw_key  # Return the raw key for testing


@pytest.fixture
def test_dataset(db, researcher_user):
    """Create test dataset with access for researcher."""
    dataset = Dataset.objects.using('datasets_db').create(
        name='Test Dataset',
        description='Test description',
        file_path='/tmp/test.csv',
        medical_domain='cardiology',
        patient_count=100,
        data_type='tabular',
        anonymized=True,
        file_size=1024,
        file_format='csv',
        columns_count=5,
        rows_count=100,
        checksum_sha256='abc123',
        uploaded_by_id=researcher_user.id
    )

    # Grant access
    DatasetAccess.objects.using('datasets_db').create(
        dataset=dataset,
        user_id=researcher_user.id,
        assigned_by_id=researcher_user.id,
        can_train=True,
        can_view_metadata=True
    )

    return dataset


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestTrainingConfigValidation:
    """Test training configuration validation."""

    def test_valid_config_accepted(self, api_key, test_dataset):
        """Test that valid configuration is accepted."""
        client = Client()

        valid_config = {
            "model_json": {
                "model": {
                    "dataset": {
                        "selected_datasets": [{
                            "dataset_id": test_dataset.id,
                            "dataset_name": "Test Dataset"
                        }]
                    }
                }
            },
            "server_address": "localhost:8080",
            "client_id": "test_client",
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(valid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        # Should not fail validation (may fail later for other reasons)
        assert response.status_code != 400 or 'Invalid configuration' not in response.json().get('error', '')

    def test_missing_model_json_rejected(self, api_key):
        """Test that missing model_json is rejected."""
        client = Client()

        invalid_config = {
            "server_address": "localhost:8080",
            "client_id": "test_client"
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(invalid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 400
        assert 'Invalid configuration' in response.json()['error']

    def test_invalid_server_address_format_rejected(self, api_key, test_dataset):
        """Test that invalid server address format is rejected."""
        client = Client()

        invalid_config = {
            "model_json": {
                "model": {
                    "dataset": {
                        "selected_datasets": [{
                            "dataset_id": test_dataset.id,
                            "dataset_name": "Test Dataset"
                        }]
                    }
                }
            },
            "server_address": "not-a-valid-server",  # Missing port
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(invalid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 400
        assert 'Invalid configuration' in response.json()['error']

    def test_external_server_allowed(self, api_key, test_dataset):
        """Test that external public server addresses are allowed."""
        client = Client()

        valid_config = {
            "model_json": {
                "model": {
                    "dataset": {
                        "selected_datasets": [{
                            "dataset_id": test_dataset.id,
                            "dataset_name": "Test Dataset"
                        }]
                    }
                }
            },
            "server_address": "research-hospital.org:8080",  # External server
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(valid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        # Should not fail on server validation (may fail elsewhere)
        assert response.status_code != 403 or 'Internal server' not in response.json().get('error', '')

    def test_localhost_blocked_regardless_of_setting(self, api_key, test_dataset, settings):
        """Test that localhost is blocked even with private networks enabled."""
        client = Client()

        # Enable private networks but localhost should still be blocked
        settings.ALLOW_PRIVATE_FL_SERVERS = True
        # The dev settings enable localhost FL servers; disable that here so the
        # SSRF localhost block is exercised.
        settings.ALLOW_LOCALHOST_FL_SERVERS = False

        ssrf_config = {
            "model_json": {
                "model": {
                    "dataset": {
                        "selected_datasets": [{
                            "dataset_id": test_dataset.id,
                            "dataset_name": "Test Dataset"
                        }]
                    }
                }
            },
            "server_address": "localhost:8080",
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(ssrf_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 403
        assert 'Localhost' in response.json()['error']

    def test_oversized_config_rejected(self, api_key, test_dataset):
        """Test that oversized JSON payload is rejected."""
        client = Client()

        # Create model_json with excessive properties to exceed limit
        huge_properties = {f"prop_{i}": f"value_{i}" * 1000 for i in range(200)}
        huge_properties.update({
            "model": {
                "dataset": {
                    "selected_datasets": [{
                        "dataset_id": test_dataset.id,
                        "dataset_name": "Test Dataset"
                    }]
                }
            }
        })

        oversized_config = {
            "model_json": huge_properties,
            "server_address": "localhost:8080",
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(oversized_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        # Should be rejected for either size or property count
        assert response.status_code in [400, 500]

    def test_model_json_not_object_rejected(self, api_key):
        """Test that non-object model_json is rejected."""
        client = Client()

        invalid_config = {
            "model_json": "not an object",  # Should be object
            "server_address": "localhost:8080",
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(invalid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 400
        assert 'Invalid configuration' in response.json()['error']

    def test_excessive_properties_in_model_json_rejected(self, api_key, test_dataset):
        """Test that model_json with excessive properties is rejected."""
        client = Client()

        # Create model_json with >100 properties
        huge_properties = {f"prop_{i}": f"value_{i}" for i in range(101)}

        invalid_config = {
            "model_json": huge_properties,
            "server_address": "localhost:8080",
            "ssl_enabled": False
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(invalid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 400
        assert 'Invalid configuration' in response.json()['error']

    def test_localhost_always_blocked(self, api_key, test_dataset, settings):
        """Test that localhost addresses are always blocked (even with ALLOW_PRIVATE_FL_SERVERS=True)."""
        client = Client()

        # Even with private networks enabled, localhost should be blocked
        settings.ALLOW_PRIVATE_FL_SERVERS = True
        # The dev settings enable localhost FL servers; disable that here so the
        # SSRF localhost block is exercised.
        settings.ALLOW_LOCALHOST_FL_SERVERS = False

        localhost_addresses = [
            "localhost:8080",
            "127.0.0.1:8080",
            "0.0.0.0:8080"
        ]

        for server_address in localhost_addresses:
            config = {
                "model_json": {
                    "model": {
                        "dataset": {
                            "selected_datasets": [{
                                "dataset_id": test_dataset.id,
                                "dataset_name": "Test Dataset"
                            }]
                        }
                    }
                },
                "server_address": server_address,
                "ssl_enabled": False
            }

            response = client.post(
                '/api/v2/start-client',
                data=json.dumps(config),
                content_type='application/json',
                HTTP_X_API_KEY=api_key,
                HTTP_X_CLIENT_IP='127.0.0.1'
            )

            assert response.status_code == 403, f"Expected 403 for {server_address}"
            assert 'Localhost' in response.json()['error']

    def test_private_network_blocked_by_default(self, api_key, test_dataset, settings):
        """Test that private network addresses are blocked by default."""
        client = Client()

        # Ensure setting is False (default)
        settings.ALLOW_PRIVATE_FL_SERVERS = False

        private_addresses = [
            "192.168.1.100:8080",  # Private network
            "10.0.0.1:8080",       # Private network
            "172.16.0.1:8080"      # Private network
        ]

        for server_address in private_addresses:
            config = {
                "model_json": {
                    "model": {
                        "dataset": {
                            "selected_datasets": [{
                                "dataset_id": test_dataset.id,
                                "dataset_name": "Test Dataset"
                            }]
                        }
                    }
                },
                "server_address": server_address,
                "ssl_enabled": False
            }

            response = client.post(
                '/api/v2/start-client',
                data=json.dumps(config),
                content_type='application/json',
                HTTP_X_API_KEY=api_key,
                HTTP_X_CLIENT_IP='127.0.0.1'
            )

            assert response.status_code == 403, f"Expected 403 for {server_address}"
            assert 'Private network' in response.json()['error']

    def test_private_network_allowed_when_configured(self, api_key, test_dataset, settings):
        """Test that private network addresses are allowed when configured."""
        client = Client()

        # Enable private networks
        settings.ALLOW_PRIVATE_FL_SERVERS = True

        private_addresses = [
            "192.168.1.100:8080",  # Private network
            "10.0.0.1:8080",       # Private network
            "172.16.0.1:8080"      # Private network
        ]

        for server_address in private_addresses:
            config = {
                "model_json": {
                    "model": {
                        "dataset": {
                            "selected_datasets": [{
                                "dataset_id": test_dataset.id,
                                "dataset_name": "Test Dataset"
                            }]
                        }
                    }
                },
                "server_address": server_address,
                "ssl_enabled": False
            }

            response = client.post(
                '/api/v2/start-client',
                data=json.dumps(config),
                content_type='application/json',
                HTTP_X_API_KEY=api_key,
                HTTP_X_CLIENT_IP='127.0.0.1'
            )

            # Should not fail on private network validation
            assert response.status_code != 403 or 'Private network' not in response.json().get('error', '')

    def test_invalid_ssl_enabled_type_rejected(self, api_key, test_dataset):
        """Test that non-boolean ssl_enabled is rejected."""
        client = Client()

        invalid_config = {
            "model_json": {
                "model": {
                    "dataset": {
                        "selected_datasets": [{
                            "dataset_id": test_dataset.id,
                            "dataset_name": "Test Dataset"
                        }]
                    }
                }
            },
            "server_address": "localhost:8080",
            "ssl_enabled": "not a boolean"  # Should be boolean
        }

        response = client.post(
            '/api/v2/start-client',
            data=json.dumps(invalid_config),
            content_type='application/json',
            HTTP_X_API_KEY=api_key,
            HTTP_X_CLIENT_IP='127.0.0.1'
        )

        assert response.status_code == 400
        assert 'Invalid configuration' in response.json()['error']
