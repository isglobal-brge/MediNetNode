"""
Integration tests for the /api/v2/start-client endpoint.

Validates authentication, SSRF protection, payload validation,
and successful Process dispatch without starting a real Flower client.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# python-magic loads a native libmagic DLL at import time.  On Windows dev
# machines without libmagic installed this causes a fatal access violation when
# Django lazily loads dataset.urls → dataset.views → dataset.uploader → magic.
# Stub the module out before any Django URL resolution can trigger it.
if "magic" not in sys.modules:
    sys.modules["magic"] = MagicMock()


_SERVER_ADDRESS = "192.168.1.100:8080"  # private IP — allowed via _fl_env_vars


def _build_model_json(dataset_id: int) -> dict:
    """Minimal model_json that passes schema and permissions validation."""
    return {
        "model": {
            "metadata": {"model_type": "dl", "framework": "pytorch"},
            "dataset": {
                "selected_datasets": [
                    {"dataset_id": dataset_id, "dataset_name": "test"}
                ]
            },
        }
    }


def _post_start_client(client, auth_headers: dict, body: dict):
    return client.post(
        "/api/v2/start-client",
        data=json.dumps(body),
        content_type="application/json",
        **auth_headers,
    )


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestStartClientAuth:
    """Request-level auth and permission guards."""

    def test_missing_api_key_returns_401(self, client, heart_attack_dataset) -> None:
        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "ssl_enabled": False,
        }
        response = client.post(
            "/api/v2/start-client",
            data=json.dumps(body),
            content_type="application/json",
            REMOTE_ADDR="127.0.0.1",
        )
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client, heart_attack_dataset) -> None:
        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "ssl_enabled": False,
        }
        response = client.post(
            "/api/v2/start-client",
            data=json.dumps(body),
            content_type="application/json",
            HTTP_X_API_KEY="totally-invalid-key",
            REMOTE_ADDR="127.0.0.1",
        )
        assert response.status_code == 401


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestStartClientValidation:
    """Payload and SSRF guard tests."""

    @pytest.fixture(autouse=True)
    def _allow_private(self, settings) -> None:
        settings.ALLOW_PRIVATE_FL_SERVERS = True

    def test_missing_model_json_returns_400(self, client, auth_headers) -> None:
        body = {"server_address": _SERVER_ADDRESS, "ssl_enabled": False}
        response = _post_start_client(client, auth_headers, body)
        assert response.status_code == 400

    def test_localhost_server_address_blocked(
        self, client, auth_headers, heart_attack_dataset
    ) -> None:
        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": "localhost:8080",
            "ssl_enabled": False,
        }
        response = _post_start_client(client, auth_headers, body)
        assert response.status_code == 403

    def test_loopback_ip_blocked(
        self, client, auth_headers, heart_attack_dataset
    ) -> None:
        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": "127.0.0.1:8080",
            "ssl_enabled": False,
        }
        response = _post_start_client(client, auth_headers, body)
        assert response.status_code == 403

    def test_ssl_required_without_ca_cert_returns_400(
        self, client, auth_headers, heart_attack_dataset
    ) -> None:
        """ssl_enabled=True (default) without ca_cert must be rejected."""
        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            # ssl_enabled not specified — defaults to True
        }
        response = _post_start_client(client, auth_headers, body)
        assert response.status_code == 400
        data = response.json()
        assert "ca_cert" in data.get("error", "").lower() or "certificate" in data.get("error", "").lower()


@pytest.mark.django_db(databases=["default", "datasets_db"])
class TestStartClientSuccess:
    """End-to-end success path with a mocked Flower Process."""

    @pytest.fixture(autouse=True)
    def _allow_private(self, settings) -> None:
        settings.ALLOW_PRIVATE_FL_SERVERS = True

    @patch("api.views.Process")
    def test_returns_200_and_flower_client_started_status(
        self, mock_process_cls, client, auth_headers, heart_attack_dataset
    ) -> None:
        mock_proc = MagicMock()
        mock_process_cls.return_value = mock_proc

        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "client_id": "test-client-001",
            "ssl_enabled": False,
        }
        response = _post_start_client(client, auth_headers, body)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Flower Client started"

    @patch("api.views.Process")
    def test_response_includes_client_id_and_server_address(
        self, mock_process_cls, client, auth_headers, heart_attack_dataset
    ) -> None:
        mock_process_cls.return_value = MagicMock()

        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "client_id": "test-client-abc",
            "ssl_enabled": False,
        }
        response = _post_start_client(client, auth_headers, body)

        data = response.json()
        assert data["client_id"] == "test-client-abc"
        assert data["server_address"] == _SERVER_ADDRESS

    @patch("api.views.Process")
    def test_process_start_called_once(
        self, mock_process_cls, client, auth_headers, heart_attack_dataset
    ) -> None:
        mock_proc = MagicMock()
        mock_process_cls.return_value = mock_proc

        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "ssl_enabled": False,
        }
        _post_start_client(client, auth_headers, body)

        mock_proc.start.assert_called_once()

    @patch("api.views.Process")
    def test_training_session_created_in_db(
        self, mock_process_cls, client, auth_headers, heart_attack_dataset
    ) -> None:
        """A TrainingSession record must exist after a successful start-client call."""
        from trainings.models import TrainingSession

        mock_process_cls.return_value = MagicMock()

        body = {
            "model_json": _build_model_json(heart_attack_dataset.id),
            "server_address": _SERVER_ADDRESS,
            "client_id": "session-test-client",
            "ssl_enabled": False,
        }
        response = _post_start_client(client, auth_headers, body)
        assert response.status_code == 200

        sessions = TrainingSession.objects.filter(
            server_address=_SERVER_ADDRESS
        )
        assert sessions.exists(), "TrainingSession was not persisted after start-client"
