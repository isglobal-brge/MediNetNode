"""
Local Flower integration tests (slow E2E).

Each test starts a real Flower server in a subprocess and runs the
Node's actual client for 1 round.  No mocks — pure end-to-end.

Run only with:
    pytest -m slow tests/integration/test_node_flower_training.py -v
"""
import socket
import subprocess
import sys
import threading
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

# Prevent python-magic crash when Django lazy-loads dataset.urls on Windows.
if "magic" not in sys.modules:
    sys.modules["magic"] = MagicMock()

# ---------------------------------------------------------------------------
# Flower server helpers
# ---------------------------------------------------------------------------

# Inline script run in a subprocess to start the Flower server.
# Running it out-of-process avoids all signal/threading conflicts with pytest.
_SERVER_SCRIPT = """
import sys
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

port = int(sys.argv[1])
num_rounds = int(sys.argv[2])

strategy = FedAvg(
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)
start_server(
    server_address=f"0.0.0.0:{port}",
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
"""


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll until the port accepts TCP connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _start_server_subprocess(port: int, num_rounds: int = 1) -> subprocess.Popen:
    """Start a Flower server in a subprocess and wait until the port is bound."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER_SCRIPT, str(port), str(num_rounds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    bound = _wait_for_port("127.0.0.1", port, timeout=20.0)
    if not bound:
        _out, _err = proc.communicate(timeout=3)
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError(
            f"Flower server did not bind on port {port} within 20 s\n"
            f"stdout: {_out.decode(errors='replace')}\n"
            f"stderr: {_err.decode(errors='replace')}"
        )
    # Allow Flower's server-side logic to fully initialize after the port binds.
    time.sleep(5)
    return proc


def _run_client_in_thread(fn, *args, timeout: float = 120.0):
    """Run fn(*args) in a daemon thread.

    Allows thread DB sharing so the client thread can see the test's
    uncommitted records (pytest-django uses non-transactional wrapping).
    Raises on timeout or if fn raised.
    """
    from django.db import connections

    for alias in connections:
        try:
            connections[alias].allow_thread_sharing = True
        except Exception:
            pass

    error_box: list = []

    def _target():
        try:
            fn(*args)
        except Exception as exc:
            error_box.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"Flower client thread did not finish within {timeout}s")
    if error_box:
        raise error_box[0]


# ---------------------------------------------------------------------------
# Port constants — use high, distinct ports to avoid conflicts
# ---------------------------------------------------------------------------
_PORT_DL = 19101
_PORT_DL2 = 19102
_PORT_SVM = 19201
_PORT_SVM2 = 19202
_PORT_DPRF = 19301


# ---------------------------------------------------------------------------
# DL end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.django_db(transaction=True, databases=["default", "datasets_db"])
class TestDLFlowerIntegration:
    """DL Flower client trains with real heart attack data for 1 round."""

    @pytest.fixture(autouse=True)
    def _allow_private(self, settings) -> None:
        settings.ALLOW_PRIVATE_FL_SERVERS = True

    def _dl_model_json(self, dataset_id: int, dataset_name: str) -> dict:
        # client.py extracts model_config = MODEL_JSON['model'] and passes it to
        # DynamicModel.  DynamicModel reads model_config['architecture']['layers'],
        # so 'architecture' must be a top-level key inside 'model', not under
        # 'config_json'.  52 feature columns match heart_attack_prediction_dataset.
        return {
            "model": {
                "metadata": {"model_type": "dl", "framework": "pytorch"},
                "dataset": {
                    "selected_datasets": [
                        {"dataset_id": dataset_id, "dataset_name": dataset_name}
                    ]
                },
                "architecture": {
                    "layers": [
                        {
                            "id": "layer_0",
                            "type": "Linear",
                            "params": {"in_features": 52, "out_features": 1},
                            "inputs": ["input_data"],
                        }
                    ]
                },
                "training": {
                    "loss_function": "bce_with_logits",
                    "optimizer": {
                        "type": "Adam",
                        "learning_rate": 0.01,
                        "weight_decay": 0,
                        "differential_privacy": {
                            "noise_multiplier": 1.0,
                            "max_grad_norm": 1.0,
                            "random_seed": 42,
                        },
                    },
                },
            },
            "train": {"rounds": 1, "epochs": 1, "batch_size": 32},
        }

    def test_dl_client_completes_one_round(
        self, heart_attack_dataset, integration_researcher_user
    ) -> None:
        from api.federated.client import start_flower_client
        from trainings.models import TrainingSession

        model_json = self._dl_model_json(
            heart_attack_dataset.id, heart_attack_dataset.name
        )
        client_id = f"dl-e2e-{uuid.uuid4().hex[:8]}"
        server_addr = f"127.0.0.1:{_PORT_DL}"

        session = TrainingSession.objects.create(
            client_id=client_id,
            user=integration_researcher_user,
            dataset_id=heart_attack_dataset.id,
            dataset_name=heart_attack_dataset.name,
            model_config=model_json,
            server_address=server_addr,
            total_rounds=1,
        )

        srv = _start_server_subprocess(_PORT_DL, num_rounds=1)
        try:
            _run_client_in_thread(
                start_flower_client,
                model_json, server_addr, client_id,
                integration_researcher_user, session.session_id, None,
            )
        finally:
            srv.terminate()
            srv.wait(timeout=10)

        session.refresh_from_db()
        assert session.status in ("COMPLETED", "ACTIVE", "STARTING"), (
            f"Unexpected status: {session.status}"
        )

    def test_dl_training_creates_training_round_record(
        self, heart_attack_dataset, integration_researcher_user
    ) -> None:
        from api.federated.client import start_flower_client
        from trainings.models import TrainingRound, TrainingSession

        model_json = self._dl_model_json(
            heart_attack_dataset.id, heart_attack_dataset.name
        )
        client_id = f"dl-round-{uuid.uuid4().hex[:8]}"
        server_addr = f"127.0.0.1:{_PORT_DL2}"

        session = TrainingSession.objects.create(
            client_id=client_id,
            user=integration_researcher_user,
            dataset_id=heart_attack_dataset.id,
            dataset_name=heart_attack_dataset.name,
            model_config=model_json,
            server_address=server_addr,
            total_rounds=1,
        )

        srv = _start_server_subprocess(_PORT_DL2, num_rounds=1)
        try:
            _run_client_in_thread(
                start_flower_client,
                model_json, server_addr, client_id,
                integration_researcher_user, session.session_id, None,
            )
        finally:
            srv.terminate()
            srv.wait(timeout=10)

        rounds = TrainingRound.objects.filter(session=session)
        assert rounds.count() >= 1, "Expected at least one TrainingRound after training"


# ---------------------------------------------------------------------------
# SVM end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.django_db(transaction=True, databases=["default", "datasets_db"])
class TestSVMFlowerIntegration:
    """SVM Flower client trains with real tabular data for 1 round."""

    @pytest.fixture(autouse=True)
    def _allow_private(self, settings) -> None:
        settings.ALLOW_PRIVATE_FL_SERVERS = True

    def _svm_model_json(self, dataset_id: int, dataset_name: str) -> dict:
        return {
            "model": {
                "metadata": {"model_type": "ml"},
                "dataset": {
                    "selected_datasets": [
                        {"dataset_id": dataset_id, "dataset_name": dataset_name}
                    ]
                },
                "training": {
                    "ml_method": "fedsvm",
                    "C": 1.0,
                    "kernel_config": {"kernel": "rbf", "gamma": 0.1},
                    "val_size": 0.2,
                    "random_state": 42,
                },
            },
        }

    def test_svm_client_completes_one_round(
        self, tabular_dataset, integration_researcher_user
    ) -> None:
        from api.federated.client import start_flower_client
        from trainings.models import TrainingSession

        model_json = self._svm_model_json(
            tabular_dataset.id, tabular_dataset.name
        )
        client_id = f"svm-e2e-{uuid.uuid4().hex[:8]}"
        server_addr = f"127.0.0.1:{_PORT_SVM}"

        session = TrainingSession.objects.create(
            client_id=client_id,
            user=integration_researcher_user,
            dataset_id=tabular_dataset.id,
            dataset_name=tabular_dataset.name,
            model_config=model_json,
            server_address=server_addr,
            total_rounds=1,
        )

        srv = _start_server_subprocess(_PORT_SVM, num_rounds=1)
        try:
            _run_client_in_thread(
                start_flower_client,
                model_json, server_addr, client_id,
                integration_researcher_user, session.session_id, None,
            )
        finally:
            srv.terminate()
            srv.wait(timeout=10)

        session.refresh_from_db()
        assert session.status in ("COMPLETED", "ACTIVE", "STARTING")

    def test_svm_client_with_linear_kernel(
        self, tabular_dataset, integration_researcher_user
    ) -> None:
        from api.federated.client import start_flower_client
        from trainings.models import TrainingSession

        model_json = {
            "model": {
                "metadata": {"model_type": "ml"},
                "dataset": {
                    "selected_datasets": [
                        {
                            "dataset_id": tabular_dataset.id,
                            "dataset_name": tabular_dataset.name,
                        }
                    ]
                },
                "training": {
                    "ml_method": "fedsvm",
                    "C": 2.0,
                    "kernel_config": {"kernel": "linear"},
                    "val_size": 0.2,
                    "random_state": 0,
                },
            },
        }
        client_id = f"svm-lin-{uuid.uuid4().hex[:8]}"
        server_addr = f"127.0.0.1:{_PORT_SVM2}"

        session = TrainingSession.objects.create(
            client_id=client_id,
            user=integration_researcher_user,
            dataset_id=tabular_dataset.id,
            dataset_name=tabular_dataset.name,
            model_config=model_json,
            server_address=server_addr,
            total_rounds=1,
        )

        srv = _start_server_subprocess(_PORT_SVM2, num_rounds=1)
        try:
            _run_client_in_thread(
                start_flower_client,
                model_json, server_addr, client_id,
                integration_researcher_user, session.session_id, None,
            )
        finally:
            srv.terminate()
            srv.wait(timeout=10)

        session.refresh_from_db()
        assert session.status in ("COMPLETED", "ACTIVE", "STARTING")


# ---------------------------------------------------------------------------
# DP Random Forest end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.django_db(transaction=True, databases=["default", "datasets_db"])
class TestDPRFFlowerIntegration:
    """DP Random Forest Flower client trains with real tabular data."""

    @pytest.fixture(autouse=True)
    def _allow_private(self, settings) -> None:
        settings.ALLOW_PRIVATE_FL_SERVERS = True

    def _dprf_model_json(self, dataset_id: int, dataset_name: str) -> dict:
        return {
            "model": {
                "metadata": {"model_type": "ml"},
                "dataset": {
                    "selected_datasets": [
                        {"dataset_id": dataset_id, "dataset_name": dataset_name}
                    ]
                },
                "training": {
                    "ml_method": "dp_random_forest",
                    "n_trees_per_client": 5,
                    "max_depth": 5,
                    "epsilon_total": 1.0,
                    "feature_bounds": {
                        "min": [0.0, 0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0, 1.0],
                    },
                },
            },
            # _validate_and_sanitize_config reads top-level 'training' key
            "training": {
                "ml_method": "dp_random_forest",
                "n_trees_per_client": 5,
                "max_depth": 5,
                "epsilon_total": 1.0,
                "feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
                "global_feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
            },
        }

    def test_dprf_client_completes_one_round(
        self, tabular_dataset, integration_researcher_user
    ) -> None:
        from api.federated.client import start_flower_client
        from trainings.models import TrainingSession

        model_json = self._dprf_model_json(
            tabular_dataset.id, tabular_dataset.name
        )
        client_id = f"dprf-e2e-{uuid.uuid4().hex[:8]}"
        server_addr = f"127.0.0.1:{_PORT_DPRF}"

        session = TrainingSession.objects.create(
            client_id=client_id,
            user=integration_researcher_user,
            dataset_id=tabular_dataset.id,
            dataset_name=tabular_dataset.name,
            model_config=model_json,
            server_address=server_addr,
            total_rounds=1,
        )

        srv = _start_server_subprocess(_PORT_DPRF, num_rounds=1)
        try:
            _run_client_in_thread(
                start_flower_client,
                model_json, server_addr, client_id,
                integration_researcher_user, session.session_id, None,
            )
        finally:
            srv.terminate()
            srv.wait(timeout=10)

        session.refresh_from_db()
        assert session.status in ("COMPLETED", "ACTIVE", "STARTING")

    def test_dprf_low_epsilon_trains_without_error(self) -> None:
        """epsilon_total=0.1 (min valid) should not raise an exception."""
        from api.federated.algorithms import get_algorithm

        np.random.seed(0)
        X = np.random.rand(80, 4)
        y = (X[:, 0] > 0.5).astype(int)

        config = {
            "model": {
                "training": {
                    "ml_method": "dp_random_forest",
                    "n_trees_per_client": 3,
                    "max_depth": 3,
                    "epsilon_total": 0.1,
                    "feature_bounds": {
                        "min": [0.0, 0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0, 1.0],
                    },
                }
            },
            "training": {
                "ml_method": "dp_random_forest",
                "n_trees_per_client": 3,
                "max_depth": 3,
                "epsilon_total": 0.1,
                "feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
                "global_feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
            },
        }

        AlgClass = get_algorithm("dp_random_forest")
        alg = AlgClass(X, y, config, X, y)
        params, metrics = alg.fit([])
        assert params is not None

    def test_dprf_epsilon_1_is_accepted(self) -> None:
        """epsilon_total=1.0 is within the valid range."""
        from api.federated.algorithms import get_algorithm

        np.random.seed(42)
        X = np.random.rand(80, 4)
        y = (X[:, 0] > 0.5).astype(int)

        config = {
            "model": {
                "training": {
                    "ml_method": "dp_random_forest",
                    "n_trees_per_client": 5,
                    "max_depth": 5,
                    "epsilon_total": 1.0,
                    "feature_bounds": {
                        "min": [0.0, 0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0, 1.0],
                    },
                }
            },
            "training": {
                "ml_method": "dp_random_forest",
                "n_trees_per_client": 5,
                "max_depth": 5,
                "epsilon_total": 1.0,
                "feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
                "global_feature_bounds": {
                    "min": [0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0, 1.0],
                },
            },
        }

        AlgClass = get_algorithm("dp_random_forest")
        alg = AlgClass(X, y, config, X, y)
        params, _ = alg.fit([])
        assert params is not None
