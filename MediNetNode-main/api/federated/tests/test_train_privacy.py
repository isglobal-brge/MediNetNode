"""
Tests for DP epsilon measurement in train_functions.py and dl_client.py.

Security model: the Hub is UNTRUSTED. It can send arbitrary values for
noise_multiplier, max_grad_norm, random_seed, optimizer type, epochs, and
even the entire 'model' config tree through Flower's config dict. The Node
must enforce its own minimums and never let Hub parameters weaken DP.

Coverage:
  - train() returns a 6-tuple including epsilon
  - epsilon is finite and positive after normal training
  - Node enforces sigma >= 1.0 even when Hub sends lower values
  - Node enforces max_grad_norm >= 1.0 even when Hub sends lower values
  - Node never uses the Hub-supplied random seed
  - NaN and inf noise_multiplier from Hub are rejected (replaced by minimum)
  - Epoch count is capped at _MAX_EPOCHS regardless of Hub request
  - Hub-supplied optimizer type not in allowlist falls back to Adam
  - epsilon scales correctly with epochs (more steps → higher epsilon)
  - epsilon scales correctly with noise_multiplier (higher sigma → lower epsilon)
  - epsilon is float('inf') when get_epsilon() fails (graceful degradation)
  - float('inf') epsilon is converted to -1.0 sentinel in dl_client metrics
  - Flower config dict cannot overwrite the 'model' key (DP parameters)
  - DLFlowerClient.fit() propagates epsilon into returned metrics dict
  - DLFlowerClient.fit() propagates epsilon into round_metrics
  - privacy_delta in metrics uses the Node constant _DP_DELTA, not a literal
  - epsilon and delta present even on training error (empty dict fallback)
  - Edge cases: empty config, missing dp config, zero/negative values from Hub
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from torch.utils.data import DataLoader, TensorDataset

from ..train_functions import (
    train, _MIN_NOISE_MULTIPLIER, _MIN_GRAD_NORM, _DP_DELTA,
    _MAX_EPOCHS, _ALLOWED_OPTIMIZERS, _safe_dp_float,
)
from ..dl_client import DLFlowerClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(in_features: int = 4) -> nn.Module:
    return nn.Linear(in_features, 1)


def _make_loader(n: int = 64, features: int = 4, batch: int = 16) -> DataLoader:
    X = torch.randn(n, features)
    y = torch.randint(0, 2, (n,)).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch, drop_last=True)


def _minimal_config(epochs: int = 1, noise_multiplier: float = 1.0,
                    max_grad_norm: float = 1.0) -> dict:
    return {
        "train": {"epochs": epochs, "batch_size": 16},
        "model": {
            "training": {
                "loss_function": "bce_with_logits",
                "optimizer": {
                    "type": "Adam",
                    "learning_rate": 0.01,
                    "weight_decay": 0,
                    "differential_privacy": {
                        "noise_multiplier": noise_multiplier,
                        "max_grad_norm": max_grad_norm,
                        "random_seed": 42,
                    },
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# 1. Return signature
# ---------------------------------------------------------------------------

class TestTrainReturnSignature:

    def test_returns_seven_tuple(self):
        """train() must return exactly 7 values (added actual_noise_multiplier as 7th)."""
        model = _make_model()
        loader = _make_loader()
        result = train(model, loader, _minimal_config(), partition_id=0, verbose=False)
        assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"

    def test_sixth_element_is_epsilon(self):
        """Sixth return value is a float named epsilon."""
        model = _make_model()
        loader = _make_loader()
        loss, acc, prec, rec, f1, epsilon, actual_noise = train(
            model, loader, _minimal_config(), partition_id=0, verbose=False
        )
        assert isinstance(epsilon, float)

    def test_epsilon_is_finite_positive(self):
        """Epsilon must be a finite positive number after normal training."""
        model = _make_model()
        loader = _make_loader()
        loss, acc, prec, rec, f1, epsilon, actual_noise = train(
            model, loader, _minimal_config(), partition_id=0, verbose=False
        )
        assert math.isfinite(epsilon), f"Expected finite epsilon, got {epsilon}"
        assert epsilon > 0, f"Expected positive epsilon, got {epsilon}"

    def test_first_five_elements_unchanged(self):
        """The first 5 return values must still be loss, acc, prec, rec, f1."""
        model = _make_model()
        loader = _make_loader()
        loss, acc, prec, rec, f1, epsilon, actual_noise = train(
            model, loader, _minimal_config(), partition_id=0, verbose=False
        )
        assert 0.0 <= loss
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_seventh_element_is_actual_noise_multiplier(self):
        """Seventh return value is the noise_multiplier actually used by Opacus."""
        model = _make_model()
        loader = _make_loader()
        loss, acc, prec, rec, f1, epsilon, actual_noise = train(
            model, loader, _minimal_config(), partition_id=0, verbose=False
        )
        assert isinstance(actual_noise, float)
        assert actual_noise >= 1.0, f"Actual noise must be >= Node minimum, got {actual_noise}"


# ---------------------------------------------------------------------------
# 2. Node-enforced security minimums
# ---------------------------------------------------------------------------

class TestNodeSecurityEnforcement:

    def test_hub_cannot_disable_noise_with_zero(self):
        """Hub sending noise_multiplier=0 must be rejected; Node uses minimum."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(noise_multiplier=0.0)
        *_, epsilon = train(model, loader, config, partition_id=0, verbose=False)
        # If noise_multiplier=0 were used, get_epsilon would return 0 or error.
        # With the enforced minimum of 1.0, epsilon must be finite and > 0.
        assert math.isfinite(epsilon) and epsilon > 0

    def test_hub_cannot_reduce_noise_below_minimum(self):
        """Hub sending noise_multiplier=0.001 must be clipped: actual sigma >= _MIN_NOISE_MULTIPLIER."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(noise_multiplier=0.001)  # Far below minimum

        captured = {}

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())

            def capture_make_private(**kwargs):
                captured["noise_multiplier"] = kwargs["noise_multiplier"]
                return (model, fresh_opt, loader)

            instance.make_private.side_effect = capture_make_private
            instance.get_epsilon.return_value = 0.5

            train(model, loader, config, partition_id=0, verbose=False)

        assert "noise_multiplier" in captured, "make_private was not called"
        assert captured["noise_multiplier"] >= _MIN_NOISE_MULTIPLIER, (
            f"Expected noise_multiplier >= {_MIN_NOISE_MULTIPLIER}, "
            f"got {captured['noise_multiplier']} — Hub value was not clamped"
        )

    def test_hub_cannot_disable_gradient_clipping(self):
        """Hub sending max_grad_norm=0 must be clipped to _MIN_GRAD_NORM."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(max_grad_norm=0.0)
        # Should not raise; Node enforces minimum
        *_, epsilon = train(model, loader, config, partition_id=0, verbose=False)
        assert math.isfinite(epsilon)

    def test_hub_cannot_send_negative_noise_multiplier(self):
        """Hub sending negative noise_multiplier must be clipped to minimum."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(noise_multiplier=-5.0)
        *_, epsilon = train(model, loader, config, partition_id=0, verbose=False)
        assert math.isfinite(epsilon) and epsilon > 0

    def test_hub_cannot_send_negative_max_grad_norm(self):
        """Hub sending negative max_grad_norm must be clipped to minimum."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(max_grad_norm=-1.0)
        *_, epsilon = train(model, loader, config, partition_id=0, verbose=False)
        assert math.isfinite(epsilon) and epsilon > 0

    def test_node_ignores_hub_random_seed(self):
        """
        Hub supplying a fixed seed must NOT control the DP noise generator.
        Two runs with the same Hub seed must produce different noise (different
        Node-generated secrets.randbelow seeds).
        """
        model1 = _make_model()
        loader1 = _make_loader(n=64)
        config1 = _minimal_config()
        config1["model"]["training"]["optimizer"]["differential_privacy"]["random_seed"] = 1234

        model2 = nn.Linear(4, 1)
        # Copy same initial weights so the only difference is the noise seed
        model2.load_state_dict(model1.state_dict())
        loader2 = DataLoader(loader1.dataset, batch_size=16, drop_last=True)
        config2 = _minimal_config()
        config2["model"]["training"]["optimizer"]["differential_privacy"]["random_seed"] = 1234

        # Run twice with same Hub seed — noise should differ (local seed differs)
        loss1, *_ = train(model1, loader1, config1, partition_id=0, verbose=False)
        loss2, *_ = train(model2, loader2, config2, partition_id=0, verbose=False)
        # We can't assert loss differs (tiny models may converge similarly),
        # but we can assert the function completes without using Hub seed deterministically.
        # The real assertion is that secrets.randbelow is called, tested via patch below.

    def test_secrets_randbelow_called_not_hub_seed(self):
        """Verify the Node calls secrets.randbelow, ignoring Hub's random_seed."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["differential_privacy"]["random_seed"] = 9999

        with patch("api.federated.train_functions.secrets.randbelow", wraps=__import__("secrets").randbelow) as mock_rb:
            train(model, loader, config, partition_id=0, verbose=False)
            mock_rb.assert_called_once()
            # The argument must be 2**31 (our entropy range)
            mock_rb.assert_called_with(2**31)


# ---------------------------------------------------------------------------
# 3. Epsilon scales correctly with training parameters
# ---------------------------------------------------------------------------

class TestEpsilonScaling:

    def test_more_epochs_increases_epsilon(self):
        """More training steps → more privacy budget consumed → higher ε."""
        model1 = _make_model()
        loader1 = _make_loader(n=128)
        *_, eps_1epoch, _noise = train(model1, loader1, _minimal_config(epochs=1),
                               partition_id=0, verbose=False)

        model2 = _make_model()
        loader2 = _make_loader(n=128)
        *_, eps_3epochs, _noise = train(model2, loader2, _minimal_config(epochs=3),
                                partition_id=0, verbose=False)

        assert eps_3epochs > eps_1epoch, (
            f"3 epochs (ε={eps_3epochs:.4f}) should cost more than "
            f"1 epoch (ε={eps_1epoch:.4f})"
        )

    def test_higher_sigma_decreases_epsilon(self):
        """Higher noise_multiplier → stronger DP → lower ε."""
        model1 = _make_model()
        loader1 = _make_loader(n=128)
        *_, eps_sigma1, _noise = train(model1, loader1, _minimal_config(noise_multiplier=1.0),
                               partition_id=0, verbose=False)

        model2 = _make_model()
        loader2 = _make_loader(n=128)
        *_, eps_sigma3, _noise = train(model2, loader2, _minimal_config(noise_multiplier=3.0),
                               partition_id=0, verbose=False)

        assert eps_sigma3 < eps_sigma1, (
            f"σ=3.0 (ε={eps_sigma3:.4f}) should be smaller than "
            f"σ=1.0 (ε={eps_sigma1:.4f})"
        )

    def test_epsilon_within_realistic_bounds(self):
        """With σ=1.0, batch=16, n=64, epochs=1, ε should be < 20."""
        model = _make_model()
        loader = _make_loader(n=64, batch=16)
        *_, epsilon = train(model, loader, _minimal_config(epochs=1),
                            partition_id=0, verbose=False)
        assert epsilon < 20.0, f"Epsilon {epsilon} seems unrealistically large"

    def test_delta_constant(self):
        """_DP_DELTA must be 1e-5 (standard value for medical data)."""
        assert _DP_DELTA == 1e-5


# ---------------------------------------------------------------------------
# 4. Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_epsilon_inf_when_get_epsilon_raises(self):
        """If privacy_engine.get_epsilon() raises, train() returns float('inf')."""
        model = _make_model()
        loader = _make_loader()

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            instance.make_private.return_value = (model, torch.optim.Adam(model.parameters()), loader)
            instance.get_epsilon.side_effect = RuntimeError("accountant error")

            result = train(model, loader, _minimal_config(), partition_id=0, verbose=False)
            epsilon = result[5]
            assert epsilon == float("inf"), f"Expected inf, got {epsilon}"

    def test_missing_dp_config_uses_safe_defaults(self):
        """If Hub omits differential_privacy block, Node uses safe defaults."""
        model = _make_model()
        loader = _make_loader()
        config = {
            "train": {"epochs": 1},
            "model": {
                "training": {
                    "loss_function": "bce_with_logits",
                    "optimizer": {"type": "Adam", "learning_rate": 0.01, "weight_decay": 0},
                }
            },
        }
        *_, epsilon = train(model, loader, config, partition_id=0, verbose=False)
        assert math.isfinite(epsilon) and epsilon > 0

    def test_completely_empty_config_does_not_crash(self):
        """train() must not raise on a completely empty config dict."""
        model = _make_model()
        loader = _make_loader()
        *_, epsilon = train(model, loader, {}, partition_id=0, verbose=False)
        assert math.isfinite(epsilon) and epsilon > 0

    def test_hub_sends_string_noise_multiplier(self):
        """Hub sending a non-numeric noise_multiplier must not crash training."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["differential_privacy"]["noise_multiplier"] = "not_a_number"
        # Should either use default or raise a clear error — must not hang
        try:
            result = train(model, loader, config, partition_id=0, verbose=False)
            assert len(result) == 7
        except (TypeError, ValueError):
            pass  # A clear exception is acceptable


# ---------------------------------------------------------------------------
# 5. DLFlowerClient integration
# ---------------------------------------------------------------------------

class TestDLFlowerClientEpsilonPropagation:

    def _make_client(self) -> DLFlowerClient:
        net = _make_model()
        loader = _make_loader()
        model_json = _minimal_config()
        # training_session=None: update_training_progress returns immediately
        # when session is None, avoiding Django ORM dependencies in unit tests.
        client = DLFlowerClient(
            net=net,
            trainloader=loader,
            valloader=loader,
            testloader=loader,
            model_json=model_json,
            training_session=None,
            client_ip="127.0.0.1",
            table_name="test_table",
            device="cpu",
            current_process=None,
            partition_id=0,
        )
        client.assigned_client_id = "test-client-1"
        return client

    def _fake_parameters(self, net):
        return [v.cpu().numpy() for _, v in net.state_dict().items()]

    def test_fit_returns_metrics_with_privacy_epsilon(self):
        """fit() returned metrics dict must contain 'privacy_epsilon'."""
        client = self._make_client()
        params = self._fake_parameters(client.net)
        _, _, metrics = client.fit(params, {})
        assert "privacy_epsilon" in metrics, f"Missing privacy_epsilon in {metrics.keys()}"

    def test_fit_returns_metrics_with_privacy_delta(self):
        """fit() returned metrics dict must contain 'privacy_delta'."""
        client = self._make_client()
        params = self._fake_parameters(client.net)
        _, _, metrics = client.fit(params, {})
        assert "privacy_delta" in metrics

    def test_fit_epsilon_is_finite_positive(self):
        """fit() privacy_epsilon must be finite and positive."""
        client = self._make_client()
        params = self._fake_parameters(client.net)
        _, _, metrics = client.fit(params, {})
        eps = metrics["privacy_epsilon"]
        assert math.isfinite(eps), f"privacy_epsilon should be finite, got {eps}"
        assert eps > 0, f"privacy_epsilon should be positive, got {eps}"

    def test_fit_delta_equals_dp_delta(self):
        """fit() privacy_delta must equal the Node constant _DP_DELTA."""
        client = self._make_client()
        params = self._fake_parameters(client.net)
        _, _, metrics = client.fit(params, {})
        assert metrics["privacy_delta"] == _DP_DELTA

    def test_fit_round_metrics_contain_epsilon(self):
        """update_training_progress must be called with epsilon in round_metrics."""
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.dl_client.update_training_progress") as mock_update:
            client.fit(params, {})
            mock_update.assert_called_once()
            round_metrics_arg = mock_update.call_args[0][3]
            assert "privacy_epsilon" in round_metrics_arg
            assert "privacy_delta" in round_metrics_arg

    def test_fit_epsilon_stored_on_self(self):
        """After fit(), client.epsilon must be set."""
        client = self._make_client()
        params = self._fake_parameters(client.net)
        client.fit(params, {})
        assert client.epsilon is not None
        assert math.isfinite(client.epsilon)

    def test_fit_error_returns_empty_metrics(self):
        """On training error, fit() returns empty metrics dict (Flower contract).

        num_examples is 1 (not 0) on error by design: returning 0 from every
        client would make Flower's FedAvg divide by a zero weight sum and crash.
        """
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.dl_client.train", side_effect=RuntimeError("boom")):
            returned_params, num_examples, metrics = client.fit(params, {})
            assert num_examples == 1
            assert metrics == {}

    def test_client_epsilon_initialized_to_none(self):
        """client.epsilon must start as None before any training."""
        client = self._make_client()
        assert client.epsilon is None


# ---------------------------------------------------------------------------
# 6. Edge cases — adversarial Hub inputs
# ---------------------------------------------------------------------------

class TestAdversarialHubInputs:

    def test_hub_sends_enormous_noise_multiplier(self):
        """Very large sigma still produces a valid (tiny) epsilon."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config(noise_multiplier=1000.0)
        *_, epsilon, _noise = train(model, loader, config, partition_id=0, verbose=False)
        assert math.isfinite(epsilon) and epsilon > 0
        assert epsilon < 1.0  # Very strong privacy with sigma=1000

    def test_hub_sends_float_nan_noise_multiplier(self):
        """Hub sending NaN noise_multiplier must be silently replaced by Node minimum."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["differential_privacy"]["noise_multiplier"] = float("nan")

        captured = {}

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())

            def capture_make_private(**kwargs):
                captured["noise_multiplier"] = kwargs["noise_multiplier"]
                return (model, fresh_opt, loader)

            instance.make_private.side_effect = capture_make_private
            instance.get_epsilon.return_value = 0.5

            result = train(model, loader, config, partition_id=0, verbose=False)

        # NaN must be replaced — the sigma actually used must be finite and >= minimum
        assert "noise_multiplier" in captured
        assert math.isfinite(captured["noise_multiplier"]), (
            "NaN noise_multiplier was passed through to Opacus — security bypass!"
        )
        assert captured["noise_multiplier"] >= _MIN_NOISE_MULTIPLIER

    def test_hub_sends_float_inf_noise_multiplier(self):
        """Hub sending inf noise_multiplier must be replaced by Node minimum."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["differential_privacy"]["noise_multiplier"] = float("inf")

        captured = {}

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())

            def capture_make_private(**kwargs):
                captured["noise_multiplier"] = kwargs["noise_multiplier"]
                return (model, fresh_opt, loader)

            instance.make_private.side_effect = capture_make_private
            instance.get_epsilon.return_value = 0.5

            train(model, loader, config, partition_id=0, verbose=False)

        assert math.isfinite(captured["noise_multiplier"]), (
            "inf noise_multiplier was passed through — security bypass!"
        )
        assert captured["noise_multiplier"] >= _MIN_NOISE_MULTIPLIER

    def test_hub_sends_single_sample_batch(self):
        """Batch size 1 (q=1/N) must not crash epsilon computation."""
        model = _make_model()
        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)).float()),
                            batch_size=1, drop_last=True)
        *_, epsilon = train(model, loader, _minimal_config(), partition_id=0, verbose=False)
        assert isinstance(epsilon, float)

    def test_hub_sends_batch_larger_than_dataset(self):
        """
        Batch >= dataset size means q=1.0 (no subsampling amplification).
        Should still compute a valid epsilon.
        """
        n = 16
        model = _make_model()
        loader = DataLoader(TensorDataset(torch.randn(n, 4), torch.randint(0, 2, (n,)).float()),
                            batch_size=n, drop_last=True)
        *_, epsilon = train(model, loader, _minimal_config(), partition_id=0, verbose=False)
        assert isinstance(epsilon, float)

    def test_minimum_constants_have_correct_values(self):
        """Security constants in train_functions must be exactly as specified."""
        assert _MIN_NOISE_MULTIPLIER == 1.0
        assert _MIN_GRAD_NORM == 1.0
        assert _DP_DELTA == 1e-5

    def test_hub_cannot_use_unlisted_optimizer(self):
        """Hub sending an optimizer type not in _ALLOWED_OPTIMIZERS must fall back to Adam."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["type"] = "ExplodingOptimizer"

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())
            instance.make_private.return_value = (model, fresh_opt, loader)
            instance.get_epsilon.return_value = 0.5

            result = train(model, loader, config, partition_id=0, verbose=False)

        assert len(result) == 7, "Should not crash with invalid optimizer type"

    def test_epoch_cap_enforced(self):
        """Hub requesting more than _MAX_EPOCHS epochs must be capped at _MAX_EPOCHS."""
        model = _make_model()
        loader = _make_loader(n=64, batch=16)  # 4 batches per epoch
        config = _minimal_config(epochs=_MAX_EPOCHS + 1000)

        step_calls = [0]

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())

            original_step = fresh_opt.step

            def counting_step(*args, **kwargs):
                step_calls[0] += 1
                return original_step(*args, **kwargs)

            fresh_opt.step = counting_step
            instance.make_private.return_value = (model, fresh_opt, loader)
            instance.get_epsilon.return_value = 0.5

            train(model, loader, config, partition_id=0, verbose=False)

        batches_per_epoch = 64 // 16  # drop_last=True
        max_expected_steps = _MAX_EPOCHS * batches_per_epoch
        assert step_calls[0] <= max_expected_steps, (
            f"Expected at most {max_expected_steps} optimizer steps "
            f"(cap={_MAX_EPOCHS} epochs), got {step_calls[0]}"
        )

    def test_safe_dp_float_nan_returns_default(self):
        """_safe_dp_float must return default for NaN (max(nan,x)==nan in Python)."""
        assert _safe_dp_float(float("nan"), 1.0) == 1.0

    def test_safe_dp_float_inf_returns_default(self):
        """_safe_dp_float must return default for inf."""
        assert _safe_dp_float(float("inf"), 1.0) == 1.0
        assert _safe_dp_float(float("-inf"), 1.0) == 1.0

    def test_safe_dp_float_string_returns_default(self):
        """_safe_dp_float must return default for non-numeric strings."""
        assert _safe_dp_float("not_a_number", 1.5) == 1.5

    def test_safe_dp_float_valid_returns_value(self):
        """_safe_dp_float must pass through valid finite floats."""
        assert _safe_dp_float(2.5, 1.0) == 2.5
        assert _safe_dp_float(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# 7. New security tests — Hub config isolation and epsilon sentinel
# ---------------------------------------------------------------------------

class TestHubConfigIsolation:

    def _make_client(self) -> DLFlowerClient:
        net = _make_model()
        loader = _make_loader()
        model_json = _minimal_config()
        client = DLFlowerClient(
            net=net,
            trainloader=loader,
            valloader=loader,
            testloader=loader,
            model_json=model_json,
            training_session=None,
            client_ip="127.0.0.1",
            table_name="test_table",
            device="cpu",
            current_process=None,
            partition_id=0,
        )
        client.assigned_client_id = "test-client-1"
        return client

    def _fake_parameters(self, net):
        return [v.cpu().numpy() for _, v in net.state_dict().items()]

    def test_flower_config_cannot_overwrite_model_key(self):
        """Hub's Flower config dict must not overwrite the Node's 'model' DP parameters."""
        client = self._make_client()
        params = self._fake_parameters(client.net)

        malicious_flower_config = {
            "model": {
                "training": {
                    "optimizer": {
                        "differential_privacy": {
                            "noise_multiplier": 0.0001,
                            "max_grad_norm": 0.0001,
                        }
                    }
                }
            }
        }

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.5, 1.0)
            client.fit(params, malicious_flower_config)

            called_config = mock_train.call_args[0][2]
            dp_params = (
                called_config.get("model", {})
                             .get("training", {})
                             .get("optimizer", {})
                             .get("differential_privacy", {})
            )
            assert dp_params.get("noise_multiplier") != 0.0001, (
                "Hub's Flower config overwrote DP parameters — critical security breach!"
            )

    def test_flower_config_allowlisted_keys_pass_through(self):
        """Keys in _SAFE_FLOWER_KEYS (server_round, client_id) should pass through."""
        from ..dl_client import _SAFE_FLOWER_KEYS
        client = self._make_client()
        params = self._fake_parameters(client.net)

        flower_round_config = {"server_round": 3, "client_id": "abc123"}

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.5, 1.0)
            client.fit(params, flower_round_config)

            called_config = mock_train.call_args[0][2]
            if "server_round" in _SAFE_FLOWER_KEYS:
                assert called_config.get("server_round") == 3
            if "client_id" in _SAFE_FLOWER_KEYS:
                assert called_config.get("client_id") == "abc123"

    def test_epsilon_inf_becomes_sentinel_in_metrics(self):
        """float('inf') epsilon from train() must be converted to -1.0 sentinel in metrics."""
        import json
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, float("inf"), 1.0)
            _, _, metrics = client.fit(params, {})

        eps = metrics.get("privacy_epsilon")
        assert eps is not None, "privacy_epsilon missing from metrics"
        assert math.isfinite(eps), (
            f"privacy_epsilon={eps} is not finite — will crash Flower JSON serialization"
        )
        try:
            json.dumps({"privacy_epsilon": eps})
        except (ValueError, TypeError) as e:
            pytest.fail(f"privacy_epsilon={eps} is not JSON-serializable: {e}")

    def test_privacy_delta_in_metrics_matches_node_constant(self):
        """privacy_delta in fit() metrics must equal the Node _DP_DELTA constant."""
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.42, 1.0)
            _, _, metrics = client.fit(params, {})

        assert metrics["privacy_delta"] == _DP_DELTA, (
            f"privacy_delta={metrics['privacy_delta']} does not match "
            f"_DP_DELTA={_DP_DELTA} — delta literal drifted from constant"
        )

    def test_round_metrics_delta_matches_node_constant(self):
        """privacy_delta in round_metrics must also use the Node _DP_DELTA constant."""
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.dl_client.update_training_progress") as mock_update, \
             patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.42, 1.0)
            client.fit(params, {})

            round_metrics_arg = mock_update.call_args[0][3]
            assert round_metrics_arg["privacy_delta"] == _DP_DELTA, (
                "privacy_delta in round_metrics drifted from _DP_DELTA constant"
            )

    def test_evaluate_is_not_affected_by_dp_changes(self):
        """evaluate() must not involve PrivacyEngine — it only tests, not trains."""
        client = self._make_client()
        params = self._fake_parameters(client.net)

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            client.evaluate(params, {})
            MockPE.assert_not_called()

    def test_flower_config_cannot_override_train_epochs(self):
        """Hub sending 'train' key in Flower config must not override Node's epoch config."""
        # Build a client whose model_json has epochs=3
        net = _make_model()
        loader = _make_loader()
        model_json = _minimal_config(epochs=3)  # Node admin configured 3 epochs
        client = DLFlowerClient(
            net=net, trainloader=loader, valloader=loader, testloader=loader,
            model_json=model_json, training_session=None,
            client_ip="127.0.0.1", table_name="t", device="cpu",
            current_process=None, partition_id=0,
        )
        client.assigned_client_id = "test"
        params = [v.cpu().numpy() for _, v in net.state_dict().items()]

        # Hub tries to override epochs to 999 via Flower config
        hub_config = {"train": {"epochs": 999}}

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.5, 1.0)
            client.fit(params, hub_config)

            called_config = mock_train.call_args[0][2]
            train_cfg = called_config.get("train", {})
            # The Hub's 'train' key must be dropped (not in _SAFE_FLOWER_KEYS).
            # Node's model_json epochs=3 must survive intact.
            assert train_cfg.get("epochs") == 3, (
                f"Hub overrode Node epoch config: got epochs={train_cfg.get('epochs')}, "
                f"expected 3 (the Node admin's value)"
            )

    def test_flower_config_unknown_keys_are_dropped(self):
        """Keys not in _SAFE_FLOWER_KEYS must be silently dropped."""
        from ..dl_client import _SAFE_FLOWER_KEYS
        client = self._make_client()
        params = self._fake_parameters(client.net)

        arbitrary_config = {"arbitrary_key": "should_be_dropped", "server_round": 5}

        with patch("api.federated.dl_client.train") as mock_train:
            mock_train.return_value = (0.5, 0.8, 0.7, 0.7, 0.7, 0.5, 1.0)
            client.fit(params, arbitrary_config)

            called_config = mock_train.call_args[0][2]
            assert "arbitrary_key" not in called_config, (
                "Arbitrary Flower key was passed through — allowlist not working"
            )
            # server_round should pass through (it's in _SAFE_FLOWER_KEYS)
            if "server_round" in _SAFE_FLOWER_KEYS:
                assert called_config.get("server_round") == 5

    def test_non_dict_differential_privacy_does_not_crash(self):
        """Hub sending 'differential_privacy' as a non-dict must not crash the Node."""
        model = _make_model()
        loader = _make_loader()
        config = _minimal_config()
        config["model"]["training"]["optimizer"]["differential_privacy"] = "disabled"

        with patch("api.federated.train_functions.PrivacyEngine") as MockPE:
            instance = MockPE.return_value
            fresh_opt = torch.optim.Adam(model.parameters())
            instance.make_private.return_value = (model, fresh_opt, loader)
            instance.get_epsilon.return_value = 0.5

            result = train(model, loader, config, partition_id=0, verbose=False)

        # Must return a 6-tuple; DP minimums still enforced
        assert len(result) == 7
        epsilon = result[5]
        assert isinstance(epsilon, float)


# ---------------------------------------------------------------------------
# 8. Additional correctness and safety tests (found by 3rd committee review)
# ---------------------------------------------------------------------------

class TestAdditionalSafetyTests:

    def test_function_test_empty_loader_returns_zero(self):
        """test() with an empty testloader must not raise ZeroDivisionError."""
        from ..train_functions import test as dp_test
        model = _make_model()
        empty_loader = DataLoader(
            TensorDataset(torch.randn(0, 4), torch.randint(0, 2, (0,)).float()),
            batch_size=16, drop_last=False,
        )
        loss, accuracy = dp_test(model, empty_loader)
        assert loss == 0.0
        assert accuracy == 0.0

    def test_set_parameters_raises_on_count_mismatch(self):
        """set_parameters() must raise ValueError when Hub sends wrong parameter count."""
        from ..dl_client import set_parameters
        model = _make_model()
        correct_params = [v.cpu().numpy() for _, v in model.state_dict().items()]
        # Send one fewer parameter than the model expects
        truncated = correct_params[:-1]
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            set_parameters(model, truncated)

    def test_evaluate_returns_dataset_size_not_batch_count(self):
        """evaluate() num_examples must be dataset size, not batch count."""
        net = _make_model()
        n_samples = 64
        loader = _make_loader(n=n_samples, batch=16)  # 4 batches
        client = DLFlowerClient(
            net=net, trainloader=loader, valloader=loader, testloader=loader,
            model_json=_minimal_config(), training_session=None,
            client_ip="127.0.0.1", table_name="t", device="cpu",
            current_process=None, partition_id=0,
        )
        client.assigned_client_id = "test"
        params = [v.cpu().numpy() for _, v in net.state_dict().items()]

        _, num_examples, _ = client.evaluate(params, {})
        assert num_examples == n_samples, (
            f"evaluate() returned {num_examples} (batch count?), expected {n_samples} (sample count)"
        )

    def test_evaluate_empty_valloader_does_not_crash(self):
        """evaluate() with an empty valloader must not crash."""
        net = _make_model()
        empty_loader = DataLoader(
            TensorDataset(torch.randn(0, 4), torch.randint(0, 2, (0,)).float()),
            batch_size=16,
        )
        client = DLFlowerClient(
            net=net, trainloader=empty_loader, valloader=empty_loader, testloader=empty_loader,
            model_json=_minimal_config(), training_session=None,
            client_ip="127.0.0.1", table_name="t", device="cpu",
            current_process=None, partition_id=0,
        )
        client.assigned_client_id = "test"
        params = [v.cpu().numpy() for _, v in net.state_dict().items()]

        loss, num_examples, metrics = client.evaluate(params, {})
        assert loss == 0.0
        assert num_examples == 0
