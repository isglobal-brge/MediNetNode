"""
FedSVM Algorithm — differentially private federated SVM.

PRIVACY REDESIGN (2026-07-04): the previous implementation shared clients'
support vectors (near-raw patient rows) across hospitals with only a utility-
preserving displacement and NO calibrated noise, while reporting a false epsilon.
That exported real patient records with zero differential privacy.

This version is a genuine (ε,δ)-DP federated SVM:
  * Each client maps its features through a SHARED Random Fourier Feature map
    (fixed seed → the feature space is identical across hospitals), approximating
    the RBF kernel.
  * A linear soft-margin SVM (hinge loss) is trained on those features with
    DP-SGD (Opacus): per-sample gradient clipping + calibrated Gaussian noise.
  * Only the aggregated MODEL WEIGHTS cross the network (FedAvg on the Hub) —
    never data points — and the epsilon reported is the REAL value from the
    Opacus accountant, so the budget accounting is truthful.
"""

import collections
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_approximation import RBFSampler

from .base import FederatedMLAlgorithm
from ..train_functions import (
    _MIN_NOISE_MULTIPLIER, _DP_DELTA, _MAX_EPOCHS, _TRAINING_BATCH_SIZE,
    _safe_dp_float,
)
from .fedsvm_core import evaluate_fedsvm

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPACUS_AVAILABLE = False

# The RFF map must be byte-for-byte identical on every hospital or the shared
# weights are meaningless. A fixed seed guarantees that (RBFSampler only samples
# random projection weights; it does not use the data).
_RFF_SEED = 42


class FedSVMAlgorithm(FederatedMLAlgorithm):
    """Differentially private federated SVM (RFF + DP-SGD, weight-sharing)."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any],
                 X_val: np.ndarray = None, y_val: np.ndarray = None):
        super().__init__(X_train, y_train, config, X_val, y_val)

        tcfg = config.get('training', {}) or {}
        kcfg = tcfg.get('kernel_config', {}) or {}
        # RFF gamma: use the data-scale heuristic (sklearn's 'scale': 1/(n_features·var)).
        # The RBFSampler gamma is very scale-sensitive and does NOT match the SVC-kernel
        # gamma a user might set, so we derive it from the (standardised) data.
        n_features = int(np.asarray(X_train).shape[1])
        _var = float(np.asarray(X_train, dtype=np.float64).var()) or 1.0
        self.gamma = float(kcfg.get('gamma_rff') or (1.0 / (n_features * _var)))
        self.n_components = int(kcfg.get('n_components', 500))
        self.C = float(tcfg.get('C', 1.0))
        self.local_epochs = min(max(int(tcfg.get('local_epochs', 5)), 1), _MAX_EPOCHS)

        # DP parameters, floored to the Node minimums (the Hub is untrusted and
        # cannot weaken privacy below these).
        dp = (tcfg.get('differential_privacy')
              or tcfg.get('optimizer', {}).get('differential_privacy')
              or {})
        self.noise_multiplier = max(
            _safe_dp_float(dp.get('noise_multiplier'), _MIN_NOISE_MULTIPLIER),
            _MIN_NOISE_MULTIPLIER,
        )
        self.max_grad_norm = _safe_dp_float(dp.get('max_grad_norm'), 1.0)
        self.device = torch.device('cpu')

        # Shared RFF feature map (RBF approximation).
        self.rff = RBFSampler(gamma=self.gamma, n_components=self.n_components,
                              random_state=_RFF_SEED)
        self.rff.fit(X_train)
        self.Z_train = self.rff.transform(X_train).astype(np.float32)

        self.n_classes = max(int(np.max(y_train)) + 1, 2)
        self.model = nn.Linear(self.n_components, self.n_classes).to(self.device)

        cnt = collections.Counter(int(v) for v in y_train)
        total = len(y_train)
        weights = [total / (self.n_classes * max(cnt.get(c, 0), 1))
                   for c in range(self.n_classes)]
        self._class_weight = torch.tensor(weights, dtype=torch.float32, device=self.device)

        print(f"[FEDSVM-DP] RFF dim={self.n_components} gamma={self.gamma} "
              f"classes={self.n_classes} sigma={self.noise_multiplier} C={self.C}")

    # ── Parameter (weight) exchange ──────────────────────────────────────────
    def get_parameters(self) -> List[np.ndarray]:
        return [self.model.weight.detach().cpu().numpy(),
                self.model.bias.detach().cpu().numpy()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        if not parameters or np.asarray(parameters[0]).size == 0:
            return
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(
                np.asarray(parameters[0], dtype=np.float32)).reshape(self.model.weight.shape))
            if len(parameters) > 1 and np.asarray(parameters[1]).size:
                self.model.bias.copy_(torch.tensor(
                    np.asarray(parameters[1], dtype=np.float32)).reshape(self.model.bias.shape))

    def _loader(self):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(self.Z_train, dtype=torch.float32),
            torch.tensor(np.asarray(self.y_train), dtype=torch.long))
        bs = min(_TRAINING_BATCH_SIZE, len(self.Z_train))
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    # ── One DP-SGD local round ───────────────────────────────────────────────
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        if not OPACUS_AVAILABLE:
            raise RuntimeError("Opacus not available — FedSVM requires DP-SGD")

        self.set_parameters(parameters)
        loader = self._loader()
        # Train on a FRESH copy each round: Opacus attaches grad-sample hooks to
        # the module in place, so re-using self.model would double-hook. self.model
        # stays the clean canonical holder of the weights.
        train_model = nn.Linear(self.n_components, self.n_classes).to(self.device)
        train_model.load_state_dict(self.model.state_dict())
        train_model.train()  # Opacus requires train mode before make_private
        optimizer = torch.optim.Adam(train_model.parameters(), lr=0.01)
        # Soft-margin SVM: multiclass hinge loss, class-weighted for imbalance.
        criterion = nn.MultiMarginLoss(weight=self._class_weight)

        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, loader = privacy_engine.make_private(
            module=train_model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm)
        total_loss, n_batches = 0.0, 0
        for _ in range(self.local_epochs):
            for Xb, yb in loader:
                if Xb.shape[0] == 0:  # Opacus Poisson sampling can yield empty batches
                    continue
                optimizer.zero_grad()
                loss = self.C * criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1

        # Copy DP-trained weights back into the canonical (unhooked) model.
        with torch.no_grad():
            self.model.weight.copy_(train_model.weight.detach())
            self.model.bias.copy_(train_model.bias.detach())

        try:
            epsilon = float(privacy_engine.get_epsilon(delta=_DP_DELTA))
            if not np.isfinite(epsilon):
                epsilon = -1.0
        except Exception as exc:
            print(f"[FEDSVM-DP] epsilon accounting failed: {exc}")
            epsilon = -1.0

        metrics = self._metrics_on_train()
        metrics['loss'] = (total_loss / n_batches) if n_batches else float('inf')
        metrics['privacy_epsilon'] = epsilon
        print(f"[FEDSVM-DP] round done: acc={metrics['accuracy']:.4f} eps={epsilon:.4f}")
        return self.get_parameters(), metrics

    # ── Evaluation / prediction ──────────────────────────────────────────────
    def _predict_Z(self, Z: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(Z, dtype=torch.float32))
            return logits.argmax(dim=1).cpu().numpy()

    def _metrics_on_train(self) -> Dict[str, float]:
        return evaluate_fedsvm(np.asarray(self.y_train), self._predict_Z(self.Z_train))

    def evaluate(self, parameters: List[np.ndarray], X_val: np.ndarray,
                 y_val: np.ndarray) -> Tuple[float, float]:
        self.set_parameters(parameters)
        Zv = self.rff.transform(X_val).astype(np.float32)
        m = evaluate_fedsvm(np.asarray(y_val), self._predict_Z(Zv))
        return 1.0 - m['accuracy'], m['accuracy']

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_Z(self.rff.transform(X).astype(np.float32))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "algorithm": "FedSVM-DP",
            "n_train_samples": len(self.X_train),
            "n_features": self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1,
            "kernel": "rbf(rff)", "n_components": self.n_components, "gamma": self.gamma,
            "differentially_private": True, "noise_multiplier": self.noise_multiplier,
            "config": self.config,
        }
