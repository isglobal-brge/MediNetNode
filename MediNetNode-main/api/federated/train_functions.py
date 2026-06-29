
import math
import secrets
import time
import torch
import numpy as np
from . import utils
from opacus import PrivacyEngine

# Conservative fallback used when the dynamic RDP calculation cannot run
# (e.g. Opacus not available, or dataset size unknown).  The actual floor
# for each job is computed analytically by compute_min_noise_multiplier().
_MIN_NOISE_MULTIPLIER = 1.0
_MIN_GRAD_NORM = 1.0
_DP_DELTA = 1e-5
# Hard epoch cap: Hub cannot exhaust the privacy budget in a single call.
_MAX_EPOCHS = 50
# Node-fixed training batch size. The Hub cannot control this value because a
# smaller batch_size lowers the DP sample_rate, which understates estimated ε
# and makes the pre-flight budget check easier to pass.  Both the epsilon
# estimator (api/views.py) and the Flower client (client.py) must import this
# constant so they always use the same value.
_TRAINING_BATCH_SIZE = 32

# Only these PyTorch optimizer types are allowed. Hub-controlled strings must
# not be passed blindly to getattr(torch.optim, ...).
_ALLOWED_OPTIMIZERS = frozenset({"Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "AdamW"})

# Loss mode classification — drives output shape, targets dtype, and metric
# computation throughout train() and test().
_BINARY_LOSSES = frozenset({"bce", "bce_with_logits"})
_MULTICLASS_LOSSES = frozenset({"cross_entropy", "nll", "nll_loss"})
_REGRESSION_LOSSES = frozenset({"mse", "mae", "l1"})


def compute_min_noise_multiplier(
    n: int,
    batch_size: int,
    epochs: int,
    target_epsilon: float,
    delta: float = _DP_DELTA,
) -> float:
    """Return the minimum σ such that DP-SGD consumes at most *target_epsilon*.

    Uses the RDP accountant (same as Opacus at runtime) to derive the floor
    analytically from the actual job parameters rather than relying on an
    arbitrary constant.  This means the floor adapts to dataset size: a larger
    hospital with more records can afford a lower σ for the same ε budget.

    Falls back to ``_MIN_NOISE_MULTIPLIER`` if the computation fails (e.g.
    Opacus unavailable, non-positive inputs).

    Args:
        n:                Number of training samples.
        batch_size:       Mini-batch size used during training.
        epochs:           Number of local epochs per federated round.
        target_epsilon:   Maximum ε budget the job may consume (per_job_max).
        delta:            DP delta (default: 1e-5).

    Returns:
        Minimum noise multiplier σ ≥ 0.01.
    """
    try:
        if n <= 0 or batch_size <= 0 or epochs <= 0 or target_epsilon <= 0:
            return _MIN_NOISE_MULTIPLIER
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(delta),
            sample_rate=batch_size / n,
            epochs=epochs,
        )
        # Clamp to a physically meaningful range — values outside [0.01, 10.0]
        # indicate degenerate inputs and should fall back to the constant.
        return float(max(0.01, min(10.0, sigma)))
    except Exception as exc:
        print(f"[DP] compute_min_noise_multiplier failed ({exc}); using fallback {_MIN_NOISE_MULTIPLIER}")
        return _MIN_NOISE_MULTIPLIER


def _get_loss_mode(loss_function: str) -> str:
    """
    Return one of 'binary', 'multiclass', or 'regression' based on the
    loss function name coming from the Hub model config.
    Defaults to 'binary' so existing deployments keep working.
    """
    lf = loss_function.lower().replace(" ", "_")
    if lf in _MULTICLASS_LOSSES:
        return "multiclass"
    if lf in _REGRESSION_LOSSES:
        return "regression"
    return "binary"


def _safe_dp_float(value, default: float) -> float:
    """
    Convert value to float, returning default if NaN, inf, or non-numeric.

    max(float('nan'), 1.0) == nan in Python — callers must not rely on max()
    alone to reject NaN. This helper performs the isfinite guard first.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _build_weighted_criterion(criterion_class, loss_mode: str, trainloader, device: str):
    """
    Build a loss criterion with class-frequency weights to handle imbalanced data.

    For *binary* tasks (BCEWithLogitsLoss) we compute ``pos_weight = n_neg / n_pos``
    which up-weights the minority class in the gradient signal.

    For *multiclass* tasks (CrossEntropyLoss) we compute per-class weights using
    the inverse-frequency formula: ``w_c = total / (n_classes * count_c)``.

    For *regression* tasks no weighting is applied.

    If weight computation fails for any reason (empty loader, all-same-class, …)
    the function falls back to an unweighted criterion rather than crashing.
    """
    if loss_mode == "regression":
        return criterion_class()

    try:
        # Collect all targets from the loader without running any forward pass
        all_targets: list = []
        for _, targets in trainloader:
            all_targets.extend(targets.numpy().flatten().tolist())

        if not all_targets:
            return criterion_class()

        target_arr = np.array(all_targets)

        if loss_mode == "binary":
            n_pos = float((target_arr == 1).sum())
            n_neg = float((target_arr == 0).sum())
            if n_pos == 0 or n_neg == 0:
                print("[IMBALANCE] All targets are the same class — skipping pos_weight")
                return criterion_class()
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
            print(f"[IMBALANCE] Binary: pos_weight={pos_weight.item():.3f} "
                  f"(n_pos={int(n_pos)}, n_neg={int(n_neg)})")
            return criterion_class(pos_weight=pos_weight)

        else:  # multiclass
            classes = np.unique(target_arr.astype(int))
            n_classes = len(classes)
            total = len(target_arr)
            weights = []
            for cls in range(n_classes):
                count = float((target_arr.astype(int) == cls).sum())
                w = total / (n_classes * count) if count > 0 else 1.0
                weights.append(w)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"[IMBALANCE] Multiclass weights: {[f'{w:.3f}' for w in weights]}")
            return criterion_class(weight=weight_tensor)

    except Exception as exc:
        print(f"[IMBALANCE] Weight computation failed ({exc}); using unweighted criterion")
        return criterion_class()


def train(net, trainloader, config, partition_id, verbose=True, device='cpu'):
    
    """
    Train the model.

    Args:
        net: The model.
        trainloader: The training data loader.
        config: The training configuration.
        partition_id: The partition ID.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        tuple: The training loss and accuracy.
    """

    if "train" not in config:
        config["train"] = {"epochs": 3}

    # Cap epochs: Hub cannot exhaust the privacy budget in a single call.
    epochs = min(int(config["train"].get("epochs", 3)), _MAX_EPOCHS)

    # Fix random seed for reproducibility across training sessions.
    # Using a deterministic seed lets the paper cite exact metric values.
    # The seed is derived from the partition_id so each client gets a unique
    # but reproducible initialisation, avoiding all-identical starting points.
    _seed = 42 + (partition_id if isinstance(partition_id, int) else 0)
    import random
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    try:
        loss_function = config.get("model", {}).get("training", {}).get("loss_function", "bce_with_logits")

        loss_mapping = {
            "bce_with_logits": "BCEWithLogitsLoss",
            "bce":             "BCEWithLogitsLoss",
            "cross_entropy":   "CrossEntropyLoss",
            "nll":             "NLLLoss",
            "nll_loss":        "NLLLoss",
            "mse":             "MSELoss",
            "mae":             "L1Loss",
            "l1":              "L1Loss",
        }

        criterion_name = loss_mapping.get(loss_function.lower(), "BCEWithLogitsLoss")
        loss_mode = _get_loss_mode(loss_function)
        print(f"[TRAIN] loss_function='{loss_function}', mode='{loss_mode}', criterion='{criterion_name}'")
    except Exception as e:
        print(f"[TRAIN] Error resolving loss function: {e}")
        criterion_name = "BCEWithLogitsLoss"
        loss_mode = "binary"

    try:
        criterion_class = getattr(torch.nn, criterion_name)

        # ── Class-imbalance compensation ─────────────────────────────────────
        # Clinical datasets are frequently highly skewed (e.g. rare disease:
        # 95 negatives / 5 positives).  We compute per-class frequencies from
        # the trainloader on the fly and pass a weight tensor to the criterion.
        # This only touches the loss function — DP, model weights, and all other
        # training settings are unchanged.
        criterion = _build_weighted_criterion(criterion_class, loss_mode, trainloader, device)
        # ─────────────────────────────────────────────────────────────────────
    except Exception as e:
        print(f"[TRAIN] Could not build weighted criterion ({e}); using unweighted fallback")
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_mode = "binary"

    # Pre-initialize opt_dp_params so the name is always defined even if the
    # try block raises before assigning it. The ACTUAL security enforcement is
    # the _safe_dp_float + max() clamping further below — not these defaults.
    # Do not remove the clamping lines thinking this initialization is the guard.
    opt_dp_params: dict = {
        "noise_multiplier": _MIN_NOISE_MULTIPLIER,
        "max_grad_norm": _MIN_GRAD_NORM,
    }

    try:
        opt_config = config.get("model", {}).get("training", {}).get("optimizer", {})
        _raw_dp = opt_config.get("differential_privacy", {})
        if not isinstance(_raw_dp, dict):
            # Hub sent a non-dict value (e.g., "disabled") — fall back to safe defaults.
            print(f"[SECURITY] differential_privacy is not a dict ({type(_raw_dp).__name__}); using Node defaults")
            _raw_dp = {}
        opt_dp_params = _raw_dp

        # Validate optimizer type against allowlist — Hub-controlled string must
        # not be passed blindly to getattr(torch.optim, ...).
        raw_opt_type = str(opt_config.get("type", "Adam")).capitalize()
        if raw_opt_type not in _ALLOWED_OPTIMIZERS:
            print(f"[SECURITY] Optimizer '{raw_opt_type}' not in allowlist; falling back to Adam")
            raw_opt_type = "Adam"
        opt_type = raw_opt_type

        opt_params = {
            "lr": opt_config.get("learning_rate", 0.01),
            "weight_decay": opt_config.get("weight_decay", 0)
        }

        print(f"Using optimizer: {opt_type} with params: {opt_params}")

        model_params = list(net.parameters())
        print(f"[SEARCH] DEBUG: Model has {len(model_params)} parameter groups")
        print(f"[SEARCH] DEBUG: Total parameters: {sum(p.numel() for p in model_params)}")

        if len(model_params) == 0:
            print(f"[ERROR] Model has no parameters! This will cause 'empty parameter list' error.")
            print(f"[ERROR] Model structure: {net}")
            print(f"[ERROR] Model state_dict keys: {list(net.state_dict().keys())}")
            raise ValueError("Model has no parameters to optimize")

        opt_class = getattr(torch.optim, opt_type)
        opt = opt_class(net.parameters(), **opt_params)
    except Exception as e:
        print(f"[ERROR] Error in optimizer configuration: {e}")
        print(f"[ERROR] Available config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
        opt = torch.optim.Adam(net.parameters(), lr=0.01)

    # Security: derive the Node-side noise floor analytically from the actual
    # job parameters using the RDP accountant, then clamp whatever the Hub sent.
    # This guarantees ε ≤ per_job_max regardless of the Hub's σ request.
    _n_train    = len(trainloader.dataset)
    _batch_sz   = trainloader.batch_size or _TRAINING_BATCH_SIZE
    _epochs_val = config.get("train", {}).get("epochs", 10)

    # Resolve target_epsilon: prefer explicit key in opt_dp_params, then look up
    # the dataset's privacy policy, then fall back to the constant floor.
    _target_eps = _safe_dp_float(opt_dp_params.get("target_epsilon"), None)
    if _target_eps is None:
        try:
            from dataset.models import DatasetPrivacyPolicy
            _ds_name = (
                config.get("model", {})
                      .get("dataset", {})
                      .get("selected_datasets", [{}])[0]
                      .get("dataset_name")
            )
            if _ds_name:
                _target_eps = DatasetPrivacyPolicy.objects.get(
                    dataset__name=_ds_name
                ).max_epsilon_per_job
        except Exception as _exc:
            print(f"[DP] Could not look up per_job_max ({_exc}); using fallback floor")

    _dynamic_min = (
        compute_min_noise_multiplier(
            n=_n_train,
            batch_size=_batch_sz,
            epochs=_epochs_val,
            target_epsilon=_target_eps,
        )
        if _target_eps is not None
        else _MIN_NOISE_MULTIPLIER
    )

    _noise_multiplier = max(
        _safe_dp_float(opt_dp_params.get("noise_multiplier"), _dynamic_min),
        _dynamic_min,
    )
    print(f"[DP] n={_n_train} batch={_batch_sz} epochs={_epochs_val} "
          f"target_eps={_target_eps} -> floor_sigma={_dynamic_min:.4f} "
          f"requested_sigma={opt_dp_params.get('noise_multiplier')} "
          f"effective_sigma={_noise_multiplier:.4f}")
    _max_grad_norm = max(
        _safe_dp_float(opt_dp_params.get("max_grad_norm"), _MIN_GRAD_NORM),
        _MIN_GRAD_NORM,
    )
    # Security: never use a Hub-supplied seed — a chosen seed makes Gaussian
    # noise predictable, potentially leaking gradient information.
    _noise_seed = secrets.randbelow(2**31)

    privacy_engine = PrivacyEngine(secure_mode=False)
    net, opt, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=opt,
        data_loader=trainloader,
        noise_multiplier=_noise_multiplier,
        max_grad_norm=_max_grad_norm,
        noise_generator=torch.Generator().manual_seed(_noise_seed))
    net.train()

    all_predictions = []
    all_targets = []
    total_loss = 0.0

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        train_acc = 0.0
        train_batches = 0
        
        for batch_idx, (features, targets) in enumerate(trainloader):
            try:
                # Opacus Poisson sampling can yield empty batches — skip them.
                # Use .shape[0] rather than len() for PyTorch tensor robustness.
                if targets.shape[0] == 0:
                    continue

                features = features.to(device)
                targets = targets.to(device)
                opt.zero_grad()
                outputs = net(features)

                # ── Branch on loss mode ──────────────────────────────────────
                if loss_mode == "multiclass":
                    # outputs: [B, num_classes] logits
                    # targets: [B] integer class indices
                    loss = criterion(outputs, targets.long())
                    predictions = outputs.argmax(dim=1)
                    correct = (predictions == targets.long()).sum().item()
                elif loss_mode == "regression":
                    # outputs: [B, 1] or [B]
                    out_squeezed = outputs.squeeze(-1)
                    loss = criterion(out_squeezed, targets.float())
                    predictions = out_squeezed
                    correct = 0  # not applicable for regression
                else:
                    # binary — original behaviour
                    loss = criterion(outputs, targets.float().unsqueeze(1))
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    correct = (predictions == targets.float().unsqueeze(1)).sum().item()
                # ────────────────────────────────────────────────────────────

                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                train_acc += correct / len(targets)
                train_batches += 1

                if epoch == epochs - 1:
                    all_predictions.extend(predictions.cpu().detach().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch idx: {batch_idx}")
                print(f"Features shape: {features.shape if 'features' in locals() else 'N/A'}")
                print(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
                raise  # bare raise preserves original traceback

        if train_batches == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss /= train_batches
            epoch_acc = train_acc / train_batches
        total_loss = epoch_loss
        
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {round(epoch_loss, 3)}, accuracy {round(epoch_acc, 3)}")
    
    # Final precision/recall/F1 are only meaningful for classification (binary / multiclass).
    try:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        if loss_mode == "regression":
            # For regression report RMSE instead of classification metrics.
            rmse = float(np.sqrt(np.mean((all_predictions - all_targets) ** 2))) if len(all_targets) > 0 else 0.0
            precision = recall = f1 = 0.0
            if verbose:
                print(f"Final metrics (regression): RMSE={rmse:.4f}")
        elif loss_mode == "multiclass":
            # Micro-averaged precision / recall / F1 across all classes.
            classes = np.unique(all_targets).astype(int)
            tp_total = fp_total = fn_total = 0
            for cls in classes:
                tp_total += int(np.sum((all_predictions == cls) & (all_targets == cls)))
                fp_total += int(np.sum((all_predictions == cls) & (all_targets != cls)))
                fn_total += int(np.sum((all_predictions != cls) & (all_targets == cls)))
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if verbose:
                print(f"Final metrics (multiclass micro): precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        else:
            # Binary — original TP/FP/FN calculation
            tp = np.sum((all_predictions == 1) & (all_targets == 1))
            fp = np.sum((all_predictions == 1) & (all_targets == 0))
            fn = np.sum((all_predictions == 0) & (all_targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if verbose:
                print(f"Final metrics (binary): precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    # Measure the actual privacy cost incurred during this training run.
    # get_epsilon() queries the RDP accountant that Opacus updated on every
    # optimizer.step() call — so this is the exact (ε, δ)-DP guarantee.
    try:
        epsilon = privacy_engine.get_epsilon(delta=_DP_DELTA)
        print(f"[DP] ε={epsilon:.4f} (δ={_DP_DELTA}, σ={_noise_multiplier}, C={_max_grad_norm})")
    except Exception as e:
        print(f"[DP] Could not compute epsilon: {e}")
        epsilon = float("inf")

    # Return the noise_multiplier that Opacus actually used during this run so
    # the caller can verify it matches the approved configuration value.
    actual_noise_multiplier = getattr(privacy_engine, 'noise_multiplier', _noise_multiplier)

    return total_loss, epoch_acc, precision, recall, f1, epsilon, actual_noise_multiplier

def test(net, testloader, device='cpu', loss_function: str = "bce_with_logits"):
    """
    Evaluate the model.

    Args:
        net: The model.
        testloader: The testing data loader.
        device: Torch device string.
        loss_function: Loss function name matching what was used during training
            (e.g. 'bce_with_logits', 'cross_entropy', 'mse').  Drives output
            shape and metric computation — must match train() to get comparable
            numbers.

    Returns:
        tuple: (loss, accuracy) — for regression tasks accuracy is the RMSE.
    """
    loss_mapping = {
        "bce_with_logits": "BCEWithLogitsLoss",
        "bce":             "BCEWithLogitsLoss",
        "cross_entropy":   "CrossEntropyLoss",
        "nll":             "NLLLoss",
        "nll_loss":        "NLLLoss",
        "mse":             "MSELoss",
        "mae":             "L1Loss",
        "l1":              "L1Loss",
    }
    loss_mode = _get_loss_mode(loss_function)
    criterion_name = loss_mapping.get(loss_function.lower(), "BCEWithLogitsLoss")
    try:
        criterion = getattr(torch.nn, criterion_name)()
    except Exception:
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_mode = "binary"

    total_loss = 0.0
    test_acc = 0.0
    test_batches = 0
    all_predictions: list = []
    all_targets_list: list = []

    net.eval()
    with torch.no_grad():
        for features, targets in testloader:
            features = features.to(device)
            targets  = targets.to(device)

            outputs = net(features)

            # ── Branch on loss mode ──────────────────────────────────────
            if loss_mode == "multiclass":
                batch_loss = criterion(outputs, targets.long())
                predictions = outputs.argmax(dim=1)
                correct = (predictions == targets.long()).sum().item()
            elif loss_mode == "regression":
                out_squeezed = outputs.squeeze(-1)
                batch_loss = criterion(out_squeezed, targets.float())
                predictions = out_squeezed
                correct = 0
            else:
                # binary
                batch_loss = criterion(outputs, targets.float().unsqueeze(1))
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            # ────────────────────────────────────────────────────────────

            total_loss += batch_loss.item()
            test_acc   += correct / len(targets)
            test_batches += 1

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets_list.extend(targets.cpu().numpy().flatten())

    if test_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / test_batches

    if loss_mode == "regression":
        preds_arr = np.array(all_predictions)
        tgts_arr  = np.array(all_targets_list)
        accuracy  = float(np.sqrt(np.mean((preds_arr - tgts_arr) ** 2)))  # RMSE
    else:
        accuracy = test_acc / test_batches

    return avg_loss, accuracy