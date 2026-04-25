
import math
import secrets
import time
import torch
import numpy as np
from . import utils
from opacus import PrivacyEngine

# Node-enforced DP minimums — the Hub cannot override these downward.
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

    try:
        loss_function = config.get("model", {}).get("training", {}).get("loss_function", "bce_with_logits")
        
        # Mappejar noms de loss functions
        loss_mapping = {
            "bce_with_logits": "BCEWithLogitsLoss",
            "cross_entropy": "CrossEntropyLoss",
            "mse": "MSELoss",
            "mae": "L1Loss"
        }
        
        criterion_name = loss_mapping.get(loss_function, "BCEWithLogitsLoss")
        # Loss function configured
    except Exception as e:
        # Error in loss configuration
        criterion_name = "BCEWithLogitsLoss"
    
    try:
        criterion_class = getattr(torch.nn, criterion_name)
        criterion = criterion_class()
    except Exception:
        criterion = torch.nn.BCEWithLogitsLoss()

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

    # Security: enforce Node-side minimums regardless of what the Hub sent.
    # Use _safe_dp_float to guard against NaN/inf/non-numeric before max().
    # max(float('nan'), 1.0) == nan in Python — explicit isfinite is required.
    _noise_multiplier = max(
        _safe_dp_float(opt_dp_params.get("noise_multiplier"), _MIN_NOISE_MULTIPLIER),
        _MIN_NOISE_MULTIPLIER,
    )
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

    # Variables per calcular mètriques finals
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
                loss = criterion(outputs, targets.float().unsqueeze(1))
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                correct = (predictions == targets.float().unsqueeze(1)).sum().item()
                train_acc += correct / len(targets)
                train_batches += 1

                if epoch == epochs - 1:
                    all_predictions.extend(predictions.cpu().numpy().flatten())
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
    
    # Calcular precision, recall i F1 de la darrera època
    try:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # True Positives, False Positives, False Negatives
        tp = np.sum((all_predictions == 1) & (all_targets == 1))
        fp = np.sum((all_predictions == 1) & (all_targets == 0))
        fn = np.sum((all_predictions == 0) & (all_targets == 1))
        
        # Evitar divisió per zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if verbose:
            print(f"Final metrics: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
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

def test(net, testloader, device='cpu'):
    """
    Evaluate the model.
    Args:
        net: The model.
        testloader: The testing data loader.
    Returns:
        tuple: The testing loss and accuracy.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    test_acc = 0.0
    test_batches = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(testloader):
            # Mou les dades al dispositiu
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = net(features)
            
            # Calcula la pèrdua
            loss += criterion(outputs, targets.float().unsqueeze(1)).item()
            
            # Calcula la precisió
            predictions = torch.sigmoid(outputs) >= 0.5
            correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            test_acc += correct / len(targets)
            test_batches += 1
            
    if test_batches == 0:
        return 0.0, 0.0
    loss /= test_batches
    accuracy = test_acc / test_batches
    return loss, accuracy