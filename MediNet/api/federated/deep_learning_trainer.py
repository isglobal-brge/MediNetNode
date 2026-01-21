import numpy as np
import torch
import time
from typing import List
from opacus import PrivacyEngine
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dl_set_parameters(net, parameters: List[np.ndarray]):
    """
    Set the model parameters.

    Args:
        net: The model.
        parameters (List[np.ndarray]): The model parameters.
    """
    try:
        # Parameter setup
        
        # Obtenir les claus del model
        keys = list(net.state_dict().keys())
        # Parameter count verification
        
        # Verificar que tenim el mateix nombre de paràmetres
        if len(keys) != len(parameters):
            # Podem truncar a la més curta per evitar errors
            min_len = min(len(keys), len(parameters))
            keys = keys[:min_len]
            parameters = parameters[:min_len]
        
        # Crear el nou state_dict amb parells de clau-valor verificats
        state_dict = OrderedDict()
        for i, (key, param) in enumerate(zip(keys, parameters)):
            try:
                state_dict[key] = torch.Tensor(param)
            except Exception as e:
                # Parameter processing error
                raise
        
        # Carregar l'state_dict actualitzat
        net.load_state_dict(state_dict, strict=False)  # Canviat a strict=False per ser més permissiu
        
    except Exception as e:
        # Parameter setting error
        raise e

def dl_get_parameters(net) -> List[np.ndarray]:
    """
    Get the model parameters.

    Args:
        net: The model.

    Returns:
        List[np.ndarray]: The model parameters.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def dl_train(net, trainloader, config, partition_id, verbose=True):
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
    # Client training initialized
    if "train" not in config:
        config["train"] = {"epochs": 3}
    
    epochs = config["train"].get("epochs", 3)
    # Training epochs configured
    try:    
        # Accedir a la loss function des de model.training.loss_function
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
    except Exception as e:
        criterion = torch.nn.BCEWithLogitsLoss()

    try:
        # Accedir a l'optimitzador des de model.training.optimizer
        opt_config = config.get("model", {}).get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").capitalize()  # "adam" -> "Adam"
        opt_dp_params = opt_config.get("differential_privacy", {"noise_multiplier": 1, "max_grad_norm": 1, "random_seed": time.time()})
        # Construir paràmetres de l'optimitzador
        opt_params = {
            "lr": opt_config.get("learning_rate", 0.01),
            "weight_decay": opt_config.get("weight_decay", 0)
        }
        
        print(f"Using optimizer: {opt_type} with params: {opt_params}")
        
        # Debug: Check if model has parameters
        model_params = list(net.parameters())
        print(f"🔍 DEBUG: Model has {len(model_params)} parameter groups")
        print(f"🔍 DEBUG: Total parameters: {sum(p.numel() for p in model_params)}")
        
        if len(model_params) == 0:
            print(f"❌ ERROR: Model has no parameters! This will cause 'empty parameter list' error.")
            print(f"❌ Model structure: {net}")
            print(f"❌ Model state_dict keys: {list(net.state_dict().keys())}")
            raise ValueError("Model has no parameters to optimize")
        
        opt_class = getattr(torch.optim, opt_type)
        opt = opt_class(net.parameters(), **opt_params)
    except Exception as e:
        print(f"Error in optimizer configuration: {e}")
        print(f"Available config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine(secure_mode=False)
    net, opt, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=opt,
        data_loader=trainloader,
        noise_multiplier=max(opt_dp_params["noise_multiplier"], 1.0),
        max_grad_norm=max(opt_dp_params["max_grad_norm"], 1), 
        noise_generator=torch.Generator().manual_seed(opt_dp_params["random_seed"])))
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
                # Mou les dades al dispositiu correcte
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
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
                
                # Guardar prediccions i targets per última època
                if epoch == epochs - 1:
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch idx: {batch_idx}")
                print(f"Features shape: {features.shape if 'features' in locals() else 'N/A'}")
                print(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
                raise e

        epoch_loss /= train_batches
        epoch_acc = train_acc / train_batches
        total_loss = epoch_loss  # Última època
        
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {round(epoch_loss, 3)}, accuracy {round(epoch_acc, 3)}")
    
    # Calcular precision, recall i F1 de la darrera època
    try:
        import numpy as np
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
    
    return total_loss, epoch_acc, precision, recall, f1

def dl_test(net, testloader):
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
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            outputs = net(features)
            
            # Calcula la pèrdua
            loss += criterion(outputs, targets.float().unsqueeze(1)).item()
            
            # Calcula la precisió
            predictions = torch.sigmoid(outputs) >= 0.5
            correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            test_acc += correct / len(targets)
            test_batches += 1
            
    # Calcula mitjanes
    loss /= test_batches
    accuracy = test_acc / test_batches
    
    return loss, accuracy