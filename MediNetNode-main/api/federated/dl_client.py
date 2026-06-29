
import math
import numpy as np
import torch
from collections import OrderedDict
from typing import List
from .train_functions import train, test, _DP_DELTA
from .utils import update_training_progress, fail_training_session
from flwr.client import NumPyClient

# Allowlist of Flower per-round config keys the Node accepts. Any key not in
# this set is silently dropped before merging with model_json. This prevents
# the Hub from overwriting Node-controlled training parameters (epochs, DP
# settings, model architecture) through the Flower config dict.
_SAFE_FLOWER_KEYS = frozenset({"server_round", "client_id"})

def set_parameters(net, parameters: List[np.ndarray]):
    """
    Set the model parameters.

    Args:
        net: The model.
        parameters (List[np.ndarray]): The model parameters.
    """
    try:
        keys = list(net.state_dict().keys())

        # Fail fast on parameter count mismatch. Silent truncation would train
        # on partially-stale local weights without any signal, which is a
        # data-integrity attack vector when the Hub sends a short parameter list.
        if len(keys) != len(parameters):
            raise ValueError(
                f"Parameter count mismatch: model has {len(keys)} layers, "
                f"Hub sent {len(parameters)}. Refusing to load partial parameters."
            )

        state_dict = OrderedDict()
        for i, (key, param) in enumerate(zip(keys, parameters)):
            try:
                state_dict[key] = torch.Tensor(param)
            except Exception as e:
                raise

        # strict=True is intentional: the count guard above already validated
        # that len(keys) == len(parameters), so any shape mismatch indicates a
        # Hub/Node model definition mismatch that should fail loudly rather than
        # silently loading mismatched tensors.
        net.load_state_dict(state_dict, strict=True)

    except Exception as e:
        raise e

def get_parameters(net):
    """
    Get the model parameters.

    Args:
        net: The model.

    Returns:
        List[np.ndarray]: The model parameters.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class DLFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, model_json, training_session, 
                 client_ip, table_name, device, current_process ,partition_id=0):
        """
        Initialize the Flower client.
        Args:
            net: The model.
            trainloader: The training data loader.
            valloader: The validation data loader.
            partition_id (int, optional): The partition ID. Defaults to 0.
        """
        self.accuracy = None
        self.loss = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.epsilon = None
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.testloader = testloader
        self.assigned_client_id = None  # Variable temporal para el ID
        self.model_json = model_json
        self.training_session = training_session
        self.client_ip = client_ip
        self.table_name = table_name
        self.device = device
        self.current_process = current_process

    def set_client_id(self, client_id):
        """Método para asignar ID desde la configuración"""
        self.assigned_client_id = client_id
        print(f"CLIENT_ID_SET: Client assigned ID: {client_id}")

    def get_parameters(self, config):
        """
        Get the model parameters.
        Args:
            config: The configuration.
        Returns:
            List[np.ndarray]: The model parameters.
        """
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """
        Train the model.
        Args:
            parameters: The model parameters.
            config: The training configuration.
        Returns:
            tuple: The updated model parameters, the number of training samples, and an empty dictionary.
        """
        
        try:
            print(f"DEBUG FIT: Received {len(parameters)} parameters")
            
            set_parameters(self.net, parameters)
            
            # Safe config merge: only allowlisted Flower metadata keys are merged.
            # All other keys (including 'model', 'train', 'optimizer') are dropped.
            # Allowlist approach is safer than blocklist — new Hub-sent keys default
            # to blocked, preventing future bypass via unblocked top-level keys.
            complete_config = self.model_json.copy()
            if config:
                safe_flower_keys = {k: v for k, v in config.items() if k in _SAFE_FLOWER_KEYS}
                complete_config.update(safe_flower_keys)
                        
            train_results = train(self.net, self.trainloader, complete_config, self.partition_id, self.device)
            self.loss, self.accuracy, self.precision, self.recall, self.f1, self.epsilon, actual_noise = train_results

            # Verify the noise_multiplier Opacus actually used matches the approved
            # configuration. A mismatch indicates tampering between config validation
            # and training execution — abort immediately.
            expected_noise = (
                complete_config.get('model', {})
                .get('training', {})
                .get('dp', {})
                .get('noise_multiplier')
            )
            if expected_noise is not None and actual_noise is not None:
                if abs(actual_noise - expected_noise) > 1e-4:
                    fail_training_session(
                        self.training_session,
                        f"DP parameters tampered: expected noise_multiplier={expected_noise}, "
                        f"actual={actual_noise}. Training aborted for security.",
                    )
                    # Return 1 (not 0) to avoid ZeroDivisionError in Hub aggregation.
                    return self.get_parameters({}), 1, {}

            # float("inf") is not JSON-serializable; use -1.0 as sentinel meaning
            # "epsilon measurement failed — treat as unbounded privacy cost".
            epsilon_serializable = self.epsilon if math.isfinite(self.epsilon) else -1.0

            round_metrics = {
                'loss': float(self.loss),
                'accuracy': float(self.accuracy),
                'precision': float(self.precision),
                'recall': float(self.recall),
                'f1': float(self.f1),
                'privacy_epsilon': float(epsilon_serializable),
                'privacy_delta': float(_DP_DELTA),
            }
            
            # Get persistent round counter from training session (survives Flower client restarts)
            if self.training_session:
                current_round = self.training_session.current_round + 1
                print(f"[SYNC] Flower round {current_round} (from persistent session state)")
            else:
                # Fallback if no training session
                current_round = getattr(self, '_round_counter', 0) + 1
                setattr(self, '_round_counter', current_round)
                print(f"[SYNC] Local round {current_round} (fallback)")
            
            update_training_progress(self.training_session, current_round, self.current_process, round_metrics)
            
            metrics = {
                "accuracy": float(self.accuracy),
                "loss": float(self.loss),
                "precision": float(self.precision),
                "recall": float(self.recall),
                "f1": float(self.f1),
                "privacy_epsilon": float(epsilon_serializable),
                "privacy_delta": float(_DP_DELTA),
                "client_name": f"Client_{self.partition_id}",
                "client_ip": self.client_ip,
                "dataset_name": self.table_name,
                "client_id": self.assigned_client_id,
                "train_samples": len(self.trainloader.dataset) if self.trainloader else 0,
            }

            print(f"[INFO] CLIENT_METRICS: client_id='{self.assigned_client_id}' | acc={self.accuracy:.3f} | loss={self.loss:.3f} | f1={self.f1:.3f} | ε={epsilon_serializable:.4f}")
            print(f"[SEARCH] DEBUG FIT: Metrics sent for client_id: {self.assigned_client_id}")
            print(f"[SEARCH] DEBUG FIT: Trainloader length: {len(self.trainloader)}")
            
            # Flower uses num_examples to weight each client in FedAvg aggregation.
            # Use dataset size (samples), not len(trainloader) (batches).
            num_examples = len(self.trainloader.dataset) if self.trainloader else 0
            if num_examples == 0:
                print(f"[ERROR] WARNING: Trainloader dataset is empty! Using 1 to avoid division by zero.")
                num_examples = 1
            #time.sleep(30)

            return get_parameters(self.net), num_examples, metrics
        except Exception as e:
            print(f"Error in fit: {e}")
            import traceback
            tb = traceback.format_exc()
            print(f"Traceback:\n{tb}")
            fail_training_session(self.training_session, str(e), tb)
            # Return num_examples=1 (never 0) — returning 0 causes a ZeroDivisionError
            # in the Hub's FedAvg weighted aggregation which would crash the entire
            # Flower server and prevent other clients from completing their rounds.
            return parameters, 1, {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model.

        Args:
            parameters: The model parameters.
            config: The evaluation configuration.

        Returns:
            tuple: The loss, the number of testing samples, and a dictionary with accuracy and loss.
        """        
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)

        print(f"DEBUG EVALUATE: Using TABLE_NAME: {self.table_name}")

        # Pass the same loss_function that was used during training so test()
        # branches on the correct output shape and metric mode.
        loss_function = self.model_json.get("model", {}).get("training", {}).get("loss_function", "bce_with_logits")
        loss, accuracy = test(self.net, self.valloader, self.device, loss_function=loss_function)
        num_val_samples = len(self.valloader.dataset) if self.valloader else 0
        return float(loss), num_val_samples, {"accuracy": float(accuracy), "loss": float(loss)}
