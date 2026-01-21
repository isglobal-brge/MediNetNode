
import numpy as np
import torch
from collections import OrderedDict
from typing import List
from .train_functions import train, test
from .utils import update_training_progress, fail_training_session
from flwr.client import NumPyClient

def set_parameters(net, parameters: List[np.ndarray]):
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
            # Parameter mismatch handling
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
        print(f"🆔 CLIENT_ID_SET: Client assigned ID: {client_id}")

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
            
            # Merge Flower config with global MODEL_JSON for complete configuration
            complete_config = self.model_json.copy()
            if config:
                # Override with any Flower-specific config
                complete_config.update(config)
                        
            train_results = train(self.net, self.trainloader, complete_config, self.partition_id, self.device)
            self.loss, self.accuracy, self.precision, self.recall, self.f1 = train_results
            
            # Update training progress with round metrics
            round_metrics = {
                'loss': float(self.loss),
                'accuracy': float(self.accuracy),
                'precision': float(self.precision),
                'recall': float(self.recall),
                'f1': float(self.f1)
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
                "client_name": f"Client_{self.partition_id}",
                "client_ip": self.client_ip,
                "dataset_name": self.table_name,
                "client_id": self.assigned_client_id,  # ← KEY: ID en métricas
                "train_samples": len(self.trainloader.dataset) if self.trainloader else 0
            }
            
            print(f"[INFO] CLIENT_METRICS: client_id='{self.assigned_client_id}' | acc={self.accuracy:.3f} | loss={self.loss:.3f} | f1={self.f1:.3f}")
            print(f"[SEARCH] DEBUG FIT: Metrics sent for client_id: {self.assigned_client_id}")
            print(f"[SEARCH] DEBUG FIT: Trainloader length: {len(self.trainloader)}")
            
            # Ensure we don't return 0 for num_examples as this causes division by zero
            num_examples = len(self.trainloader)
            if num_examples == 0:
                print(f"[ERROR] WARNING: Trainloader is empty! This will cause division by zero error.")
                # Return at least 1 to avoid division by zero
                num_examples = 1
            #time.sleep(30)

            return get_parameters(self.net), num_examples, metrics
        except Exception as e:
            print(f"Error in fit: {e}")
            # Mark training session as failed
            import traceback
            fail_training_session(self.training_session, str(e), traceback.format_exc())
            return parameters, 0, {}

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
        
        # Use global TABLE_NAME for consistent data loading
        print(f"DEBUG EVALUATE: Using TABLE_NAME: {self.table_name}")
        
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss": float(loss)}
