import numpy as np
import torch
import warnings
import os
import sys
import time
from . import utils
from .utils import complete_training_session, fail_training_session
from datasets.utils.logging import disable_progress_bar
from flwr.client import start_client
from flwr.common import Context
from .data_loaders import create_train_val_loaders, load_ml_data
from .model_builder import DynamicModel, SequentialModel
from datetime import datetime
from .dl_client import DLFlowerClient
from .ml_client import MLFlowerClient
from .algorithms import get_algorithm
# Add Django project to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
warnings.filterwarnings("ignore")

try:
    import django
    django.setup()
    from trainings.models import TrainingSession, TrainingRound
    from django.contrib.auth import get_user_model
    DJANGO_AVAILABLE = True
    User = get_user_model()
    
except ImportError as e:
    print(f"Warning: Django models not available for training tracking: {e}")
    DJANGO_AVAILABLE = False
    

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
MODEL_JSON = dict()
MODEL_VALIDATED = False
TABLE_NAME = None
CLIENT_IP = "localhost"
ASSIGNED_CLIENT_ID = None  # Global variable for client_id
TRAINING_SESSION = None  # Global training session instance
CURRENT_USER = None  # Global user context
CURRENT_PROCESS = None  # Current training process
disable_progress_bar()




def client_fn(context: Context):
    """
    Create a Flower client instance.
    Args:
        context (Context): The context for the client.
    Returns:
        FlowerClient: The Flower client instance.
    """
    # Try to recover training session if not already loaded (Flower restart)
    global MODEL_VALIDATED, MODEL_JSON, TABLE_NAME, ASSIGNED_CLIENT_ID, TRAINING_SESSION

    if not TRAINING_SESSION and ASSIGNED_CLIENT_ID and DJANGO_AVAILABLE:
        try:
            existing_session = TrainingSession.objects.get(client_id=ASSIGNED_CLIENT_ID, status__in=['STARTING', 'ACTIVE'])
            TRAINING_SESSION = existing_session
            print(f"[SYNC] Recovered training session in client_fn: {existing_session.session_id} (Round {existing_session.current_round})")
        except TrainingSession.DoesNotExist:
            print(f"[SEARCH] No existing session found in client_fn for client_id: {ASSIGNED_CLIENT_ID}")
        except Exception as e:
            print(f"[ERROR] Error recovering session in client_fn: {e}")
    elif TRAINING_SESSION:
        print(f"[LIST] Using existing training session: {TRAINING_SESSION.session_id} (Round {TRAINING_SESSION.current_round})")
    
    # Extract only the model part from the full JSON structure
    print(f"MODEL_JSON KEYS: {MODEL_JSON.keys()}")
    print(f"MODEL_JSON TYPE: {MODEL_JSON['model']['metadata']['model_type']}")
    model_config = MODEL_JSON.get('model', MODEL_JSON)
    model_type = model_config['metadata']['model_type']
    partition_id = int(context.node_config.get("partition-id", 0))

    # ==================== ML CLIENT INITIALIZATION ====================
    if model_type == 'ml':
        print(f"\n{'='*70}")
        print(f"🤖 INITIALIZING ML CLIENT (Machine Learning)")
        print(f"{'='*70}")

        # Extract ML algorithm configuration
        training_config = model_config.get('training', {})
        ml_method = training_config.get('ml_method', 'fedsvm').lower()

        print(f"[INFO] ML Algorithm: {ml_method}")
        print(f"[PACKAGE] Dataset: {TABLE_NAME}")

        # Load ML data (NumPy arrays, no batch_size)
        print(f"\n[SYNC] Loading ML data...")
        (X_train, y_train), (X_val, y_val) = load_ml_data(
            dataset_id=TABLE_NAME,
            val_size=training_config.get('val_size', 0.2),
            random_state=training_config.get('random_state', 42)
        )

        # Get algorithm class from registry
        print(f"\n[SEARCH] Loading algorithm '{ml_method}' from registry...")
        try:
            AlgorithmClass = get_algorithm(ml_method)
            print(f"[OK] Algorithm class loaded: {AlgorithmClass.__name__}")
        except ValueError as e:
            print(f"[ERROR] {e}")
            raise

        # Initialize algorithm instance with validation data
        print(f"\n[CONFIG] Initializing {ml_method} algorithm...")
        algorithm_instance = AlgorithmClass(X_train, y_train, MODEL_JSON, X_val, y_val)
        print(f"[OK] Algorithm initialized successfully")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")

        # Create ML Flower client
        print(f"\n[FLOWER] Creating MLFlowerClient...")
        flower_client = MLFlowerClient(
            algorithm_instance=algorithm_instance,
            validation_data=(X_val, y_val),
            model_json=MODEL_JSON,
            training_session=TRAINING_SESSION,
            client_ip=CLIENT_IP,
            table_name=TABLE_NAME,
            current_process=CURRENT_PROCESS,
            partition_id=partition_id
        )

        print(f"[OK] MLFlowerClient created successfully")
        print(f"{'='*70}\n")

    # ==================== DL CLIENT INITIALIZATION ====================
    else:
        print(f"\n{'='*70}")
        print(f"🧠 INITIALIZING DL CLIENT (Deep Learning)")
        print(f"{'='*70}")

        net = DynamicModel(model_config).to(DEVICE)

        # [SEARCH] DEBUG: Check model after creation
        print(f"DEBUG: Model created with {sum(p.numel() for p in net.parameters()) if net.parameters() else 0} parameters")
        print(f"DEBUG: Model state_dict keys: {list(net.state_dict().keys()) if hasattr(net, 'state_dict') else 'No state_dict'}")
        print(f"DEBUG: Net type: {type(net)}")
        print(f"DEBUG: Net structure: {net}")

        if not MODEL_VALIDATED:
            net = utils.check_model(net)
            MODEL_VALIDATED = True
        print("##################### MODEL VALIDATED #####################")

        # Log data loading attempt
        with open('client_debug.log', 'a', encoding='utf-8') as f:
            f.write(f"Loading data for table: {TABLE_NAME}\n")

        trainloader, valloader = create_train_val_loaders(TABLE_NAME, batch_size=32)

        # Log data loading results
        with open('client_debug.log', 'a', encoding='utf-8') as f:
            f.write(f"Trainloader length: {len(trainloader) if trainloader else 'None'}\n")
            f.write(f"Valloader length: {len(valloader) if valloader else 'None'}\n")

        # Create DL Flower client
        flower_client = DLFlowerClient(
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            testloader=None,
            model_json=MODEL_JSON,
            training_session=TRAINING_SESSION,
            client_ip=CLIENT_IP,
            table_name=TABLE_NAME,
            device=DEVICE,
            current_process=CURRENT_PROCESS,
            partition_id=partition_id
        )

        print(f"[OK] DLFlowerClient created successfully")
        print(f"{'='*70}\n")

    # Set the client_id if available (common for both ML and DL)
    if ASSIGNED_CLIENT_ID:
        flower_client.set_client_id(ASSIGNED_CLIENT_ID)
        print(f"🆔 FlowerClient ID set to: {ASSIGNED_CLIENT_ID}")
    else:
        print("[WARNING] Warning: No client_id assigned")

    return flower_client

def start_flower_client(model_json, server_address="localhost:8080", client_id=None, user=None, session_id=None, ca_cert=None):
    """
    Start the Flower client.
    Args:
        model_json (dict): The JSON configuration of the model.
        server_address (str, optional): The address of the server. Defaults to "localhost:8080".
        client_id (str): The unique client ID assigned by server.
        user: Django User instance for tracking.
        session_id: UUID of the training session.
        ca_cert (str): CA certificate for SSL/TLS connection (PEM format).
    """
    global MODEL_JSON, MODEL_VALIDATED, TABLE_NAME, CLIENT_IP, ASSIGNED_CLIENT_ID, CURRENT_USER, TRAINING_SESSION

    # Set the assigned client_id and user
    ASSIGNED_CLIENT_ID = client_id
    CURRENT_USER = user
    print(f"🆔 CLIENT_ID assigned globally: {ASSIGNED_CLIENT_ID}")
    
    #Get client IP automatically
    import socket
    try:
        # Connect to server to get local IP that will be used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((server_address.split(':')[0], int(server_address.split(':')[1])))
            CLIENT_IP = s.getsockname()[0]
    except:
        CLIENT_IP = "localhost"
    
    print(f"🌐 Client IP detected: {CLIENT_IP}")
    
    # Pass the complete model configuration, not just layers
    MODEL_JSON = model_json  # Full config instead of just layers
    MODEL_VALIDATED = False
    
    # Access dataset from the correct path based on debug_received_from_server.json structure
    TABLE_NAME = int(model_json['model']['dataset']['selected_datasets'][0]['dataset_id'])
    print(f"DEBUG: Model config loaded with {len(model_json.keys()) if isinstance(model_json, dict) else 0} sections")
    print(f"DEBUG: TABLE_NAME set to: {TABLE_NAME}")
    print(f"DEBUG: SERVER_ADDRESS set to: {server_address}")
    
    # Load training session from session_id (created in API layer)
    if session_id and DJANGO_AVAILABLE:
        try:
            TRAINING_SESSION = TrainingSession.objects.get(session_id=session_id)
            print(f"[OK] Loaded training session: {TRAINING_SESSION.session_id} (Round {TRAINING_SESSION.current_round})")
            print(f"Session status: {TRAINING_SESSION.status}, User: {TRAINING_SESSION.user.username}")
        except TrainingSession.DoesNotExist:
            print(f"[ERROR] Training session not found: {session_id}")
            TRAINING_SESSION = None
        except Exception as e:
            print(f"[ERROR] Error loading training session: {e}")
            TRAINING_SESSION = None
    else:
        print("[WARNING] Warning: No session_id provided for training tracking")
        TRAINING_SESSION = None
    
    # Force write to file for debugging
    with open('client_debug.log', 'w', encoding='utf-8') as f:
        f.write(f"MODEL_JSON: {model_json}\n")
        f.write(f"TABLE_NAME: {TABLE_NAME}\n")
        f.write(f"SERVER_ADDRESS: {server_address}\n")
        f.write(f"CLIENT_IP: {CLIENT_IP}\n")
        f.write(f"CLIENT_ID: {client_id}\n")
        f.write(f"USER: {user.username if user else 'None'}\n")
    
    try:
        # Convert CA certificate to bytes for Flower SSL connection
        root_certificates = ca_cert.encode() if ca_cert else None

        if root_certificates:
            print(f"[LOCKED] Starting Flower client with SSL/TLS (certificate provided)")
        else:
            print(f"[WARNING] Starting Flower client without SSL (no certificate)")

        start_client(
            server_address=server_address,
            client_fn=client_fn,
            root_certificates=root_certificates
        )
        # If we reach here, training completed successfully
        complete_training_session(TRAINING_SESSION)
    except Exception as e:
        # Training failed
        import traceback
        fail_training_session(TRAINING_SESSION, str(e), traceback.format_exc())
        raise

