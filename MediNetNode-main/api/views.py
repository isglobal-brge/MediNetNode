"""
API views for RESEARCHER users - stateless authentication.
Compatible with existing client_api.py structure.
"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import os
import logging
from .federated import client
from dataset.models import Dataset, DatasetAccess, ResearcherEpsilonBudget
from trainings.models import TrainingSession
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view
from multiprocessing import Process
from datetime import datetime
from medinet.error_handlers import SafeErrorResponse

logger = logging.getLogger(__name__)

CLIENT_VERSION = "0.1" # Version of the client API, used for versioning and compatibility checks

MAX_CONCURRENT_TRAINING_SESSIONS = 2

# JSON schema for training configuration validation
TRAINING_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model_json": {
            "type": "object",
            "maxProperties": 100
        },
        "server_address": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9.-]+:[0-9]{1,5}$",
            "maxLength": 100
        },
        "client_id": {
            "type": "string",
            "maxLength": 100
        },
        "ca_cert": {
            "type": "string",
            "maxLength": 10000
        },
        "ssl_enabled": {
            "type": "boolean"
        }
    },
    "required": ["model_json"],
    "additionalProperties": True
}

# Maximum JSON payload size (5MB)
MAX_JSON_SIZE = 5 * 1024 * 1024

# Blocked server patterns (SSRF protection)
ALWAYS_BLOCKED_PATTERNS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0'
]

PRIVATE_NETWORK_PATTERNS = [
    '192.168.',
    '10.',
    '172.16.',
    '172.17.',
    '172.18.',
    '172.19.',
    '172.20.',
    '172.21.',
    '172.22.',
    '172.23.',
    '172.24.',
    '172.25.',
    '172.26.',
    '172.27.',
    '172.28.',
    '172.29.',
    '172.30.',
    '172.31.'
]


def validate_training_config(data, request_body_size):
    """
    Validate training configuration against security rules.

    Args:
        data: Parsed JSON configuration
        request_body_size: Size of request body in bytes

    Returns:
        JsonResponse with error if validation fails, None if passes
    """
    from jsonschema import validate, ValidationError

    # Size limit check
    if request_body_size > MAX_JSON_SIZE:
        logger.warning(f"Training config exceeds size limit: {request_body_size} bytes")
        return JsonResponse({
            'error': 'Configuration too large'
        }, status=400)

    # Schema validation
    try:
        validate(instance=data, schema=TRAINING_CONFIG_SCHEMA)
    except ValidationError as e:
        logger.warning(f"Training config schema validation failed: {e.message}")
        return JsonResponse({
            'error': 'Invalid configuration format'
        }, status=400)

    # SSRF protection - block localhost always
    server_address = data.get("server_address", "localhost:8080")
    server_host = server_address.split(':')[0].lower()

    # Always block localhost/loopback
    for blocked_pattern in ALWAYS_BLOCKED_PATTERNS:
        if server_host.startswith(blocked_pattern):
            logger.warning(f"Blocked localhost server address: {server_address}")
            return JsonResponse({
                'error': 'Localhost addresses not allowed'
            }, status=403)

    # Block private networks only if not explicitly allowed (check dynamically)
    allow_private = getattr(settings, 'ALLOW_PRIVATE_FL_SERVERS', False)
    if not allow_private:
        for blocked_pattern in PRIVATE_NETWORK_PATTERNS:
            if server_host.startswith(blocked_pattern):
                logger.warning(f"Blocked private network server address: {server_address}")
                return JsonResponse({
                    'error': 'Private network addresses not allowed. Set ALLOW_PRIVATE_FL_SERVERS=True in settings to enable.'
                }, status=403)

    return None


def api_view_required(view_func):
    """Decorator to ensure API authentication middleware has run."""
    def wrapper(request, *args, **kwargs):
        # Check if API authentication middleware has run
        if not hasattr(request, 'api_key') or not hasattr(request, 'api_user'):
            return JsonResponse(
                {'error': 'API authentication required'},
                status=401
            )
        return view_func(request, *args, **kwargs)
    return wrapper


@require_http_methods(["GET"])
@api_view_required
def ping(request):
    """
    Health check endpoint compatible with client_api.py.
    
    Returns:
        JsonResponse: {'status': 'pong'}
    """
    logger.info(f"Ping request from user {request.api_user.username}")
    return JsonResponse({'status': 'pong'})


@require_http_methods(["GET"])
@api_view_required
def get_data_info(request):
    """
    Retrieve dataset metadata for authorized datasets.
    Compatible with client_api.py get_data_info endpoint.
    
    Returns:
        JsonResponse: Dataset metadata in the format expected by clients
    """
    try:
        user = request.api_user
        logger.info(f"get_data_info request from user {user.username}")
        
        # Get datasets accessible to this user
        accessible_datasets = get_user_datasets(user)
        
        if not accessible_datasets:
            logger.warning(f"No datasets accessible to user {user.username}")
            return JsonResponse({
                'error': 'No datasets available for this user'
            }, status=403)
        
        # Format data to match client_api.py structure
        data_dict = format_datasets_for_client(accessible_datasets)
        logger.info(f"Returning {len(accessible_datasets)} datasets to user {user.username}")
        return JsonResponse(data_dict)
        
    except Exception as e:
        return SafeErrorResponse.internal_error(request, e, "get_data_info")


@csrf_exempt
@require_http_methods(["POST"])
@api_view_required
def start_client(request):
    """
    Start federated learning client endpoint.
    Compatible with client_api.py start-client endpoint.
    
    Accepts JSON payload with model configuration and initiates training.
    """
    try:
        user = request.api_user
        # Parse JSON request body
        try:
            data = json.loads(request.body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Invalid JSON in start_client request: {str(e)}")
            return JsonResponse({
                'error': 'Invalid JSON format'
            }, status=400)
        logger.info(f"start_client request from user {user.username}")
        logger.debug(f"Request data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

        # Limit concurrent active training sessions per researcher
        active_sessions = TrainingSession.objects.filter(
            user=request.api_user,
            status__in=['STARTING', 'ACTIVE'],
        ).count()
        if active_sessions >= MAX_CONCURRENT_TRAINING_SESSIONS:
            return JsonResponse(
                {
                    'error': (
                        f'Límite de sesiones simultáneas alcanzado ({MAX_CONCURRENT_TRAINING_SESSIONS}). '
                        f'Espera a que termine un entrenamiento antes de iniciar otro.'
                    )
                },
                status=429,
            )

        # Validate training configuration
        validation_error = validate_training_config(data, len(request.body))
        if validation_error is not None:
            return validation_error
        
        # Extract and validate required parameters
        model_json = data.get("model_json")
        server_address = data.get("server_address", "localhost:8080")
        client_id = data.get("client_id")
        ca_cert = data.get("ca_cert")
        ssl_enabled = data.get("ssl_enabled", True)

        if not model_json:
            logger.error("Missing model_json in start_client request")
            return JsonResponse({
                'error': 'model_json is required'
            }, status=400)

        # SSL certificate validation - MANDATORY for secure connection
        if ssl_enabled and not ca_cert:
            logger.error(f"Missing ca_cert in start_client request from user {user.username}")
            return JsonResponse({
                'error': 'CA certificate (ca_cert) required for secure connection'
            }, status=400)

        # Comprehensive security validation
        validation_result = validate_training_permissions(user, model_json)
        if validation_result is not None:
            return validation_result
        
        # Log training initiation for audit
        logger.info(f"Training initiated by user {user.username}, client_id: {client_id}")
        
        # Create training session BEFORE starting client (ensures it exists)
        import psutil
        
        try:
            # Extract dataset info from model_json 
            # revisar hauria d'agafar el dataset no de model json 
            dataset_config = model_json.get('model', {}).get('dataset', {})
            selected_datasets = dataset_config.get('selected_datasets', [])
            dataset_id = None
            dataset_name = "unknown"
            
            if selected_datasets and len(selected_datasets) > 0:
                first_dataset = selected_datasets[0]
                dataset_id = first_dataset.get('dataset_id')
                dataset_name = first_dataset.get('dataset_name', 'unknown')
            
            # Get current process for tracking
            current_process = psutil.Process()
            
            # Create training session
            training_session = TrainingSession(
                client_id=client_id or f"client_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user=user,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                model_config=model_json,
                server_address=server_address,
                status='STARTING',
                process_id=current_process.pid
            )
            training_session.save()
            
            logger.info(f"[OK] Training session created: {training_session.session_id}")
            
        except Exception as e:
            return SafeErrorResponse.internal_error(request, e, "create_training_session")
        
        # Save training request JSON to documentation folder for ML testing
        try:
            # Create training_requests subdirectory if it doesn't exist
            doc_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documentacion')
            training_requests_dir = os.path.join(doc_base, 'training_requests')
            os.makedirs(training_requests_dir, exist_ok=True)

            # Create filename with timestamp and session_id
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_short = str(training_session.session_id)[:8]
            filename = f'training_request_{timestamp_str}_{session_short}.json'
            doc_file = os.path.join(training_requests_dir, filename)

            # Prepare comprehensive debug data for ML testing
            debug_data = {
                "client_id": client_id,
                "server_address": server_address,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "session_id": str(training_session.session_id),
                "timestamp": str(datetime.now()),
                "model_type": model_json.get('model_type', 'unknown'),
                "ml_method": model_json.get('ml_method', None),
                "model_config": model_json
            }

            # Save to documentation folder
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📄 Training request saved to: {filename}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to save training request to documentation: {e}")
        
        # Pass training_session and SSL certificate to flower client
        process = Process(target=client.start_flower_client, args=(model_json, server_address, client_id, user, training_session.session_id, ca_cert), daemon=True)
        process.start()
        
        response_data = {
            'status': 'Flower Client started',
            'client_id': client_id,
            'server_address': server_address,
            'user': user.username
        }
        
        return JsonResponse(response_data, status=200)
        
    except Exception as e:
        return SafeErrorResponse.internal_error(request, e, "start_client")


def extract_dataset_id_from_model(model_json):
    """
    Extract dataset ID from model configuration.
    
    Args:
        model_json (dict): Model configuration JSON
        
    Returns:
        int: Dataset ID or None if not found
    """
    try:
        if isinstance(model_json, dict):
            # Check model.dataset.selected_datasets[0] structure
            model_config = model_json.get('model', {})
            dataset_config = model_config.get('dataset', {})
            selected_datasets = dataset_config.get('selected_datasets', [])
            
            if selected_datasets and len(selected_datasets) > 0:
                first_dataset = selected_datasets[0]
                if isinstance(first_dataset, dict):
                    dataset_id = first_dataset.get('dataset_id')
                    if dataset_id:
                        return int(dataset_id)
            
            # Check direct dataset_id reference
            dataset_id = model_json.get('dataset_id')
            if dataset_id:
                return int(dataset_id)
                
    except Exception as e:
        logger.error(f"Error extracting dataset ID from model config: {str(e)}")
        return None
    
    return None


def estimate_job_epsilon(config: dict, dataset_size: int) -> float:
    """
    Estimate ε for a training job using RDPAccountant with Node-enforced minimums.

    Mirrors the clamping logic from train_functions.py so the pre-flight check
    uses the same ε that actual training would produce.  Returns float('inf') on
    any failure so callers can treat an estimation failure as "budget unknown"
    and decide how to proceed.
    """
    import math
    try:
        from api.federated.train_functions import (
            _MIN_NOISE_MULTIPLIER, _DP_DELTA, _MAX_EPOCHS, _safe_dp_float,
            _TRAINING_BATCH_SIZE,
        )
        from opacus.accountants import RDPAccountant
    except ImportError as exc:
        logger.warning("[DP] Cannot import DP dependencies for epsilon estimation: %s", exc)
        return float('inf')

    if dataset_size <= 0:
        return float('inf')

    opt_config = config.get('model', {}).get('training', {}).get('optimizer', {})
    _raw_dp = opt_config.get('differential_privacy', {})
    if not isinstance(_raw_dp, dict):
        _raw_dp = {}

    # Apply the same Node-enforced floor as train_functions.train()
    noise_multiplier = max(
        _safe_dp_float(_raw_dp.get('noise_multiplier'), _MIN_NOISE_MULTIPLIER),
        _MIN_NOISE_MULTIPLIER,
    )

    # Use the Node-fixed batch size, not the Hub-supplied one. The Hub could
    # send a tiny batch_size to lower the estimated sample_rate and understate ε.
    batch_size = min(_TRAINING_BATCH_SIZE, dataset_size)

    try:
        raw_epochs = int(config.get('train', {}).get('epochs', 3))
    except (TypeError, ValueError):
        raw_epochs = 3
    epochs = min(max(raw_epochs, 1), _MAX_EPOCHS)

    sample_rate = batch_size / dataset_size
    steps = max(int(epochs * dataset_size / batch_size), 1)

    try:
        acc = RDPAccountant()
        acc.history = [(noise_multiplier, sample_rate, steps)]
        eps = float(acc.get_epsilon(delta=_DP_DELTA))
        return eps if math.isfinite(eps) else float('inf')
    except Exception as exc:
        logger.warning("[DP] RDPAccountant epsilon estimation failed: %s", exc)
        return float('inf')


def validate_training_permissions(user, model_json):
    """
    Comprehensive validation of user permissions for federated learning training.
    
    Args:
        user: CustomUser instance
        model_json (dict): Model configuration JSON
        
    Returns:
        JsonResponse: Error response if validation fails, None if validation passes
    """
    # 1. Validate general training permission
    if not user.has_permission('dataset.train'):
        logger.warning(f"User {user.username} lacks general training permission")
        return JsonResponse({
            'error': 'User does not have training permissions'
        }, status=403)
    
    # 2. Extract dataset ID from model configuration
    dataset_id = extract_dataset_id_from_model(model_json)
    
    if dataset_id is None:
        logger.warning(f"No dataset ID found in model configuration for user {user.username}")
        return JsonResponse({
            'error': 'No valid dataset ID found in model configuration'
        }, status=400)
    
    # 3. Validate access to the specific dataset
    try:
        from dataset.models import DatasetAccess

        # Check if user has access to this specific dataset
        try:
            access = DatasetAccess.objects.using('datasets_db').get(
                user_id=user.id,
                dataset_id=dataset_id
            )

            # Check if user has training permission for this dataset
            if not access.can_train:
                logger.warning(
                    f"User {user.username} lacks training permission for dataset {dataset_id}"
                )
                return JsonResponse({
                    'error': f'Training permission denied for dataset {dataset_id}'
                }, status=403)

            # Check if the dataset is ACTIVE (is_active=True)
            if not access.dataset.is_active:
                logger.warning(
                    f"Dataset {dataset_id} is paused/inactive - training not allowed for user {user.username}"
                )
                return JsonResponse({
                    'error': f'Dataset {dataset_id} is currently paused and unavailable for training'
                }, status=403)

        except DatasetAccess.DoesNotExist:
            logger.warning(
                f"User {user.username} has no access record for dataset {dataset_id}"
            )
            return JsonResponse({
                'error': f'Access denied to dataset {dataset_id}'
            }, status=403)
        
        # 4. Log successful validation for audit
        logger.info(
            f"Training permissions validated for user {user.username}, "
            f"dataset: {dataset_id}"
        )

        # 4b. Experimental job fast-path: skip ALL epsilon budget checks.
        # The researcher is operating on the small experiment subset which has no
        # budget tracking by design.  We still validate access (step 3) above.
        use_experiment = bool(model_json.get('use_experiment', False))
        if use_experiment:
            try:
                dataset_obj = Dataset.objects.using('datasets_db').get(id=dataset_id)
                if dataset_obj.experiment_file_path:
                    logger.info(
                        "[EXP] Experimental job for user %s on dataset %s — skipping budget checks.",
                        user.username, dataset_id,
                    )
                    return None  # Validation passed — no budget consumed
                else:
                    logger.warning(
                        "[EXP] use_experiment=True but dataset %s has no experiment_file_path — "
                        "falling through to normal budget checks.",
                        dataset_id,
                    )
            except Dataset.DoesNotExist:
                pass  # Fall through to normal budget checks

        # 5. Privacy budget pre-check (Node protects itself; Hub is untrusted).
        # Design invariants:
        #   • Missing policy → BLOCK (fail-closed): datasets must have a policy.
        #   • Estimation failure → can_accept_job(inf) handles it → BLOCK.
        #   • DB / system error → 503 (fail-closed): if we cannot prove budget
        #     remains, we must not allow access to patient data.
        try:
            from dataset.models import DatasetPrivacyPolicy
            try:
                policy = DatasetPrivacyPolicy.objects.get(dataset_id=dataset_id)
            except DatasetPrivacyPolicy.DoesNotExist:
                logger.warning(
                    "[DP] No privacy policy for dataset %s — blocking training. "
                    "Administrator must configure a DatasetPrivacyPolicy.",
                    dataset_id,
                )
                return JsonResponse(
                    {'error': (
                        f'Dataset {dataset_id} has no privacy policy configured. '
                        'An administrator must create a privacy policy before '
                        'training is permitted on this dataset.'
                    )},
                    status=403,
                )
            dataset_size = access.dataset.patient_count or 0
            estimated_eps = estimate_job_epsilon(model_json, dataset_size)
            can_proceed, reason = policy.can_accept_job(estimated_eps)
            if not can_proceed:
                logger.warning(
                    "[DP] Budget check failed for user %s, dataset %s: %s",
                    user.username, dataset_id, reason,
                )
                return JsonResponse({'error': reason}, status=403)
            logger.info(
                "[DP] Budget check passed: estimated ε=%.4f (remaining: %.4f) "
                "for user %s, dataset %s",
                estimated_eps, policy.remaining_budget,
                user.username, dataset_id,
            )

            # --- Chequeo de presupuesto por researcher ---
            try:
                researcher_budget, _ = ResearcherEpsilonBudget.get_or_create_for(
                    dataset=access.dataset,
                    researcher_id=user.id,
                    policy=policy,
                )
                can_proceed, reason = researcher_budget.can_accept_job(estimated_eps)
                if not can_proceed:
                    logger.warning(
                        "Researcher %s rechazado por presupuesto personal agotado: %s",
                        user.username, reason
                    )
                    return JsonResponse(
                        {'error': f'Presupuesto de privacidad del researcher agotado: {reason}'},
                        status=403,
                    )
            except Exception as exc:
                logger.error("Error verificando presupuesto del researcher: %s", exc)
                return JsonResponse(
                    {'error': 'Error verificando presupuesto de privacidad del researcher.'},
                    status=500,
                )

        except Exception as dp_exc:
            logger.error(
                "[DP] Privacy budget system error for dataset %s: %s", dataset_id, dp_exc
            )
            return JsonResponse(
                {'error': 'Privacy budget check unavailable. Contact administrator.'},
                status=503,
            )

        return None  # Validation passed
        
    except Exception as e:
        return SafeErrorResponse.internal_error(request, e, "validate_dataset_permissions")


def get_user_datasets(user):
    """
    Get datasets accessible to the user.
    Args:
        user: CustomUser instance
    Returns:
        list: List of Dataset objects accessible to the user (only active datasets)
    """
    try:
        # Get dataset access records for this user
        # Using user_id since we have cross-database relationships
        dataset_accesses = DatasetAccess.objects.using('datasets_db').filter(
            user_id=user.id,
        )

        if not dataset_accesses.exists():
            return []

        # Get the actual datasets - use the dataset relationship
        # Only include datasets that are ACTIVE (is_active=True)
        datasets = []
        for access in dataset_accesses:
            if access.can_view_metadata and access.dataset.is_active:  # Check permission and active status
                datasets.append(access.dataset)

        return list(datasets)
        
    except Exception as e:
        logger.error(f"Error retrieving user datasets: {str(e)}")
        return []


def format_datasets_for_client(datasets):
    """
    Format datasets to match the structure expected by client_api.py.
    
    Args:
        datasets: List of Dataset objects
        
    Returns:
        dict: Formatted data compatible with client expectations
    """
    data_dict = {
        'dataset_id': [],
        'dataset_name': [],
        'medical_domain': [],
        'patient_count': [],
        'data_type': [],
        'file_size': [],
        'description': [],
        'target_column': [],
        'num_columns': [],
        'created_at': [],
        'metadata': []
    }
    
    for dataset in datasets:
        data_dict['dataset_id'].append(dataset.id)
        data_dict['dataset_name'].append(dataset.name)
        data_dict['medical_domain'].append(dataset.get_medical_domain_display())
        data_dict['patient_count'].append(dataset.patient_count or 0)
        data_dict['data_type'].append(dataset.get_data_type_display())
        data_dict['file_size'].append(dataset.file_size)
        data_dict['description'].append(dataset.description or '')
        data_dict['target_column'].append(dataset.target_column or '')
        data_dict['num_columns'].append(dataset.columns_count or 0)
        data_dict['created_at'].append(dataset.uploaded_at.isoformat() if dataset.uploaded_at else '')
        
        # Get metadata if available
        metadata_info = {}
        try:
            if hasattr(dataset, 'metadata') and dataset.metadata:
                metadata_info = {
                    'statistical_summary': dataset.metadata.statistical_summary or {},
                    'missing_values': dataset.metadata.missing_values or {},
                    'data_distribution': dataset.metadata.data_distribution or {},
                    'quality_score': dataset.metadata.quality_score,
                    'completeness_percentage': dataset.metadata.completeness_percentage,
                    'generated_at': dataset.metadata.generated_at.isoformat() if dataset.metadata.generated_at else None,
                    'updated_at': dataset.metadata.updated_at.isoformat() if dataset.metadata.updated_at else None
                }
        except Exception as e:
            logger.error(f"Error retrieving metadata for dataset {dataset.id}: {str(e)}")
            metadata_info = {}
        
        data_dict['metadata'].append(metadata_info)
        #data_dict['client_version'] = CLIENT_VERSION
    
    return data_dict


@csrf_exempt
@require_http_methods(["POST"])
@api_view_required
def cancel_training(request, session_id):
    """
    Cancel an active training session by killing its process.

    Args:
        request: HTTP request with API authentication
        session_id: UUID of the training session to cancel

    Returns:
        JsonResponse with cancellation status
    """
    import psutil

    user = request.api_user
    logger.info(f"Cancel training request from user {user.username} for session {session_id}")

    # Get training session
    session = TrainingSession.objects.filter(session_id=session_id).first()

    if not session:
        return JsonResponse({
            'error': 'Training session not found'
        }, status=404)

    # Verify user owns this session
    if session.user_id != user.id:
        logger.warning(f"User {user.username} attempted to cancel session {session_id} owned by user {session.user_id}")
        return JsonResponse({
            'error': 'Not authorized to cancel this training session'
        }, status=403)

    # Check if session is cancellable
    if session.status in ['COMPLETED', 'FAILED', 'CANCELLED']:
        return JsonResponse({
            'error': f'Training session already {session.status.lower()}',
            'status': session.status
        }, status=400)

    # Kill the process if it exists
    process_killed = False
    if session.process_id:
        if psutil.pid_exists(session.process_id):
            process = psutil.Process(session.process_id)
            process.terminate()
            process.wait(timeout=5)
            process_killed = True
            logger.info(f"Process {session.process_id} terminated for session {session_id}")
        else:
            logger.warning(f"Process {session.process_id} no longer exists for session {session_id}")

    # Update session status
    session.status = 'CANCELLED'
    session.error_message = f"Training cancelled by user {user.username}"
    session.save(update_fields=['status', 'error_message'])

    logger.info(f"Training session {session_id} cancelled successfully")

    return JsonResponse({
        'status': 'success',
        'message': 'Training session cancelled',
        'session_id': str(session_id),
        'process_killed': process_killed
    }, status=200)