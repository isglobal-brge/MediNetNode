"""
API views for RESEARCHER users - stateless authentication.
Compatible with existing client_api.py structure.
"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.utils import timezone
import json
import os
import logging
from dataset.models import Dataset, DatasetAccess, ResearcherEpsilonBudget, DatasetPrivacyPolicy
from trainings.models import TrainingSession
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view
from multiprocessing import Process
from datetime import datetime
from medinet.error_handlers import SafeErrorResponse

logger = logging.getLogger(__name__)

CLIENT_VERSION = "0.1"

MAX_CONCURRENT_TRAINING_SESSIONS = 3

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

    if request_body_size > MAX_JSON_SIZE:
        logger.warning(f"Training config exceeds size limit: {request_body_size} bytes")
        return JsonResponse({
            'error': 'Configuration too large'
        }, status=400)

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

    # Always block localhost/loopback (unless explicitly allowed for local dev/testing)
    allow_localhost = getattr(settings, 'ALLOW_LOCALHOST_FL_SERVERS', False)
    if not allow_localhost:
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
    Health check endpoint. Returns status and API key expiry info so the Hub
    can warn researchers before their key expires.

    Returns:
        JsonResponse: {
            'status': 'ok',
            'api_key': {'expires_at': <iso>, 'days_remaining': <int>} | null
        }
    """
    logger.info(f"Ping request from user {request.api_user.username}")

    api_key_info = None
    api_key = request.api_key
    if api_key.expires_at:
        delta = api_key.expires_at - timezone.now()
        days_remaining = max(0, delta.days)
        api_key_info = {
            'expires_at': api_key.expires_at.isoformat(),
            'days_remaining': days_remaining,
        }

    return JsonResponse({'status': 'ok', 'api_key': api_key_info})


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

        accessible_datasets = get_user_datasets(user)
        
        if not accessible_datasets:
            logger.warning(f"No datasets accessible to user {user.username}")
            return JsonResponse({
                'error': 'No datasets available for this user'
            }, status=403)
        
        data_dict = format_datasets_for_client(accessible_datasets, researcher_user=user)
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

        validation_error = validate_training_config(data, len(request.body))
        if validation_error is not None:
            return validation_error

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

        validation_result = validate_training_permissions(user, model_json)
        if validation_result is not None:
            return validation_result

        # Log training initiation for audit
        logger.info(f"Training initiated by user {user.username}, client_id: {client_id}")

        # Create training session BEFORE starting client (ensures it exists)
        import psutil

        try:
            # revisar hauria d'agafar el dataset no de model json
            dataset_config = model_json.get('model', {}).get('dataset', {})
            selected_datasets = dataset_config.get('selected_datasets', [])
            dataset_id = None
            dataset_name = "unknown"
            
            if selected_datasets and len(selected_datasets) > 0:
                first_dataset = selected_datasets[0]
                dataset_id = first_dataset.get('dataset_id')
                dataset_name = first_dataset.get('dataset_name', 'unknown')

            current_process = psutil.Process()

            training_session = TrainingSession(
                client_id=client_id or f"client_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user=user,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                model_config=model_json,
                server_address=server_address,
                status='STARTING',
                process_id=current_process.pid,
                use_experiment=bool(model_json.get('use_experiment', False)),
            )
            training_session.save()
            
            logger.info(f"[OK] Training session created: {training_session.session_id}")
            
        except Exception as e:
            return SafeErrorResponse.internal_error(request, e, "create_training_session")
        
        # Save training request JSON to documentation folder for ML testing
        try:
            doc_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documentacion')
            training_requests_dir = os.path.join(doc_base, 'training_requests')
            os.makedirs(training_requests_dir, exist_ok=True)

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_short = str(training_session.session_id)[:8]
            filename = f'training_request_{timestamp_str}_{session_short}.json'
            doc_file = os.path.join(training_requests_dir, filename)

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

            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Training request saved to: {filename}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to save training request to documentation: {e}")
        
        # Pass training_session and SSL certificate to flower client.
        # Imported lazily: keeps the heavy ML stack (torch/numpy/flwr) out of
        # module import, so the HTTP/permission layer (e.g. budget enforcement)
        # can be imported and tested without it.
        from .federated import client
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
    if not user.has_permission('dataset.train'):
        logger.warning(f"User {user.username} lacks general training permission")
        return JsonResponse({
            'error': 'User does not have training permissions'
        }, status=403)

    dataset_id = extract_dataset_id_from_model(model_json)

    if dataset_id is None:
        logger.warning(f"No dataset ID found in model configuration for user {user.username}")
        return JsonResponse({
            'error': 'No valid dataset ID found in model configuration'
        }, status=400)
    
    try:
        from dataset.models import DatasetAccess

        try:
            access = DatasetAccess.objects.using('datasets_db').get(
                user_id=user.id,
                dataset_id=dataset_id
            )

            if not access.can_train:
                logger.warning(
                    f"User {user.username} lacks training permission for dataset {dataset_id}"
                )
                return JsonResponse({
                    'error': f'Training permission denied for dataset {dataset_id}'
                }, status=403)

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
        # Design invariants (see docs/superpowers/specs/
        # 2026-06-28-researcher-budget-enforcement-design.md):
        #   • Missing policy → BLOCK (fail-closed): datasets must have a policy
        #     (it is the source of the per-researcher limits).
        #   • The per-researcher ResearcherEpsilonBudget is the ENFORCEMENT GATE.
        #     The dataset-level policy counter is audit-only and does NOT block,
        #     so researchers never contend over a shared dataset pool.
        #   • Estimation failure → researcher can_accept_job(inf) handles it → BLOCK.
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

            # The dataset-level policy is the configuration source + audit
            # aggregate ONLY; it no longer blocks. Enforcement is the
            # per-researcher quota below (the policy supplies its limits).

            # --- Enforcement gate: per-researcher epsilon budget ---
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
                        {
                            'error': f'Presupuesto de privacidad del researcher agotado: {reason}',
                            'budget_exhausted': True,
                            'dataset_id': dataset_id,
                        },
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
        # Using user_id since we have cross-database relationships
        dataset_accesses = DatasetAccess.objects.using('datasets_db').filter(
            user_id=user.id,
        )

        if not dataset_accesses.exists():
            return []

        datasets = []
        for access in dataset_accesses:
            if access.can_view_metadata and access.dataset.is_active:
                datasets.append(access.dataset)

        return list(datasets)
        
    except Exception as e:
        logger.error(f"Error retrieving user datasets: {str(e)}")
        return []


def format_datasets_for_client(datasets, researcher_user=None):
    """
    Format datasets to match the structure expected by client_api.py.

    Args:
        datasets:         List of Dataset objects
        researcher_user:  Optional api_user — when supplied, per-researcher
                          epsilon budgets are included in the response.

    Returns:
        dict: Formatted data compatible with client expectations
    """
    from dataset.models import DatasetPrivacyPolicy

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
        'metadata': [],
        'privacy_policy': [],
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
                    'updated_at': dataset.metadata.updated_at.isoformat() if dataset.metadata.updated_at else None,
                }
        except Exception as e:
            logger.error(f"Error retrieving metadata for dataset {dataset.id}: {str(e)}")
        data_dict['metadata'].append(metadata_info)

        # Dataset-level policy (set by Node admin)
        privacy_info = None
        try:
            policy = DatasetPrivacyPolicy.objects.get(dataset_id=dataset.id)
            privacy_info = {
                'sensitivity':        policy.sensitivity,
                'max_epsilon_per_job': policy.max_epsilon_per_job,
                'lifetime_budget':    policy.lifetime_budget,
                'spent_epsilon':      round(policy.spent_epsilon, 4),
                'remaining_budget':   round(policy.remaining_budget, 4),
            }

            # Per-researcher budget (finer-grained, if it exists)
            if researcher_user is not None:
                try:
                    rb = ResearcherEpsilonBudget.objects.get(
                        dataset_id=dataset.id,
                        researcher_id=researcher_user.id,
                    )
                    privacy_info['researcher_budget'] = {
                        'lifetime_budget':  rb.lifetime_budget,
                        'spent_epsilon':    round(rb.spent_epsilon, 4),
                        'remaining_budget': round(rb.remaining_budget, 4),
                        'max_epsilon_per_job': rb.max_epsilon_per_job,
                        'period':           rb.period,
                    }
                except ResearcherEpsilonBudget.DoesNotExist:
                    pass  # No per-researcher record yet — dataset policy is enough

        except DatasetPrivacyPolicy.DoesNotExist:
            # Dataset has no policy configured yet
            privacy_info = None
        except Exception as e:
            logger.error(f"Error retrieving privacy policy for dataset {dataset.id}: {str(e)}")

        data_dict['privacy_policy'].append(privacy_info)

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

    session = TrainingSession.objects.filter(session_id=session_id).first()

    if not session:
        return JsonResponse({
            'error': 'Training session not found'
        }, status=404)

    if session.user_id != user.id:
        logger.warning(f"User {user.username} attempted to cancel session {session_id} owned by user {session.user_id}")
        return JsonResponse({
            'error': 'Not authorized to cancel this training session'
        }, status=403)

    if session.status in ['COMPLETED', 'FAILED', 'CANCELLED']:
        return JsonResponse({
            'error': f'Training session already {session.status.lower()}',
            'status': session.status
        }, status=400)

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


@csrf_exempt
@require_http_methods(["GET"])
def budget_status(request):
    """
    GET /api/v2/budget-status/

    Returns the researcher's current ε budget for every dataset they have
    access to (or a single dataset when ?dataset_name=X is supplied).

    Response shape:
        {
          "datasets": [
            {
              "dataset_id": 3,
              "dataset_name": "Hospital_BCN",
              "is_experimental": false,
              "spent_epsilon": 1.23,
              "remaining_budget": 3.77,
              "lifetime_budget": 5.0,
              "max_epsilon_per_job": 1.0,
              "period": "annual"
            },
            ...
          ]
        }

    Experimental datasets are included but flagged — budget is always null for
    them because no tracking takes place on the experimental split.
    """
    user = getattr(request, 'api_user', None)
    if not user:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    dataset_name_filter = request.GET.get('dataset_name', '').strip()

    try:
        accesses = DatasetAccess.objects.using('datasets_db').filter(
            user_id=user.id,
            can_train=True,
        ).select_related('dataset')

        if dataset_name_filter:
            accesses = accesses.filter(dataset__name=dataset_name_filter)

        results = []
        for access in accesses:
            ds = access.dataset
            budget = ResearcherEpsilonBudget.objects.using('datasets_db').filter(
                dataset=ds,
                researcher_id=user.id,
            ).first()

            if budget:
                entry = {
                    'dataset_id': ds.id,
                    'dataset_name': ds.name,
                    'is_experimental': False,
                    'spent_epsilon': round(budget.spent_epsilon, 6),
                    'remaining_budget': round(budget.remaining_budget, 6),
                    'lifetime_budget': round(budget.lifetime_budget, 6),
                    'max_epsilon_per_job': round(budget.max_epsilon_per_job, 6),
                    'period': budget.period,
                }
            else:
                # Access exists but no budget record yet — treat as fresh
                from dataset.models import DatasetPrivacyPolicy
                policy = DatasetPrivacyPolicy.objects.using('datasets_db').filter(
                    dataset=ds
                ).first()
                entry = {
                    'dataset_id': ds.id,
                    'dataset_name': ds.name,
                    'is_experimental': False,
                    'spent_epsilon': 0.0,
                    'remaining_budget': policy.lifetime_budget if policy else None,
                    'lifetime_budget': policy.lifetime_budget if policy else None,
                    'max_epsilon_per_job': policy.max_epsilon_per_job if policy else None,
                    'period': 'annual',
                }
            results.append(entry)

        return JsonResponse({'datasets': results})

    except Exception as exc:
        logger.error("[budget_status] Error: %s", exc)
        return JsonResponse({'error': 'Internal error fetching budget status'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def estimate_epsilon(request):
    """
    POST /api/v2/estimate-epsilon/

    Returns the pre-flight ε estimate for a given training configuration
    without starting any training session.  Reuses the same
    `estimate_job_epsilon()` logic used by the real training gate.

    Request body:
        {
          "dataset_name": "Hospital_BCN",
          "model_json": { ... }   // same shape as start-client
        }

    Response:
        {
          "estimated_epsilon": 0.87,
          "delta": 1e-5,
          "dataset_id": 3,
          "dataset_size": 4200
        }

    Returns 422 when estimation is not possible (missing DP config, import
    error, etc.) so the Hub can display "unable to estimate" rather than crash.
    """
    user = getattr(request, 'api_user', None)
    if not user:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    dataset_name = (body.get('dataset_name') or '').strip()
    model_json = body.get('model_json')

    if not dataset_name:
        return JsonResponse({'error': 'dataset_name is required'}, status=400)
    if not isinstance(model_json, dict):
        return JsonResponse({'error': 'model_json must be an object'}, status=400)

    try:
        access = DatasetAccess.objects.using('datasets_db').filter(
            user_id=user.id,
            can_train=True,
            dataset__name=dataset_name,
        ).select_related('dataset').first()

        if not access:
            return JsonResponse({'error': f'Dataset "{dataset_name}" not found or access denied'}, status=404)

        ds = access.dataset
        dataset_size = ds.patient_count or 0

        if dataset_size <= 0:
            return JsonResponse({'error': 'Dataset has no patient records — cannot estimate ε'}, status=422)

        estimated_eps = estimate_job_epsilon(model_json, dataset_size)

        import math
        if not math.isfinite(estimated_eps):
            return JsonResponse(
                {'error': 'Could not estimate ε — check DP parameters (noise_multiplier, epochs)'},
                status=422,
            )

        from api.federated.train_functions import _DP_DELTA
        return JsonResponse({
            'estimated_epsilon': round(estimated_eps, 6),
            'delta': _DP_DELTA,
            'dataset_id': ds.id,
            'dataset_size': dataset_size,
        })

    except Exception as exc:
        logger.error("[estimate_epsilon] Error: %s", exc)
        return JsonResponse({'error': 'Internal error during epsilon estimation'}, status=500)


@require_http_methods(["GET"])
@api_view_required
def min_noise_multiplier(request):
    """
    GET /api/v2/min-noise-multiplier/

    Returns the minimum noise multiplier (σ) the Node will enforce for a given
    dataset and training configuration, derived analytically from the RDP
    accountant so that DP-SGD consumes at most the dataset's per_job_max
    epsilon budget.

    The Hub calls this endpoint as the researcher configures batch_size and
    epochs, displaying the value in real time so they can set σ ≥ the floor.
    At runtime the Node recalculates and clamps regardless of what the Hub sends.

    Query parameters:
        dataset_name    (required) — name of the dataset
        batch_size      (required) — integer, mini-batch size
        epochs          (required) — integer, local epochs per federated round
        use_experiment  (optional) — "true"/"1" to compute σ for the experimental
                                     partition (experiment_row_count) instead of the
                                     full dataset. A smaller partition requires a higher
                                     σ floor to maintain the same ε guarantee, so this
                                     must match the actual training mode.

    Response:
        {
          "min_noise_multiplier": 0.957,
          "target_epsilon":       4.5,
          "dataset_name":         "test_split_3",
          "dataset_size":         1752,
          "use_experiment":       true,
          "batch_size":           32,
          "epochs":               20,
          "delta":                1e-5,
          "note":                 "..."
        }
    """
    try:
        dataset_name   = (request.GET.get("dataset_name") or "").strip()
        batch_size     = request.GET.get("batch_size")
        epochs         = request.GET.get("epochs")
        use_experiment = request.GET.get("use_experiment", "false").lower() in ("1", "true", "yes")

        if not dataset_name:
            return JsonResponse({"error": "dataset_name is required"}, status=400)
        if not batch_size or not epochs:
            return JsonResponse(
                {"error": "batch_size and epochs are required query parameters"},
                status=400,
            )

        try:
            batch_size = int(batch_size)
            epochs     = int(epochs)
        except ValueError:
            return JsonResponse(
                {"error": "batch_size and epochs must be integers"},
                status=400,
            )

        if batch_size <= 0 or epochs <= 0:
            return JsonResponse(
                {"error": "batch_size and epochs must be positive integers"},
                status=400,
            )

        try:
            dataset = Dataset.objects.select_related("privacy_policy").get(
                name=dataset_name
            )
        except Dataset.DoesNotExist:
            return JsonResponse(
                {"error": f"Dataset '{dataset_name}' not found"},
                status=404,
            )

        if not hasattr(dataset, "privacy_policy"):
            return JsonResponse(
                {"error": f"Dataset '{dataset_name}' has no privacy policy configured"},
                status=400,
            )

        # Use the experimental partition size when requested and available.
        # This matters because a smaller n requires a higher σ floor to achieve
        # the same ε guarantee — computing against the full dataset would
        # underestimate the required noise and silently violate the budget.
        exp_n        = dataset.experiment_row_count or 0
        is_exp_mode  = use_experiment and exp_n > 0
        n_records    = exp_n if is_exp_mode else (dataset.rows_count or dataset.patient_count or 0)
        target_epsilon = dataset.privacy_policy.max_epsilon_per_job

        if n_records <= 0:
            return JsonResponse(
                {"error": "Dataset has no record count — upload may be incomplete"},
                status=400,
            )

        from api.federated.train_functions import compute_min_noise_multiplier, _DP_DELTA
        min_sigma = compute_min_noise_multiplier(
            n=n_records,
            batch_size=batch_size,
            epochs=epochs,
            target_epsilon=target_epsilon,
        )

        partition_label = "experimental partition" if is_exp_mode else "full dataset"
        return JsonResponse({
            "min_noise_multiplier": round(min_sigma, 6),
            "target_epsilon":       target_epsilon,
            "dataset_name":         dataset_name,
            "dataset_size":         n_records,
            "use_experiment":       is_exp_mode,
            "batch_size":           batch_size,
            "epochs":               epochs,
            "delta":                _DP_DELTA,
            "note": (
                f"Minimum sigma such that DP-SGD with batch={batch_size}, "
                f"epochs={epochs} on n={n_records} records ({partition_label}) consumes "
                f"at most epsilon={target_epsilon} (delta={_DP_DELTA}) per job, "
                "derived from the RDP accountant."
            ),
        })

    except Exception as exc:
        logger.error("[min_noise_multiplier] Error: %s", exc)
        return SafeErrorResponse.internal_error(request, exc, "min_noise_multiplier")