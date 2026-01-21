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
from dataset.models import Dataset, DatasetAccess
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view
from multiprocessing import Process
from datetime import datetime

logger = logging.getLogger(__name__)

CLIENT_VERSION = "0.1" # Version of the client API, used for versioning and compatibility checks


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
        logger.error(f"Error in get_data_info for user {request.api_user.username}: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error retrieving dataset information'
        }, status=500)


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
        from trainings.models import TrainingSession
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
            logger.error(f"[ERROR] Error creating training session: {e}")
            return JsonResponse({
                'error': 'Failed to create training session'
            }, status=500)
        
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
        logger.error(f"[ERROR]Error in start_client for user {request.api_user.username}: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error starting client'
        }, status=500)


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
        
        return None  # Validation passed
        
    except Exception as e:
        logger.error(f"Error validating dataset permissions for user {user.username}: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error validating permissions'
        }, status=500)


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
    from trainings.models import TrainingSession

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