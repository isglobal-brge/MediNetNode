"""
Dataset views with secure upload functionality.
"""

import os
import logging
from typing import Dict, Any
from django.shortcuts import render, redirect, get_object_or_404
from django.core.exceptions import PermissionDenied
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Sum, Q, Min
from django.core.paginator import Paginator
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import get_user_model
from .models import Dataset, DatasetAccess, DatasetMetadata
from .uploader import SecureDatasetUploader
from .forms import DatasetMetadataForm, DatasetEditForm
from users.decorators import require_role

# Configure logging
logger = logging.getLogger(__name__)


@login_required
@require_role('ADMIN')
def dataset_detail(request, dataset_id):
    """
    Display detailed information about a dataset including metadata and permissions.
    
    Args:
        request: The HTTP request
        dataset_id: The ID of the dataset to display
        
    Returns:
        Rendered template with dataset details
        
    Raises:
        PermissionDenied: If user doesn't have access to this dataset
        Http404: If dataset doesn't exist
    """
    # Get dataset with related metadata
    dataset = get_object_or_404(
        Dataset.objects.using('datasets_db'),
        id=dataset_id
    )
    # Obtener el usuario que subió el dataset manualmente desde la base de usuarios
    uploaded_by_user = None
    if dataset.uploaded_by_id:
        UserModel = get_user_model()
        try:
            uploaded_by_user = UserModel.objects.using('default').get(id=dataset.uploaded_by_id)
        except UserModel.DoesNotExist:
            uploaded_by_user = None
    
    # Check permissions
    if not request.user.is_superuser and (not request.user.role or request.user.role.name != 'ADMIN'):
        # For researchers, check if they have access
        if not DatasetAccess.objects.using('datasets_db').filter(
            dataset_id=dataset_id,
            user_id=request.user.id
        ).exists():
            logger.warning(
                f"User {request.user.username} attempted to access dataset {dataset_id} "
                f"without permission"
            )
            raise PermissionDenied("You don't have access to this dataset")
    columns_count = dataset.columns_count
    # Get metadata
    metadata_obj = DatasetMetadata.objects.using('datasets_db').filter(
        dataset_id=dataset_id
    ).order_by('-generated_at').first()
    

    
    # Initialize metadata with proper structure
    metadata = {
        'data_quality_score': None,
        'privacy_score': None,
        'k_anonymity_verified': False,
        'nulls_verified_zero': False,
        'extraction_timestamp': None,
        'statistical_summary': {},
        'column_types': {},
        'column_info': {}
    }
    
    # If we have metadata object, extract the data properly
    if metadata_obj:
        # Check if statistical_summary contains the full metadata structure
        if metadata_obj.statistical_summary and isinstance(metadata_obj.statistical_summary, dict):
            # Check if this is actually the full metadata object
            if 'file_type' in metadata_obj.statistical_summary:
                print("[DEBUG] Found full metadata in statistical_summary field")
                # This field contains the full metadata, extract it
                full_metadata = metadata_obj.statistical_summary
                metadata.update({
                    'k_anonymity_verified': full_metadata.get('k_anonymity_verified', False),
                    'nulls_verified_zero': full_metadata.get('nulls_verified_zero', False),
                    'extraction_timestamp': full_metadata.get('extraction_timestamp'),
                    'statistical_summary': full_metadata.get('statistical_summary', {}),
                    'column_types': full_metadata.get('column_types', {})
                    
                })
            else:
                # This is actually statistical summary data
                metadata['statistical_summary'] = metadata_obj.statistical_summary
        
        # Check other fields
        if metadata_obj.quality_score is not None:
            metadata['data_quality_score'] = int(metadata_obj.quality_score * 100)
        
        if metadata_obj.completeness_percentage is not None:
            metadata['privacy_score'] = int(metadata_obj.completeness_percentage)
    
    
    # Get access history with user details
    # Obtener access_history sin select_related cruzado
    access_history_raw = DatasetAccess.objects.using('datasets_db').filter(
        dataset_id=dataset_id
    ).order_by('-assigned_at')[:10]
    # Para cada acceso, obtener el usuario y el asignador manualmente
    UserModel = get_user_model()
    access_history = []
    for access in access_history_raw:
        user = None
        assigned_by = None
        if access.user_id:
            try:
                user = UserModel.objects.using('default').get(id=access.user_id)
            except UserModel.DoesNotExist:
                user = None
        if access.assigned_by_id:
            try:
                assigned_by = UserModel.objects.using('default').get(id=access.assigned_by_id)
            except UserModel.DoesNotExist:
                assigned_by = None
        access_history.append({
            'access': access,
            'user': user,
            'assigned_by': assigned_by
        })
    
    # Calculate usage statistics
    stats = {
        'total_accesses': dataset.access_count,
        'unique_users': DatasetAccess.objects.using('datasets_db')
            .filter(dataset_id=dataset_id)
            .values('user_id')
            .distinct()
            .count(),
        'last_accessed': dataset.last_accessed,
        'days_since_upload': (timezone.now() - dataset.uploaded_at).days
    }
    
    # If it's a POST request, handle metadata updates
    if request.method == 'POST' and (request.user.is_superuser or 
            (request.user.role and request.user.role.name == 'ADMIN')):
        form = DatasetMetadataForm(request.POST, instance=metadata_obj)
        if form.is_valid():
            metadata_obj = form.save(commit=False)
            metadata_obj.dataset_id = dataset_id
            # Note: updated_by field might not exist in the model
            metadata_obj.save()
            messages.success(request, 'Metadata updated successfully')
            return redirect('dataset:detail', dataset_id=dataset_id)
    else:
        form = DatasetMetadataForm(instance=metadata_obj)
    
    # Format file size for display
    def format_file_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
    
    formatted_size = format_file_size(dataset.file_size)
    context = {
        'dataset': dataset,
        'columns_count': columns_count,
        'uploaded_by_user': uploaded_by_user,
        'metadata': metadata,
        'access_history': access_history,
        'stats': stats,
        'formatted_size': formatted_size,
        'form': form,
        'can_edit': request.user.is_superuser or 
                   (request.user.role and request.user.role.name == 'ADMIN')
    }
    
    return render(request, 'dataset/detail.html', context)


@login_required
@require_role('ADMIN')
def dataset_list(request):
    """List all datasets with advanced filtering, search, and pagination."""

    # Base queryset depending on user role (show ALL datasets, not filtered by is_active)
    if request.user.is_superuser or (request.user.role and request.user.role.name == 'ADMIN'):
        queryset = Dataset.objects.using('datasets_db').all()
    else:
        accessible_dataset_ids = DatasetAccess.objects.using('datasets_db').filter(
            user_id=request.user.id
        ).values_list('dataset_id', flat=True)

        queryset = Dataset.objects.using('datasets_db').filter(
            id__in=accessible_dataset_ids
        )
    
    # Search functionality
    search_query = request.GET.get('search', '').strip()
    if search_query:
        queryset = queryset.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(medical_domain__icontains=search_query)
        )
    
    # Filter by medical domain
    domain_filter = request.GET.get('domain', '').strip()
    if domain_filter:
        queryset = queryset.filter(medical_domain=domain_filter)
    
    # Filter by data type
    type_filter = request.GET.get('data_type', '').strip()
    if type_filter:
        queryset = queryset.filter(data_type=type_filter)
    
    # Filter by file format
    format_filter = request.GET.get('format', '').strip()
    if format_filter:
        queryset = queryset.filter(file_format=format_filter)

    # Filter by status (Active/Paused)
    status_filter = request.GET.get('status', '').strip()
    if status_filter:
        if status_filter == 'active':
            queryset = queryset.filter(is_active=True)
        elif status_filter == 'paused':
            queryset = queryset.filter(is_active=False)

    # Filter by size range
    size_min = request.GET.get('size_min', '').strip()
    size_max = request.GET.get('size_max', '').strip()
    
    if size_min:
        try:
            size_min_bytes = int(size_min) * 1024 * 1024  # Convert MB to bytes
            queryset = queryset.filter(file_size__gte=size_min_bytes)
        except ValueError:
            pass
    
    if size_max:
        try:
            size_max_bytes = int(size_max) * 1024 * 1024  # Convert MB to bytes
            queryset = queryset.filter(file_size__lte=size_max_bytes)
        except ValueError:
            pass
    
    # Filter by date range
    date_from = request.GET.get('date_from', '').strip()
    date_to = request.GET.get('date_to', '').strip()
    
    if date_from:
        try:
            from datetime import datetime
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
            queryset = queryset.filter(uploaded_at__date__gte=date_from_obj)
        except ValueError:
            pass
    
    if date_to:
        try:
            from datetime import datetime
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
            queryset = queryset.filter(uploaded_at__date__lte=date_to_obj)
        except ValueError:
            pass
    
    # Sorting
    sort_by = request.GET.get('sort', 'uploaded_at')
    sort_order = request.GET.get('order', 'desc')
    
    valid_sort_fields = ['name', 'uploaded_at', 'file_size', 'medical_domain', 'patient_count']
    if sort_by in valid_sort_fields:
        if sort_order == 'asc':
            queryset = queryset.order_by(sort_by)
        else:
            queryset = queryset.order_by(f'-{sort_by}')
    else:
        queryset = queryset.order_by('-uploaded_at')
    
    # Pagination
    page_size = request.GET.get('page_size', '10')
    try:
        page_size = int(page_size)
        if page_size not in [10, 25, 50, 100]:
            page_size = 10
    except ValueError:
        page_size = 10
    
    paginator = Paginator(queryset, page_size)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get filter options for dropdowns (show all, not just active)
    all_domains = Dataset.objects.using('datasets_db').values_list(
        'medical_domain', flat=True
    ).distinct()

    all_data_types = Dataset.objects.using('datasets_db').values_list(
        'data_type', flat=True
    ).distinct()

    all_formats = Dataset.objects.using('datasets_db').values_list(
        'file_format', flat=True
    ).distinct()
    
    # Statistics for the filtered results
    total_datasets = queryset.count()
    total_size = queryset.aggregate(total=Sum('file_size'))['total'] or 0
    
    def format_file_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    context = {
        'page_obj': page_obj,
        'datasets': page_obj.object_list,
        'user_role': 'ADMIN' if request.user.is_superuser else (request.user.role.name if request.user.role else None),
        'search_query': search_query,
        'domain_filter': domain_filter,
        'type_filter': type_filter,
        'format_filter': format_filter,
        'status_filter': status_filter,
        'size_min': size_min,
        'size_max': size_max,
        'date_from': date_from,
        'date_to': date_to,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'page_size': page_size,
        'all_domains': all_domains,
        'all_data_types': all_data_types,
        'all_formats': all_formats,
        'total_datasets': total_datasets,
        'total_size_formatted': format_file_size(total_size),
        'has_filters': any([search_query, domain_filter, type_filter, format_filter, status_filter,
                           size_min, size_max, date_from, date_to]),
    }
    
    return render(request, 'dataset/dataset_list.html', context)


@login_required
@require_role('ADMIN')
def dataset_upload(request):
    """Upload new dataset with secure processing."""
    
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name')
            description = request.POST.get('description')
            medical_domain = request.POST.get('medical_domain')
            data_type = request.POST.get('data_type', 'tabular')
            target_column = request.POST.get('target_column')
            
            # Validate required fields
            if not name or not description or not medical_domain:
                return JsonResponse({'success': False, 'error': 'All fields are required'}, status=400)
            
            # Get uploaded file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'success': False, 'error': 'No file provided'}, status=400)
            
            # Save file temporarily
            temp_path = _save_temp_file(uploaded_file)
            
            # Create uploader and process synchronously
            uploader = SecureDatasetUploader(request.user)
            
            try:
                # Perform upload synchronously
                dataset, upload_info = uploader.upload_dataset(
                    file_path=temp_path,
                    name=name,
                    description=description,
                    medical_domain=medical_domain,
                    data_type=data_type,
                    target_column=target_column
                )
                
                # Cleanup temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    temp_dir = os.path.dirname(temp_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                
                # Prepare success message
                success_message = f'Dataset "{name}" uploaded successfully!'
                phi_info = {}
                
                # Add information about removed PHI columns
                if upload_info.get('phi_columns_removed'):
                    removed_columns = upload_info['phi_columns_removed']
                    removed_names = [col['name'] for col in removed_columns]
                    success_message += f' Note: {len(removed_names)} PHI column(s) were automatically removed for security: {", ".join(removed_names)}'
                    phi_info = {
                        'phi_columns_removed': removed_columns,
                        'original_columns': upload_info.get('original_columns', 0),
                        'final_columns': upload_info.get('final_columns', 0)
                    }
                
                messages.success(request, success_message)
                response_data = {
                    'success': True,
                    'message': success_message,
                    'dataset_id': dataset.id
                }
                
                # Add PHI information if columns were removed
                if phi_info:
                    response_data['phi_info'] = phi_info
                
                return JsonResponse(response_data)
                
            except Exception as upload_error:
                # Cleanup temporary file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    temp_dir = os.path.dirname(temp_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                
                logger.error(f"Dataset upload failed: {str(upload_error)}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Upload failed: {str(upload_error)}'
                }, status=400)
            
        except Exception as e:
            logger.error(f"Upload initialization failed: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Upload failed: {str(e)}'}, status=400)
    
    context = {
        'max_file_size': None,  # No size limit for medical datasets
        'allowed_extensions': list(SecureDatasetUploader.ALLOWED_EXTENSIONS.keys())
    }
    
    return render(request, 'dataset/dataset_upload.html', context)


# Helper functions

def _save_temp_file(uploaded_file) -> str:
    """Save uploaded file to temporary location."""
    import tempfile
    
    temp_dir = tempfile.mkdtemp(prefix='upload_')
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return temp_path


def _create_upload_session(user) -> str:
    """Create a unique upload session ID."""
    import uuid
    session_id = str(uuid.uuid4())
    
    # Store initial progress
    _update_upload_progress(session_id, {
        'status': 'initializing',
        'message': 'Preparing upload...',
        'progress': 0,
        'user_id': user.id
    })
    
    return session_id


def _start_async_upload(user, temp_path: str, name: str, description: str, 
                       medical_domain: str, data_type: str, session_id: str):
    """Start asynchronous upload process."""
    
    def progress_callback(status: str, message: str):
        """Callback to update progress."""
        progress_map = {
            'initializing': 10,
            'validating': 20,
            'extracting_metadata': 40,
            'validating_metadata': 60,
            'calculating_checksums': 70,
            'storing': 85,
            'saving': 95,
            'completed': 100
        }
        
        progress_data = {
            'status': status,
            'message': message,
            'progress': progress_map.get(status, 0),
            'user_id': user.id
        }
        
        _update_upload_progress(session_id, progress_data)
        _send_progress_update(session_id, progress_data)
    
    # Create uploader with progress callback
    uploader = SecureDatasetUploader(user, progress_callback)
    
    try:
        # Perform upload
        dataset = uploader.upload_dataset(
            file_path=temp_path,
            name=name,
            description=description,
            medical_domain=medical_domain,
            data_type=data_type
        )
        
        # Final success update
        final_progress = {
            'status': 'completed',
            'message': 'Upload completed successfully!',
            'progress': 100,
            'dataset_id': dataset.id,
            'user_id': user.id
        }
        
        _update_upload_progress(session_id, final_progress)
        _send_progress_update(session_id, final_progress)
        
    except Exception as e:
        # Handle upload error
        error_progress = {
            'status': 'error',
            'message': str(e),
            'progress': 0,
            'error': True,
            'user_id': user.id
        }
        
        _update_upload_progress(session_id, error_progress)
        _send_progress_update(session_id, error_progress)
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                temp_dir = os.path.dirname(temp_path)
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass


def _update_upload_progress(session_id: str, progress_data: Dict[str, Any]):
    """Update upload progress in cache."""
    from django.core.cache import cache
    cache.set(f'upload_progress_{session_id}', progress_data, timeout=3600)


def _get_upload_progress(session_id: str) -> Dict[str, Any]:
    """Get upload progress from cache."""
    from django.core.cache import cache
    return cache.get(f'upload_progress_{session_id}')


def _send_progress_update(session_id: str, progress_data: Dict[str, Any]):
    """Send progress update via WebSocket."""
    # WebSocket functionality temporarily disabled for testing
    pass
    # try:
    #     channel_layer = get_channel_layer()
    #     if channel_layer:
    #         async_to_sync(channel_layer.group_send)(
    #             f'upload_{session_id}',
    #             {
    #                 'type': 'upload_progress',
    #                 'progress': progress_data
    #             }
    #         )
    # except Exception as e:
    #     logger.error(f"Failed to send WebSocket progress update: {str(e)}")


# API Views for AJAX requests

@login_required
@require_http_methods(["GET"])
def upload_progress(request, session_id):
    """Get upload progress for a session."""
    
    progress_data = _get_upload_progress(session_id)
    
    if not progress_data:
        return JsonResponse({'error': 'Session not found'}, status=404)
    
    return JsonResponse(progress_data)


@csrf_exempt
@require_http_methods(["POST"])
@login_required
@require_role('ADMIN')  # Only ADMIN can validate files via web API
def api_validate_file(request):
    """API endpoint to validate file before upload."""
    
    if 'file' not in request.FILES:
        return JsonResponse({'valid': False, 'error': 'No file provided'})
    
    uploaded_file = request.FILES['file']
    
    # Basic validation (only check for empty files)
    if uploaded_file.size == 0:
        return JsonResponse({
            'valid': False, 
            'error': 'File cannot be empty'
        })
    
    # Check extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in SecureDatasetUploader.ALLOWED_EXTENSIONS:
        return JsonResponse({
            'valid': False,
            'error': f'File type not allowed. Allowed: {", ".join(SecureDatasetUploader.ALLOWED_EXTENSIONS.keys())}'
        })
    
    return JsonResponse({'valid': True})


@csrf_exempt
@require_http_methods(["POST"])
@login_required
@require_role('ADMIN')  # Only ADMIN can detect columns via web API
def api_detect_columns(request):
    """API endpoint to detect columns from CSV file for target selection."""
    
    if 'file' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No file provided'})
    
    uploaded_file = request.FILES['file']
    
    # Check if it's a CSV file
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext != '.csv':
        return JsonResponse({
            'success': False, 
            'error': 'Column detection is only available for CSV files'
        })
    
    try:
        # Save file temporarily
        temp_path = _save_temp_file(uploaded_file)
        
        # Create uploader instance and detect columns
        uploader = SecureDatasetUploader(request.user)
        columns = uploader.get_csv_columns(temp_path)
        
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            temp_dir = os.path.dirname(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        
        return JsonResponse({
            'success': True,
            'columns': columns,
            'message': f'Detected {len(columns)} columns'
        })
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            temp_dir = os.path.dirname(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        
        logger.error(f"Column detection failed: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to detect columns: {str(e)}'
        })


@login_required
@require_http_methods(["POST"])
@require_role('ADMIN')  # Only ADMIN can cancel uploads via web API
def api_cancel_upload(request, session_id):
    """API endpoint to cancel an ongoing upload."""
    
    progress_data = {
        'status': 'cancelled',
        'message': 'Upload cancelled by user',
        'progress': 0,
        'error': True,
        'user_id': request.user.id
    }
    
    _update_upload_progress(session_id, progress_data)
    _send_progress_update(session_id, progress_data)
    
    return JsonResponse({'success': True, 'message': 'Upload cancelled'})


User = get_user_model()


@login_required
@require_role('ADMIN')
def dataset_manage_access(request, dataset_id):
    """Manage access permissions for a dataset."""
    
    # Get dataset
    dataset = get_object_or_404(
        Dataset.objects.using('datasets_db'),
        id=dataset_id
    )
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'grant_access':
            user_id = request.POST.get('user_id')
            can_train = request.POST.get('can_train') == 'on'
            can_view_metadata = request.POST.get('can_view_metadata') == 'on'
            

            if not user_id:
                messages.error(request, 'No user selected')
                return redirect('dataset:manage_access', dataset_id=dataset_id)
            
            try:
                
                # Get user from main database
                UserModel = get_user_model()
                user = UserModel.objects.using('default').get(id=user_id)
                
                # Create or update access
                try:
                    access, created = DatasetAccess.objects.using('datasets_db').get_or_create(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        defaults={
                            'assigned_by_id': request.user.id,
                            'can_train': can_train,
                            'can_view_metadata': can_view_metadata
                        }
                    )

                except Exception as e:
                    access = None
                    created = False
                    print(f"Error creating or updating access: {str(e)}")

                if not created:
                    # Update existing access
                    access.can_train = can_train
                    access.can_view_metadata = can_view_metadata
                    access.assigned_by_id = request.user.id
                    access.save()
                
                message = f'Access {"updated" if not created else "granted"} for {user.username}'
                messages.success(request, message)
                
            except Exception as e:
                messages.error(request, f'Error granting access: {str(e)}')
        
        elif action == 'revoke_access':
            user_id = request.POST.get('user_id')
            try:
                DatasetAccess.objects.using('datasets_db').filter(
                    dataset_id=dataset_id,
                    user_id=user_id
                ).delete()
                messages.success(request, 'Access revoked successfully')
            except Exception as e:
                messages.error(request, f'Error revoking access: {str(e)}')
        
        elif action == 'update_permissions':
            user_id = request.POST.get('user_id')
            can_train = request.POST.get('can_train') == 'on'
            can_view_metadata = request.POST.get('can_view_metadata') == 'on'
            
            try:
                access = DatasetAccess.objects.using('datasets_db').get(
                    dataset_id=dataset_id,
                    user_id=user_id
                )
                access.can_train = can_train
                access.can_view_metadata = can_view_metadata
                access.save()
                messages.success(request, 'Permissions updated successfully')
            except Exception as e:
                messages.error(request, f'Error updating permissions: {str(e)}')
        
        return redirect('dataset:manage_access', dataset_id=dataset_id)
    
    # Get current access list with user details
    current_access_raw = DatasetAccess.objects.using('datasets_db').filter(
        dataset_id=dataset_id
    ).order_by('-assigned_at')
    
    # Get user details for each access
    UserModel = get_user_model()
    current_access = []
    for access in current_access_raw:
        user = None
        assigned_by = None
        try:
            user = UserModel.objects.using('default').get(id=access.user_id)
        except UserModel.DoesNotExist:
            user = None
        try:
            assigned_by = UserModel.objects.using('default').get(id=access.assigned_by_id)
        except UserModel.DoesNotExist:
            assigned_by = None
        
        if user:  # Only include if user exists
            current_access.append({
                'access': access,
                'user': user,
                'assigned_by': assigned_by
            })
    
    # Get all users for granting access (exclude those who already have access)
    current_user_ids = [access['user'].id for access in current_access if access['user']]
    try:
        # Get all active users excluding those who already have access
        available_users = UserModel.objects.using('default').filter(
            is_active=True
        ).exclude(
            id__in=current_user_ids
        )
        
        # Prefer non-admin users but include admins if no other users available
        non_admin_users = available_users.exclude(is_superuser=True)
        if non_admin_users.exists():
            # If we have non-admin users, show them first but still include admins
            preferred_users = non_admin_users.exclude(id=request.user.id)
            admin_users = available_users.filter(is_superuser=True).exclude(id=request.user.id)
            
            # Combine: non-admins first, then admins
            from django.db.models import Case, When, Value, IntegerField
            available_users = available_users.exclude(id=request.user.id).annotate(
                priority=Case(
                    When(is_superuser=False, then=Value(1)),
                    When(is_superuser=True, then=Value(2)),
                    default=Value(3),
                    output_field=IntegerField()
                )
            ).order_by('priority', 'first_name', 'username')
        else:
            # If only admins available, exclude current user but show other admins
            available_users = available_users.exclude(id=request.user.id).order_by('first_name', 'username')
        
        # Ensure we have at least some users to show (for debugging)
        if not available_users.exists():
            logger.warning(f"No available users found. Current user IDs excluded: {current_user_ids + [request.user.id]}")
            logger.warning(f"Total active users in system: {UserModel.objects.using('default').filter(is_active=True).count()}")
        
    except Exception as e:
        logger.error(f"Error fetching available users: {str(e)}")
        available_users = UserModel.objects.none()
    
    context = {
        'dataset': dataset,
        'current_access': current_access,
        'available_users': available_users,
        'total_access': len(current_access)
    }
    
    return render(request, 'dataset/manage_access.html', context)


@login_required
@require_role('ADMIN')
def datasets_dashboard(request):
    """Dashboard principal con métricas generales de datasets."""
    
    # Métricas básicas
    total_datasets = Dataset.objects.using('datasets_db').filter(is_active=True).count()
    total_size_bytes = Dataset.objects.using('datasets_db').filter(is_active=True).aggregate(
        total_size=Sum('file_size')
    )['total_size'] or 0
    
    # Convertir bytes a formato legible
    def format_file_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    total_size_formatted = format_file_size(total_size_bytes)    
    from django.db.models.functions import Lower
    datasets_by_domain = Dataset.objects.using('datasets_db').filter(
        is_active=True,
        medical_domain__isnull=False
    ).exclude(
        medical_domain__exact=''
    ).values(
        'medical_domain'
    ).annotate(
        domain_lower=Lower('medical_domain'),
        count=Count('id')
    ).values(
        'domain_lower'
    ).annotate(
        total_count=Count('id'),
        display_name=Min('medical_domain')  # Use the first occurrence for display
    ).order_by('-total_count')
    
    # Datasets recientes (últimos 7 días)
    week_ago = timezone.now() - timedelta(days=7)
    recent_datasets = Dataset.objects.using('datasets_db').filter(
        is_active=True,
        uploaded_at__gte=week_ago
    ).order_by('-uploaded_at')[:5]
    
    # Estadísticas de acceso
    total_assignments = DatasetAccess.objects.using('datasets_db').count()
    active_researchers = DatasetAccess.objects.using('datasets_db').values('user_id').distinct().count()
    
    # Top 5 datasets más accedidos
    top_datasets = Dataset.objects.using('datasets_db').filter(is_active=True).annotate(
        access_assignments=Count('access_permissions')
    ).order_by('-access_count')[:5]
    
    # Datos para gráficos (últimos 30 días, INCLUYENDO HOY)
    today = timezone.now().date()
    daily_uploads = []
    for i in range(29, -1, -1):  # Desde 29 días atrás hasta hoy (0)
        date = today - timedelta(days=i)
        count = Dataset.objects.using('datasets_db').filter(
            uploaded_at__date=date
        ).count()
        daily_uploads.append({
            'date': date.strftime('%Y-%m-%d'),
            'count': count
        })
    
    context = {
        'total_datasets': total_datasets,
        'total_size_formatted': total_size_formatted,
        'total_size_bytes': total_size_bytes,
        'datasets_by_domain': datasets_by_domain,
        'recent_datasets': recent_datasets,
        'total_assignments': total_assignments,
        'active_researchers': active_researchers,
        'top_datasets': top_datasets,
        'daily_uploads': daily_uploads,
    }
    
    return render(request, 'dataset/dashboard.html', context)


@login_required
@require_role('ADMIN')
def dataset_edit(request, dataset_id):
    """
    Edit existing dataset information.
    Only ADMIN and RESEARCHER (owner) can edit datasets.
    """
    # Get dataset from datasets_db
    dataset = get_object_or_404(Dataset.objects.using('datasets_db'), pk=dataset_id, is_active=True)

    # Check permissions: ADMIN can edit any, RESEARCHER can edit own
    if request.user.role.name == 'RESEARCHER' and dataset.uploaded_by_id != request.user.id:
        messages.error(request, 'You do not have permission to edit this dataset.')
        return redirect('dataset:list')

    if request.method == 'POST':
        form = DatasetEditForm(request.POST, instance=dataset)
        if form.is_valid():
            try:
                # Check if target_column changed
                old_target_column = dataset.target_column
                new_target_column = form.cleaned_data.get('target_column')
                target_column_changed = old_target_column != new_target_column

                # Save to datasets_db
                updated_dataset = form.save(commit=False)
                updated_dataset.save(using='datasets_db')

                # Regenerate metadata if target_column changed
                if target_column_changed and new_target_column:
                    logger.info(f'Target column changed from "{old_target_column}" to "{new_target_column}", regenerating metadata...')

                    try:
                        from .uploader import SecureDatasetUploader
                        import pandas as pd

                        # Create uploader instance
                        uploader = SecureDatasetUploader(user=request.user)

                        # Read the dataset file
                        df = pd.read_csv(updated_dataset.file_path)

                        # Get existing metadata
                        dataset_metadata = DatasetMetadata.objects.using('datasets_db').get(dataset=updated_dataset)
                        existing_metadata = dataset_metadata.statistical_summary

                        # Get column_info from existing metadata
                        column_info = existing_metadata.get('column_info', {})

                        # Regenerate target_info with new target column
                        target_info = uploader._analyze_target_column(df, new_target_column, column_info)

                        # Update metadata with new target_info
                        existing_metadata['target_info'] = target_info
                        dataset_metadata.statistical_summary = existing_metadata
                        dataset_metadata.save(using='datasets_db')

                        messages.success(
                            request,
                            f'Dataset "{updated_dataset.name}" updated successfully. '
                            f'Metadata regenerated for new target column: "{new_target_column}".'
                        )
                        logger.info(f'Metadata regenerated for dataset {dataset_id} with new target: {new_target_column}')

                    except Exception as metadata_error:
                        logger.error(f'Error regenerating metadata: {str(metadata_error)}')
                        messages.warning(
                            request,
                            f'Dataset updated but metadata regeneration failed: {str(metadata_error)}'
                        )
                else:
                    messages.success(
                        request,
                        f'Dataset "{updated_dataset.name}" has been updated successfully.'
                    )

                logger.info(f'Dataset {dataset_id} edited by user {request.user.username}')
                return redirect('dataset:list')

            except Exception as e:
                logger.error(f'Error updating dataset {dataset_id}: {str(e)}')
                messages.error(
                    request,
                    f'An error occurred while updating the dataset: {str(e)}'
                )
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = DatasetEditForm(instance=dataset)

    # Get uploader info from default database
    try:
        UserModel = get_user_model()
        uploader = UserModel.objects.using('default').get(id=dataset.uploaded_by_id)
        uploader_name = uploader.username
    except:
        uploader_name = f'User ID: {dataset.uploaded_by_id}'

    context = {
        'form': form,
        'dataset': dataset,
        'uploader_name': uploader_name,
        'page_title': f'Edit Dataset: {dataset.name}'
    }

    return render(request, 'dataset/edit.html', context)


@login_required
@require_role('ADMIN')
@require_http_methods(["POST"])
def dataset_toggle_active(request, dataset_id):
    """
    Toggle dataset active status (pause/activate).
    Only ADMIN can perform this action.
    When paused (is_active=False), dataset becomes unavailable to all users.
    """
    dataset = get_object_or_404(Dataset.objects.using('datasets_db'), pk=dataset_id)

    # Toggle the is_active status
    new_status = not dataset.is_active
    dataset.is_active = new_status
    dataset.save(using='datasets_db')

    # Prepare success message
    action = "activated" if new_status else "paused"
    messages.success(
        request,
        f'Dataset "{dataset.name}" has been {action} successfully. '
        f'{"It is now available to authorized users." if new_status else "It is no longer available to any users."}'
    )

    return redirect('dataset:list')
