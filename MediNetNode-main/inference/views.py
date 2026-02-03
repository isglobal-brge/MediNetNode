"""
Views for inference app (MEMBER-facing).

Handles model management and inference execution for MEMBER users.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum, Count, Q, Avg
from django.core.paginator import Paginator
from django.http import Http404
from datetime import timedelta
from users.decorators import require_role
from inference.models import DeployedModel, PredictionAudit
from inference.forms import ModelUploadForm
from dataset.models import Dataset


@login_required
@require_role('MEMBER', 'ADMIN')
def member_dashboard(request):
    """
    MEMBER dashboard home page.

    Shows:
    - Quick stats (datasets, models, predictions, storage)
    - Recent models table
    - Recent activity feed
    - Recommended actions
    """
    user = request.user

    # ===== STATISTICS CALCULATION =====

    # 1. My Datasets (using uploaded_by_id due to dual-database architecture)
    my_datasets = Dataset.objects.using('datasets_db').filter(uploaded_by_id=user.id)
    my_datasets_count = my_datasets.count()
    my_datasets_recent = my_datasets.filter(
        uploaded_at__gte=timezone.now() - timedelta(days=7)
    ).count()

    # 2. My Models
    my_models = DeployedModel.objects.filter(uploaded_by=user)
    my_models_count = my_models.count()
    my_models_active = my_models.filter(status='approved').count()

    # 3. Predictions This Month
    month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    predictions_this_month = PredictionAudit.objects.filter(
        user=user,
        timestamp__gte=month_start
    ).count()
    predictions_today = PredictionAudit.objects.filter(
        user=user,
        timestamp__gte=timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ).count()

    # 4. Pending Models
    pending_models_count = my_models.filter(status='pending').count()

    # 5. Total Storage
    datasets_size = my_datasets.aggregate(total=Sum('file_size'))['total'] or 0
    models_size = my_models.aggregate(total=Sum('file_size'))['total'] or 0
    total_storage_bytes = datasets_size + models_size

    # Format storage
    if total_storage_bytes >= 1024**3:  # GB
        storage_formatted = f"{total_storage_bytes / (1024**3):.1f} GB"
    elif total_storage_bytes >= 1024**2:  # MB
        storage_formatted = f"{total_storage_bytes / (1024**2):.1f} MB"
    elif total_storage_bytes >= 1024:  # KB
        storage_formatted = f"{total_storage_bytes / 1024:.1f} KB"
    else:
        storage_formatted = f"{total_storage_bytes} B"

    # Storage percentage (assume 5 GB quota for now)
    storage_quota = 5 * 1024**3  # 5 GB in bytes
    storage_percentage = int((total_storage_bytes / storage_quota) * 100) if storage_quota > 0 else 0

    # ===== RECENT MODELS TABLE =====
    recent_models = my_models.order_by('-created_at')[:5]

    # ===== RECENT ACTIVITY FEED =====
    # Combine recent predictions and model events
    recent_predictions = PredictionAudit.objects.filter(user=user).order_by('-timestamp')[:5]
    recent_model_uploads = my_models.order_by('-created_at')[:5]
    recent_datasets_uploads = my_datasets.order_by('-uploaded_at')[:3]

    # Build activity feed (simplified for now)
    activities = []

    for pred in recent_predictions:
        activities.append({
            'type': 'prediction',
            'timestamp': pred.timestamp,
            'description': f'Prediction completed on {pred.model.name if pred.model else "Unknown Model"}',
            'icon': 'bi-check-circle',
            'color': 'success'
        })

    for model in recent_model_uploads:
        activities.append({
            'type': 'model_upload',
            'timestamp': model.created_at,
            'description': f'Model uploaded: {model.name}',
            'status': model.status,
            'icon': 'bi-upload',
            'color': 'primary'
        })

    for dataset in recent_datasets_uploads:
        size_mb = dataset.file_size / (1024**2) if dataset.file_size else 0
        activities.append({
            'type': 'dataset_upload',
            'timestamp': dataset.uploaded_at,
            'description': f'Dataset uploaded: {dataset.name}',
            'size': f'{size_mb:.1f} MB',
            'icon': 'bi-database',
            'color': 'purple'
        })

    # Sort by timestamp and take top 10
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_activities = activities[:10]

    # ===== RECOMMENDED ACTIONS =====
    recommendations = []

    # 1. Pending models alert
    if pending_models_count > 0:
        recommendations.append({
            'type': 'warning',
            'icon': 'bi-hourglass',
            'message': f'You have {pending_models_count} pending model{"s" if pending_models_count > 1 else ""} waiting for validation',
            'action_text': 'Review Models',
            'action_url': '/inference/models/?status=pending'
        })

    # 2. Storage alert
    if storage_percentage >= 75:
        level = 'danger' if storage_percentage >= 90 else 'warning'
        recommendations.append({
            'type': level,
            'icon': 'bi-exclamation-triangle',
            'message': f'Your storage is {storage_percentage}% full ({storage_formatted} / 5.0 GB)',
            'action_text': 'Manage Storage',
            'action_url': '/datasets/'
        })

    # 3. First time user guide
    if my_models_count == 0:
        recommendations.append({
            'type': 'info',
            'icon': 'bi-lightbulb',
            'message': 'Get started by uploading your first model',
            'action_text': 'Upload Model Tutorial',
            'action_url': '/inference/models/upload/'
        })

    # 4. New public models (mock for now - would need logic to track "new")
    # public_models_count = DeployedModel.objects.filter(is_public=True, status='approved').count()

    context = {
        'page_title': 'Dashboard',
        'username': user.username,
        'last_login': user.last_login,

        # Stats
        'my_datasets_count': my_datasets_count,
        'my_datasets_recent': my_datasets_recent,
        'my_models_count': my_models_count,
        'my_models_active': my_models_active,
        'predictions_this_month': predictions_this_month,
        'predictions_today': predictions_today,
        'pending_models_count': pending_models_count,
        'total_storage_bytes': total_storage_bytes,
        'storage_formatted': storage_formatted,
        'storage_percentage': storage_percentage,

        # Tables
        'recent_models': recent_models,
        'recent_activities': recent_activities,

        # Recommendations
        'recommendations': recommendations,
    }

    return render(request, 'inference/member_dashboard.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def my_models(request):
    """
    Display user's uploaded models with filtering and pagination.

    Features:
    - Filter by status (all, pending, approved, rejected, deprecated)
    - Filter by domain (cardiology, neurology, oncology, etc.)
    - Search by model name or description
    - Pagination (10 models per page)
    - Sort by recent (default)
    """
    user = request.user

    # Get filter parameters from query string
    status_filter = request.GET.get('status', 'all')
    domain_filter = request.GET.get('domain', 'all')
    search_query = request.GET.get('q', '').strip()
    sort_by = request.GET.get('sort', 'recent')

    # Base queryset: user's models only
    models = DeployedModel.objects.filter(uploaded_by=user)

    # Apply status filter
    if status_filter and status_filter != 'all':
        models = models.filter(status=status_filter)

    # Apply domain filter
    if domain_filter and domain_filter != 'all':
        models = models.filter(domain=domain_filter)

    # Apply search filter
    if search_query:
        models = models.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(version__icontains=search_query)
        )

    # Apply sorting
    if sort_by == 'name':
        models = models.order_by('name')
    elif sort_by == 'status':
        # Custom ordering: approved, pending, rejected, deprecated
        from django.db.models import Case, When, IntegerField
        status_order = Case(
            When(status='approved', then=0),
            When(status='pending', then=1),
            When(status='rejected', then=2),
            When(status='deprecated', then=3),
            default=4,
            output_field=IntegerField()
        )
        models = models.annotate(status_order=status_order).order_by('status_order', '-created_at')
    elif sort_by == 'domain':
        models = models.order_by('domain', '-created_at')
    else:  # Default: recent
        models = models.order_by('-created_at')

    # Get total count before pagination
    total_count = models.count()

    # Stats by status
    all_user_models = DeployedModel.objects.filter(uploaded_by=user)
    stats = {
        'total': all_user_models.count(),
        'approved': all_user_models.filter(status='approved').count(),
        'pending': all_user_models.filter(status='pending').count(),
        'rejected': all_user_models.filter(status='rejected').count(),
        'deprecated': all_user_models.filter(status='deprecated').count(),
    }

    # Get unique domains for filter dropdown
    domains = DeployedModel.objects.filter(uploaded_by=user).values_list(
        'domain', flat=True
    ).distinct().order_by('domain')

    # Pagination (10 models per page)
    paginator = Paginator(models, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': 'My Models',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'My Models', 'url': None}
        ],
        # Models data
        'models': page_obj,
        'total_count': total_count,
        'stats': stats,
        # Filter states
        'status_filter': status_filter,
        'domain_filter': domain_filter,
        'search_query': search_query,
        'sort_by': sort_by,
        # Filter options
        'domains': domains,
        'status_choices': [
            ('all', 'All Status'),
            ('approved', 'Active'),
            ('pending', 'Pending'),
            ('rejected', 'Rejected'),
            ('deprecated', 'Deprecated'),
        ],
        'sort_choices': [
            ('recent', 'Most Recent'),
            ('name', 'Name (A-Z)'),
            ('status', 'Status'),
            ('domain', 'Domain'),
        ],
    }

    return render(request, 'inference/my_models.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def upload_model(request):
    """
    Upload a new model for inference.

    Features:
    - ONNX model file upload with validation
    - Input/output schema definition (JSON)
    - Optional accuracy field (0-100%)
    - Automatic status assignment (pending for MEMBER, approved for ADMIN)
    """
    if request.method == 'POST':
        form = ModelUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                # Save model instance without committing
                model = form.save(commit=False)

                # Set uploaded_by to current user
                model.uploaded_by = request.user

                # Set status based on user role
                # ADMIN uploads are auto-approved, MEMBER uploads require approval
                if request.user.role and request.user.role.name == 'ADMIN':
                    model.status = 'approved'
                    model.approved_by = request.user
                    model.approved_at = timezone.now()
                else:
                    model.status = 'pending'

                # Save the model (checksum and file_size calculated in model.save())
                model.save()

                # Success message
                if model.status == 'approved':
                    messages.success(
                        request,
                        f'Model "{model.name}" uploaded successfully and is ready for inference.'
                    )
                else:
                    messages.success(
                        request,
                        f'Model "{model.name}" uploaded successfully. It will be available after admin approval.'
                    )

                # Redirect to My Models page
                return redirect('inference:my_models')

            except Exception as e:
                messages.error(
                    request,
                    f'Error saving model: {str(e)}'
                )
        else:
            # Form has validation errors
            messages.error(
                request,
                'Please correct the errors below.'
            )
    else:
        # GET request - display empty form
        form = ModelUploadForm()

    context = {
        'page_title': 'Upload Model',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'Models', 'url': 'inference:my_models'},
            {'name': 'Upload', 'url': None}
        ],
        'form': form,
    }
    return render(request, 'inference/upload_model.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def public_models(request):
    """
    Browse public models available for inference.

    Shows all approved public models that users can use for predictions.
    Features:
    - Filter by domain
    - Search by name/description
    - Sort by recent, name, domain, popularity
    - Pagination (12 per page for grid layout)
    """
    user = request.user

    # Get filter parameters
    domain_filter = request.GET.get('domain', 'all')
    search_query = request.GET.get('q', '').strip()
    sort_by = request.GET.get('sort', 'recent')

    # Base queryset: public and approved models only
    models = DeployedModel.objects.filter(
        is_public=True,
        status='approved'
    )

    # Apply domain filter
    if domain_filter and domain_filter != 'all':
        models = models.filter(domain=domain_filter)

    # Apply search filter
    if search_query:
        models = models.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(domain__icontains=search_query)
        )

    # Apply sorting
    if sort_by == 'name':
        models = models.order_by('name')
    elif sort_by == 'domain':
        models = models.order_by('domain', 'name')
    elif sort_by == 'popular':
        models = models.order_by('-total_predictions', '-created_at')
    else:  # Default: recent
        models = models.order_by('-created_at')

    # Get total count before pagination
    total_count = models.count()

    # Stats by domain (for all public approved models)
    all_public_models = DeployedModel.objects.filter(is_public=True, status='approved')
    domain_stats = all_public_models.values('domain').annotate(
        count=Count('id')
    ).order_by('domain')

    # Convert to dict for easy template access
    stats_by_domain = {item['domain']: item['count'] for item in domain_stats}
    total_public = all_public_models.count()

    # Get unique domains for filter dropdown
    domains = all_public_models.values_list('domain', flat=True).distinct().order_by('domain')

    # Domain display names mapping
    domain_names = dict(DeployedModel.DOMAIN_CHOICES)

    # Pagination (12 per page for 3x4 or 4x3 grid)
    paginator = Paginator(models, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    # Check which models the user owns (for UI differentiation)
    user_model_ids = set(
        DeployedModel.objects.filter(uploaded_by=user).values_list('id', flat=True)
    )

    context = {
        'page_title': 'Public Models',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'Public Models', 'url': None}
        ],
        # Models data
        'models': page_obj,
        'total_count': total_count,
        'total_public': total_public,
        'stats_by_domain': stats_by_domain,
        'user_model_ids': user_model_ids,
        # Filter states
        'domain_filter': domain_filter,
        'search_query': search_query,
        'sort_by': sort_by,
        # Filter options
        'domains': domains,
        'domain_names': domain_names,
        'sort_choices': [
            ('recent', 'Most Recent'),
            ('popular', 'Most Popular'),
            ('name', 'Name (A-Z)'),
            ('domain', 'Domain'),
        ],
    }

    return render(request, 'inference/public_models.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def new_prediction(request):
    """
    Create a new prediction using a deployed model.

    This is Step 1 of the prediction wizard: Model Selection.
    Users can select from their own approved models or public approved models.

    Query params:
    - model: Pre-select a model by ID (from model_detail page)
    - domain: Filter models by domain
    - q: Search models by name
    """
    user = request.user

    # Get filter/search parameters
    domain_filter = request.GET.get('domain', 'all')
    search_query = request.GET.get('q', '').strip()
    preselected_model_id = request.GET.get('model', '')

    # Get user's approved models
    my_models = DeployedModel.objects.filter(
        uploaded_by=user,
        status='approved'
    )

    # Get public approved models (excluding user's own)
    public_models = DeployedModel.objects.filter(
        is_public=True,
        status='approved'
    ).exclude(uploaded_by=user)

    # Apply domain filter
    if domain_filter and domain_filter != 'all':
        my_models = my_models.filter(domain=domain_filter)
        public_models = public_models.filter(domain=domain_filter)

    # Apply search filter
    if search_query:
        search_q = Q(name__icontains=search_query) | Q(description__icontains=search_query)
        my_models = my_models.filter(search_q)
        public_models = public_models.filter(search_q)

    # Order by name
    my_models = my_models.order_by('name')
    public_models = public_models.order_by('name')

    # Get preselected model if provided
    preselected_model = None
    if preselected_model_id:
        try:
            model_id = int(preselected_model_id)
            # Check if user can access this model
            preselected_model = DeployedModel.objects.filter(
                Q(uploaded_by=user, status='approved') |
                Q(is_public=True, status='approved')
            ).filter(id=model_id).first()
        except (ValueError, TypeError):
            pass

    # Get unique domains for filter
    all_accessible_models = DeployedModel.objects.filter(
        Q(uploaded_by=user, status='approved') |
        Q(is_public=True, status='approved')
    )
    domains = all_accessible_models.values_list('domain', flat=True).distinct().order_by('domain')

    context = {
        'page_title': 'New Prediction',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'New Prediction', 'url': None}
        ],
        'wizard_step': 1,
        'wizard_steps': [
            {'num': 1, 'name': 'Select Model', 'icon': 'bi-box-seam'},
            {'num': 2, 'name': 'Load Data', 'icon': 'bi-file-earmark-arrow-up'},
            {'num': 3, 'name': 'Results', 'icon': 'bi-graph-up'},
        ],
        # Models
        'my_models': my_models,
        'public_models': public_models,
        'preselected_model': preselected_model,
        # Filters
        'domain_filter': domain_filter,
        'search_query': search_query,
        'domains': domains,
    }

    return render(request, 'inference/new_prediction.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def prediction_load_data(request):
    """
    Step 2 of prediction wizard: Load Data.

    Handles both GET (display upload form) and POST (process uploaded data).
    Users must select a model in Step 1 before reaching this step.
    """
    user = request.user

    # Get model_id from POST (coming from step 1) or GET (reload page)
    model_id = request.POST.get('model_id') or request.GET.get('model_id')

    if not model_id:
        messages.error(request, 'Please select a model first.')
        return redirect('inference:new_prediction')

    try:
        model_id = int(model_id)
    except (ValueError, TypeError):
        messages.error(request, 'Invalid model selected.')
        return redirect('inference:new_prediction')

    # Verify user has access to this model
    model = DeployedModel.objects.filter(
        Q(uploaded_by=user, status='approved') |
        Q(is_public=True, status='approved')
    ).filter(id=model_id).first()

    if not model:
        messages.error(request, 'Model not found or not accessible.')
        return redirect('inference:new_prediction')

    # Build context for the template
    context = {
        'page_title': 'Load Data',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'New Prediction', 'url': 'inference:new_prediction'},
            {'name': 'Load Data', 'url': None}
        ],
        'wizard_step': 2,
        'wizard_steps': [
            {'num': 1, 'name': 'Select Model', 'icon': 'bi-box-seam'},
            {'num': 2, 'name': 'Load Data', 'icon': 'bi-file-earmark-arrow-up'},
            {'num': 3, 'name': 'Results', 'icon': 'bi-graph-up'},
        ],
        'model': model,
        # Placeholder for data preview (will be populated via AJAX or form submission)
        'data_preview': None,
        'data_errors': [],
    }

    return render(request, 'inference/prediction_load_data.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def run_prediction(request):
    """
    Step 3 of prediction wizard: Execute prediction and show results.

    Receives:
    - model_id: The model to use
    - data_file: CSV or JSON file with input data

    Process:
    1. Validate model access
    2. Parse and validate input data against model schema
    3. Run ONNX inference
    4. Create PredictionAudit record
    5. Display results
    """
    import time
    import csv
    import json
    import hashlib
    import numpy as np

    if request.method != 'POST':
        messages.error(request, 'Invalid request method.')
        return redirect('inference:new_prediction')

    user = request.user
    model_id = request.POST.get('model_id')
    data_file = request.FILES.get('data_file')

    # Validate inputs
    if not model_id or not data_file:
        messages.error(request, 'Model and data file are required.')
        return redirect('inference:new_prediction')

    try:
        model_id = int(model_id)
    except (ValueError, TypeError):
        messages.error(request, 'Invalid model selected.')
        return redirect('inference:new_prediction')

    # Get model and verify access
    model = DeployedModel.objects.filter(
        Q(uploaded_by=user, status='approved') |
        Q(is_public=True, status='approved')
    ).filter(id=model_id).first()

    if not model:
        messages.error(request, 'Model not found or not accessible.')
        return redirect('inference:new_prediction')

    # Track execution time
    start_time = time.time()
    errors = []
    results = None
    input_data = None

    try:
        # Parse the uploaded file
        file_content = data_file.read().decode('utf-8')
        file_ext = data_file.name.split('.')[-1].lower()

        if file_ext == 'csv':
            # Parse CSV
            reader = csv.DictReader(file_content.splitlines())
            rows = list(reader)
            if not rows:
                raise ValueError("CSV file is empty")
            columns = list(rows[0].keys())
        elif file_ext == 'json':
            # Parse JSON
            parsed = json.loads(file_content)
            rows = parsed if isinstance(parsed, list) else [parsed]
            if not rows:
                raise ValueError("JSON file is empty")
            columns = list(rows[0].keys())
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Validate batch size
        if len(rows) > model.max_batch_size:
            raise ValueError(
                f"Batch size ({len(rows)}) exceeds maximum allowed ({model.max_batch_size})"
            )

        # Get expected features from model schema
        expected_features = []
        if 'features' in model.input_schema:
            expected_features = [f['name'].strip('"') for f in model.input_schema['features']]
        elif 'feature_names' in model.input_schema:
            expected_features = [f.strip('"') for f in model.input_schema['feature_names']]

        # Prepare input array for ONNX
        # Clean column names (remove quotes if present)
        clean_columns = [c.strip('"') for c in columns]

        # Build numpy array with features in correct order
        num_features = len(expected_features) if expected_features else len(clean_columns)
        input_array = np.zeros((len(rows), num_features), dtype=np.float32)

        for i, row in enumerate(rows):
            for j, feat in enumerate(expected_features if expected_features else clean_columns):
                # Find matching column (handle quoted/unquoted variations)
                value = None
                for col in columns:
                    if col.strip('"') == feat:
                        value = row[col]
                        break

                if value is not None:
                    try:
                        input_array[i, j] = float(value)
                    except (ValueError, TypeError):
                        input_array[i, j] = 0.0

        # Load and run ONNX model
        import onnxruntime as ort

        model_path = model.model_file.path
        session = ort.InferenceSession(model_path)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: input_array})

        # Parse outputs
        output_names = [o.name for o in session.get_outputs()]
        results = {
            'rows': len(rows),
            'columns': columns[:10],  # First 10 for preview
            'predictions': [],
            'output_type': model.output_schema.get('type', 'unknown'),
        }

        # Process each row's prediction
        for i in range(len(rows)):
            pred_result = {'row_index': i + 1}

            for idx, out_name in enumerate(output_names):
                if 'label' in out_name.lower():
                    # Classification label (may be overridden by probability check)
                    pred_result['model_label'] = int(outputs[idx][i])
                    pred_result['label'] = int(outputs[idx][i])
                elif 'probability' in out_name.lower():
                    # Probabilities (could be dict/map)
                    probs = outputs[idx]
                    if isinstance(probs, list) and len(probs) > i:
                        prob_dict = probs[i]
                        if isinstance(prob_dict, dict):
                            # Convert from 0-1 to percentage (0-100)
                            pred_result['probabilities'] = {
                                str(k): float(v) * 100 for k, v in prob_dict.items()
                            }
                            # Override label based on max probability (fix ONNX inconsistency)
                            if prob_dict:
                                max_class = max(prob_dict.keys(), key=lambda k: prob_dict[k])
                                pred_result['label'] = int(max_class)
                else:
                    # Generic output (regression value)
                    if hasattr(outputs[idx], '__len__') and len(outputs[idx]) > i:
                        pred_result['value'] = float(outputs[idx][i])
                    else:
                        pred_result['value'] = float(outputs[idx])

            results['predictions'].append(pred_result)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Compute input hash for audit
        input_hash = hashlib.sha256(file_content.encode()).hexdigest()

        # Create PredictionAudit record
        audit = PredictionAudit.objects.create(
            user=user,
            ip_address=request.META.get('REMOTE_ADDR', '127.0.0.1'),
            model=model,
            model_name=model.name,
            model_version=model.version,
            model_domain=model.domain,
            records_count=len(rows),
            execution_time_ms=execution_time_ms,
            rate_limit_remaining=model.max_requests_per_minute,  # Simplified
            input_hash=input_hash,
            success=True,
            dp_noise_applied=model.enable_differential_privacy,
        )

        # Update model stats
        model.total_predictions += 1
        model.last_prediction_at = timezone.now()
        model.save(update_fields=['total_predictions', 'last_prediction_at'])

        # Add metadata to results
        results['execution_time_ms'] = execution_time_ms
        results['audit_id'] = str(audit.id)

    except Exception as e:
        errors.append(str(e))
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Create failed audit record
        try:
            input_hash = hashlib.sha256(
                data_file.read() if hasattr(data_file, 'read') else b''
            ).hexdigest()
        except Exception:
            input_hash = hashlib.sha256(b'error').hexdigest()

        PredictionAudit.objects.create(
            user=user,
            ip_address=request.META.get('REMOTE_ADDR', '127.0.0.1'),
            model=model,
            model_name=model.name,
            model_version=model.version,
            model_domain=model.domain,
            records_count=0,
            execution_time_ms=execution_time_ms,
            rate_limit_remaining=model.max_requests_per_minute,
            input_hash=input_hash,
            success=False,
            error_message=str(e),
        )

    # Extract class labels from output_schema (if defined)
    # Format: {'classes': {0: 'Control', 1: 'Case'}} or {'classes': ['Control', 'Case']}
    class_labels = {}
    if model.output_schema:
        classes = model.output_schema.get('classes', {})
        if isinstance(classes, dict):
            # Format: {0: 'Control', 1: 'Case'} or {'0': 'Control', '1': 'Case'}
            class_labels = {str(k): v for k, v in classes.items()}
        elif isinstance(classes, list):
            # Format: ['Control', 'Case'] - index is the class label
            class_labels = {str(i): name for i, name in enumerate(classes)}

    context = {
        'page_title': 'Prediction Results',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'New Prediction', 'url': 'inference:new_prediction'},
            {'name': 'Results', 'url': None}
        ],
        'wizard_step': 3,
        'wizard_steps': [
            {'num': 1, 'name': 'Select Model', 'icon': 'bi-box-seam'},
            {'num': 2, 'name': 'Load Data', 'icon': 'bi-file-earmark-arrow-up'},
            {'num': 3, 'name': 'Results', 'icon': 'bi-graph-up'},
        ],
        'model': model,
        'results': results,
        'errors': errors,
        'class_labels': class_labels,
    }

    return render(request, 'inference/prediction_results.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def my_history(request):
    """
    View prediction history for the current user.

    Features:
    - Filter by model, domain, status (success/failed), date range
    - Search by model name
    - Sort by date, execution time, records count
    - Pagination (20 per page)
    - Stats summary (total, success rate, avg execution time)
    """
    user = request.user

    # Get filter parameters
    model_filter = request.GET.get('model', 'all')
    domain_filter = request.GET.get('domain', 'all')
    status_filter = request.GET.get('status', 'all')
    search_query = request.GET.get('q', '').strip()
    sort_by = request.GET.get('sort', 'recent')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')

    # Base queryset: user's predictions only
    predictions = PredictionAudit.objects.filter(user=user)

    # Apply model filter
    if model_filter and model_filter != 'all':
        predictions = predictions.filter(model_id=model_filter)

    # Apply domain filter
    if domain_filter and domain_filter != 'all':
        predictions = predictions.filter(model_domain=domain_filter)

    # Apply status filter
    if status_filter == 'success':
        predictions = predictions.filter(success=True)
    elif status_filter == 'failed':
        predictions = predictions.filter(success=False)

    # Apply search filter (model name)
    if search_query:
        predictions = predictions.filter(
            Q(model_name__icontains=search_query) |
            Q(model_version__icontains=search_query)
        )

    # Apply date filters
    if date_from:
        try:
            from datetime import datetime
            date_from_parsed = datetime.strptime(date_from, '%Y-%m-%d')
            predictions = predictions.filter(timestamp__date__gte=date_from_parsed)
        except ValueError:
            pass

    if date_to:
        try:
            from datetime import datetime
            date_to_parsed = datetime.strptime(date_to, '%Y-%m-%d')
            predictions = predictions.filter(timestamp__date__lte=date_to_parsed)
        except ValueError:
            pass

    # Apply sorting
    if sort_by == 'model':
        predictions = predictions.order_by('model_name', '-timestamp')
    elif sort_by == 'records':
        predictions = predictions.order_by('-records_count', '-timestamp')
    elif sort_by == 'time':
        predictions = predictions.order_by('-execution_time_ms', '-timestamp')
    elif sort_by == 'oldest':
        predictions = predictions.order_by('timestamp')
    else:  # Default: recent
        predictions = predictions.order_by('-timestamp')

    # Get total count before pagination
    total_count = predictions.count()

    # Calculate stats for all user predictions
    all_user_predictions = PredictionAudit.objects.filter(user=user)
    stats = all_user_predictions.aggregate(
        total=Count('id'),
        successful=Count('id', filter=Q(success=True)),
        failed=Count('id', filter=Q(success=False)),
        total_records=Sum('records_count'),
        avg_execution_time=Avg('execution_time_ms'),
    )

    # Calculate success rate
    if stats['total'] and stats['total'] > 0:
        stats['success_rate'] = (stats['successful'] / stats['total']) * 100
    else:
        stats['success_rate'] = 0

    # Format avg execution time
    if stats['avg_execution_time']:
        stats['avg_execution_time'] = round(stats['avg_execution_time'], 1)
    else:
        stats['avg_execution_time'] = 0

    # Get unique models for filter dropdown (from user's history)
    user_models = all_user_predictions.values(
        'model_id', 'model_name', 'model_version'
    ).distinct().order_by('model_name')

    # Get unique domains for filter dropdown
    domains = all_user_predictions.values_list(
        'model_domain', flat=True
    ).distinct().order_by('model_domain')

    # Pagination (20 per page)
    paginator = Paginator(predictions, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': 'My Prediction History',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'My History', 'url': None}
        ],
        # Predictions data
        'predictions': page_obj,
        'total_count': total_count,
        'stats': stats,
        # Filter states
        'model_filter': model_filter,
        'domain_filter': domain_filter,
        'status_filter': status_filter,
        'search_query': search_query,
        'sort_by': sort_by,
        'date_from': date_from,
        'date_to': date_to,
        # Filter options
        'user_models': user_models,
        'domains': domains,
        'status_choices': [
            ('all', 'All Status'),
            ('success', 'Successful'),
            ('failed', 'Failed'),
        ],
        'sort_choices': [
            ('recent', 'Most Recent'),
            ('oldest', 'Oldest First'),
            ('model', 'By Model'),
            ('records', 'Most Records'),
            ('time', 'Longest Execution'),
        ],
    }

    return render(request, 'inference/my_history.html', context)


@login_required
@require_role('MEMBER', 'ADMIN')
def model_detail(request, model_id):
    """
    View detailed information about a specific model.

    Access control:
    - ADMIN can view all models
    - MEMBER can view their own models + public approved models
    """
    user = request.user
    model = get_object_or_404(DeployedModel, id=model_id)

    # Access control
    is_owner = model.uploaded_by == user
    is_admin = user.role and user.role.name == 'ADMIN'
    is_public_approved = model.is_public and model.status == 'approved'

    if not (is_owner or is_admin or is_public_approved):
        raise Http404("Model not found")

    # Get prediction stats for this model
    prediction_stats = PredictionAudit.objects.filter(model=model).aggregate(
        total_predictions=Count('id'),
        total_records=Sum('records_count'),
    )

    # Recent predictions (last 10)
    recent_predictions = PredictionAudit.objects.filter(
        model=model
    ).order_by('-timestamp')[:10]

    # If owner, get their own predictions count
    user_predictions = 0
    if is_owner or is_admin:
        user_predictions = PredictionAudit.objects.filter(
            model=model,
            user=user
        ).count()

    context = {
        'page_title': f'{model.name} - Details',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': 'inference:member_dashboard'},
            {'name': 'My Models', 'url': 'inference:my_models'},
            {'name': model.name, 'url': None}
        ],
        'model': model,
        'is_owner': is_owner,
        'is_admin': is_admin,
        'prediction_stats': prediction_stats,
        'recent_predictions': recent_predictions,
        'user_predictions': user_predictions,
    }

    return render(request, 'inference/model_detail.html', context)
