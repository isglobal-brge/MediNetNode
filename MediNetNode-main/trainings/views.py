from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count, Avg
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
import csv
from django.http import HttpResponse

from django.contrib.auth import get_user_model

from users.decorators import require_role
from .models import TrainingSession, TrainingRound, BudgetResetRequest
from dataset.models import Dataset, ResearcherEpsilonBudget

User = get_user_model()

"""
Training monitoring views.
All views are restricted to ADMIN and AUDITOR users only.
RESEARCHER users have NO ACCESS to web interfaces.
"""

@login_required
@require_role('ADMIN', 'AUDITOR')
def dashboard(request):
    """
    Training monitoring dashboard with overview statistics and active sessions.
    """
    now = timezone.now()
    today = now.date()
    week_ago = now - timedelta(days=7)

    active_count = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).count()

    today_completed = TrainingSession.objects.filter(
        completed_at__date=today,
        status='COMPLETED'
    ).count()

    week_sessions = TrainingSession.objects.filter(
        started_at__gte=week_ago,
        status__in=['COMPLETED', 'FAILED']
    )

    if week_sessions.count() > 0:
        success_rate = (week_sessions.filter(status='COMPLETED').count() / week_sessions.count()) * 100
    else:
        success_rate = 0

    completed_sessions = TrainingSession.objects.filter(
        completed_at__gte=week_ago,
        status='COMPLETED'
    )

    avg_duration_seconds = 0
    if completed_sessions.exists():
        durations = []
        for session in completed_sessions:
            if session.duration:
                durations.append(session.duration.total_seconds())

        if durations:
            avg_duration_seconds = sum(durations) / len(durations)

    avg_duration_minutes = int(avg_duration_seconds / 60)
    avg_duration_hours = int(avg_duration_minutes / 60)
    avg_duration_display = f"{avg_duration_hours}h {avg_duration_minutes % 60}m"

    active_sessions = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).select_related('user').order_by('-started_at')[:4]

    recent_sessions = TrainingSession.objects.filter(
        status__in=['COMPLETED', 'FAILED', 'CANCELLED']
    ).select_related('user').order_by('-completed_at')[:4]
    
    context = {
        'active_count': active_count,
        'today_completed': today_completed,
        'success_rate': round(success_rate, 1),
        'avg_duration': avg_duration_display,
        'active_sessions': active_sessions,
        'recent_sessions': recent_sessions,
        'pending_count': BudgetResetRequest.objects.filter(status='pending').count(),
        'page_title': 'Training Dashboard',
    }

    return render(request, 'trainings/dashboard.html', context)


@login_required
@require_role('ADMIN', 'AUDITOR')
def active_sessions(request):
    """
    Real-time active training sessions monitoring.
    """
    active_sessions = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).select_related('user').prefetch_related('rounds').order_by('-started_at')
    
    context = {
        'active_sessions': active_sessions,
        'page_title': 'Active Training Sessions'
    }
    
    return render(request, 'trainings/active_sessions.html', context)


@login_required
@require_role('ADMIN', 'AUDITOR')
def training_history(request):
    """
    Training history with filtering and pagination.
    """
    sessions = TrainingSession.objects.select_related('user').order_by('-started_at')

    status_filter = request.GET.get('status')
    user_filter = request.GET.get('user_id')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    search = request.GET.get('search')

    user_display = None
    if user_filter:
        try:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            selected_user = User.objects.get(id=user_filter)
            user_display = selected_user.username
        except (User.DoesNotExist, ValueError):
            user_filter = None

    if status_filter:
        sessions = sessions.filter(status=status_filter)

    if user_filter:
        sessions = sessions.filter(user_id=user_filter)
    
    if date_from:
        try:
            date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
            sessions = sessions.filter(started_at__date__gte=date_from)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
            sessions = sessions.filter(started_at__date__lte=date_to)
        except ValueError:
            pass
    
    if search:
        sessions = sessions.filter(
            Q(client_id__icontains=search) |
            Q(dataset_name__icontains=search) |
            Q(user__username__icontains=search)
        )

    paginator = Paginator(sessions, 20)
    page = request.GET.get('page')
    sessions_page = paginator.get_page(page)

    from django.contrib.auth import get_user_model
    User = get_user_model()
    available_users = User.objects.filter(
        trainingsession__isnull=False
    ).distinct().values('id', 'username')
    
    context = {
        'sessions': sessions_page,
        'available_users': available_users,
        'current_filters': {
            'status': status_filter,
            'user': user_filter,
            'user_display': user_display,
            'date_from': date_from,
            'date_to': date_to,
            'search': search,
        },
        'status_choices': TrainingSession.STATUS_CHOICES,
        'page_title': 'Training History'
    }
    
    return render(request, 'trainings/history.html', context)


@login_required
@require_role('ADMIN', 'AUDITOR')
def session_detail(request, session_id):
    """
    Detailed view of a training session.
    """
    session = get_object_or_404(
        TrainingSession.objects.select_related('user').prefetch_related('rounds'),
        session_id=session_id
    )

    rounds = session.rounds.order_by('round_number')
    
    context = {
        'session': session,
        'rounds': rounds,
        'can_cancel': session.is_active and request.user.role.name == 'ADMIN',
        'page_title': f'Training Session {str(session_id)[:8]}'
    }
    
    return render(request, 'trainings/session_detail.html', context)


@login_required
@require_role('ADMIN')
@require_POST
def cancel_session(request, session_id):
    """
    Cancel active training session (ADMIN only).
    """
    session = get_object_or_404(TrainingSession, session_id=session_id)
    
    if session.cancel_training():
        messages.success(request, f'Training session {str(session_id)[:8]} has been cancelled.')
    else:
        messages.error(request, f'Cannot cancel session {str(session_id)[:8]} - it may already be completed.')
    
    return redirect('trainings:session_detail', session_id=session_id)


@login_required
@require_role('ADMIN', 'AUDITOR')
def export_training_history(request):
    """
    Export training history as CSV.
    """
    sessions = TrainingSession.objects.select_related('user').order_by('-started_at')

    # Apply same filters as history view
    status_filter = request.GET.get('status')
    user_filter = request.GET.get('user_id')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    search = request.GET.get('search')
    
    if status_filter:
        sessions = sessions.filter(status=status_filter)
    
    if user_filter:
        sessions = sessions.filter(user_id=user_filter)
    
    if date_from:
        try:
            date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
            sessions = sessions.filter(started_at__date__gte=date_from)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
            sessions = sessions.filter(started_at__date__lte=date_to)
        except ValueError:
            pass
    
    if search:
        sessions = sessions.filter(
            Q(client_id__icontains=search) |
            Q(dataset_name__icontains=search) |
            Q(user__username__icontains=search)
        )

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="training_history_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Session ID', 'Client ID', 'User', 'Dataset Name', 'Status',
        'Started At', 'Completed At', 'Duration (seconds)', 'Current Round',
        'Total Rounds', 'Progress %', 'Final Accuracy', 'Final Loss',
        'Final F1', 'CPU Usage %', 'Memory Usage MB', 'Error Message'
    ])
    
    for session in sessions:
        duration_seconds = ''
        if session.duration:
            duration_seconds = int(session.duration.total_seconds())
        
        writer.writerow([
            str(session.session_id),
            session.client_id or '',
            session.user.username,
            session.dataset_name,
            session.get_status_display(),
            session.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            session.completed_at.strftime('%Y-%m-%d %H:%M:%S') if session.completed_at else '',
            duration_seconds,
            session.current_round,
            session.total_rounds,
            session.progress_percentage,
            session.final_accuracy or '',
            session.final_loss or '',
            session.final_f1 or '',
            session.cpu_usage or '',
            session.memory_usage or '',
            session.error_message or ''
        ])
    
    return response


@login_required
@require_role('ADMIN', 'AUDITOR')
@require_http_methods(["GET"])
def dashboard_stats_api(request):
    """
    API endpoint for dashboard statistics refresh.
    """
    now = timezone.now()
    today = now.date()
    week_ago = now - timedelta(days=7)
    
    active_count = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).count()
    
    today_completed = TrainingSession.objects.filter(
        completed_at__date=today,
        status='COMPLETED'
    ).count()
    
    week_sessions = TrainingSession.objects.filter(
        started_at__gte=week_ago,
        status__in=['COMPLETED', 'FAILED']
    )
    
    if week_sessions.count() > 0:
        success_rate = (week_sessions.filter(status='COMPLETED').count() / week_sessions.count()) * 100
    else:
        success_rate = 0
    
    return JsonResponse({
        'active_count': active_count,
        'today_completed': today_completed,
        'success_rate': round(success_rate, 1),
        'timestamp': timezone.now().isoformat()
    })


@login_required
@require_role('ADMIN', 'AUDITOR')
@require_http_methods(["GET"])
def active_sessions_api(request):
    """
    API endpoint for active sessions refresh.
    """
    active_sessions = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).select_related('user').order_by('-started_at')
    
    sessions_data = []
    for session in active_sessions:
        sessions_data.append({
            'session_id': str(session.session_id),
            'client_id': session.client_id,
            'user': session.user.username,
            'dataset_name': session.dataset_name,
            'status': session.status,
            'current_round': session.current_round,
            'total_rounds': session.total_rounds,
            'progress_percentage': session.progress_percentage,
            'cpu_usage': session.cpu_usage,
            'memory_usage': session.memory_usage,
            'duration': str(session.duration) if session.duration else None,
            'started_at': session.started_at.isoformat()
        })
    
    return JsonResponse({
        'sessions': sessions_data,
        'count': len(sessions_data),
        'timestamp': timezone.now().isoformat()
    })


@login_required
@require_role('ADMIN', 'AUDITOR')
@require_http_methods(["GET"])
def session_status_api(request, session_id):
    """
    API endpoint for individual session status refresh.
    """
    try:
        session = TrainingSession.objects.select_related('user').get(session_id=session_id)

        rounds_count = session.rounds.count()
        
        return JsonResponse({
            'session_id': str(session.session_id),
            'status': session.status,
            'current_round': session.current_round,
            'total_rounds': session.total_rounds,
            'progress_percentage': session.progress_percentage,
            'cpu_usage': session.cpu_usage,
            'memory_usage': session.memory_usage,
            'is_active': session.is_active,
            'is_finished': session.is_finished,
            'rounds_count': rounds_count,
            'timestamp': timezone.now().isoformat()
        })
    
    except TrainingSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)


@login_required
@require_role('ADMIN', 'AUDITOR')
def active_sessions_refresh(request):
    """
    Partial refresh for active sessions (returns HTML fragment).
    """
    active_sessions = TrainingSession.objects.filter(
        status__in=['STARTING', 'ACTIVE']
    ).select_related('user').prefetch_related('rounds').order_by('-started_at')
    
    context = {
        'active_sessions': active_sessions
    }

    return render(request, 'trainings/partials/active_sessions_table.html', context)


@login_required
@require_role('ADMIN')
def budget_reset_list(request):
    """
    List all budget reset requests, newest first.
    ADMIN only.
    """
    status_filter = request.GET.get('status', '')
    qs = BudgetResetRequest.objects.all()
    if status_filter in ('pending', 'approved', 'rejected'):
        qs = qs.filter(status=status_filter)

    researcher_ids = list(qs.values_list('researcher_id', flat=True).distinct())
    user_map = {u.id: u for u in User.objects.filter(id__in=researcher_ids)}

    # Enrich with dataset names (using datasets_db router if applicable)
    dataset_ids = list(qs.values_list('dataset_id', flat=True).distinct())
    dataset_map = {d.id: d for d in Dataset.objects.filter(id__in=dataset_ids)}

    requests_enriched = []
    for r in qs:
        requests_enriched.append({
            'obj': r,
            'researcher': user_map.get(r.researcher_id),
            'dataset': dataset_map.get(r.dataset_id),
        })

    paginator = Paginator(requests_enriched, 25)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'pending_count': BudgetResetRequest.objects.filter(status='pending').count(),
        'page_title': 'Privacy Budget Reset Requests',
    }
    return render(request, 'trainings/budget_reset_list.html', context)


@login_required
@require_role('ADMIN')
def budget_reset_detail(request, request_id):
    """
    Detail view for a single budget reset request.
    Shows full activity dashboard for the researcher on the affected dataset.
    Approve/Reject actions handled here via POST.
    ADMIN only.
    """
    reset_req = get_object_or_404(BudgetResetRequest, pk=request_id)

    researcher = User.objects.filter(id=reset_req.researcher_id).first()
    try:
        dataset = Dataset.objects.get(id=reset_req.dataset_id)
    except Dataset.DoesNotExist:
        dataset = None

    training_sessions = TrainingSession.objects.filter(
        user_id=reset_req.researcher_id,
        dataset_id=reset_req.dataset_id,
    ).order_by('-started_at')

    epsilon_budget = ResearcherEpsilonBudget.objects.filter(
        dataset_id=reset_req.dataset_id,
        researcher_id=reset_req.researcher_id,
    ).first()

    if request.method == 'POST':
        action = request.POST.get('action')
        notes = request.POST.get('notes', '').strip()

        if action == 'approve':
            try:
                reset_req.approve(admin=request.user, notes=notes)
                if epsilon_budget:
                    epsilon_budget.reset_period()
                messages.success(request, 'Budget reset request approved and budget has been reset.')
            except ValueError as exc:
                messages.error(request, str(exc))
        elif action == 'reject':
            try:
                reset_req.reject(admin=request.user, notes=notes)
                messages.success(request, 'Budget reset request rejected.')
            except ValueError as exc:
                messages.error(request, str(exc))
        else:
            messages.error(request, 'Unknown action.')

        return redirect('trainings:budget_reset_detail', request_id=request_id)

    context = {
        'reset_req': reset_req,
        'researcher': researcher,
        'dataset': dataset,
        'training_sessions': training_sessions,
        'epsilon_budget': epsilon_budget,
        'page_title': f'Budget Reset Request #{request_id}',
    }
    return render(request, 'trainings/budget_reset_detail.html', context)
