from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, Http404
from django.core.paginator import Paginator
from django.db.models import Count, Q, Avg, Max, Min
from django.utils import timezone
from datetime import timedelta, datetime
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import json
import csv

from users.decorators import require_role
from .models import AuditEvent, SecurityIncident, DataAccessLog
from users.models import CustomUser


@require_role('AUDITOR')
def auditor_dashboard(request):
    """Dashboard principal para auditores con métricas y visualizaciones."""
    
    # Configurar período de análisis (por defecto 7 días)
    days = int(request.GET.get('days', 7))
    if days not in [7, 30, 90]:
        days = 7
        
    start_date = timezone.now() - timedelta(days=days)
    
    # Métricas principales
    total_events = AuditEvent.objects.filter(timestamp__gte=start_date).count()
    security_events = AuditEvent.objects.filter(
        timestamp__gte=start_date, 
        severity__in=['CRITICAL', 'SECURITY']
    ).count()
    failed_events = AuditEvent.objects.filter(
        timestamp__gte=start_date, 
        success=False
    ).count()
    
    # Incidentes de seguridad
    open_incidents = SecurityIncident.objects.filter(state__in=['OPEN', 'INVESTIGATING']).count()
    total_incidents = SecurityIncident.objects.count()
    
    # Eventos que requieren revisión
    events_pending_review = AuditEvent.objects.filter(
        requires_review=True, 
        reviewed_at__isnull=True
    ).count()
    
    # Usuarios más activos en el período
    top_users = AuditEvent.objects.filter(
        timestamp__gte=start_date,
        user__isnull=False
    ).values(
        'user__username', 'user__role__name'
    ).annotate(
        event_count=Count('id'),
        avg_risk=Avg('risk_score')
    ).order_by('-event_count')[:10]
    
    # Eventos por categoría para gráfico pie
    events_by_category = AuditEvent.objects.filter(
        timestamp__gte=start_date
    ).values('category').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Timeline de eventos críticos (últimos 10)
    critical_events = AuditEvent.objects.filter(
        timestamp__gte=start_date,
        severity__in=['CRITICAL', 'SECURITY']
    ).select_related('user').order_by('-timestamp')[:10]
    
    # Datos para gráfico de tendencias (últimos 7 días)
    trends_data = []
    for i in range(6, -1, -1):  # Últimos 7 días
        date = timezone.now().date() - timedelta(days=i)
        day_events = AuditEvent.objects.filter(
            timestamp__date=date
        )
        
        trends_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'total': day_events.count(),
            'security': day_events.filter(severity__in=['CRITICAL', 'SECURITY']).count(),
            'failed': day_events.filter(success=False).count()
        })
    
    # Análisis de patrones sospechosos
    suspicious_patterns = _detect_suspicious_patterns(start_date)
    
    context = {
        'days': days,
        'total_events': total_events,
        'security_events': security_events,
        'failed_events': failed_events,
        'open_incidents': open_incidents,
        'total_incidents': total_incidents,
        'events_pending_review': events_pending_review,
        'top_users': top_users,
        'events_by_category': list(events_by_category),
        'critical_events': critical_events,
        'trends_data': json.dumps(trends_data),
        'suspicious_patterns': suspicious_patterns,
    }
    
    return render(request, 'audit/auditor_dashboard.html', context)


@require_role('AUDITOR')
def audit_search(request):
    """Búsqueda avanzada de eventos de auditoría."""
    
    events = AuditEvent.objects.select_related('user', 'reviewed_by').all()
    
    # Filtros
    category_filter = request.GET.get('category')
    if category_filter and category_filter != 'ALL':
        events = events.filter(category=category_filter)
    
    severity_filter = request.GET.get('severity')
    if severity_filter and severity_filter != 'ALL':
        events = events.filter(severity=severity_filter)
        
    user_filter = request.GET.get('user')
    if user_filter:
        events = events.filter(
            Q(user__username__icontains=user_filter) |
            Q(user__email__icontains=user_filter)
        )
    
    ip_filter = request.GET.get('ip_address')
    if ip_filter:
        events = events.filter(ip_address__icontains=ip_filter)
    
    # Filtro por rango de fechas
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            events = events.filter(timestamp__date__gte=start)
        except ValueError:
            pass
            
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            events = events.filter(timestamp__date__lte=end)
        except ValueError:
            pass
    
    # Filtro por risk score mínimo
    min_risk = request.GET.get('min_risk')
    if min_risk:
        try:
            min_risk_value = int(min_risk)
            events = events.filter(risk_score__gte=min_risk_value)
        except ValueError:
            pass
    
    # Búsqueda en texto libre
    search_text = request.GET.get('search')
    if search_text:
        events = events.filter(
            Q(action__icontains=search_text) |
            Q(resource__icontains=search_text) |
            Q(details__icontains=search_text)
        )
    
    # Solo eventos que requieren revisión
    if request.GET.get('requires_review') == 'true':
        events = events.filter(requires_review=True, reviewed_at__isnull=True)
    
    # Ordenamiento
    order_by = request.GET.get('order_by', '-timestamp')
    if order_by in ['-timestamp', 'timestamp', '-risk_score', 'risk_score', 'category', 'severity']:
        events = events.order_by(order_by)
    else:
        events = events.order_by('-timestamp')
    
    # Paginación
    paginator = Paginator(events, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Opciones para filtros
    categories = AuditEvent.CATEGORY_CHOICES
    severities = AuditEvent.SEVERITY_CHOICES
    
    context = {
        'page_obj': page_obj,
        'categories': categories,
        'severities': severities,
        'current_filters': request.GET,
        'total_results': paginator.count
    }
    
    return render(request, 'audit/audit_search.html', context)


@require_role('AUDITOR')
def dataset_analysis(request):
    """Análisis especializado de accesos a datasets."""
    
    # Filtrar solo eventos de acceso a datos
    data_events = AuditEvent.objects.filter(category='DATA_ACCESS').select_related('user')
    
    # Filtros específicos
    days = int(request.GET.get('days', 30))
    start_date = timezone.now() - timedelta(days=days)
    data_events = data_events.filter(timestamp__gte=start_date)
    
    medical_domain = request.GET.get('medical_domain')
    if medical_domain:
        # Filtrar por dominio médico usando DataAccessLog relacionado
        data_events = data_events.filter(
            data_access_log__medical_domain__icontains=medical_domain
        )
    
    # Estadísticas generales
    total_accesses = data_events.count()
    unique_users = data_events.values('user').distinct().count()
    unique_datasets = data_events.values('resource').distinct().count()
    
    # Top datasets más accedidos - filtrar solo accesos reales a datasets
    top_datasets = data_events.filter(
        resource__startswith='dataset:'
    ).values('resource').annotate(
        access_count=Count('id'),
        unique_users=Count('user', distinct=True)
    ).order_by('-access_count')[:10]
    
    # Clean dataset names (remove 'dataset:' prefix)
    for dataset in top_datasets:
        dataset['dataset_name'] = dataset['resource'].replace('dataset:', '')
        dataset['resource_display'] = dataset['dataset_name']
    
    # Get available medical domains from dataset models
    from dataset.models import Dataset
    available_domains = Dataset.objects.using('datasets_db').values_list('medical_domain', flat=True).distinct().exclude(medical_domain__isnull=True).exclude(medical_domain__exact='')
    
    # Accesos por dominio médico
    domain_stats = DataAccessLog.objects.filter(
        audit_event__timestamp__gte=start_date
    ).values('medical_domain').annotate(
        access_count=Count('id'),
        avg_sensitivity=Avg('data_sensitivity_level')
    ).order_by('-access_count')
    
    # Patrones de acceso sospechosos específicos para datasets
    suspicious_data_patterns = _detect_suspicious_data_patterns(start_date)
    
    # Accesos por hora del día (heatmap)
    hourly_access = {}
    for hour in range(24):
        hourly_access[hour] = data_events.filter(
            timestamp__hour=hour
        ).count()
    
    context = {
        'days': days,
        'total_accesses': total_accesses,
        'unique_users': unique_users,
        'unique_datasets': unique_datasets,
        'top_datasets': top_datasets,
        'domain_stats': domain_stats,
        'suspicious_data_patterns': suspicious_data_patterns,
        'hourly_access': hourly_access,
        'current_filters': request.GET,
        'available_domains': list(available_domains),
    }
    
    return render(request, 'audit/dataset_analysis.html', context)


@require_role('AUDITOR')
def security_incidents(request):
    """Gestión de incidentes de seguridad."""
    
    incidents = SecurityIncident.objects.select_related('assigned_to').all()
    
    # Filtros
    state_filter = request.GET.get('state')
    # Default filter: show only open and investigating incidents if no filter specified
    if not state_filter:
        incidents = incidents.filter(state__in=['OPEN', 'INVESTIGATING'])
    elif state_filter != 'ALL':
        incidents = incidents.filter(state=state_filter)
    
    severity_filter = request.GET.get('severity')
    if severity_filter:
        try:
            severity_value = int(severity_filter)
            incidents = incidents.filter(severity=severity_value)
        except ValueError:
            pass
    
    # Ordenamiento
    incidents = incidents.order_by('-created_at')
    
    # Paginación
    paginator = Paginator(incidents, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Estadísticas de incidentes
    incident_stats = {
        'total': SecurityIncident.objects.count(),
        'open': SecurityIncident.objects.filter(state='OPEN').count(),
        'investigating': SecurityIncident.objects.filter(state='INVESTIGATING').count(),
        'resolved': SecurityIncident.objects.filter(state='RESOLVED').count(),
        'closed': SecurityIncident.objects.filter(state='CLOSED').count(),
    }
    
    context = {
        'page_obj': page_obj,
        'incident_stats': incident_stats,
        'states': SecurityIncident.STATE_CHOICES,
        'severity_levels': SecurityIncident.SEVERITY_LEVELS,
        'current_filters': request.GET,
    }
    
    return render(request, 'audit/security_incidents.html', context)


@require_role('AUDITOR')
@require_http_methods(['POST'])
def update_incident_state(request, incident_id):
    """Actualizar estado de un incidente de seguridad."""
    
    incident = get_object_or_404(SecurityIncident, id=incident_id)
    new_state = request.POST.get('state')
    
    if new_state not in dict(SecurityIncident.STATE_CHOICES):
        return JsonResponse({'success': False, 'error': 'Estado no válido'})
    
    old_state = incident.state
    incident.state = new_state
    incident.save()
    
    # Log de la acción del auditor
    from .audit_logger import AuditLogger
    AuditLogger.log_event(
        action='INCIDENT_STATE_UPDATE',
        resource=f'incident:{incident.id}',
        user=request.user,
        success=True,
        details={
            'old_state': old_state,
            'new_state': new_state,
            'incident_type': incident.incident_type
        }
    )
    
    messages.success(request, f'Incidente {incident.id} actualizado a {new_state}')
    return JsonResponse({'success': True, 'message': 'Estado actualizado correctamente'})


@require_role('AUDITOR')
@require_http_methods(['POST'])
def mark_event_reviewed(request, event_id):
    """Mark an audit event as reviewed."""
    try:
        event = get_object_or_404(AuditEvent, id=event_id)
        
        if event.reviewed_at:
            return JsonResponse({'success': False, 'error': 'Event already reviewed'})
        
        # Mark as reviewed
        event.mark_reviewed(request.user)
        
        # Log this action
        from .audit_logger import AuditLogger
        AuditLogger.log_event(
            action='MARK_EVENT_REVIEWED',
            resource=f'event:{event_id}',
            user=request.user,
            success=True,
            details={
                'original_action': event.action,
                'original_risk_score': event.risk_score,
                'event_id': event_id
            }
        )
        
        return JsonResponse({'success': True, 'message': 'Event marked as reviewed'})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@require_role('AUDITOR')
def export_audit_report(request):
    """Exportar reporte de auditoría en CSV."""
    
    # Construir queryset basado en filtros
    events = AuditEvent.objects.select_related('user', 'reviewed_by').all()
    
    # Aplicar los mismos filtros que en audit_search
    category_filter = request.GET.get('category')
    if category_filter and category_filter != 'ALL':
        events = events.filter(category=category_filter)
    
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            events = events.filter(timestamp__date__gte=start)
        except ValueError:
            pass
            
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            events = events.filter(timestamp__date__lte=end)
        except ValueError:
            pass
    
    # Crear respuesta CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="audit_report_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Timestamp', 'Category', 'Action', 'User', 'Resource', 'IP Address',
        'Success', 'Risk Score', 'Severity', 'Requires Review', 'Details'
    ])
    
    for event in events[:5000]:  # Limitar a 5000 registros para evitar timeouts
        writer.writerow([
            event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            event.category,
            event.action,
            event.user.username if event.user else 'N/A',
            event.resource,
            event.ip_address or 'N/A',
            'Yes' if event.success else 'No',
            event.risk_score,
            event.severity,
            'Yes' if event.requires_review else 'No',
            json.dumps(event.details) if event.details else ''
        ])
    
    return response


def _detect_suspicious_patterns(start_date):
    """Detectar patrones sospechosos en los eventos de auditoría."""
    patterns = []
    
    # 1. Accesos fuera de horario laboral (22:00-06:00) - exclude ADMIN/AUDITOR
    night_accesses = AuditEvent.objects.filter(
        timestamp__gte=start_date,
        timestamp__hour__in=[22, 23, 0, 1, 2, 3, 4, 5, 6]
    ).exclude(
        user__role__name__in=['ADMIN', 'AUDITOR']
    ).count()
    
    if night_accesses > 0:
        patterns.append({
            'type': 'NIGHT_ACCESS',
            'title': 'Accesos fuera de horario laboral',
            'count': night_accesses,
            'severity': 'medium' if night_accesses < 10 else 'high'
        })
    
    # 2. Usuarios con múltiples fallos de autenticación
    failed_auth_users = AuditEvent.objects.filter(
        timestamp__gte=start_date,
        category='AUTH',
        success=False
    ).values('user__username').annotate(
        fail_count=Count('id')
    ).filter(fail_count__gte=3)
    
    for user in failed_auth_users:
        patterns.append({
            'type': 'MULTIPLE_AUTH_FAILURES',
            'title': f'Multiple authentication failures: {user["user__username"]}',
            'count': user['fail_count'],
            'severity': 'high'
        })
    
    # 3. Anomalous activity spikes by user - exclude ADMIN/AUDITOR who have legitimate high activity
    high_activity_users = AuditEvent.objects.filter(
        timestamp__gte=start_date
    ).exclude(
        user__role__name__in=['ADMIN', 'AUDITOR']
    ).values('user__username').annotate(
        event_count=Count('id')
    ).filter(event_count__gte=100)  # More than 100 events in the period
    
    for user in high_activity_users:
        patterns.append({
            'type': 'HIGH_ACTIVITY',
            'title': f'High anomalous activity: {user["user__username"]}',
            'count': user['event_count'],
            'severity': 'medium'
        })
    
    return patterns


@require_role('AUDITOR')
def dataset_real_data_analysis(request):
    """Vista especializada para análisis de datos reales de datasets - SOLO AUDITORES."""
    from dataset.models import Dataset
    import pandas as pd
    import os
    from django.utils import timezone
    import hashlib
    
    # Log del acceso a datos reales para auditoría
    from .audit_logger import AuditLogger
    AuditLogger.log_event(
        action='REAL_DATA_ACCESS_REQUEST',
        resource='dataset_real_data_analysis',
        user=request.user,
        success=True,
        details={
            'watermark_applied': True,
            'access_level': 'AUDITOR_REAL_DATA',
            'max_rows': 100
        }
    )
    
    # Obtener todos los datasets disponibles
    datasets = Dataset.objects.using('datasets_db').all()
    
    selected_dataset_id = request.GET.get('dataset_id')
    dataset_data = None
    watermark_info = None
    compliance_analysis = None
    anonymization_score = None
    
    if selected_dataset_id:
        try:
            selected_dataset = Dataset.objects.using('datasets_db').get(id=selected_dataset_id)
            
            # Crear watermark único para trazabilidad
            watermark_info = {
                'auditor': request.user.username,
                'timestamp': timezone.now(),
                'dataset_id': selected_dataset_id,
                'dataset_name': selected_dataset.name,
                'access_hash': hashlib.md5(
                    f"{request.user.id}-{selected_dataset_id}-{timezone.now().isoformat()}".encode()
                ).hexdigest()[:8].upper()
            }
            
            # Verificar que el archivo existe
            if os.path.exists(selected_dataset.file_path):
                try:
                    # Leer solo las primeras 100 filas (límite de seguridad)
                    if selected_dataset.file_path.endswith('.csv'):
                        df = pd.read_csv(selected_dataset.file_path, nrows=100)
                    elif selected_dataset.file_path.endswith('.json'):
                        df = pd.read_json(selected_dataset.file_path, lines=True).head(100)
                    else:
                        df = None
                    
                    if df is not None:
                        # Análisis de anonimización automática
                        anonymization_score = _analyze_anonymization_quality(df)
                        
                        # Análisis de cumplimiento
                        compliance_analysis = _analyze_dataset_compliance(df, selected_dataset)
                        
                        # Convertir a diccionario para template con límite de filas
                        dataset_data = {
                            'dataframe': df,
                            'columns': list(df.columns),
                            'row_count': len(df),
                            'total_rows_in_file': selected_dataset.rows_count if selected_dataset.rows_count else 'Unknown',
                            'data_preview': df.head(100).to_dict('records'),  # Máximo 100 filas
                            'column_types': df.dtypes.to_dict(),
                            'missing_values': df.isnull().sum().to_dict(),
                            'unique_counts': df.nunique().to_dict()
                        }
                        
                        # Log específico del acceso a datos reales
                        AuditLogger.log_event(
                            action='REAL_DATA_PREVIEW_ACCESSED',
                            resource=f'dataset:{selected_dataset.name}',
                            user=request.user,
                            success=True,
                            details={
                                'dataset_id': selected_dataset_id,
                                'rows_accessed': len(df),
                                'columns_accessed': list(df.columns),
                                'watermark_hash': watermark_info['access_hash'],
                                'medical_domain': selected_dataset.medical_domain,
                                'anonymization_score': anonymization_score,
                                'compliance_status': compliance_analysis.get('overall_score', 0) if compliance_analysis else 0
                            }
                        )
                    
                except Exception as e:
                    dataset_data = {'error': f'Error reading dataset: {str(e)}'}
            else:
                dataset_data = {'error': 'Dataset file not found on filesystem'}
                
        except Dataset.DoesNotExist:
            dataset_data = {'error': 'Dataset not found'}
    
    # Estadísticas de uso por dominio médico
    domain_usage_stats = _get_domain_usage_statistics()
    
    # Detección de accesos anómalos
    anomaly_detection = _detect_anomalous_access_patterns()
    
    # Análisis de cumplimiento de políticas
    policy_compliance = _analyze_policy_compliance()
    
    # Datasets sin uso reciente (unused)
    unused_datasets = _find_unused_datasets()
    
    context = {
        'datasets': datasets,
        'selected_dataset_id': selected_dataset_id,
        'dataset_data': dataset_data,
        'watermark_info': watermark_info,
        'compliance_analysis': compliance_analysis,
        'anonymization_score': anonymization_score,
        'domain_usage_stats': domain_usage_stats,
        'anomaly_detection': anomaly_detection,
        'policy_compliance': policy_compliance,
        'unused_datasets': unused_datasets,
        'max_preview_rows': 100,
        'current_user': request.user,
    }
    
    return render(request, 'audit/medical_data_analysis.html', context)


def _analyze_anonymization_quality(df):
    """Analizar calidad de anonimización del dataset."""
    if df is None or df.empty:
        return {'score': 0, 'issues': ['No data available']}
    
    score = 100
    issues = []
    
    # Detectar columnas potencialmente identificadoras
    potential_identifiers = ['name', 'email', 'phone', 'address', 'id', 'ssn', 'dni', 'cedula']
    identifier_columns = [col for col in df.columns if any(identifier in col.lower() for identifier in potential_identifiers)]
    
    if identifier_columns:
        score -= 30
        issues.append(f'Potential identifier columns: {", ".join(identifier_columns)}')
    
    # Verificar valores únicos altos (posibles identificadores)
    for col in df.columns:
        if df[col].dtype == 'object':  # Columnas de texto
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # Más del 90% valores únicos
                score -= 10
                issues.append(f'High uniqueness in column "{col}": {unique_ratio:.2%}')
    
    # Verificar patrones de email, teléfono, etc.
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().astype(str).head(10)
            for value in sample_values:
                if '@' in value and '.' in value:  # Posible email
                    score -= 20
                    issues.append(f'Email-like pattern in column "{col}"')
                    break
                if len(value) >= 9 and value.replace('-', '').replace(' ', '').isdigit():  # Posible teléfono
                    score -= 15
                    issues.append(f'Phone-like pattern in column "{col}"')
                    break
    
    return {
        'score': max(0, score),
        'level': 'EXCELLENT' if score >= 90 else 'GOOD' if score >= 70 else 'FAIR' if score >= 50 else 'POOR',
        'issues': issues if issues else ['No anonymization issues detected']
    }


def _analyze_dataset_compliance(df, dataset):
    """Analizar cumplimiento del dataset con regulaciones médicas."""
    if df is None or df.empty:
        return {'overall_score': 0, 'checks': []}
    
    compliance_checks = []
    total_score = 0
    max_score = 0
    
    # Check 1: K-anonymity básico
    max_score += 20
    k_value = df.groupby(list(df.columns)).size().min() if not df.empty else 0
    if k_value >= 3:
        total_score += 20
        compliance_checks.append({'check': 'K-Anonymity (k≥3)', 'status': 'PASS', 'score': 20})
    else:
        compliance_checks.append({'check': 'K-Anonymity (k≥3)', 'status': 'FAIL', 'score': 0})
    
    # Check 2: Datos sensibles encriptados/anonimizados
    max_score += 25
    sensitive_patterns = ['password', 'ssn', 'credit', 'bank']
    sensitive_found = False
    for col in df.columns:
        if any(pattern in col.lower() for pattern in sensitive_patterns):
            sensitive_found = True
            break
    
    if not sensitive_found:
        total_score += 25
        compliance_checks.append({'check': 'No Sensitive Data Patterns', 'status': 'PASS', 'score': 25})
    else:
        compliance_checks.append({'check': 'No Sensitive Data Patterns', 'status': 'FAIL', 'score': 0})
    
    # Check 3: Completeness (menos del 20% valores faltantes)
    max_score += 20
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio <= 0.2:
        total_score += 20
        compliance_checks.append({'check': 'Data Completeness (≤20% missing)', 'status': 'PASS', 'score': 20})
    else:
        compliance_checks.append({'check': 'Data Completeness (≤20% missing)', 'status': 'FAIL', 'score': 0})
    
    # Check 4: Dominio médico válido
    max_score += 15
    valid_domains = ['cardiology', 'neurology', 'oncology', 'radiology', 'pathology', 'general']
    if dataset.medical_domain and any(domain in dataset.medical_domain.lower() for domain in valid_domains):
        total_score += 15
        compliance_checks.append({'check': 'Valid Medical Domain', 'status': 'PASS', 'score': 15})
    else:
        compliance_checks.append({'check': 'Valid Medical Domain', 'status': 'FAIL', 'score': 0})
    
    # Check 5: Tamaño mínimo de dataset
    max_score += 20
    if len(df) >= 50:  # Mínimo 50 registros
        total_score += 20
        compliance_checks.append({'check': 'Minimum Dataset Size (≥50 records)', 'status': 'PASS', 'score': 20})
    else:
        compliance_checks.append({'check': 'Minimum Dataset Size (≥50 records)', 'status': 'FAIL', 'score': 0})
    
    overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
    
    return {
        'overall_score': round(overall_score, 1),
        'total_points': total_score,
        'max_points': max_score,
        'compliance_level': 'EXCELLENT' if overall_score >= 90 else 'GOOD' if overall_score >= 70 else 'FAIR' if overall_score >= 50 else 'POOR',
        'checks': compliance_checks
    }


def _get_domain_usage_statistics():
    """Obtener estadísticas de uso por dominio médico."""
    from datetime import timedelta
    
    end_date = timezone.now()
    start_date = end_date - timedelta(days=90)  # Últimos 90 días
    
    domain_stats = DataAccessLog.objects.filter(
        audit_event__timestamp__gte=start_date,
        audit_event__timestamp__lte=end_date
    ).values('medical_domain').annotate(
        access_count=Count('id'),
        unique_users=Count('audit_event__user', distinct=True),
        avg_sensitivity=Avg('data_sensitivity_level'),
        total_records=Count('records_accessed'),
        avg_patients=Avg('patient_count_accessed')
    ).order_by('-access_count')
    
    return list(domain_stats)


def _detect_anomalous_access_patterns():
    """Detectar patrones anómalos de acceso a datos."""
    from datetime import timedelta
    
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)  # Últimos 30 días
    
    anomalies = []
    
    # 1. Accesos fuera de horario (22:00 - 06:00)
    night_accesses = AuditEvent.objects.filter(
        category='DATA_ACCESS',
        timestamp__gte=start_date,
        timestamp__hour__in=[22, 23, 0, 1, 2, 3, 4, 5, 6]
    ).select_related('user').values(
        'user__username', 'timestamp__date'
    ).annotate(
        night_access_count=Count('id')
    ).filter(night_access_count__gte=3)
    
    for access in night_accesses:
        anomalies.append({
            'type': 'NIGHT_ACCESS',
            'severity': 'HIGH',
            'description': f'User {access["user__username"]} accessed data {access["night_access_count"]} times during night hours',
            'user': access['user__username'],
            'date': access['timestamp__date']
        })
    
    # 2. Volumen anormal de accesos por usuario
    high_volume_users = AuditEvent.objects.filter(
        category='DATA_ACCESS',
        timestamp__gte=start_date
    ).values('user__username').annotate(
        access_count=Count('id')
    ).filter(access_count__gte=50)  # Más de 50 accesos en 30 días
    
    for user in high_volume_users:
        anomalies.append({
            'type': 'HIGH_VOLUME',
            'severity': 'MEDIUM',
            'description': f'User {user["user__username"]} has unusually high access volume: {user["access_count"]} accesses',
            'user': user['user__username'],
            'count': user['access_count']
        })
    
    # 3. Acceso a múltiples dominios médicos por usuario
    cross_domain_users = DataAccessLog.objects.filter(
        audit_event__timestamp__gte=start_date
    ).values('audit_event__user__username').annotate(
        domain_count=Count('medical_domain', distinct=True)
    ).filter(domain_count__gte=4)
    
    for user in cross_domain_users:
        anomalies.append({
            'type': 'CROSS_DOMAIN',
            'severity': 'MEDIUM',
            'description': f'User {user["audit_event__user__username"]} accessed {user["domain_count"]} different medical domains',
            'user': user['audit_event__user__username'],
            'count': user['domain_count']
        })
    
    return anomalies


def _analyze_policy_compliance():
    """Analizar cumplimiento de políticas de acceso."""
    from datetime import timedelta
    
    end_date = timezone.now()
    start_date = end_date - timedelta(days=90)  # Últimos 90 días
    
    compliance_issues = []
    
    # 1. Usuarios con permisos excesivos (acceso a >5 datasets diferentes) - exclude ADMIN/AUDITOR
    users_with_excessive_access = AuditEvent.objects.filter(
        category='DATA_ACCESS',
        timestamp__gte=start_date
    ).exclude(
        user__role__name__in=['ADMIN', 'AUDITOR']
    ).values('user__username', 'user__role__name').annotate(
        dataset_count=Count('resource', distinct=True)
    ).filter(dataset_count__gte=5)
    
    for user in users_with_excessive_access:
        compliance_issues.append({
            'type': 'EXCESSIVE_PERMISSIONS',
            'severity': 'MEDIUM',
            'user': user['user__username'],
            'role': user['user__role__name'],
            'description': f'User has access to {user["dataset_count"]} datasets (policy limit: 5)',
            'count': user['dataset_count']
        })
    
    # 2. Violaciones de segregation of duties (mismo usuario acceso + administración) - exclude ADMIN/AUDITOR who legitimately have both
    users_with_mixed_roles = AuditEvent.objects.filter(
        timestamp__gte=start_date,
        user__isnull=False
    ).exclude(
        user__role__name__in=['ADMIN', 'AUDITOR']
    ).values('user__username').annotate(
        has_data_access=Count('id', filter=Q(category='DATA_ACCESS')),
        has_admin_access=Count('id', filter=Q(category='USER_MGMT'))
    ).filter(has_data_access__gt=0, has_admin_access__gt=0)
    
    for user in users_with_mixed_roles:
        compliance_issues.append({
            'type': 'SEGREGATION_VIOLATION',
            'severity': 'HIGH',
            'user': user['user__username'],
            'description': 'User has both data access and administrative privileges',
            'data_accesses': user['has_data_access'],
            'admin_accesses': user['has_admin_access']
        })
    
    return compliance_issues


def _find_unused_datasets():
    """Encontrar datasets sin uso reciente."""
    from datetime import timedelta
    from dataset.models import Dataset
    
    end_date = timezone.now()
    cutoff_date = end_date - timedelta(days=90)  # Sin uso en 90 días
    
    # Datasets que han sido accedidos recientemente
    recently_accessed_datasets = set(
        AuditEvent.objects.filter(
            category='DATA_ACCESS',
            timestamp__gte=cutoff_date
        ).values_list('resource', flat=True)
    )
    
    # Obtener todos los datasets
    all_datasets = Dataset.objects.using('datasets_db').values('id', 'name', 'medical_domain', 'uploaded_at')
    
    unused_datasets = []
    for dataset in all_datasets:
        dataset_resource = f"dataset:{dataset['name']}"
        if dataset_resource not in recently_accessed_datasets:
            days_unused = (end_date.date() - dataset['uploaded_at'].date()).days if dataset['uploaded_at'] else 0
            unused_datasets.append({
                'id': dataset['id'],
                'name': dataset['name'],
                'medical_domain': dataset['medical_domain'],
                'days_unused': days_unused,
                'severity': 'HIGH' if days_unused > 180 else 'MEDIUM' if days_unused > 90 else 'LOW'
            })
    
    return sorted(unused_datasets, key=lambda x: x['days_unused'], reverse=True)


def _detect_suspicious_data_patterns(start_date):
    """Detectar patrones sospechosos específicos para acceso a datos."""
    patterns = []
    
    # Usuarios accediendo a múltiples dominios médicos
    multi_domain_users = DataAccessLog.objects.filter(
        audit_event__timestamp__gte=start_date
    ).values('audit_event__user__username').annotate(
        domain_count=Count('medical_domain', distinct=True)
    ).filter(domain_count__gte=3)
    
    for user in multi_domain_users:
        patterns.append({
            'type': 'MULTI_DOMAIN_ACCESS',
            'title': f'Acceso a múltiples dominios: {user["audit_event__user__username"]}',
            'count': user['domain_count'],
            'severity': 'medium'
        })
    
    return patterns