from django.urls import path
from . import views

app_name = 'audit'

urlpatterns = [
    # Main auditor dashboard
    path('dashboard/', views.auditor_dashboard, name='auditor_dashboard'),
    
    # Advanced search and filtering
    path('search/', views.audit_search, name='audit_search'),
    
    # Dataset analysis (specialized view)
    path('dataset-analysis/', views.dataset_analysis, name='dataset_analysis'),
    
    # CRITICAL: Medical data analysis - AUDITOR ONLY
    path('medical-data/', views.dataset_real_data_analysis, name='medical_data_analysis'),
    
    # Security incident management
    path('incidents/', views.security_incidents, name='security_incidents'),
    path('incidents/<int:incident_id>/update-state/', views.update_incident_state, name='update_incident_state'),
    
    # Reports and exports
    path('export/', views.export_audit_report, name='export_audit_report'),
    
    # Event management
    path('events/<int:event_id>/mark-reviewed/', views.mark_event_reviewed, name='mark_event_reviewed'),
]