"""
URL configuration for trainings app.
All views restricted to ADMIN and AUDITOR users only.
RESEARCHER users have NO ACCESS to these endpoints.
"""
from django.urls import path
from . import views

app_name = 'trainings'

urlpatterns = [
    # Dashboard - Overview and active trainings
    path('', views.dashboard, name='dashboard'),
    
    # Active Sessions - Real-time training monitoring
    path('active/', views.active_sessions, name='active_sessions'),
    path('active/refresh/', views.active_sessions_refresh, name='active_sessions_refresh'),
    
    # Training History - Completed/failed trainings
    path('history/', views.training_history, name='history'),
    path('history/export/', views.export_training_history, name='export_history'),
    
    # Training Session Details
    path('session/<uuid:session_id>/', views.session_detail, name='session_detail'),
    path('session/<uuid:session_id>/cancel/', views.cancel_session, name='cancel_session'),
    
    # AJAX endpoints for real-time updates
    path('api/dashboard-stats/', views.dashboard_stats_api, name='dashboard_stats_api'),
    path('api/active-sessions/', views.active_sessions_api, name='active_sessions_api'),
    path('api/session-status/<uuid:session_id>/', views.session_status_api, name='session_status_api'),
]