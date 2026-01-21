"""
URL configuration for dataset app.
"""

from django.urls import path
from . import views

app_name = 'dataset'

urlpatterns = [
    # Main views
    path('dashboard/', views.datasets_dashboard, name='dashboard'),
    path('', views.dataset_list, name='list'),
    path('upload/', views.dataset_upload, name='upload'),
    path('<int:dataset_id>/detail/', views.dataset_detail, name='detail'),
    path('<int:dataset_id>/edit/', views.dataset_edit, name='edit'),
    path('<int:dataset_id>/manage-access/', views.dataset_manage_access, name='manage_access'),
    path('<int:dataset_id>/toggle-active/', views.dataset_toggle_active, name='toggle_active'),
    # API endpoints for progress tracking
    path('api/validate-file/', views.api_validate_file, name='api_validate_file'),
    path('api/detect-columns/', views.api_detect_columns, name='api_detect_columns'),
    path('api/upload-progress/<str:session_id>/', views.upload_progress, name='upload_progress'),
    path('api/cancel-upload/<str:session_id>/', views.api_cancel_upload, name='api_cancel_upload'),
]
