from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('admin/', views.admin_dashboard, name='admin_dashboard'),
    
    # User management
    path('users/', views.user_list, name='user_list'),
    path('users/create/', views.create_user, name='create_user'),
    path('users/<int:user_id>/', views.user_detail, name='user_detail'),
    path('users/<int:user_id>/edit/', views.update_user, name='update_user'),
    path('users/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    path('users/<int:user_id>/password/', views.change_user_password, name='change_user_password'),
    path('users/<int:user_id>/logs/', views.user_activity_logs, name='user_activity_logs'),
    
    # System logs
    path('system/logs/', views.system_audit_logs, name='system_audit_logs'),
    
    # User creation success
    path('users/created/success/', views.user_created_success, name='user_created_success'),
    path('users/created/download/', views.download_user_info, name='download_user_info'),

    # Export
    path('users/export/', views.export_users_csv, name='export_users_csv'),

    # Researcher info
    path('info/researcher/', views.researcher_info, name='researcher_info'),
]