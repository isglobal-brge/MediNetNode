"""
URL routing for API endpoints.
Compatible with existing client_api.py structure.
"""
from django.urls import path
from . import views
from .budget_views import request_budget_reset, approve_budget_reset, reject_budget_reset

app_name = 'api'

urlpatterns = [
    # Health check endpoint
    path('v2/ping', views.ping, name='ping'),

    # Dataset metadata endpoint
    path('v2/get-data-info', views.get_data_info, name='get_data_info'),

    # Federated learning client start endpoint
    path('v2/start-client', views.start_client, name='start_client'),

    # Cancel training endpoint
    path('v2/cancel-training/<uuid:session_id>', views.cancel_training, name='cancel_training'),

    # Budget reset endpoints
    path('v2/budget-reset/', request_budget_reset, name='budget_reset_request'),
    path('v2/budget-reset/<int:request_id>/approve/', approve_budget_reset, name='budget_reset_approve'),
    path('v2/budget-reset/<int:request_id>/reject/', reject_budget_reset, name='budget_reset_reject'),

    # Budget visibility endpoints
    path('v2/budget-status/', views.budget_status, name='budget_status'),
    path('v2/estimate-epsilon/', views.estimate_epsilon, name='estimate_epsilon'),

    # DP configuration helper — returns the RDP-derived minimum noise multiplier
    # for a given dataset + training config; used by the Hub UI in real time.
    path('v2/min-noise-multiplier/', views.min_noise_multiplier, name='min_noise_multiplier'),
]