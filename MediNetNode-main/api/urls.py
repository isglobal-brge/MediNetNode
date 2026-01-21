"""
URL routing for API endpoints.
Compatible with existing client_api.py structure.
"""
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # Health check endpoint
    path('v1/ping', views.ping, name='ping'),

    # Dataset metadata endpoint
    path('v1/get-data-info', views.get_data_info, name='get_data_info'),

    # Federated learning client start endpoint
    path('v1/start-client', views.start_client, name='start_client'),

    # Cancel training endpoint
    path('v1/cancel-training/<uuid:session_id>', views.cancel_training, name='cancel_training'),
]