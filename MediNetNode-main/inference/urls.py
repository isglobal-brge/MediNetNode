"""
URL configuration for inference app.

Handles MEMBER-facing pages for model management and inference execution.
"""
from django.urls import path
from . import views

app_name = 'inference'

urlpatterns = [
    # Dashboard (MEMBER & ADMIN)
    path('dashboard/', views.member_dashboard, name='member_dashboard'),

    # Model Management (MEMBER & ADMIN)
    path('models/', views.my_models, name='my_models'),
    path('models/upload/', views.upload_model, name='upload_model'),
    path('models/public/', views.public_models, name='public_models'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),

    # Inference Execution (MEMBER & ADMIN)
    path('predict/', views.new_prediction, name='new_prediction'),
    path('predict/load-data/', views.prediction_load_data, name='prediction_load_data'),
    path('predict/run/', views.run_prediction, name='run_prediction'),
    path('history/', views.my_history, name='my_history'),
]
