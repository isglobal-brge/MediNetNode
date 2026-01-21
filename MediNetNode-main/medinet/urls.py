
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth.views import LogoutView
from auth_system.views import login_view, login_page, logout_view
from core.views import InitialSetupView, InitialSetupPageView, RootRedirectView, system_settings_view
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Swagger/OpenAPI schema configuration
schema_view = get_schema_view(
    openapi.Info(
        title="MediNet RESEARCHER API",
        default_version='v1',
        description="Stateless API for RESEARCHER users - Federated Learning Platform",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="admin@medinet.com"),
        license=openapi.License(name="Proprietary License"),
    ),
    public=False,  # Only authenticated users can access
    permission_classes=[permissions.AllowAny],  # Allow access to API docs
    patterns=[
        path('api/', include('api.urls')),
    ],
)

urlpatterns = [
    # Initial Setup - Must be first, only accessible without users
    path('setup/', InitialSetupPageView.as_view(), name='initial-setup-page'),
    path('api/setup/', InitialSetupView.as_view(), name='initial-setup-api'),

    path('django-admin/', admin.site.urls),  # Move Django admin to different URL
    path('auth/login/', login_page, name='login'),
    path('auth/logout/', logout_view, name='logout'),

    # System Settings
    path('settings/', system_settings_view, name='system_settings'),

    # Root redirect must be after all specific paths
    path('', RootRedirectView.as_view(), name='root_redirect'),
    path('', include('users.urls')),
    path('datasets/', include('dataset.urls')),
    path('audit/', include('audit.urls')),  # Audit dashboard for AUDITOR users
    path('trainings/', include('trainings.urls')),  # Training monitoring for ADMIN/AUDITOR users
    path('api/', include('api.urls')),  # API endpoints for RESEARCHER users
    # Swagger/API Documentation URLs
    path('api/docs/swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api/docs/redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('api/docs/swagger.json', schema_view.without_ui(cache_timeout=0), name='schema-json'),
]
