
import os

from django.contrib import admin
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.db.models import Q
from django.http import FileResponse, Http404
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth.views import LogoutView
from auth_system.views import login_view, login_page, logout_view
from core.views import InitialSetupView, InitialSetupPageView, RootRedirectView, system_settings_view
from rest_framework import permissions
from rest_framework.authentication import SessionAuthentication
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


def _user_can_access_media(user, rel_path):
    """
    Authorize a user for a specific media file (H7 — IDOR fix). Fail-closed.

    The only DB-managed content under MEDIA_ROOT is inference model files
    (``inference/models/...``). A user may fetch one only when they own it
    (``uploaded_by``) or it is public *and* approved — the same rule enforced
    by the inference views (see inference/views.py run_prediction). This closes
    the IDOR where any authenticated user could enumerate and download other
    users' private/pending ONNX models.

    Any other media path is restricted to ADMIN, so unknown files cannot be
    enumerated by lower-privilege roles. Superusers are NOT auto-granted here
    (consistent with the H9 RBAC hardening); the platform admin has the ADMIN
    role and passes on that basis.
    """
    role_name = getattr(getattr(user, 'role', None), 'name', None)

    if rel_path.startswith('inference/models/'):
        if role_name not in ('MEMBER', 'ADMIN'):
            return False
        from inference.models import DeployedModel
        return DeployedModel.objects.filter(
            Q(uploaded_by=user) | Q(is_public=True, status='approved'),
            model_file=rel_path,
        ).exists()

    # Unknown/other media: ADMIN only (fail-closed).
    return role_name == 'ADMIN'


@login_required
def protected_media(request, path):
    """
    Serve media files only to authenticated *and authorized* users.

    Hardening (H7):
    - Path-traversal: resolve symlinks with realpath and require the file to
      live strictly inside MEDIA_ROOT. The trailing os.sep prevents a sibling
      directory whose name merely starts with 'media' (e.g. 'media_backup')
      from passing a bare startswith check.
    - Authorization: per-file ownership/role check (see _user_can_access_media).
    - Fail-closed: every rejection returns 404 so files cannot be enumerated.
    """
    media_root = os.path.realpath(str(settings.MEDIA_ROOT))
    file_path = os.path.realpath(os.path.join(media_root, path))

    # Containment: must be MEDIA_ROOT itself or strictly inside it.
    if not (file_path == media_root or file_path.startswith(media_root + os.sep)):
        raise Http404
    if not os.path.isfile(file_path):
        raise Http404

    # FileField stores names relative to MEDIA_ROOT with POSIX separators.
    rel_path = os.path.relpath(file_path, media_root).replace(os.sep, '/')
    if not _user_can_access_media(request.user, rel_path):
        raise Http404

    return FileResponse(open(file_path, 'rb'))

# Swagger/OpenAPI schema configuration
schema_view = get_schema_view(
    openapi.Info(
        title="MediNet RESEARCHER API",
        default_version='v2',
        description="Stateless API for RESEARCHER users - Federated Learning Platform",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="admin@medinet.com"),
        license=openapi.License(name="Proprietary License"),
    ),
    public=False,  # Only authenticated users can access
    # Require authentication to view the API schema/docs — AllowAny exposed the
    # full API surface to anonymous users (info disclosure). SessionAuthentication
    # lets browser-authenticated staff (ADMIN/MEMBER) load the docs; anonymous
    # requests are rejected.
    permission_classes=[permissions.IsAuthenticated],
    authentication_classes=[SessionAuthentication],
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
    path('inference/', include('inference.urls')),  # Inference management for MEMBER/ADMIN users
    path('api/', include('api.urls')),  # API endpoints for RESEARCHER users
]

urlpatterns += [
    # Swagger/API Documentation URLs
    path('api/docs/swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api/docs/redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('api/docs/swagger.json', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    # Media files: always require authentication (datasets, models, patient data)
    path('media/<path:path>', protected_media, name='protected_media'),
]
