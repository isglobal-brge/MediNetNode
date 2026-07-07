"""
Regression tests for H7 — IDOR in protected media serving.

Verify that the per-file authorization added to ``medinet.urls.protected_media``
closes the IDOR where any authenticated user could enumerate and download other
users' private/pending inference model files.
"""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command

from users.models import Role, CustomUser
from inference.models import DeployedModel
from medinet.urls import _user_can_access_media


@pytest.fixture
def roles(db):
    call_command('setup_roles', '--force')


def _make_model(owner, *, is_public, status, name):
    fake_onnx = SimpleUploadedFile(
        f"{name}.onnx", b"fake onnx", content_type="application/octet-stream"
    )
    return DeployedModel.objects.create(
        name=name, version="1.0.0", description="t", domain="cardiology",
        model_file=fake_onnx,
        input_schema={"feature_names": ["a"], "dtypes": {}, "shape": []},
        output_schema={"output_names": ["o"], "dtypes": {}, "shape": []},
        uploaded_by=owner, is_public=is_public, status=status,
    )


@pytest.mark.django_db
def test_member_cannot_access_other_users_private_model(roles):
    member = Role.objects.get(name='MEMBER')
    owner = CustomUser.objects.create_user(username='owner_h7', password='x', role=member)
    other = CustomUser.objects.create_user(username='other_h7', password='x', role=member)
    model = _make_model(owner, is_public=False, status='pending', name='private_h7')
    rel = model.model_file.name

    # IDOR closed: a different MEMBER cannot reach the private/pending model.
    assert _user_can_access_media(other, rel) is False
    # Owner can still access their own model.
    assert _user_can_access_media(owner, rel) is True


@pytest.mark.django_db
def test_public_approved_model_is_accessible_to_members(roles):
    member = Role.objects.get(name='MEMBER')
    owner = CustomUser.objects.create_user(username='owner_pub', password='x', role=member)
    other = CustomUser.objects.create_user(username='other_pub', password='x', role=member)
    model = _make_model(owner, is_public=True, status='approved', name='public_h7')
    assert _user_can_access_media(other, model.model_file.name) is True


@pytest.mark.django_db
def test_non_member_roles_denied_model_media(roles):
    researcher = CustomUser.objects.create_user(
        username='res_h7', password='x', role=Role.objects.get(name='RESEARCHER')
    )
    owner = CustomUser.objects.create_user(
        username='owner_res', password='x', role=Role.objects.get(name='MEMBER')
    )
    model = _make_model(owner, is_public=True, status='approved', name='pub_for_res')
    # Only MEMBER/ADMIN roles handle inference models — even a public one.
    assert _user_can_access_media(researcher, model.model_file.name) is False


@pytest.mark.django_db
def test_unknown_media_path_is_admin_only(roles):
    admin = CustomUser.objects.create_user(
        username='admin_h7', password='x', role=Role.objects.get(name='ADMIN')
    )
    member = CustomUser.objects.create_user(
        username='member_h7', password='x', role=Role.objects.get(name='MEMBER')
    )
    # A path mapping to no DeployedModel: fail-closed to ADMIN only.
    assert _user_can_access_media(admin, 'datasets/secret.csv') is True
    assert _user_can_access_media(member, 'datasets/secret.csv') is False


@pytest.mark.django_db
def test_http_member_cannot_download_other_users_model(roles, client):
    member = Role.objects.get(name='MEMBER')
    owner = CustomUser.objects.create_user(username='owner_http', password='Pass123!', role=member)
    other = CustomUser.objects.create_user(username='other_http', password='Pass123!', role=member)
    model = _make_model(owner, is_public=False, status='pending', name='http_priv')
    url = model.model_file.url  # /media/inference/models/YYYY/MM/http_priv.onnx

    client.force_login(other)
    assert client.get(url).status_code == 404   # IDOR closed end-to-end

    client.force_login(owner)
    assert client.get(url).status_code == 200    # owner allowed
