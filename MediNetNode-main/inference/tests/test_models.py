"""
Tests for DeployedModel and custom manager.
"""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from users.models import Role, CustomUser
from inference.models import DeployedModel


@pytest.fixture(scope='class')
def setup_roles(django_db_setup, django_db_blocker):
    """Setup roles before running tests."""
    with django_db_blocker.unblock():
        call_command('setup_roles', '--force')


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestDeployedModel:
    """Test DeployedModel model functionality."""

    def test_create_deployed_model(self):
        """Test creating a DeployedModel instance."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_model_test',
            password='testpass123',
            role=admin_role
        )

        # Create fake ONNX file
        fake_onnx = SimpleUploadedFile(
            "test_model.onnx",
            b"fake onnx content for testing",
            content_type="application/octet-stream"
        )

        model = DeployedModel.objects.create(
            name="Test Cardiology Model",
            version="1.0.0",
            description="A test model for cardiology predictions",
            domain="cardiology",
            model_file=fake_onnx,
            input_schema={
                "feature_names": ["age", "blood_pressure", "cholesterol"],
                "dtypes": {"age": "int", "blood_pressure": "float", "cholesterol": "float"},
                "shape": [None, 3]
            },
            output_schema={
                "output_names": ["risk_score"],
                "dtypes": {"risk_score": "float"},
                "shape": [None, 1]
            },
            uploaded_by=admin_user
        )

        assert model.name == "Test Cardiology Model"
        assert model.version == "1.0.0"
        assert model.domain == "cardiology"
        assert model.status == "pending"
        assert model.uploaded_by == admin_user
        assert model.checksum != ""  # Should auto-compute
        assert model.file_size > 0  # Should auto-compute

    def test_model_approval_workflow(self):
        """Test approve/reject/deprecate methods."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_workflow_test',
            password='testpass123',
            role=admin_role
        )

        fake_onnx = SimpleUploadedFile(
            "workflow_model.onnx",
            b"fake onnx content",
            content_type="application/octet-stream"
        )

        model = DeployedModel.objects.create(
            name="Workflow Test Model",
            version="1.0.0",
            description="Testing approval workflow",
            domain="neurology",
            model_file=fake_onnx,
            input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
            output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
            uploaded_by=admin_user
        )

        # Test approval
        assert model.status == "pending"
        model.approve(admin_user)
        assert model.status == "approved"
        assert model.approved_by == admin_user
        assert model.approved_at is not None

        # Test deprecation
        model.deprecate()
        assert model.status == "deprecated"

        # Test rejection
        model.status = "pending"
        model.save()
        model.reject(admin_user, reason="Test rejection")
        assert model.status == "rejected"
        assert "REJECTED: Test rejection" in model.validation_notes

    def test_increment_predictions(self):
        """Test prediction counter increment."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_stats_test',
            password='testpass123',
            role=admin_role
        )

        fake_onnx = SimpleUploadedFile(
            "stats_model.onnx",
            b"fake onnx content",
            content_type="application/octet-stream"
        )

        model = DeployedModel.objects.create(
            name="Stats Test Model",
            version="1.0.0",
            description="Testing prediction stats",
            domain="oncology",
            model_file=fake_onnx,
            input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
            output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
            uploaded_by=admin_user
        )

        assert model.total_predictions == 0
        assert model.last_prediction_at is None

        model.increment_predictions()
        assert model.total_predictions == 1
        assert model.last_prediction_at is not None

        model.increment_predictions()
        assert model.total_predictions == 2


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestDeployedModelManager:
    """Test custom DeployedModelManager."""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        """Create test models for manager tests."""
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.create_user(
            username='admin_manager_test',
            password='testpass123',
            role=admin_role
        )

        # Create models in different domains
        for domain in ['cardiology', 'neurology', 'oncology']:
            fake_onnx = SimpleUploadedFile(
                f"{domain}_model.onnx",
                f"fake onnx for {domain}".encode(),
                content_type="application/octet-stream"
            )

            model = DeployedModel.objects.create(
                name=f"{domain.capitalize()} Model",
                version="1.0.0",
                description=f"Test model for {domain}",
                domain=domain,
                model_file=fake_onnx,
                input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
                output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
                uploaded_by=admin_user,
                status='approved',
                is_public=True
            )

    def test_member_sees_all_approved_models(self):
        """MEMBER with scope ALL should see all approved public models."""
        member_role = Role.objects.get(name='MEMBER')
        member_user = CustomUser.objects.create_user(
            username='member_manager_test',
            password='testpass123',
            role=member_role
        )

        accessible = DeployedModel.objects.accessible_by_user(member_user)
        assert accessible.count() == 3  # All three approved public models

    def test_researcher_sees_all_approved_models(self):
        """RESEARCHER with scope ALL should see all approved public models."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_manager_test',
            password='testpass123',
            role=researcher_role
        )

        accessible = DeployedModel.objects.accessible_by_user(researcher_user)
        assert accessible.count() == 3

    def test_limited_scope_user_sees_only_allowed_domains(self):
        """User with limited scope should only see models in allowed domains."""
        limited_role = Role.objects.create(
            name='LIMITED_CARDIOLOGIST',
            permissions={
                'api.access': True,
                'inference.execute': {'scope': ['cardiology']},
            }
        )
        limited_user = CustomUser.objects.create_user(
            username='limited_manager_test',
            password='testpass123',
            role=limited_role
        )

        accessible = DeployedModel.objects.accessible_by_user(limited_user)
        assert accessible.count() == 1
        assert accessible.first().domain == 'cardiology'

    def test_user_without_permission_sees_nothing(self):
        """User without inference.execute permission sees no models."""
        auditor_role = Role.objects.get(name='AUDITOR')
        auditor_user = CustomUser.objects.create_user(
            username='auditor_manager_test',
            password='testpass123',
            role=auditor_role
        )

        # AUDITOR has inference.view but not inference.execute
        accessible = DeployedModel.objects.accessible_by_user(auditor_user)
        assert accessible.count() == 0

    def test_superuser_sees_all_models(self):
        """Superuser should see all models regardless of status."""
        superuser = CustomUser.objects.create_superuser(
            username='superuser_manager_test',
            password='testpass123',
            email='super@test.com'
        )

        # Create a pending model
        admin_role = Role.objects.get(name='ADMIN')
        admin_user = CustomUser.objects.get(username='admin_manager_test')

        fake_onnx = SimpleUploadedFile(
            "pending_model.onnx",
            b"fake onnx pending",
            content_type="application/octet-stream"
        )

        DeployedModel.objects.create(
            name="Pending Model",
            version="1.0.0",
            description="Pending test model",
            domain="radiology",
            model_file=fake_onnx,
            input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
            output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
            uploaded_by=admin_user,
            status='pending',
            is_public=False
        )

        accessible = DeployedModel.objects.accessible_by_user(superuser)
        # Should see all models including pending and non-public
        assert accessible.count() >= 4
