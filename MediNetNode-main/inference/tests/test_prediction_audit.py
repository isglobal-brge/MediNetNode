"""
Tests for PredictionAudit model.
"""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from users.models import Role, CustomUser, APIKey
from inference.models import DeployedModel, PredictionAudit


@pytest.fixture(scope='class')
def setup_roles(django_db_setup, django_db_blocker):
    """Setup roles before running tests."""
    with django_db_blocker.unblock():
        call_command('setup_roles', '--force')


@pytest.mark.django_db
@pytest.mark.usefixtures('setup_roles')
class TestPredictionAudit:
    """Test PredictionAudit model functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_model(self):
        """Create a test model for audit tests."""
        admin_role = Role.objects.get(name='ADMIN')
        self.admin_user = CustomUser.objects.create_user(
            username='admin_audit_test',
            password='testpass123',
            role=admin_role
        )

        fake_onnx = SimpleUploadedFile(
            "audit_test_model.onnx",
            b"fake onnx content for audit tests",
            content_type="application/octet-stream"
        )

        self.test_model = DeployedModel.objects.create(
            name="Audit Test Model",
            version="1.0.0",
            description="Model for testing prediction audits",
            domain="cardiology",
            model_file=fake_onnx,
            input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
            output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
            uploaded_by=self.admin_user,
            status='approved',
            is_public=True
        )

    def test_create_prediction_audit(self):
        """Test creating a PredictionAudit entry."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_audit_test',
            password='testpass123',
            role=researcher_role
        )

        # Create API key for researcher
        api_key = APIKey.objects.create(
            user=researcher_user,
            name="Test API Key",
            ip_whitelist=["0.0.0.0/0"]
        )

        # Compute input hash
        input_data = "test input data"
        input_hash = PredictionAudit.compute_input_hash(input_data)

        # Create audit entry
        audit = PredictionAudit.objects.create(
            user=researcher_user,
            api_key=api_key,
            ip_address="192.168.1.100",
            model=self.test_model,
            model_name=self.test_model.name,
            model_version=self.test_model.version,
            model_domain=self.test_model.domain,
            records_count=10,
            execution_time_ms=250,
            rate_limit_remaining=59,
            input_hash=input_hash,
            success=True
        )

        assert audit.id is not None
        assert audit.user == researcher_user
        assert audit.api_key == api_key
        assert audit.model == self.test_model
        assert audit.records_count == 10
        assert audit.execution_time_ms == 250
        assert audit.suspicious_score == 0.0
        assert audit.patterns_detected == []
        assert audit.success is True
        assert audit.dp_noise_applied is False

    def test_compute_input_hash(self):
        """Test input hash computation."""
        # Test with string
        hash1 = PredictionAudit.compute_input_hash("test data")
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
        assert hash1 == PredictionAudit.compute_input_hash("test data")  # Consistent

        # Test with bytes
        hash2 = PredictionAudit.compute_input_hash(b"test data")
        assert hash1 == hash2  # Same result for string and bytes

        # Different data produces different hash
        hash3 = PredictionAudit.compute_input_hash("different data")
        assert hash1 != hash3

    def test_mark_suspicious(self):
        """Test marking audit entry as suspicious."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_suspicious_test',
            password='testpass123',
            role=researcher_role
        )

        audit = PredictionAudit.objects.create(
            user=researcher_user,
            ip_address="192.168.1.101",
            model=self.test_model,
            model_name=self.test_model.name,
            model_version=self.test_model.version,
            model_domain=self.test_model.domain,
            records_count=5,
            execution_time_ms=100,
            rate_limit_remaining=50,
            input_hash=PredictionAudit.compute_input_hash("test"),
            success=True
        )

        # Initially not suspicious
        assert audit.suspicious_score == 0.0
        assert audit.patterns_detected == []

        # Mark as suspicious with patterns
        patterns = ['rapid_fire', 'exhaustive_search']
        audit.mark_suspicious(patterns)

        # Reload from database
        audit.refresh_from_db()

        assert audit.patterns_detected == patterns
        assert audit.suspicious_score == 0.4  # 2 patterns * 0.2

    def test_suspicious_score_capped_at_one(self):
        """Test that suspicious score is capped at 1.0."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_score_cap_test',
            password='testpass123',
            role=researcher_role
        )

        audit = PredictionAudit.objects.create(
            user=researcher_user,
            ip_address="192.168.1.102",
            model=self.test_model,
            model_name=self.test_model.name,
            model_version=self.test_model.version,
            model_domain=self.test_model.domain,
            records_count=5,
            execution_time_ms=100,
            rate_limit_remaining=50,
            input_hash=PredictionAudit.compute_input_hash("test"),
            success=True
        )

        # Many patterns (more than 5)
        patterns = ['pattern1', 'pattern2', 'pattern3', 'pattern4', 'pattern5', 'pattern6']
        audit.mark_suspicious(patterns)

        audit.refresh_from_db()

        # Score should be capped at 1.0 (not 1.2)
        assert audit.suspicious_score == 1.0

    def test_failed_prediction_audit(self):
        """Test audit entry for failed prediction."""
        member_role = Role.objects.get(name='MEMBER')
        member_user = CustomUser.objects.create_user(
            username='member_failed_test',
            password='testpass123',
            role=member_role
        )

        audit = PredictionAudit.objects.create(
            user=member_user,
            ip_address="192.168.1.103",
            model=self.test_model,
            model_name=self.test_model.name,
            model_version=self.test_model.version,
            model_domain=self.test_model.domain,
            records_count=10,
            execution_time_ms=50,
            rate_limit_remaining=59,
            input_hash=PredictionAudit.compute_input_hash("test"),
            success=False,
            error_message="Invalid input format"
        )

        assert audit.success is False
        assert audit.error_message == "Invalid input format"

    def test_differential_privacy_applied(self):
        """Test audit entry with differential privacy applied."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_dp_test',
            password='testpass123',
            role=researcher_role
        )

        audit = PredictionAudit.objects.create(
            user=researcher_user,
            ip_address="192.168.1.104",
            model=self.test_model,
            model_name=self.test_model.name,
            model_version=self.test_model.version,
            model_domain=self.test_model.domain,
            records_count=100,
            execution_time_ms=500,
            rate_limit_remaining=40,
            input_hash=PredictionAudit.compute_input_hash("test"),
            success=True,
            dp_noise_applied=True
        )

        assert audit.dp_noise_applied is True

    def test_audit_ordering(self):
        """Test that audits are ordered by timestamp descending."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_order_test',
            password='testpass123',
            role=researcher_role
        )

        # Create multiple audit entries
        for i in range(3):
            PredictionAudit.objects.create(
                user=researcher_user,
                ip_address="192.168.1.105",
                model=self.test_model,
                model_name=self.test_model.name,
                model_version=self.test_model.version,
                model_domain=self.test_model.domain,
                records_count=i,
                execution_time_ms=100,
                rate_limit_remaining=50,
                input_hash=PredictionAudit.compute_input_hash(f"test{i}"),
                success=True
            )

        # Get all audits
        audits = PredictionAudit.objects.filter(user=researcher_user)

        # Should be ordered by timestamp descending (newest first)
        assert audits[0].records_count == 2  # Last created
        assert audits[1].records_count == 1
        assert audits[2].records_count == 0  # First created

    def test_audit_snapshot_persists_after_model_deletion(self):
        """Test that audit entry retains model info even after model deletion."""
        researcher_role = Role.objects.get(name='RESEARCHER')
        researcher_user = CustomUser.objects.create_user(
            username='researcher_snapshot_test',
            password='testpass123',
            role=researcher_role
        )

        # Create temporary model
        fake_onnx = SimpleUploadedFile(
            "temp_model.onnx",
            b"temp onnx",
            content_type="application/octet-stream"
        )

        temp_model = DeployedModel.objects.create(
            name="Temporary Model",
            version="2.0.0",
            description="Temporary test model",
            domain="neurology",
            model_file=fake_onnx,
            input_schema={"feature_names": ["test"], "dtypes": {}, "shape": []},
            output_schema={"output_names": ["output"], "dtypes": {}, "shape": []},
            uploaded_by=self.admin_user,
            status='approved',
            is_public=True
        )

        # Create audit entry
        audit = PredictionAudit.objects.create(
            user=researcher_user,
            ip_address="192.168.1.106",
            model=temp_model,
            model_name=temp_model.name,
            model_version=temp_model.version,
            model_domain=temp_model.domain,
            records_count=10,
            execution_time_ms=200,
            rate_limit_remaining=50,
            input_hash=PredictionAudit.compute_input_hash("test"),
            success=True
        )

        # Verify model reference exists
        assert audit.model == temp_model

        # Delete the model
        temp_model.delete()

        # Reload audit entry
        audit.refresh_from_db()

        # Model reference should be null, but snapshots persist
        assert audit.model is None
        assert audit.model_name == "Temporary Model"
        assert audit.model_version == "2.0.0"
        assert audit.model_domain == "neurology"
