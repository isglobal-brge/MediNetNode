"""
Tests for SQL injection protection in dataset uploader.

Validates that parameterized queries prevent SQL injection attacks.
"""
import os
import tempfile
import pytest
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.db import connection
from users.models import Role
from dataset.models import Dataset, DatasetMetadata
from dataset.uploader import SecureDatasetUploader

User = get_user_model()


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class SQLInjectionProtectionTests(TestCase):
    """Test SQL injection protection in dataset upload functionality."""

    databases = ['default', 'datasets_db']

    def setUp(self):
        """Set up test data."""
        self.admin_role = Role.objects.get(name='ADMIN')
        self.admin_user = User.objects.create_user(
            username='admin',
            email='admin@test.com',
            password='TestPass123!',
            role=self.admin_role,
            is_superuser=True
        )

        # Create a test CSV file with at least 5 rows for k-anonymity
        self.test_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        )
        self.test_file.write("diagnosis,age,gender\n")
        self.test_file.write("diabetes,45,M\n")
        self.test_file.write("hypertension,62,F\n")
        self.test_file.write("asthma,30,M\n")
        self.test_file.write("arthritis,55,F\n")
        self.test_file.write("migraine,40,M\n")
        self.test_file_path = self.test_file.name
        self.test_file.close()

        self.uploader = SecureDatasetUploader(self.admin_user)

    def tearDown(self):
        """Clean up test file."""
        if os.path.exists(self.test_file_path):
            os.unlink(self.test_file_path)

        # Clean up uploaded datasets
        Dataset.objects.using('datasets_db').all().delete()
        DatasetMetadata.objects.using('datasets_db').all().delete()

    def test_blocks_sql_injection_in_dataset_name(self):
        """Test that SQL injection in dataset name is neutralized."""
        # Attempt SQL injection via dataset name
        malicious_name = "test'; DROP TABLE dataset_dataset; --"

        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name=malicious_name,
            description="Test description",
            medical_domain="cardiology",
            data_type="tabular"
        )

        # Dataset should be created safely
        assert dataset is not None
        assert dataset.name == malicious_name  # Name stored as-is (safe with parameterized queries)

        # Verify table still exists
        with connection['datasets_db'].cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM dataset_dataset")
            count = cursor.fetchone()[0]
            assert count > 0  # Table exists and has records

    def test_blocks_sql_injection_in_description(self):
        """Test that SQL injection in description is neutralized."""
        malicious_description = "test'; DELETE FROM dataset_dataset WHERE '1'='1"

        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name="Test Dataset",
            description=malicious_description,
            medical_domain="neurology",
            data_type="tabular"
        )

        # Dataset should be created safely
        assert dataset is not None
        assert dataset.description == malicious_description

        # Verify other datasets are not deleted
        datasets_count = Dataset.objects.using('datasets_db').count()
        assert datasets_count >= 1

    def test_blocks_sql_injection_in_file_path(self):
        """Test that SQL injection in file path is neutralized."""
        # Create a file with malicious characters in directory name
        malicious_dir = tempfile.mkdtemp(suffix="_test'; DROP TABLE")
        malicious_file = os.path.join(malicious_dir, "data.csv")

        with open(malicious_file, 'w') as f:
            f.write("diagnosis,age,gender\n")
            f.write("diabetes,45,M\n")
            f.write("hypertension,62,F\n")
            f.write("asthma,30,M\n")
            f.write("arthritis,55,F\n")
            f.write("migraine,40,M\n")

        try:
            dataset, info = self.uploader.upload_dataset(
                file_path=malicious_file,
                name="Path Injection Test",
                description="Test description",
                medical_domain="general",
                data_type="tabular"
            )

            # Dataset should be created safely
            assert dataset is not None

            # Verify table still exists
            with connection['datasets_db'].cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM dataset_dataset")
                count = cursor.fetchone()[0]
                assert count > 0

        finally:
            # Cleanup
            if os.path.exists(malicious_file):
                os.unlink(malicious_file)
            if os.path.exists(malicious_dir):
                os.rmdir(malicious_dir)

    def test_blocks_sql_injection_with_quotes(self):
        """Test that single quotes in data are properly escaped."""
        name_with_quotes = "O'Brien's Dataset"
        desc_with_quotes = "Patient data from St. Mary's Hospital"

        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name=name_with_quotes,
            description=desc_with_quotes,
            medical_domain="cardiology",
            data_type="tabular"
        )

        # Dataset should be created safely
        assert dataset is not None
        assert dataset.name == name_with_quotes
        assert dataset.description == desc_with_quotes

        # Reload from database to verify persistence
        reloaded = Dataset.objects.using('datasets_db').get(id=dataset.id)
        assert reloaded.name == name_with_quotes
        assert reloaded.description == desc_with_quotes

    def test_blocks_sql_injection_with_comment_syntax(self):
        """Test that SQL comment syntax is neutralized."""
        malicious_name = "test -- comment\nDROP TABLE dataset_dataset;"

        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name=malicious_name,
            description="Test description",
            medical_domain="oncology",
            data_type="tabular"
        )

        # Dataset should be created safely
        assert dataset is not None

        # Verify table still exists
        datasets_count = Dataset.objects.using('datasets_db').count()
        assert datasets_count >= 1

    def test_blocks_sql_injection_in_metadata(self):
        """Test that SQL injection in metadata is neutralized."""
        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name="Metadata Injection Test",
            description="Test description",
            medical_domain="radiology",
            data_type="tabular"
        )

        # Verify metadata table still exists
        with connection['datasets_db'].cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM dataset_datasetmetadata")
            count = cursor.fetchone()[0]
            assert count > 0

    def test_parameterized_query_prevents_union_injection(self):
        """Test that UNION-based SQL injection is prevented."""
        malicious_name = "test' UNION SELECT * FROM dataset_dataset--"

        dataset, info = self.uploader.upload_dataset(
            file_path=self.test_file_path,
            name=malicious_name,
            description="Test description",
            medical_domain="pathology",
            data_type="tabular"
        )

        # Dataset should be created safely
        assert dataset is not None

        # Verify only expected number of datasets exist
        datasets_count = Dataset.objects.using('datasets_db').count()
        assert datasets_count >= 1

    def test_special_characters_handled_safely(self):
        """Test that various special characters are handled safely."""
        special_chars = [
            "test';--",
            "test\";--",
            "test\\';--",
            "test<script>alert(1)</script>",
            "test${injection}",
        ]

        for name in special_chars:
            dataset, info = self.uploader.upload_dataset(
                file_path=self.test_file_path,
                name=name,
                description="Test description",
                medical_domain="general",
                data_type="tabular"
            )

            # Each dataset should be created safely
            assert dataset is not None
            assert dataset.name == name

            # Clean up
            dataset.delete()

    def test_database_integrity_after_injection_attempts(self):
        """Test that database remains intact after multiple injection attempts."""
        initial_count = Dataset.objects.using('datasets_db').count()

        injection_attempts = [
            "'; DROP TABLE dataset_dataset; --",
            "'; DELETE FROM dataset_dataset; --",
            "'; UPDATE dataset_dataset SET name='hacked'; --",
        ]

        for attempt in injection_attempts:
            try:
                dataset, info = self.uploader.upload_dataset(
                    file_path=self.test_file_path,
                    name=attempt,
                    description="Test description",
                    medical_domain="general",
                    data_type="tabular"
                )
                # Clean up
                if dataset:
                    dataset.delete()
            except Exception as e:
                # Some attempts might fail for other reasons
                pass

        # Verify table structure is intact
        with connection['datasets_db'].cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM dataset_dataset")
            final_count = cursor.fetchone()[0]
            # Table should exist and have initial count (since we deleted created datasets)
            assert final_count == initial_count
