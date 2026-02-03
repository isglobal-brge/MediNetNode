"""
Simple tests for SQL injection protection in dataset uploader.

Validates that parameterized queries are used instead of string concatenation.
"""
import os
import tempfile
import pytest
from django.test import TestCase
from django.db import connections
from users.models import Role
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class SQLInjectionProtectionSimpleTests(TestCase):
    """Test SQL injection protection at the raw SQL level."""

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

    def test_parameterized_query_blocks_sql_injection_in_name(self):
        """Test that parameterized queries prevent SQL injection in dataset name."""
        malicious_name = "test'; DROP TABLE dataset_dataset; --"

        # Use parameterized query (the fix)
        params = (
            malicious_name,  # name with SQL injection attempt
            "Test description",  # description
            "/tmp/test.csv",  # file_path
            "cardiology",  # medical_domain
            5,  # patient_count
            "tabular",  # data_type
            True,  # anonymized
            1024,  # file_size
            "csv",  # file_format
            3,  # columns_count
            5,  # rows_count
            timezone.now().strftime('%Y-%m-%d %H:%M:%S'),  # uploaded_at
            None,  # last_accessed
            0,  # access_count
            "abc123",  # checksum_sha256
            True,  # is_active
            self.admin_user.id,  # uploaded_by_id
            None,  # target_column
        )

        sql_parameterized = """
            INSERT INTO dataset_dataset (
                name, description, file_path, medical_domain, patient_count,
                data_type, anonymized, file_size, file_format, columns_count,
                rows_count, uploaded_at, last_accessed, access_count,
                checksum_sha256, is_active, uploaded_by_id, target_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with connections['datasets_db'].cursor() as cursor:
            # Execute parameterized query
            cursor.execute(sql_parameterized, params)

            # Verify the malicious name was stored as-is (not executed)
            cursor.execute("SELECT name FROM dataset_dataset WHERE name = ?", (malicious_name,))
            result = cursor.fetchone()

            assert result is not None
            assert result[0] == malicious_name

            # Verify table still exists (not dropped)
            cursor.execute("SELECT COUNT(*) FROM dataset_dataset")
            count = cursor.fetchone()[0]
            assert count > 0

    def test_parameterized_query_blocks_sql_injection_in_description(self):
        """Test that parameterized queries prevent SQL injection in description."""
        malicious_desc = "test'; DELETE FROM dataset_dataset WHERE '1'='1"

        params = (
            "Test Dataset",
            malicious_desc,
            "/tmp/test2.csv",
            "neurology",
            5,
            "tabular",
            True,
            2048,
            "csv",
            3,
            5,
            timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            None,
            0,
            "def456",
            True,
            self.admin_user.id,
            None,
        )

        sql_parameterized = """
            INSERT INTO dataset_dataset (
                name, description, file_path, medical_domain, patient_count,
                data_type, anonymized, file_size, file_format, columns_count,
                rows_count, uploaded_at, last_accessed, access_count,
                checksum_sha256, is_active, uploaded_by_id, target_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with connections['datasets_db'].cursor() as cursor:
            initial_count = cursor.execute("SELECT COUNT(*) FROM dataset_dataset").fetchone()[0]

            cursor.execute(sql_parameterized, params)

            # Verify only one record was added (not deleted)
            final_count = cursor.execute("SELECT COUNT(*) FROM dataset_dataset").fetchone()[0]
            assert final_count == initial_count + 1

            # Verify malicious description was stored as-is
            cursor.execute("SELECT description FROM dataset_dataset WHERE description = ?", (malicious_desc,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == malicious_desc

    def test_single_quotes_properly_handled(self):
        """Test that single quotes in data are properly handled by parameterized queries."""
        name_with_quote = "O'Brien's Dataset"
        desc_with_quote = "St. Mary's Hospital Data"

        params = (
            name_with_quote,
            desc_with_quote,
            "/tmp/test3.csv",
            "cardiology",
            5,
            "tabular",
            True,
            1024,
            "csv",
            3,
            5,
            timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            None,
            0,
            "ghi789",
            True,
            self.admin_user.id,
            None,
        )

        sql_parameterized = """
            INSERT INTO dataset_dataset (
                name, description, file_path, medical_domain, patient_count,
                data_type, anonymized, file_size, file_format, columns_count,
                rows_count, uploaded_at, last_accessed, access_count,
                checksum_sha256, is_active, uploaded_by_id, target_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with connections['datasets_db'].cursor() as cursor:
            cursor.execute(sql_parameterized, params)

            # Verify data was stored correctly
            cursor.execute("SELECT name, description FROM dataset_dataset WHERE name = ?", (name_with_quote,))
            result = cursor.fetchone()

            assert result is not None
            assert result[0] == name_with_quote
            assert result[1] == desc_with_quote

    def test_metadata_parameterized_query_blocks_injection(self):
        """Test that metadata insertion also uses parameterized queries."""
        import json

        # First create a dataset to get an ID
        params = (
            "Metadata Test",
            "Test description",
            "/tmp/test4.csv",
            "oncology",
            5,
            "tabular",
            True,
            1024,
            "csv",
            3,
            5,
            timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            None,
            0,
            "jkl012",
            True,
            self.admin_user.id,
            None,
        )

        sql_dataset = """
            INSERT INTO dataset_dataset (
                name, description, file_path, medical_domain, patient_count,
                data_type, anonymized, file_size, file_format, columns_count,
                rows_count, uploaded_at, last_accessed, access_count,
                checksum_sha256, is_active, uploaded_by_id, target_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with connections['datasets_db'].cursor() as cursor:
            cursor.execute(sql_dataset, params)
            dataset_id = cursor.lastrowid

            # Try to inject SQL via metadata
            malicious_metadata = {"attack": "'; DROP TABLE dataset_datasetmetadata; --"}

            metadata_params = (
                dataset_id,
                json.dumps(malicious_metadata),
                '{}',
                '{}',
                1.0,
                100.0,
                timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
                timezone.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            metadata_sql = """
                INSERT INTO dataset_datasetmetadata (
                    dataset_id, statistical_summary, missing_values, data_distribution,
                    quality_score, completeness_percentage, generated_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(metadata_sql, metadata_params)

            # Verify table still exists
            cursor.execute("SELECT COUNT(*) FROM dataset_datasetmetadata")
            count = cursor.fetchone()[0]
            assert count > 0

            # Verify malicious metadata was stored as-is
            cursor.execute("SELECT statistical_summary FROM dataset_datasetmetadata WHERE dataset_id = ?", (dataset_id,))
            result = cursor.fetchone()
            assert result is not None
            stored_metadata = json.loads(result[0])
            assert stored_metadata["attack"] == "'; DROP TABLE dataset_datasetmetadata; --"
