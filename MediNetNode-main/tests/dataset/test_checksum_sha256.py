"""
Tests for SHA-256 checksum implementation in dataset models and upload.

SECURITY FIX: Validates migration from MD5 (vulnerable to collision attacks)
to SHA-256 (cryptographically secure) for dataset integrity verification.
"""
import os
import tempfile
import hashlib
from django.test import TestCase
from django.core.exceptions import ValidationError
from dataset.models import Dataset
from users.models import Role
from django.contrib.auth import get_user_model

User = get_user_model()


class SHA256ChecksumModelTests(TestCase):
    """Test SHA-256 checksum calculation in Dataset model."""

    databases = ['default', 'datasets_db']

    def setUp(self):
        """Set up test data."""
        self.admin_role = Role.objects.get(name='ADMIN')
        self.admin_user = User.objects.create_user(
            username='admin_test',
            email='admin@test.com',
            password='TestPass123!',
            role=self.admin_role,
            is_superuser=True
        )

        self.test_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        )
        self.test_file.write("patient_id,diagnosis,age\n")
        self.test_file.write("1,diabetes,45\n")
        self.test_file.write("2,hypertension,62\n")
        self.test_file_path = self.test_file.name
        self.test_file.close()

        with open(self.test_file_path, 'rb') as f:
            self.expected_sha256 = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        """Clean up test file."""
        if os.path.exists(self.test_file_path):
            os.unlink(self.test_file_path)

    def test_calculate_checksum_returns_sha256(self):
        """Test that calculate_checksum() returns valid SHA-256 hash."""
        dataset = Dataset(
            name='Test Dataset',
            description='Test description',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=100,
            patient_count=2
        )

        checksum = dataset.calculate_checksum()

        # SHA-256 produces 64 hexadecimal characters
        self.assertEqual(len(checksum), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in checksum))
        self.assertEqual(checksum, self.expected_sha256)

    def test_checksum_sha256_field_length(self):
        """Test that checksum_sha256 field accepts 64-character hash."""
        dataset = Dataset.objects.using('datasets_db').create(
            name='Test Dataset Length',
            description='Test description',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=100,
            patient_count=2,
            checksum_sha256=self.expected_sha256
        )

        saved_dataset = Dataset.objects.using('datasets_db').get(id=dataset.id)
        self.assertEqual(saved_dataset.checksum_sha256, self.expected_sha256)
        self.assertEqual(len(saved_dataset.checksum_sha256), 64)

    def test_auto_calculate_checksum_on_save(self):
        """Test that checksum is automatically calculated on save."""
        dataset = Dataset(
            name='Auto Checksum Test',
            description='Test description',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='neurology',
            data_type='tabular',
            file_size=100,
            patient_count=2
        )

        self.assertIsNone(dataset.checksum_sha256)

        dataset.save(using='datasets_db')

        self.assertIsNotNone(dataset.checksum_sha256)
        self.assertEqual(dataset.checksum_sha256, self.expected_sha256)

    def test_checksum_raises_error_for_missing_file(self):
        """Test that calculate_checksum() raises error for missing file."""
        dataset = Dataset(
            name='Missing File Test',
            description='Test description',
            file_path='/nonexistent/file.csv',
            uploaded_by_id=self.admin_user.id,
            medical_domain='oncology',
            data_type='tabular',
            file_size=100
        )

        with self.assertRaises(ValidationError) as context:
            dataset.calculate_checksum()

        self.assertIn('File not found', str(context.exception))

    def test_checksum_detects_file_modifications(self):
        """Test that checksum changes when file content is modified."""
        dataset = Dataset.objects.using('datasets_db').create(
            name='Modification Detection Test',
            description='Test description',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='radiology',
            data_type='tabular',
            file_size=100,
            patient_count=2
        )

        original_checksum = dataset.checksum_sha256

        with open(self.test_file_path, 'a') as f:
            f.write("3,asthma,30\n")

        new_checksum = dataset.calculate_checksum()

        self.assertNotEqual(original_checksum, new_checksum)
        self.assertEqual(len(new_checksum), 64)


class SHA256ChecksumCollisionResistanceTests(TestCase):
    """Test SHA-256 collision resistance (security-focused tests)."""

    databases = ['default', 'datasets_db']

    def setUp(self):
        """Set up test data."""
        self.admin_role = Role.objects.get(name='ADMIN')
        self.admin_user = User.objects.create_user(
            username='security_admin',
            email='security@test.com',
            password='TestPass123!',
            role=self.admin_role,
            is_superuser=True
        )

    def test_different_files_produce_different_checksums(self):
        """Test that different file contents produce different SHA-256 hashes."""
        file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        file1.write("data1,value1\n")
        file1_path = file1.name
        file1.close()

        file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        file2.write("data2,value2\n")
        file2_path = file2.name
        file2.close()

        try:
            dataset1 = Dataset(
                name='File 1',
                description='First file',
                file_path=file1_path,
                uploaded_by_id=self.admin_user.id,
                medical_domain='general',
                data_type='tabular',
                file_size=100
            )

            dataset2 = Dataset(
                name='File 2',
                description='Second file',
                file_path=file2_path,
                uploaded_by_id=self.admin_user.id,
                medical_domain='general',
                data_type='tabular',
                file_size=100
            )

            checksum1 = dataset1.calculate_checksum()
            checksum2 = dataset2.calculate_checksum()

            self.assertNotEqual(checksum1, checksum2)

            self.assertEqual(len(checksum1), 64)
            self.assertEqual(len(checksum2), 64)

        finally:
            os.unlink(file1_path)
            os.unlink(file2_path)

    def test_identical_files_produce_identical_checksums(self):
        """Test that identical file contents produce identical SHA-256 hashes."""
        content = "patient_id,diagnosis\n1,diabetes\n2,hypertension\n"

        file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        file1.write(content)
        file1_path = file1.name
        file1.close()

        file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        file2.write(content)
        file2_path = file2.name
        file2.close()

        try:
            dataset1 = Dataset(
                name='Identical File 1',
                description='First file',
                file_path=file1_path,
                uploaded_by_id=self.admin_user.id,
                medical_domain='general',
                data_type='tabular',
                file_size=100
            )

            dataset2 = Dataset(
                name='Identical File 2',
                description='Second file',
                file_path=file2_path,
                uploaded_by_id=self.admin_user.id,
                medical_domain='general',
                data_type='tabular',
                file_size=100
            )

            checksum1 = dataset1.calculate_checksum()
            checksum2 = dataset2.calculate_checksum()

            self.assertEqual(checksum1, checksum2)

        finally:
            os.unlink(file1_path)
            os.unlink(file2_path)

    def test_sha256_length_prevents_md5_collision_substitution(self):
        """
        Test that SHA-256 field length (64 chars) prevents MD5 hash (32 chars) substitution.

        SECURITY: This prevents attackers from substituting MD5 hashes for SHA-256.
        """
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_file.write("data\n")
        test_file_path = test_file.name
        test_file.close()

        try:
            # Calculate MD5 (32 chars) - the old vulnerable hash
            with open(test_file_path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()

            self.assertEqual(len(md5_hash), 32)

            dataset = Dataset(
                name='MD5 Substitution Test',
                description='Test MD5 length mismatch',
                file_path=test_file_path,
                uploaded_by_id=self.admin_user.id,
                medical_domain='general',
                data_type='tabular',
                file_size=100,
                checksum_sha256=md5_hash  # Try to use MD5 (32 chars) as SHA-256
            )

            dataset.save(using='datasets_db')

            correct_sha256 = dataset.calculate_checksum()
            self.assertEqual(len(correct_sha256), 64)

            self.assertNotEqual(md5_hash, correct_sha256)
            self.assertEqual(len(dataset.checksum_sha256), 32)  # Stored MD5 length
            self.assertEqual(len(correct_sha256), 64)  # Correct SHA-256 length

        finally:
            os.unlink(test_file_path)


class SHA256ChecksumDocumentationTests(TestCase):
    """Test that SHA-256 implementation is properly documented."""

    def test_calculate_checksum_has_security_documentation(self):
        """Test that calculate_checksum() method documents SHA-256 usage."""
        from dataset.models import Dataset
        import inspect

        docstring = inspect.getdoc(Dataset.calculate_checksum)
        self.assertIsNotNone(docstring)
        self.assertIn('SHA-256', docstring)
        self.assertIn('collision', docstring.lower())

    def test_checksum_field_has_helpful_text(self):
        """Test that checksum_sha256 field has descriptive help_text."""
        from dataset.models import Dataset

        checksum_field = Dataset._meta.get_field('checksum_sha256')
        self.assertIsNotNone(checksum_field.help_text)
        self.assertIn('SHA-256', checksum_field.help_text)
