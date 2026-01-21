"""
Simplified tests for the secure dataset uploader (without heavy dependencies).
"""

import os
import tempfile
import hashlib
from django.test import TestCase
from django.db import IntegrityError
from django.contrib.auth import get_user_model
from dataset.uploader import SecureDatasetUploader
from dataset.uploader import SecurityValidationError
from dataset.models import Dataset, DatasetAccess

User = get_user_model()


class SecureUploaderBasicTest(TestCase):
    """Basic security tests without heavy dependencies."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        # Create test user
        self.user = User.objects.db_manager('default').create_user(
            username='test_uploader_simple',
            email='uploader@test.com',
            password='testpass123'
        )
    
    def test_forbidden_patterns_detection(self):
        """Test that forbidden PHI patterns are detected."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test forbidden patterns - check that regex patterns contain expected words
        expected_patterns = [
            'patient_id', 'name', 'first_name', 'last_name', 
            'email', 'phone', 'ssn', 'mrn', 'medical_record',
            'address', 'zip', 'birth_date', 'dob'
        ]
        
        # Convert regex patterns to a searchable string
        all_patterns_text = ' '.join(uploader.FORBIDDEN_PATTERNS)
        
        for pattern in expected_patterns:
            self.assertIn(pattern, all_patterns_text, 
                         f"Pattern '{pattern}' not found in forbidden patterns")
    
    def test_k_anonymity_minimum(self):
        """Test k-anonymity minimum requirement."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Verify k-anonymity minimum is 5
        self.assertEqual(uploader.MIN_K_ANONYMITY, 5)
    
    def test_allowed_extensions(self):
        """Test that only medical file formats are allowed."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Verify allowed extensions
        expected_extensions = {'.csv', '.json', '.parquet', '.h5', '.npy'}
        actual_extensions = set(uploader.ALLOWED_EXTENSIONS.keys())
        
        self.assertEqual(actual_extensions, expected_extensions)
    
    def test_no_file_size_limit(self):
        """Test that there's no file size limit for medical datasets."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Verify no file size limit
        self.assertIsNone(uploader.MAX_FILE_SIZE)
    
    def test_filename_sanitization(self):
        """Test filename sanitization for security."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test dangerous filename
        dangerous_filename = "../../etc/passwd<script>alert('xss')</script>.csv"
        safe_filename = uploader._sanitize_filename(dangerous_filename)
        
        # Should not contain dangerous characters
        self.assertNotIn('..', safe_filename)
        self.assertNotIn('<', safe_filename)
        self.assertNotIn('>', safe_filename)
        self.assertNotIn('/', safe_filename)
        self.assertNotIn('\\', safe_filename)
        
        # Should still have .csv extension
        self.assertTrue(safe_filename.endswith('.csv'))
    
    def test_checksum_calculation(self):
        """Test MD5 and SHA256 checksum calculation."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Create test file
        content = "test content for checksum validation"
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        temp_file.write(content)
        temp_file.close()
        
        try:
            md5_hash, sha256_hash = uploader._calculate_checksums(temp_file.name)
            
            # Verify checksums are correct length
            self.assertEqual(len(md5_hash), 32)  # MD5 is 32 hex chars
            self.assertEqual(len(sha256_hash), 64)  # SHA256 is 64 hex chars
            
            # Verify checksums are correct
            expected_md5 = hashlib.md5(content.encode()).hexdigest()
            expected_sha256 = hashlib.sha256(content.encode()).hexdigest()
            
            self.assertEqual(md5_hash, expected_md5)
            self.assertEqual(sha256_hash, expected_sha256)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_safe_filename_validation(self):
        """Test safe filename validation."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test safe filenames
        safe_filenames = [
            'dataset.csv',
            'medical_data_2024.json',
            'patient-records.parquet'
        ]
        
        for filename in safe_filenames:
            self.assertTrue(uploader._is_safe_filename(filename))
        
        # Test unsafe filenames
        unsafe_filenames = [
            '../../../etc/passwd',
            'file<script>.csv',
            'data|pipe.json',
            '..',
            '.',
            ''
        ]
        
        for filename in unsafe_filenames:
            self.assertFalse(uploader._is_safe_filename(filename))
    
    def test_progress_callback_functionality(self):
        """Test progress callback mechanism."""
        
        progress_updates = []
        
        def test_callback(status, message):
            progress_updates.append((status, message))
        
        uploader = SecureDatasetUploader(self.user, test_callback)
        
        # Trigger progress updates
        uploader._update_progress("validating", "Validating file...")
        uploader._update_progress("extracting_metadata", "Extracting metadata...")
        uploader._update_progress("completed", "Upload completed!")
        
        # Verify callbacks were called
        self.assertEqual(len(progress_updates), 3)
        self.assertEqual(progress_updates[0], ("validating", "Validating file..."))
        self.assertEqual(progress_updates[1], ("extracting_metadata", "Extracting metadata..."))
        self.assertEqual(progress_updates[2], ("completed", "Upload completed!"))
    
    def test_quarantine_directory_setup(self):
        """Test quarantine directory setup."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Verify quarantine directory is set up
        quarantine_dir = uploader._get_quarantine_dir()
        self.assertIsInstance(quarantine_dir, str)
        self.assertTrue(len(quarantine_dir) > 0)
    
    def test_medical_compliance_validation_structure(self):
        """Test the structure of medical compliance validation with PHI removal."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test metadata with forbidden column patterns
        metadata_with_phi = {
            'file_type': 'csv',
            'columns': 3,
            'column_info': {
                'patient_id': {'type': 'numeric'},
                'age': {'type': 'numeric'},
                'diagnosis': {'type': 'text'}
            },
            'k_anonymity_compliant': True,
            'nulls_verified_zero': True
        }
        
        # PHI columns should be automatically removed, not raise an error
        result_metadata = uploader._validate_medical_compliance(metadata_with_phi, 'test.csv')
        
        # Verify PHI columns were removed
        self.assertIn('phi_columns_removed', result_metadata)
        self.assertEqual(len(result_metadata['phi_columns_removed']), 1)
        self.assertEqual(result_metadata['phi_columns_removed'][0]['name'], 'patient_id')
        
        # Verify safe columns remain
        self.assertIn('age', result_metadata['column_info'])
        self.assertIn('diagnosis', result_metadata['column_info'])
        self.assertNotIn('patient_id', result_metadata['column_info'])
        
        # Verify column count updated
        self.assertEqual(result_metadata['columns'], 2)
    
    def test_k_anonymity_compliance_validation(self):
        """Test k-anonymity compliance validation."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test metadata with insufficient k-anonymity
        metadata_insufficient_k = {
            'file_type': 'csv',
            'column_info': {
                'age': {'type': 'numeric'},
                'score': {'type': 'numeric'}
            },
            'rows': 3,  # Less than MIN_K_ANONYMITY (5)
            'k_anonymity_compliant': False,
            'nulls_verified_zero': True
        }
        
        # Should raise SecurityValidationError for insufficient k-anonymity
        with self.assertRaises(SecurityValidationError) as context:
            uploader._validate_medical_compliance(metadata_insufficient_k, 'test.csv')
        
        self.assertIn("k-anonymity", str(context.exception))
    
    def test_null_values_validation(self):
        """Test null values validation."""
        
        uploader = SecureDatasetUploader(self.user)
        
        # Test metadata with null values
        metadata_with_nulls = {
            'file_type': 'csv',
            'column_info': {
                'age': {'type': 'numeric'},
                'score': {'type': 'numeric'}
            },
            'k_anonymity_compliant': True,
            'nulls_verified_zero': False  # Has nulls - should fail
        }
        
        # Should raise SecurityValidationError for null values
        with self.assertRaises(SecurityValidationError) as context:
            uploader._validate_medical_compliance(metadata_with_nulls, 'test.csv')
        
        self.assertIn("null values", str(context.exception))


class DatasetModelsSecurityTest(TestCase):
    """Test security features of dataset models."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.db_manager('datasets_db').create_user(
            username='model_test_user',
            email='model@test.com',
            password='testpass123'
        )
    
    def test_dataset_model_security_fields(self):
        """Test that Dataset model has required security fields."""
        
        # Create a dataset to verify fields exist
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        temp_file.write("age,score\n25,85\n30,90")
        temp_file.close()
        
        try:
            dataset = Dataset.objects.using('datasets_db').create(
                name='Security Test Dataset',
                description='Test dataset for security validation',
                file_path=temp_file.name,
                uploaded_by_id=self.user.id,
                medical_domain='general',
                data_type='tabular',
                file_format='csv',
                file_size=100,
                is_active=True
            )
            
            # Verify security fields exist
            self.assertTrue(hasattr(dataset, 'checksum_md5'))
            self.assertTrue(hasattr(dataset, 'is_active'))
            self.assertTrue(hasattr(dataset, 'access_count'))
            self.assertTrue(hasattr(dataset, 'last_accessed'))
            
            # Verify checksum is calculated
            self.assertIsNotNone(dataset.checksum_md5)
            self.assertEqual(len(dataset.checksum_md5), 32)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_dataset_access_unique_constraint(self):
        """Test that DatasetAccess has unique constraint."""
        
        # Create dataset
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        temp_file.write("age,score\n25,85")
        temp_file.close()
        
        try:
            dataset = Dataset.objects.using('datasets_db').create(
                name='Access Test Dataset',
                description='Test dataset for access control',
                file_path=temp_file.name,
                uploaded_by_id=self.user.id,
                medical_domain='general',
                data_type='tabular',
                file_format='csv'
            )
            
            # Create first access record
            DatasetAccess.objects.using('datasets_db').create(
                dataset_id=dataset.id,
                user_id=self.user.id,
                assigned_by_id=self.user.id,
                can_train=True,
                can_view_metadata=True
            )
            
            # Try to create duplicate - should fail
            with self.assertRaises(IntegrityError):
                DatasetAccess.objects.using('datasets_db').create(
                    dataset_id=dataset.id,
                    user_id=self.user.id,  # Same user
                    assigned_by_id=self.user.id,
                    can_train=False,
                    can_view_metadata=False
                )
        
        finally:
            os.unlink(temp_file.name)

