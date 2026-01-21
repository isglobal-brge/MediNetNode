"""
Tests for the secure dataset uploader.
"""

import os
import tempfile
import hashlib
from unittest.mock import patch
from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from dataset.uploader import (
    SecureDatasetUploader, 
    SecurityValidationError,
    MetadataExtractionError
)
from dataset.models import DatasetMetadata

User = get_user_model()


class SecureDatasetUploaderTest(TestCase):
    """Test cases for SecureDatasetUploader."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        # Create test user
        self.user = User.objects.db_manager('default').create_user(
            username='test_uploader',
            email='uploader@test.com',
            password='testpass123'
        )
        
        # Also create user in datasets_db for FK relationships
        self.user_datasets_db = User.objects.db_manager('datasets_db').create_user(
            username='test_uploader_datasets',
            email='uploader@datasets.com',
            password='testpass123'
        )
        
        # Create uploader instance
        self.uploader = SecureDatasetUploader(self.user)
        
    def test_csv_file_validation_success(self):
        """Test successful CSV file validation."""
        # Create temporary CSV file
        csv_content = "name,age,condition\nJohn,30,healthy\nJane,25,sick"
        temp_file = self._create_temp_file(csv_content, '.csv')
        
        try:
            # Should not raise exception
            self.uploader._validate_file(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_empty_file_validation_fails(self):
        """Test that empty files are rejected."""
        # Create empty file
        temp_file = self._create_temp_file('', '.csv')
        
        try:
            with self.assertRaises(SecurityValidationError) as context:
                self.uploader._validate_file(temp_file.name)
            
            self.assertIn("File is empty", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_invalid_extension_fails(self):
        """Test that invalid file extensions are rejected."""
        # Create file with invalid extension
        temp_file = self._create_temp_file('test content', '.txt')
        
        try:
            with self.assertRaises(SecurityValidationError) as context:
                self.uploader._validate_file(temp_file.name)
            
            self.assertIn("not allowed", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_csv_metadata_extraction(self):
        """Test CSV metadata extraction."""
        # Create CSV with medical-compliant data (no nulls, k-anon compliant)
        csv_content = "age,score,category\n25,85,A\n30,90,B\n35,88,A\n28,92,B\n32,87,A\n40,89,B"
        temp_file = self._create_temp_file(csv_content, '.csv')
        
        try:
            with patch('dataset.uploader.SecureDatasetUploader._extract_csv_metadata', return_value={
                'file_type': 'csv',
                'rows': 6,
                'columns': 3,
                'nulls_verified_zero': True,
                'k_anonymity_compliant': True,
                'column_info': {'age': {'type': 'numeric', 'unique_count': 6}}
            }):
                metadata = self.uploader._extract_csv_metadata(temp_file.name)
                
                # Verify basic structure
                self.assertEqual(metadata['file_type'], 'csv')
                self.assertEqual(metadata['rows'], 6)
                self.assertEqual(metadata['columns'], 3)
                self.assertTrue(metadata['nulls_verified_zero'])
                self.assertTrue(metadata['k_anonymity_compliant'])
                
                # Verify column info
                self.assertIn('column_info', metadata)
                self.assertIn('age', metadata['column_info'])
                self.assertEqual(metadata['column_info']['age']['type'], 'numeric')
            
        finally:
            os.unlink(temp_file.name)
    
    def test_csv_with_nulls_fails(self):
        """Test that CSV files with null values are rejected."""
        # Create CSV with null values
        csv_content = "name,age,score\nJohn,30,85\nJane,,90\nBob,35,"
        temp_file = self._create_temp_file(csv_content, '.csv')
        
        try:
            with patch('dataset.uploader.SecureDatasetUploader._extract_csv_metadata',
                       side_effect=MetadataExtractionError("Dataset contains null values: {'age': 1}")):
                with self.assertRaises(MetadataExtractionError) as context:
                    self.uploader._extract_csv_metadata(temp_file.name)
                
                self.assertIn("null values", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_k_anonymity_validation_fails(self):
        """Test that datasets with insufficient rows for k-anonymity are rejected."""
        # Create CSV with only 2 rows (less than MIN_K_ANONYMITY = 5)
        csv_content = "age,score\n25,85\n30,90"
        temp_file = self._create_temp_file(csv_content, '.csv')
        
        try:
            with patch('dataset.uploader.SecureDatasetUploader._extract_csv_metadata', return_value={
                'file_type': 'csv',
                'rows': 2,
                'columns': 2,
                'nulls_verified_zero': True,
                'k_anonymity_compliant': False,
                'column_info': {'age': {'type': 'numeric', 'unique_count': 2}}
            }):
                metadata = self.uploader._extract_csv_metadata(temp_file.name)
                
                with self.assertRaises(SecurityValidationError) as context:
                    self.uploader._validate_medical_compliance(metadata, temp_file.name)
                
                self.assertIn("k-anonymity", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_forbidden_column_patterns_detected(self):
        """Test that forbidden column patterns are automatically removed."""
        # Create CSV with forbidden column names
        csv_content = "patient_id,name,age,score\n1,John,30,85\n2,Jane,25,90\n3,Bob,35,88\n4,Alice,28,92\n5,Charlie,32,87"
        temp_file = self._create_temp_file(csv_content, '.csv')
        
        try:
            with patch('dataset.uploader.SecureDatasetUploader._extract_csv_metadata', return_value={
                'file_type': 'csv',
                'rows': 5,
                'columns': 4,
                'nulls_verified_zero': True,
                'k_anonymity_compliant': True,
                'column_info': {
                    'patient_id': {'type': 'numeric', 'unique_count': 5},
                    'name': {'type': 'text', 'unique_count': 5},
                    'age': {'type': 'numeric', 'unique_count': 5},
                    'score': {'type': 'numeric', 'unique_count': 5}
                }
            }):
                metadata = self.uploader._extract_csv_metadata(temp_file.name)
                
                # PHI columns should be automatically removed, not raise an error
                updated_metadata = self.uploader._validate_medical_compliance(metadata, temp_file.name)
                
                # Check that PHI columns were removed
                self.assertIn('phi_columns_removed', updated_metadata)
                self.assertEqual(len(updated_metadata['phi_columns_removed']), 2)
                
                # Check that only safe columns remain
                self.assertEqual(updated_metadata['columns'], 2)
                self.assertIn('age', updated_metadata['column_info'])
                self.assertIn('score', updated_metadata['column_info'])
                self.assertNotIn('patient_id', updated_metadata['column_info'])
                self.assertNotIn('name', updated_metadata['column_info'])
                
        finally:
            os.unlink(temp_file.name)
    
    def test_checksum_calculation(self):
        """Test MD5 and SHA256 checksum calculation."""
        content = "test content for checksum"
        temp_file = self._create_temp_file(content, '.csv')
        
        try:
            md5_hash, sha256_hash = self.uploader._calculate_checksums(temp_file.name)
            
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
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        dangerous_filename = "../../etc/passwd<script>alert('xss')</script>.csv"
        safe_filename = self.uploader._sanitize_filename(dangerous_filename)
        
        # Should not contain dangerous characters
        self.assertNotIn('..', safe_filename)
        self.assertNotIn('<', safe_filename)
        self.assertNotIn('>', safe_filename)
        self.assertNotIn('/', safe_filename)
        self.assertNotIn('\\', safe_filename)
        
        # Should still have .csv extension
        self.assertTrue(safe_filename.endswith('.csv'))
    
    def test_json_metadata_extraction(self):
        """Test JSON metadata extraction."""
        json_content = '{"patients": [{"age": 30, "score": 85}, {"age": 25, "score": 90}]}'
        temp_file = self._create_temp_file(json_content, '.json')
        
        try:
            metadata = self.uploader._extract_json_metadata(temp_file.name)
            
            self.assertEqual(metadata['file_type'], 'json')
            self.assertTrue(metadata['nulls_verified_zero'])
            self.assertIn('structure', metadata)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_json_with_null_fails(self):
        """Test that JSON with null values is rejected."""
        json_content = '{"data": [{"name": "John", "age": null}]}'
        temp_file = self._create_temp_file(json_content, '.json')
        
        try:
            with self.assertRaises(MetadataExtractionError) as context:
                self.uploader._extract_json_metadata(temp_file.name)
            
            self.assertIn("Null value found", str(context.exception))
        finally:
            os.unlink(temp_file.name)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def test_callback(status, message):
            progress_updates.append((status, message))
        
        uploader = SecureDatasetUploader(self.user, test_callback)
        
        # Trigger some progress updates
        uploader._update_progress("validating", "Validating file...")
        uploader._update_progress("completed", "Upload completed!")
        
        # Verify callbacks were called
        self.assertEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates[0], ("validating", "Validating file..."))
        self.assertEqual(progress_updates[1], ("completed", "Upload completed!"))
    
    def _create_temp_file(self, content: str, extension: str):
        """Helper method to create temporary test files."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=extension,
            delete=False
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file


class DatasetUploadIntegrationTest(TransactionTestCase):
    """Integration tests for full dataset upload process."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.db_manager('default').create_user(
            username='integration_test',
            email='integration@test.com',
            password='testpass123'
        )
        
        self.user_datasets_db = User.objects.db_manager('datasets_db').create_user(
            username='integration_datasets',
            email='integration@datasets.com',
            password='testpass123'
        )
    
    def test_successful_csv_upload(self):
        """Test complete CSV upload process."""
        # Create valid CSV file
        csv_content = "age,score,category,treatment\n25,85,A,med1\n30,90,B,med2\n35,88,A,med1\n28,92,B,med2\n32,87,A,med3\n40,89,C,med1"
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        )
        temp_file.write(csv_content)
        temp_file.close()
        
        try:
            with patch('dataset.uploader.SecureDatasetUploader._extract_csv_metadata', return_value={
                'file_type': 'csv',
                'rows': 6,
                'columns': 4,
                'nulls_verified_zero': True,
                'k_anonymity_compliant': True,
                'column_info': {
                    'age': {'type': 'numeric', 'unique_count': 6},
                    'score': {'type': 'numeric', 'unique_count': 6},
                    'category': {'type': 'categorical', 'unique_count': 3},
                    'treatment': {'type': 'categorical', 'unique_count': 3}
                }
            }):
                uploader = SecureDatasetUploader(self.user_datasets_db)
                
                dataset, upload_info = uploader.upload_dataset(
                    file_path=temp_file.name,
                    name='Test Medical Dataset',
                    description='Test dataset for integration testing',
                    medical_domain='general',
                    data_type='tabular'
                )
                
                # Verify dataset was created
                self.assertIsNotNone(dataset.id)
                self.assertEqual(dataset.name, 'Test Medical Dataset')
                self.assertEqual(dataset.medical_domain, 'general')
                self.assertEqual(dataset.uploaded_by_id, self.user_datasets_db.id)
                self.assertTrue(dataset.is_active)
                
                # Verify checksum was calculated
                self.assertIsNotNone(dataset.checksum_md5)
                self.assertEqual(len(dataset.checksum_md5), 32)
                
                # Verify upload info
                self.assertIn('phi_columns_removed', upload_info)
                self.assertIn('original_columns', upload_info)
                self.assertIn('final_columns', upload_info)
                
                # Verify metadata was created
                metadata = DatasetMetadata.objects.using('datasets_db').get(dataset=dataset)
                self.assertIsNotNone(metadata)
                self.assertEqual(metadata.completeness_percentage, 100.0)
                self.assertEqual(metadata.quality_score, 1.0)
                
                # Verify file was stored
                self.assertTrue(os.path.exists(dataset.file_path))
            
        finally:
            # Cleanup
            try:
                os.unlink(temp_file.name)
            except FileNotFoundError:
                pass  # File may have been moved by uploader
            
            if 'dataset' in locals() and dataset and hasattr(dataset, 'file_path') and os.path.exists(dataset.file_path):
                os.unlink(dataset.file_path)
