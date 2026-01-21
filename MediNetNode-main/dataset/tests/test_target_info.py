"""
Tests for target_info metadata generation in dataset uploader.
This tests the automatic detection of task type (regression, binary, multiclass)
and recommended neural network output configuration.
"""

import os
import tempfile
import pandas as pd
from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from dataset.uploader import SecureDatasetUploader
from dataset.models import DatasetMetadata

User = get_user_model()


class TargetInfoMetadataTest(TestCase):
    """Test target_info generation for different dataset types."""

    databases = {'default', 'datasets_db'}

    def setUp(self):
        """Set up test user."""
        self.user = User.objects.db_manager('default').create_user(
            username='test_target_info',
            email='targetinfo@test.com',
            password='testpass123'
        )
        self.uploader = SecureDatasetUploader(self.user)

    def create_temp_csv(self, data_dict, filename='test.csv'):
        """Helper to create temporary CSV file."""
        df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name

    def test_binary_classification_detection(self):
        """Test binary classification target detection (2 unique values)."""
        # Create dataset with binary target
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90],
            'has_disease': ['no', 'yes', 'no', 'yes', 'no', 'yes']
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='has_disease')

            # Verify target_info exists
            self.assertIn('target_info', metadata)
            target_info = metadata['target_info']

            # Verify binary classification detection
            self.assertEqual(target_info['task_type'], 'classification')
            self.assertEqual(target_info['task_subtype'], 'binary_classification')
            self.assertEqual(target_info['num_classes'], 2)
            self.assertEqual(target_info['output_neurons'], 1)
            self.assertEqual(target_info['data_type'], 'categorical')

            # Verify recommended PyTorch configuration
            self.assertEqual(target_info['recommended_activation'], 'sigmoid')
            self.assertEqual(target_info['recommended_loss'], 'BCEWithLogitsLoss')

            # Verify classes are detected
            self.assertIn('classes', target_info)
            self.assertEqual(len(target_info['classes']), 2)

        finally:
            os.unlink(file_path)

    def test_multiclass_classification_detection(self):
        """Test multiclass classification target detection (>2 unique values)."""
        # Create dataset with multiclass target (3 disease stages)
        data = {
            'age': [25, 30, 35, 40, 45, 50, 55, 60],
            'weight': [65, 70, 75, 80, 85, 90, 95, 100],
            'disease_stage': ['healthy', 'mild', 'severe', 'healthy', 'mild', 'severe', 'healthy', 'mild']
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='disease_stage')

            # Verify target_info exists
            self.assertIn('target_info', metadata)
            target_info = metadata['target_info']

            # Verify multiclass classification detection
            self.assertEqual(target_info['task_type'], 'classification')
            self.assertEqual(target_info['task_subtype'], 'multiclass_classification')
            self.assertEqual(target_info['num_classes'], 3)
            self.assertEqual(target_info['output_neurons'], 3)  # 3 neurons for 3 classes
            self.assertEqual(target_info['data_type'], 'categorical')

            # Verify recommended PyTorch configuration
            self.assertEqual(target_info['recommended_activation'], 'softmax')
            self.assertEqual(target_info['recommended_loss'], 'CrossEntropyLoss')

            # Verify all classes are detected
            self.assertIn('classes', target_info)
            self.assertEqual(set(target_info['classes']), {'healthy', 'mild', 'severe'})

        finally:
            os.unlink(file_path)

    def test_regression_detection(self):
        """Test regression target detection (continuous numeric values)."""
        # Create dataset with regression target
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90],
            'blood_pressure': [120.5, 130.2, 125.8, 140.1, 135.6, 128.3]
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='blood_pressure')

            # Verify target_info exists
            self.assertIn('target_info', metadata)
            target_info = metadata['target_info']

            # Verify regression detection
            self.assertEqual(target_info['task_type'], 'regression')
            self.assertEqual(target_info['task_subtype'], 'regression')
            self.assertEqual(target_info['output_neurons'], 1)
            self.assertEqual(target_info['data_type'], 'numeric')

            # Verify recommended PyTorch configuration
            self.assertEqual(target_info['recommended_activation'], 'none')
            self.assertEqual(target_info['recommended_loss'], 'MSELoss')

            # Verify value range statistics
            self.assertIn('value_range', target_info)
            self.assertIn('min', target_info['value_range'])
            self.assertIn('max', target_info['value_range'])
            self.assertIn('mean', target_info['value_range'])
            self.assertIn('std', target_info['value_range'])

        finally:
            os.unlink(file_path)

    def test_integer_binary_classification(self):
        """Test binary classification with integer labels (0, 1)."""
        # Create dataset with integer binary target
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90],
            'has_disease': [0, 1, 0, 1, 0, 1]
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='has_disease')

            # Verify target_info exists
            self.assertIn('target_info', metadata)
            target_info = metadata['target_info']

            # Verify binary classification detection (even with integers)
            self.assertEqual(target_info['task_type'], 'classification')
            self.assertEqual(target_info['task_subtype'], 'binary_classification')
            self.assertEqual(target_info['num_classes'], 2)
            self.assertEqual(target_info['output_neurons'], 1)

            # Verify classes
            self.assertEqual(set(target_info['classes']), {0, 1})

        finally:
            os.unlink(file_path)

    def test_integer_multiclass_classification(self):
        """Test multiclass classification with integer labels."""
        # Create dataset with integer multiclass target
        data = {
            'age': [25, 30, 35, 40, 45, 50, 55, 60],
            'weight': [65, 70, 75, 80, 85, 90, 95, 100],
            'risk_level': [0, 1, 2, 0, 1, 2, 0, 1]  # 3 risk levels
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='risk_level')

            # Verify target_info exists
            self.assertIn('target_info', metadata)
            target_info = metadata['target_info']

            # Verify multiclass classification detection
            self.assertEqual(target_info['task_type'], 'classification')
            self.assertEqual(target_info['task_subtype'], 'multiclass_classification')
            self.assertEqual(target_info['num_classes'], 3)
            self.assertEqual(target_info['output_neurons'], 3)

            # Verify classes
            self.assertEqual(set(target_info['classes']), {0, 1, 2})

        finally:
            os.unlink(file_path)

    def test_no_target_column_specified(self):
        """Test that target_info is not generated when no target specified."""
        # Create dataset
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90],
            'diagnosis': ['A', 'B', 'A', 'B', 'A', 'B']
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata WITHOUT target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column=None)

            # Verify target_info does NOT exist
            self.assertNotIn('target_info', metadata)

        finally:
            os.unlink(file_path)

    def test_invalid_target_column(self):
        """Test handling of non-existent target column."""
        # Create dataset
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90]
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with non-existent target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='nonexistent')

            # Verify target_info does NOT exist (column not found)
            self.assertNotIn('target_info', metadata)

        finally:
            os.unlink(file_path)

    def test_target_preservation_after_phi_removal(self):
        """Test that target column is preserved even if it matches PHI patterns."""
        # Create dataset where target matches a forbidden pattern
        data = {
            'feature1': [25, 30, 35, 40, 45, 50],
            'feature2': [65, 70, 75, 80, 85, 90],
            'patient_id': [100, 101, 102, 103, 104, 105],  # PHI column (matches \bpatient_id\b pattern)
            'diagnosis': ['A', 'B', 'A', 'B', 'A', 'B']  # Target (safe)
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='diagnosis')

            # Validate compliance (this should remove PHI but keep target)
            metadata = self.uploader._validate_medical_compliance(
                metadata, file_path, target_column='diagnosis'
            )

            # Verify PHI column was removed
            self.assertIn('phi_columns_removed', metadata)
            self.assertEqual(len(metadata['phi_columns_removed']), 1)

            # Verify target still exists and has target_info
            # Note: After PHI removal, the file is rewritten, so target_info should be regenerated
            self.assertIn('target_info', metadata)
            self.assertEqual(metadata['target_info']['column_name'], 'diagnosis')

        finally:
            os.unlink(file_path)

    def test_five_class_classification(self):
        """Test 5-class classification (common in medical staging)."""
        # Create dataset with 5 disease stages
        data = {
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'score': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'stage': ['stage0', 'stage1', 'stage2', 'stage3', 'stage4',
                     'stage0', 'stage1', 'stage2', 'stage3', 'stage4']
        }
        file_path = self.create_temp_csv(data)

        try:
            # Extract metadata with target column
            metadata = self.uploader._extract_csv_metadata(file_path, target_column='stage')

            # Verify target_info
            target_info = metadata['target_info']

            # Verify 5-class configuration
            self.assertEqual(target_info['num_classes'], 5)
            self.assertEqual(target_info['output_neurons'], 5)  # 5 output neurons for 5 classes
            self.assertEqual(target_info['task_subtype'], 'multiclass_classification')

        finally:
            os.unlink(file_path)


class TargetInfoIntegrationTest(TestCase):
    """Integration tests for target_info in full upload flow."""

    databases = {'default', 'datasets_db'}

    def setUp(self):
        """Set up test user."""
        self.user = User.objects.db_manager('default').create_user(
            username='test_integration',
            email='integration@test.com',
            password='testpass123'
        )

    def test_target_info_saved_in_metadata(self):
        """Test that target_info is properly saved in DatasetMetadata."""
        # Create a valid dataset with binary classification
        data = {
            'age': [25, 30, 35, 40, 45, 50],
            'weight': [65, 70, 75, 80, 85, 90],
            'diagnosis': ['healthy', 'diseased', 'healthy', 'diseased', 'healthy', 'diseased']
        }

        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        try:
            uploader = SecureDatasetUploader(self.user)

            # Upload dataset with target column
            dataset, upload_info = uploader.upload_dataset(
                file_path=temp_file.name,
                name='Test Dataset with Target',
                description='Test dataset for target_info validation',
                medical_domain='general',
                data_type='tabular',
                target_column='diagnosis'
            )

            # Verify dataset was created
            self.assertIsNotNone(dataset)
            self.assertEqual(dataset.target_column, 'diagnosis')

            # Verify metadata was created and contains target_info
            metadata_obj = DatasetMetadata.objects.using('datasets_db').filter(
                dataset_id=dataset.id
            ).first()

            self.assertIsNotNone(metadata_obj)

            # Parse the statistical_summary JSON field
            metadata_json = metadata_obj.statistical_summary
            self.assertIn('target_info', metadata_json)

            # Verify target_info content
            target_info = metadata_json['target_info']
            self.assertEqual(target_info['column_name'], 'diagnosis')
            self.assertEqual(target_info['task_subtype'], 'binary_classification')
            self.assertEqual(target_info['output_neurons'], 1)

        finally:
            # Cleanup
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            # Dataset file cleanup is handled by the uploader's transaction
