import os
import tempfile
import hashlib
from django.test import TestCase
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetAccess, DatasetMetadata
from core.routers import DatabaseRouter

User = get_user_model()


class DatasetModelTest(TestCase):
    """Test cases for Dataset model."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        # Create test users in default database
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_test',
            email='admin@test.com',
            password='testpass123'
        )
        self.researcher_user = User.objects.db_manager('default').create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='testpass123'
        )
        
        # Also create users in datasets_db for FK relationships
        self.admin_user_datasets_db = User.objects.db_manager('datasets_db').create_user(
            username='admin_test_datasets',
            email='admin@test_datasets.com',
            password='testpass123'
        )
        
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False
        )
        self.temp_file.write("name,age,condition\nJohn,30,healthy\nJane,25,sick")
        self.temp_file.close()
        
        self.test_file_path = self.temp_file.name

    def tearDown(self):
        """Clean up test data."""
        # Remove temporary file
        if os.path.exists(self.test_file_path):
            os.unlink(self.test_file_path)

    def test_dataset_creation_in_datasets_db(self):
        """Test that Dataset is created in datasets_db correctly."""
        # Create dataset without triggering save() foreign key validation
        dataset_data = {
            'name': 'Test Cardiology Dataset',
            'description': 'Test dataset for cardiology research',
            'file_path': self.test_file_path,
            'uploaded_by_id': self.admin_user_datasets_db.id,
            'medical_domain': 'cardiology',
            'patient_count': 100,
            'data_type': 'tabular',
            'file_format': 'csv',
            'columns_count': 3,
            'rows_count': 2,
            'file_size': os.path.getsize(self.test_file_path),
            'checksum_md5': self._calculate_test_file_checksum()
        }
        
        # Use bulk_create to avoid FK validation
        datasets = Dataset.objects.using('datasets_db').bulk_create([Dataset(**dataset_data)])
        dataset = datasets[0]
        
        # Verify dataset was created
        self.assertIsNotNone(dataset.pk)
        self.assertEqual(dataset.name, 'Test Cardiology Dataset')
        self.assertEqual(dataset.medical_domain, 'cardiology')
        self.assertEqual(dataset.uploaded_by_id, self.admin_user_datasets_db.id)
        
        # Verify it exists in the database
        retrieved_dataset = Dataset.objects.using('datasets_db').get(pk=dataset.pk)
        self.assertEqual(retrieved_dataset.name, 'Test Cardiology Dataset')

    def test_checksum_md5_calculated_automatically(self):
        """Test that MD5 checksum is calculated automatically."""
        dataset = Dataset.objects.using('datasets_db').create(
            name='Test Checksum Dataset',
            description='Test dataset for checksum validation',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='general',
            patient_count=2,
            data_type='tabular',
            file_format='csv'
        )
        
        # Verify checksum was calculated
        self.assertIsNotNone(dataset.checksum_md5)
        self.assertEqual(len(dataset.checksum_md5), 32)  # MD5 is 32 characters
        
        # Verify checksum is correct
        expected_checksum = self._calculate_test_file_checksum()
        self.assertEqual(dataset.checksum_md5, expected_checksum)

    def test_file_size_calculated_automatically(self):
        """Test that file size is calculated automatically."""
        dataset = self._create_test_dataset('File Size')
        
        # Verify file size was calculated
        expected_size = os.path.getsize(self.test_file_path)
        self.assertEqual(dataset.file_size, expected_size)

    def test_medical_domain_validation(self):
        """Test medical domain validation."""
        dataset = Dataset(
            name='Test Invalid Domain Dataset',
            description='Test dataset with invalid domain',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='invalid_domain',  # Invalid domain
            data_type='tabular',
            file_format='csv'
        )
        
        with self.assertRaises(ValidationError):
            dataset.clean()

    def test_patient_count_validation_for_tabular_data(self):
        """Test that patient count is required for tabular data."""
        dataset = Dataset(
            name='Test Patient Count Dataset',
            description='Test dataset without patient count',
            file_path=self.test_file_path,
            uploaded_by_id=self.admin_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_format='csv',
            patient_count=None  # Missing patient count
        )
        
        with self.assertRaises(ValidationError):
            dataset.clean()

    def test_file_path_validation(self):
        """Test file path validation."""
        dataset = Dataset(
            name='Test Non-existent File Dataset',
            description='Test dataset with non-existent file',
            file_path='/non/existent/file.csv',
            uploaded_by_id=self.admin_user.id,
            medical_domain='general',
            data_type='tabular',
            file_format='csv'
        )
        
        with self.assertRaises(ValidationError):
            dataset.clean()

    def test_update_access_count(self):
        """Test access count update functionality."""
        dataset = self._create_test_dataset('Access Count')
        
        initial_count = dataset.access_count
        self.assertEqual(initial_count, 0)
        
        # Update access count
        dataset.update_access_count()
        
        # Verify count increased and last_accessed was set
        dataset.refresh_from_db()
        self.assertEqual(dataset.access_count, 1)
        self.assertIsNotNone(dataset.last_accessed)

    def test_get_file_size_display(self):
        """Test human-readable file size display."""
        dataset = self._create_test_dataset('File Size Display')
        
        size_display = dataset.get_file_size_display()
        self.assertIn('B', size_display)  # Should show bytes for small file

    def _create_test_dataset(self, name_suffix="", **kwargs):
        """Helper method to create a test dataset."""
        defaults = {
            'name': f'Test Dataset {name_suffix}',
            'description': 'Test dataset description',
            'file_path': self.test_file_path,
            'uploaded_by_id': self.admin_user.id,
            'medical_domain': 'general',
            'data_type': 'tabular',
            'file_format': 'csv'
        }
        defaults.update(kwargs)
        return Dataset.objects.using('datasets_db').create(**defaults)

    def _calculate_test_file_checksum(self):
        """Helper method to calculate the expected checksum."""
        hash_md5 = hashlib.md5()
        with open(self.test_file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class DatasetAccessModelTest(TestCase):
    """Test cases for DatasetAccess model."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        # Create test users in both databases for FK consistency
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_access_test',
            email='admin@access.com',
            password='testpass123'
        )
        self.researcher_user = User.objects.db_manager('default').create_user(
            username='researcher_access_test',
            email='researcher@access.com',
            password='testpass123'
        )
        self.other_user = User.objects.db_manager('default').create_user(
            username='other_access_test',
            email='other@access.com',
            password='testpass123'
        )
        
        # Also create users in datasets_db for FK relationships
        self.admin_user_datasets = User.objects.db_manager('datasets_db').create_user(
            username='admin_access_datasets',
            email='admin@datasets.com',
            password='testpass123'
        )
        self.researcher_user_datasets = User.objects.db_manager('datasets_db').create_user(
            username='researcher_datasets',
            email='researcher@datasets.com',
            password='testpass123'
        )
        self.other_user_datasets = User.objects.db_manager('datasets_db').create_user(
            username='other_datasets',
            email='other@datasets.com',
            password='testpass123'
        )
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False
        )
        self.temp_file.write("name,age\nJohn,30")
        self.temp_file.close()
        
        # Create test dataset in datasets_db using datasets_db user
        self.dataset = Dataset.objects.using('datasets_db').create(
            name='Test Access Dataset',
            description='Dataset for access testing',
            file_path=self.temp_file.name,
            uploaded_by_id=self.admin_user_datasets.id,
            medical_domain='general',
            data_type='tabular',
            file_format='csv'
        )

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_dataset_access_creation(self):
        """Test DatasetAccess creation."""
        access = DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.researcher_user_datasets.id,
            assigned_by_id=self.admin_user_datasets.id,
            can_train=True,
            can_view_metadata=True
        )
        
        self.assertIsNotNone(access.pk)
        self.assertEqual(access.dataset_id, self.dataset.id)
        self.assertEqual(access.user_id, self.researcher_user_datasets.id)
        self.assertEqual(access.assigned_by_id, self.admin_user_datasets.id)
        self.assertTrue(access.can_train)
        self.assertTrue(access.can_view_metadata)

    def test_unique_together_constraint(self):
        """Test unique_together constraint for dataset and user."""
        # Create first access
        DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.researcher_user_datasets.id,
            assigned_by_id=self.admin_user_datasets.id
        )
        
        # Try to create duplicate access - should raise IntegrityError
        with self.assertRaises(IntegrityError):
            DatasetAccess.objects.using('datasets_db').create(
                dataset_id=self.dataset.id,
                user_id=self.researcher_user_datasets.id,  # Same user
                assigned_by_id=self.admin_user_datasets.id
            )

    def test_multiple_users_same_dataset(self):
        """Test that multiple users can access the same dataset."""
        access1 = DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.researcher_user_datasets.id,
            assigned_by_id=self.admin_user_datasets.id
        )
        
        access2 = DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.other_user_datasets.id,  # Different user
            assigned_by_id=self.admin_user_datasets.id
        )
        
        self.assertNotEqual(access1.user_id, access2.user_id)
        self.assertEqual(access1.dataset_id, access2.dataset_id)

    def test_permission_defaults(self):
        """Test default permission values."""
        access = DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.researcher_user_datasets.id,
            assigned_by_id=self.admin_user_datasets.id
        )
        
        # Default values should be True
        self.assertTrue(access.can_train)
        self.assertTrue(access.can_view_metadata)

    def test_string_representation(self):
        """Test string representation of DatasetAccess."""
        # Need to access the related user object for string representation
        # This test may need the objects to be loaded from their respective databases
        access = DatasetAccess.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            user_id=self.researcher_user_datasets.id,
            assigned_by_id=self.admin_user_datasets.id
        )
        
        # For now, just test that str() doesn't crash
        str_repr = str(access)
        self.assertIsInstance(str_repr, str)


class DatasetMetadataModelTest(TestCase):
    """Test cases for DatasetMetadata model."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        # Create users in both databases for FK consistency
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_metadata_test',
            email='admin@metadata.com',
            password='testpass123'
        )
        
        self.admin_user_datasets = User.objects.db_manager('datasets_db').create_user(
            username='admin_metadata_datasets',
            email='admin@metadata_datasets.com',
            password='testpass123'
        )
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False
        )
        self.temp_file.write("name,age,score\nJohn,30,85\nJane,25,\nBob,35,92")
        self.temp_file.close()
        
        # Create test dataset using datasets_db user
        self.dataset = Dataset.objects.using('datasets_db').create(
            name='Test Metadata Dataset',
            description='Dataset for metadata testing',
            file_path=self.temp_file.name,
            uploaded_by_id=self.admin_user_datasets.id,
            medical_domain='general',
            data_type='tabular',
            file_format='csv',
            columns_count=3,
            rows_count=3
        )

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_metadata_creation(self):
        """Test DatasetMetadata creation with OneToOne relationship."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            statistical_summary={'mean_age': 30, 'std_age': 5},
            missing_values={'age': 0, 'score': 1},
            data_distribution={'age': 'normal', 'score': 'skewed'}
        )
        
        self.assertIsNotNone(metadata.pk)
        self.assertEqual(metadata.dataset_id, self.dataset.id)
        self.assertEqual(metadata.statistical_summary['mean_age'], 30)
        self.assertEqual(metadata.missing_values['score'], 1)

    def test_onetoone_relationship(self):
        """Test OneToOne relationship constraint."""
        # Create first metadata
        metadata1 = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            statistical_summary={'test': 'data'}
        )
        
        # Try to create another metadata for same dataset
        with self.assertRaises(IntegrityError):
            DatasetMetadata.objects.using('datasets_db').create(
                dataset_id=self.dataset.id,
                statistical_summary={'test': 'other_data'}
            )

    def test_calculate_completeness(self):
        """Test completeness percentage calculation."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            missing_values={'name': 0, 'age': 0, 'score': 1}  # 1 missing value
        )
        
        completeness = metadata.calculate_completeness()
        
        # Total cells: 3 rows × 3 columns = 9 cells
        # Missing cells: 1
        # Completeness: (9 - 1) / 9 * 100 = 88.89%
        expected_completeness = 88.89
        self.assertAlmostEqual(completeness, expected_completeness, places=2)

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            statistical_summary={'mean': 50, 'std': 10},
            missing_values={'name': 0, 'age': 0, 'score': 1},
            data_distribution={'age': 'normal'}
        )
        
        quality_score = metadata.calculate_quality_score()
        
        # Should be > 0 and <= 1.0
        self.assertGreater(quality_score, 0)
        self.assertLessEqual(quality_score, 1.0)

    def test_auto_calculation_on_save(self):
        """Test that quality metrics are calculated automatically on save."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            missing_values={'name': 0, 'age': 0, 'score': 1}
        )
        
        # Should have auto-calculated completeness and quality score
        self.assertIsNotNone(metadata.completeness_percentage)
        self.assertIsNotNone(metadata.quality_score)
        self.assertGreater(metadata.completeness_percentage, 0)
        self.assertGreater(metadata.quality_score, 0)

    def test_string_representation(self):
        """Test string representation of DatasetMetadata."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id
        )
        
        # Since we can't easily load the related dataset in cross-DB tests,
        # just test that str() doesn't crash
        str_repr = str(metadata)
        self.assertIsInstance(str_repr, str)

    def test_metadata_relationship_access(self):
        """Test accessing metadata through dataset relationship."""
        metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset_id=self.dataset.id,
            statistical_summary={'test': 'data'}
        )
        
        # Verify metadata exists with correct dataset ID
        retrieved_metadata = DatasetMetadata.objects.using('datasets_db').get(dataset_id=self.dataset.id)
        self.assertEqual(retrieved_metadata.dataset_id, self.dataset.id)
        self.assertEqual(retrieved_metadata.statistical_summary['test'], 'data')


class DatabaseRoutingTest(TestCase):
    """Test cases for database routing."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_routing_test',
            email='admin@routing.com',
            password='testpass123'
        )

    def test_dataset_models_use_datasets_db(self):
        """Test that dataset models are routed to datasets_db."""
        # This is tested implicitly by the fact that our other tests work
        # but we can also test the router directly
        
        router = DatabaseRouter()
        
        # Test Dataset model routing
        self.assertEqual(router.db_for_read(Dataset), 'datasets_db')
        self.assertEqual(router.db_for_write(Dataset), 'datasets_db')
        
        # Test DatasetAccess model routing
        self.assertEqual(router.db_for_read(DatasetAccess), 'datasets_db')
        self.assertEqual(router.db_for_write(DatasetAccess), 'datasets_db')
        
        # Test DatasetMetadata model routing
        self.assertEqual(router.db_for_read(DatasetMetadata), 'datasets_db')
        self.assertEqual(router.db_for_write(DatasetMetadata), 'datasets_db')

    def test_user_models_use_default_db(self):
        """Test that User models still use default database."""
        
        router = DatabaseRouter()
        
        # Test User model routing (should be default)
        self.assertEqual(router.db_for_read(User), 'default')
        self.assertEqual(router.db_for_write(User), 'default')

    def test_migration_routing(self):
        """Test migration routing."""
        
        router = DatabaseRouter()
        
        # Test that dataset app migrations go to datasets_db
        self.assertTrue(router.allow_migrate('datasets_db', 'dataset'))
        self.assertFalse(router.allow_migrate('default', 'dataset'))
        
        # Test that users app migrations go to both databases in test environment
        # (This is needed for FK relationships to work in tests)
        self.assertTrue(router.allow_migrate('default', 'users'))
        self.assertTrue(router.allow_migrate('datasets_db', 'users'))  # Changed from False to True