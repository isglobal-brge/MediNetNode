"""
Tests for dataset pause/activate functionality and API validation.
"""
import os
import tempfile
import json
from django.db import connections
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetAccess
from users.models import Role, APIKey
from api.views import validate_training_permissions, get_user_datasets
from django.http import JsonResponse

User = get_user_model()


class DatasetToggleActiveTest(TestCase):
    """Test cases for dataset pause/activate functionality."""

    databases = {'default', 'datasets_db'}

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        super().setUpClass()
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON")
        super().tearDownClass()

    def setUp(self):
        """Set up test data."""
        self.client = Client()

        # Get or create roles
        self.admin_role, _ = Role.objects.get_or_create(
            name='ADMIN',
            defaults={
                'permissions': {
                    'dataset.view': True,
                    'dataset.upload': True,
                    'dataset.edit': True,
                    'dataset.delete': True,
                    'dataset.manage_access': True,
                    'dataset.train': True
                }
            }
        )

        # Create admin user
        self.admin_user = User.objects.create_user(
            username='admin_test',
            email='admin@test.com',
            password='testpass123',
            role=self.admin_role
        )

        # Create test dataset with temp file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.temp_file.write(b'col1,col2\nval1,val2\n')
        self.temp_file.close()

        self.dataset = Dataset.objects.using('datasets_db').create(
            name='Test Dataset',
            description='Test description',
            file_path=self.temp_file.name,
            uploaded_by_id=self.admin_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=1024,
            file_format='csv',
            checksum_md5='test_checksum',
            is_active=True
        )

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_admin_can_pause_dataset(self):
        """Test that admin can pause an active dataset."""
        self.client.login(username='admin_test', password='testpass123')

        # Verify dataset is active
        self.assertTrue(self.dataset.is_active)

        # Pause the dataset
        response = self.client.post(
            reverse('dataset:toggle_active', args=[self.dataset.id])
        )

        # Check redirect
        self.assertEqual(response.status_code, 302)

        # Verify dataset is now paused
        self.dataset.refresh_from_db()
        self.assertFalse(self.dataset.is_active)

    def test_admin_can_activate_dataset(self):
        """Test that admin can activate a paused dataset."""
        self.client.login(username='admin_test', password='testpass123')

        # Set dataset to paused
        self.dataset.is_active = False
        self.dataset.save(using='datasets_db')

        # Activate the dataset
        response = self.client.post(
            reverse('dataset:toggle_active', args=[self.dataset.id])
        )

        # Check redirect
        self.assertEqual(response.status_code, 302)

        # Verify dataset is now active
        self.dataset.refresh_from_db()
        self.assertTrue(self.dataset.is_active)

    def test_toggle_active_requires_post(self):
        """Test that toggle_active only accepts POST requests."""
        self.client.login(username='admin_test', password='testpass123')

        # Try GET request
        response = self.client.get(
            reverse('dataset:toggle_active', args=[self.dataset.id])
        )

        # Should be rejected
        self.assertEqual(response.status_code, 405)

    def test_paused_dataset_in_list(self):
        """Test that paused datasets appear in list view."""
        self.client.login(username='admin_test', password='testpass123')

        # Pause the dataset
        self.dataset.is_active = False
        self.dataset.save(using='datasets_db')

        # Get list view
        response = self.client.get(reverse('dataset:list'))

        # Paused dataset should still appear
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.dataset.name)

    def test_filter_by_status_active(self):
        """Test filtering datasets by active status."""
        self.client.login(username='admin_test', password='testpass123')

        # Create a paused dataset
        temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file2.write(b'col1,col2\nval1,val2\n')
        temp_file2.close()

        paused_dataset = Dataset.objects.using('datasets_db').create(
            name='Paused Dataset',
            description='Paused',
            file_path=temp_file2.name,
            uploaded_by_id=self.admin_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=1024,
            file_format='csv',
            checksum_md5='test_checksum2',
            is_active=False
        )

        try:
            # Filter by active status
            response = self.client.get(reverse('dataset:list') + '?status=active')

            # Should only show active dataset
            self.assertContains(response, self.dataset.name)
            self.assertNotContains(response, paused_dataset.name)
        finally:
            os.unlink(temp_file2.name)

    def test_filter_by_status_paused(self):
        """Test filtering datasets by paused status."""
        self.client.login(username='admin_test', password='testpass123')

        # Pause the dataset
        self.dataset.is_active = False
        self.dataset.save(using='datasets_db')

        # Filter by paused status
        response = self.client.get(reverse('dataset:list') + '?status=paused')

        # Should show paused dataset
        self.assertContains(response, self.dataset.name)


class DatasetAPIActiveValidationTest(TestCase):
    """Test cases for API validation of is_active status."""

    databases = {'default', 'datasets_db'}

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        super().setUpClass()
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON")
        super().tearDownClass()

    def setUp(self):
        """Set up test data."""
        self.client = Client()

        # Get or create roles
        self.researcher_role, _ = Role.objects.get_or_create(
            name='RESEARCHER',
            defaults={
                'permissions': {
                    'dataset.view': True,
                    'dataset.train': True
                }
            }
        )

        # Create researcher user
        self.researcher_user = User.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='testpass123',
            role=self.researcher_role
        )

        # Create API key for researcher
        
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            key='test_api_key_123'
        )

        # Create test datasets
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.temp_file.write(b'col1,col2\nval1,val2\n')
        self.temp_file.close()

        self.active_dataset = Dataset.objects.using('datasets_db').create(
            name='Active Dataset',
            description='Active',
            file_path=self.temp_file.name,
            uploaded_by_id=self.researcher_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=1024,
            file_format='csv',
            checksum_md5='test_checksum_active',
            is_active=True
        )

        self.temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.temp_file2.write(b'col1,col2\nval1,val2\n')
        self.temp_file2.close()

        self.paused_dataset = Dataset.objects.using('datasets_db').create(
            name='Paused Dataset',
            description='Paused',
            file_path=self.temp_file2.name,
            uploaded_by_id=self.researcher_user.id,
            medical_domain='cardiology',
            data_type='tabular',
            file_size=1024,
            file_format='csv',
            checksum_md5='test_checksum_paused',
            is_active=False
        )

        # Grant access to both datasets
        DatasetAccess.objects.using('datasets_db').create(
            dataset=self.active_dataset,
            user_id=self.researcher_user.id,
            assigned_by_id=self.researcher_user.id,
            can_train=True,
            can_view_metadata=True
        )

        DatasetAccess.objects.using('datasets_db').create(
            dataset=self.paused_dataset,
            user_id=self.researcher_user.id,
            assigned_by_id=self.researcher_user.id,
            can_train=True,
            can_view_metadata=True
        )

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        if os.path.exists(self.temp_file2.name):
            os.unlink(self.temp_file2.name)

    def test_get_data_info_excludes_paused_datasets(self):
        """Test that get_user_datasets() only returns active datasets."""

        # Call the helper function directly
        datasets = get_user_datasets(self.researcher_user)

        # Should only return active dataset
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].id, self.active_dataset.id)
        self.assertEqual(datasets[0].name, 'Active Dataset')
        self.assertTrue(datasets[0].is_active)

    def test_start_client_rejects_paused_dataset(self):
        """Test that validate_training_permissions rejects paused datasets."""
        

        # Prepare model_json with paused dataset
        model_json = {
            'model': {
                'dataset': {
                    'selected_datasets': [
                        {
                            'dataset_id': self.paused_dataset.id,
                            'dataset_name': 'Paused Dataset'
                        }
                    ]
                }
            }
        }

        # Call the validation function directly
        result = validate_training_permissions(self.researcher_user, model_json)

        # Should return JsonResponse error for paused dataset
        self.assertIsInstance(result, JsonResponse)
        self.assertEqual(result.status_code, 403)

        # Parse the response content
        data = json.loads(result.content)
        self.assertIn('error', data)
        self.assertIn('paused', data['error'].lower())

    def test_start_client_accepts_active_dataset(self):
        """Test that validate_training_permissions accepts active datasets."""

        # Prepare model_json with active dataset
        model_json = {
            'model': {
                'dataset': {
                    'selected_datasets': [
                        {
                            'dataset_id': self.active_dataset.id,
                            'dataset_name': 'Active Dataset'
                        }
                    ]
                }
            }
        }

        # Call the validation function directly
        result = validate_training_permissions(self.researcher_user, model_json)

        # Should return None (no error) for active dataset
        self.assertIsNone(result)
