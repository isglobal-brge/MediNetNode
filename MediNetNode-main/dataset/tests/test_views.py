"""
Tests for dataset views.
"""
import os
import tempfile
import json
from django.test import TestCase, Client
from django.urls import reverse
from django.db import connections
from django.utils import timezone
from django.contrib.auth import get_user_model
from unittest.mock import patch
from dataset.models import Dataset
from users.models import Role

User = get_user_model()


class DatasetDashboardViewTest(TestCase):
    """Test cases for datasets dashboard view."""
    
    # Use both databases as required by the multi-database setup
    databases = {'default', 'datasets_db'}
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        super().setUpClass()
        # Disable foreign key constraints for the entire test class
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        # Re-enable foreign key constraints
        datasets_connection = connections['datasets_db']
        with datasets_connection.cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON")
        super().tearDownClass()
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        
        # Create admin user
        self.admin_user = User.objects.create_user(
            username='admin_test',
            email='admin@test.com',
            password='testpass123'
        )
        
        # Create admin role and assign
        admin_role, _ = Role.objects.get_or_create(
            name='ADMIN'
        )
        self.admin_user.role = admin_role
        self.admin_user.save()
        
        # Create researcher user
        self.researcher_user = User.objects.create_user(
            username='researcher_test',
            email='researcher@test.com',
            password='testpass123'
        )
        
        researcher_role, _ = Role.objects.get_or_create(
            name='RESEARCHER'
        )
        self.researcher_user.role = researcher_role
        self.researcher_user.save()
        
        # Create test datasets in default database
        self.create_test_datasets()
    
    def create_test_datasets(self):
        """Create test datasets for dashboard metrics."""
        
        # Use raw SQL to avoid cross-database foreign key validation issues
        datasets_connection = connections['datasets_db']
        
        with datasets_connection.cursor() as cursor:
            
            # Insert dataset 1
            cursor.execute("""
                INSERT INTO dataset_dataset (
                    name, description, file_path, uploaded_by_id,
                    medical_domain, patient_count, data_type, anonymized,
                    file_size, file_format, columns_count, rows_count,
                    checksum_md5, is_active, uploaded_at, last_accessed, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'Cardiology Dataset 1',
                'Test cardiology dataset',
                '/test/cardio1.csv',
                self.admin_user.id,
                'cardiology',
                100,
                'tabular',
                True,
                1024*1024,  # 1MB
                'csv',
                10,
                100,
                'test_checksum_1',
                True,
                timezone.now().isoformat(),
                timezone.now().isoformat(),
                0
            ))
            dataset1_id = cursor.lastrowid
            
            # Insert dataset 2
            cursor.execute("""
                INSERT INTO dataset_dataset (
                    name, description, file_path, uploaded_by_id,
                    medical_domain, patient_count, data_type, anonymized,
                    file_size, file_format, columns_count, rows_count,
                    checksum_md5, is_active, uploaded_at, last_accessed, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'Neurology Dataset 1',
                'Test neurology dataset',
                '/test/neuro1.csv',
                self.admin_user.id,
                'neurology',
                200,
                'tabular',
                True,
                2*1024*1024,  # 2MB
                'csv',
                15,
                200,
                'test_checksum_2',
                True,
                timezone.now().isoformat(),
                timezone.now().isoformat(),
                0
            ))
            dataset2_id = cursor.lastrowid
            
            # Insert dataset access records
            cursor.execute("""
                INSERT INTO dataset_datasetaccess (
                    dataset_id, user_id, assigned_by_id, can_train,
                    can_view_metadata, assigned_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset1_id,
                self.researcher_user.id,
                self.admin_user.id,
                True,
                True,
                timezone.now().isoformat()
            ))
            
            cursor.execute("""
                INSERT INTO dataset_datasetaccess (
                    dataset_id, user_id, assigned_by_id, can_train,
                    can_view_metadata, assigned_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset2_id,
                self.researcher_user.id,
                self.admin_user.id,
                False,
                True,
                timezone.now().isoformat()
            ))
        
        # Store dataset objects for later use in tests by fetching them
        self.dataset1 = Dataset.objects.using('datasets_db').get(id=dataset1_id)
        self.dataset2 = Dataset.objects.using('datasets_db').get(id=dataset2_id)
    
    @patch('dataset.views.Dataset.objects')
    def test_dashboard_requires_admin_permission(self, mock_dataset_objects):
        """Test that dashboard requires admin permission."""
        # Mock the dataset objects to avoid database routing issues
        mock_dataset_objects.using.return_value.filter.return_value.count.return_value = 0
        mock_dataset_objects.using.return_value.filter.return_value.aggregate.return_value = {'total_size': 0}
        
        # Test without login
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Test with researcher (should be redirected to researcher info page)
        self.client.login(username='researcher_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirect to researcher info
    
    def test_dashboard_displays_correct_metrics(self):
        """Test that dashboard displays correct metrics."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset Management')
        
        # Check metrics in context
        context = response.context
        self.assertEqual(context['total_datasets'], 2)
        self.assertEqual(context['total_assignments'], 2)
        self.assertEqual(context['active_researchers'], 1)
        
        # Check datasets by domain
        domains = list(context['datasets_by_domain'])
        domain_names = [d['domain_lower'] for d in domains]
        self.assertIn('cardiology', domain_names)
        self.assertIn('neurology', domain_names)
    
    def test_dashboard_calculates_total_size(self):
        """Test that dashboard calculates total storage size correctly."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        context = response.context
        expected_size = 3 * 1024 * 1024  # 3MB total
        self.assertEqual(context['total_size_bytes'], expected_size)
        
        # Check formatted size
        self.assertIn('MB', context['total_size_formatted'])
    
    def test_dashboard_shows_recent_datasets(self):
        """Test that dashboard shows recent datasets."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        context = response.context
        recent_datasets = context['recent_datasets']
        
        # Should show both datasets (created recently)
        self.assertEqual(len(recent_datasets), 2)
        
        # Check that both expected datasets are present (order may vary)
        dataset_names = [dataset.name for dataset in recent_datasets]
        self.assertIn('Neurology Dataset 1', dataset_names)
        self.assertIn('Cardiology Dataset 1', dataset_names)
    
    def test_dashboard_shows_top_accessed_datasets(self):
        """Test that dashboard shows most accessed datasets."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        context = response.context
        top_datasets = context['top_datasets']
        
        # Should show datasets ordered by access count
        self.assertEqual(len(top_datasets), 2)
        
        # Both datasets should have access_count attribute
        for dataset in top_datasets:
            self.assertTrue(hasattr(dataset, 'access_count'))
    
    def test_dashboard_generates_daily_upload_data(self):
        """Test that dashboard generates data for upload charts."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        context = response.context
        daily_uploads = context['daily_uploads']
        
        # Should have 30 days of data
        self.assertEqual(len(daily_uploads), 30)
        
        # Each entry should have date and count
        for entry in daily_uploads:
            self.assertIn('date', entry)
            self.assertIn('count', entry)
    
    def test_dashboard_template_renders_correctly(self):
        """Test that dashboard template renders with all components."""
        self.client.login(username='admin_test', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        
        self.assertEqual(response.status_code, 200)
        
        # Check for key template elements
        self.assertContains(response, 'Total Datasets')
        self.assertContains(response, 'Storage Used')
        self.assertContains(response, 'Active Researchers')
        self.assertContains(response, 'Access Assignments')
        
        # Check for charts
        self.assertContains(response, 'domainsChart')
        
        # Check for action buttons
        self.assertContains(response, 'Upload Dataset')
        self.assertContains(response, 'View All')


class DatasetUploadViewTest(TestCase):
    """Test cases for dataset upload view."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        
        # Create admin user
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_upload',
            email='admin@upload.com',
            password='testpass123'
        )
        
        admin_role, _ = Role.objects.using('default').get_or_create(
            name='ADMIN'
        )
        self.admin_user.role = admin_role
        self.admin_user.save()
    
    def test_upload_page_requires_authentication(self):
        """Test that upload page requires authentication."""
        response = self.client.get(reverse('dataset:upload'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_upload_page_renders_correctly(self):
        """Test that upload page renders with drag&drop interface."""
        self.client.login(username='admin_upload', password='testpass123')
        response = self.client.get(reverse('dataset:upload'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset Upload')
        self.assertContains(response, 'drag-drop-zone')
        self.assertContains(response, 'Drag your file here')
        
        # Check form fields
        self.assertContains(response, 'name="name"')
        self.assertContains(response, 'name="description"')
        self.assertContains(response, 'name="medical_domain"')
        self.assertContains(response, 'name="data_type"')
        
        # Check security information
        self.assertContains(response, 'Security Validations')
        self.assertContains(response, 'K-anonymity')
    
    @patch('dataset.views.SecureDatasetUploader')
    def test_file_upload_with_valid_data(self, mock_uploader):
        """Test file upload with valid form data."""
        # Mock the uploader
        mock_instance = mock_uploader.return_value
        mock_dataset = Dataset(id=1, name='Test Dataset')
        mock_upload_info = {'phi_columns_removed': [], 'final_columns': 3, 'original_columns': 3}
        mock_instance.upload_dataset.return_value = (mock_dataset, mock_upload_info)
        
        self.client.login(username='admin_upload', password='testpass123')
        
        # Create a test file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_file.write(b'name,age,score\nJohn,30,85\nJane,25,90')
            temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(reverse('dataset:upload'), {
                    'name': 'Test Dataset',
                    'description': 'Test description',
                    'medical_domain': 'cardiology',
                    'data_type': 'tabular',
                    'file': f
                })
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore if file can't be deleted or doesn't exist
        
        # Should redirect or return success
        self.assertIn(response.status_code, [200, 302])
        
        # Check that uploader was called
        mock_uploader.assert_called_once()
    
    def test_upload_form_validation(self):
        """Test that upload form validates required fields."""
        self.client.login(username='admin_upload', password='testpass123')
        
        # Test with missing required fields
        response = self.client.post(reverse('dataset:upload'), {
            'name': '',  # Required field missing
            'description': 'Test description',
        })
        
        # Should return validation error as JSON
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'application/json')
        
        # Check error response
        response_data = json.loads(response.content)
        self.assertEqual(response_data['success'], False)
        self.assertIn('error', response_data)


class DatasetAPIViewTest(TestCase):
    """Test cases for dataset API endpoints."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        
        # Create admin user
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_api',
            email='admin@api.com',
            password='testpass123'
        )
        
        admin_role, _ = Role.objects.using('default').get_or_create(
            name='ADMIN'
        )
        self.admin_user.role = admin_role
        self.admin_user.save()
    
    def test_file_validation_api_empty_file(self):
        """Test file validation API with empty file."""
        self.client.login(username='admin_api', password='testpass123')
        
        # Create empty file
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            response = self.client.post(reverse('dataset:api_validate_file'), {
                'file': temp_file
            })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['valid'])
        self.assertIn('empty', data['error'].lower())
    
    def test_file_validation_api_invalid_extension(self):
        """Test file validation API with invalid file extension."""
        self.client.login(username='admin_api', password='testpass123')
        
        # Create file with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b'test content')
            temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(reverse('dataset:api_validate_file'), {
                    'file': f
                })
        finally:
            try:
                os.unlink(temp_file.name)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore if file can't be deleted or doesn't exist
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['valid'])
        self.assertIn('not allowed', data['error'])
    
    def test_file_validation_api_valid_file(self):
        """Test file validation API with valid file."""
        self.client.login(username='admin_api', password='testpass123')
        
        # Create valid CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_file.write(b'name,age,score\nJohn,30,85\nJane,25,90')
            temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(reverse('dataset:api_validate_file'), {
                    'file': f
                })
        finally:
            try:
                os.unlink(temp_file.name)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore if file can't be deleted or doesn't exist
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['valid'])
    
    def test_cancel_upload_api(self):
        """Test cancel upload API endpoint."""
        self.client.login(username='admin_api', password='testpass123')
        
        session_id = 'test_session_123'
        response = self.client.post(reverse('dataset:api_cancel_upload', args=[session_id]))
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('cancelled', data['message'].lower())


class DatasetPermissionTest(TestCase):
    """Test cases for dataset permission handling."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        
        # Create users with different roles
        self.admin_user = User.objects.db_manager('default').create_user(
            username='admin_perm',
            email='admin@perm.com',
            password='testpass123'
        )
        
        self.researcher_user = User.objects.db_manager('default').create_user(
            username='researcher_perm',
            email='researcher@perm.com',
            password='testpass123'
        )
        
        # Create roles
        admin_role, _ = Role.objects.using('default').get_or_create(
            name='ADMIN'
        )
        
        researcher_role, _ = Role.objects.using('default').get_or_create(
            name='RESEARCHER'
        )
        
        self.admin_user.role = admin_role
        self.admin_user.save()
        
        self.researcher_user.role = researcher_role
        self.researcher_user.save()
    
    def test_admin_can_access_dashboard(self):
        """Test that admin users can access dashboard."""
        self.client.login(username='admin_perm', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 200)
    
    def test_researcher_redirected_from_dashboard(self):
        """Test that researcher users are redirected from dashboard (API-only access)."""
        self.client.login(username='researcher_perm', password='testpass123')
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirected to researcher info page
    
    def test_admin_can_upload_datasets(self):
        """Test that admin users can access upload page."""
        self.client.login(username='admin_perm', password='testpass123')
        response = self.client.get(reverse('dataset:upload'))
        self.assertEqual(response.status_code, 200)
    
    def test_researcher_redirected_from_web_access(self):
        """Test that researcher users are redirected from web pages (API-only access)."""
        self.client.login(username='researcher_perm', password='testpass123')
        response = self.client.get(reverse('dataset:upload'))
        self.assertEqual(response.status_code, 302)  # Redirected to researcher info page
