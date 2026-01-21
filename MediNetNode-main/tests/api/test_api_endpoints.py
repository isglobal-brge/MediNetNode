"""
Tests for API endpoints - Essential endpoints for RESEARCHER users.
Tests the complete API workflow with authentication.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
import json
from users.models import CustomUser, Role, APIKey
from dataset.models import Dataset, DatasetAccess


class APIEndpointTests(TestCase):
    """Test API endpoints with proper authentication."""
    
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()

        # Get or create RESEARCHER role
        self.researcher_role, _ = Role.objects.get_or_create(
            name='RESEARCHER',
            defaults={
                'permissions': {
                    'api.access': True,
                    'dataset.view': True,
                    'dataset.train': True
                }
            }
        )
        
        # Create RESEARCHER user
        self.researcher_user = CustomUser.objects.create_user(
            username='researcher_api_test',
            email='researcher@test.com',
            password='ResearcherPass123!',
            role=self.researcher_role
        )
        
        # Create API key
        self.api_key = APIKey.objects.create(
            user=self.researcher_user,
            name='Test API Key',
            ip_whitelist=['127.0.0.1', '192.168.1.100']
        )
        
        # Create test datasets
        self.dataset1 = Dataset.objects.using('datasets_db').create(
            name='Heart Disease Dataset',
            description='Heart disease prediction dataset',
            file_path='/data/heart_disease.csv',
            uploaded_by_id=1,
            medical_domain='cardiology',
            patient_count=1000,
            data_type='tabular',
            file_size=1024000,
            file_format='csv'
        )
        
        self.dataset2 = Dataset.objects.using('datasets_db').create(
            name='Diabetes Dataset',
            description='Diabetes prediction dataset',
            file_path='/data/diabetes.csv',
            uploaded_by_id=1,
            medical_domain='general',
            patient_count=500,
            data_type='tabular',
            file_size=512000,
            file_format='csv'
        )
        
        # Grant access to datasets
        DatasetAccess.objects.using('datasets_db').create(
            dataset=self.dataset1,
            user_id=self.researcher_user.id,
            assigned_by_id=1,  # Assume admin user with ID 1
            can_train=True,
            can_view_metadata=True
        )
        
        DatasetAccess.objects.using('datasets_db').create(
            dataset=self.dataset2,
            user_id=self.researcher_user.id,
            assigned_by_id=1,
            can_train=True,
            can_view_metadata=True
        )
    
    def get_auth_headers(self):
        """Get authentication headers for API requests."""
        return {
            'HTTP_X_API_KEY': self.api_key.key,
            'HTTP_X_CLIENT_IP': '127.0.0.1'
        }
    
    def test_ping_endpoint_success(self):
        """Test ping endpoint with valid authentication."""
        response = self.client.get(
            '/api/v1/ping',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['status'], 'pong')
    
    def test_ping_endpoint_no_auth(self):
        """Test ping endpoint without authentication."""
        response = self.client.get('/api/v1/ping')
        
        self.assertEqual(response.status_code, 401)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'Missing X-API-Key header')
    
    def test_ping_endpoint_invalid_key(self):
        """Test ping endpoint with invalid API key."""
        response = self.client.get(
            '/api/v1/ping',
            HTTP_X_API_KEY='invalid_key_12345',
            HTTP_X_CLIENT_IP='127.0.0.1'
        )
        
        self.assertEqual(response.status_code, 401)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'Invalid API key')
    
    def test_ping_endpoint_wrong_ip(self):
        """Test ping endpoint from unauthorized IP."""
        response = self.client.get(
            '/api/v1/ping',
            HTTP_X_API_KEY=self.api_key.key,
            HTTP_X_CLIENT_IP='192.168.2.100'  # Not in whitelist
        )
        
        self.assertEqual(response.status_code, 403)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'IP address not authorized for this API key')
    
    def test_get_data_info_success(self):
        """Test get_data_info endpoint with valid authentication."""
        response = self.client.get(
            '/api/v1/get-data-info',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content.decode('utf-8'))
        
        # Check that we get the expected structure
        expected_keys = ['dataset_id', 'dataset_name', 'medical_domain', 
                        'patient_count', 'data_type', 'file_size', 'description', 'created_at']
        for key in expected_keys:
            self.assertIn(key, data)
        
        # Check that we get both datasets
        self.assertEqual(len(data['dataset_id']), 2)
        self.assertIn('Heart Disease Dataset', data['dataset_name'])
        self.assertIn('Diabetes Dataset', data['dataset_name'])
    
    def test_get_data_info_no_auth(self):
        """Test get_data_info endpoint without authentication."""
        response = self.client.get('/api/v1/get-data-info')
        
        self.assertEqual(response.status_code, 401)
    
    def test_get_data_info_no_datasets(self):
        """Test get_data_info when user has no dataset access."""
        # Remove dataset access
        DatasetAccess.objects.using('datasets_db').filter(
            user_id=self.researcher_user.id
        ).delete()
        
        response = self.client.get(
            '/api/v1/get-data-info',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 403)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'No datasets available for this user')
    
    def test_start_client_success(self):
        """Test start_client endpoint with valid request."""
        payload = {
            "model_json": {
                "framework": "pytorch",
                "model": {
                    "layers": [
                        {"type": "linear", "input_size": 25, "output_size": 50},
                        {"type": "relu"},
                        {"type": "linear", "input_size": 50, "output_size": 1}
                    ],
                    "dataset": {
                        "selected_datasets": [
                            {"dataset_id": self.dataset1.id, "dataset_name": "Heart Disease Dataset"}
                        ]
                    }
                }
            },
            "server_address": "localhost:8080",
            "client_id": "client_123"
        }
        
        response = self.client.post(
            '/api/v1/start-client',
            data=json.dumps(payload),
            content_type='application/json',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['status'], 'Flower Client started')
        self.assertEqual(data['client_id'], 'client_123')
        self.assertEqual(data['user'], 'researcher_api_test')
    
    def test_start_client_no_auth(self):
        """Test start_client endpoint without authentication."""
        payload = {"model_json": {"framework": "pytorch"}}
        
        response = self.client.post(
            '/api/v1/start-client',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 401)
    
    def test_start_client_missing_model_json(self):
        """Test start_client endpoint without model_json."""
        payload = {
            "server_address": "localhost:8080",
            "client_id": "client_123"
        }
        
        response = self.client.post(
            '/api/v1/start-client',
            data=json.dumps(payload),
            content_type='application/json',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'model_json is required')
    
    def test_start_client_invalid_json(self):
        """Test start_client endpoint with invalid JSON."""
        response = self.client.post(
            '/api/v1/start-client',
            data='invalid json data',
            content_type='application/json',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'Invalid JSON format')
    
    def test_start_client_no_train_permission(self):
        """Test start_client endpoint when user lacks train permission."""
        # Remove train permission
        self.researcher_role.permissions = {
            'api.access': True, 
            'dataset.view': True
            # Removed 'dataset.train': True
        }
        self.researcher_role.save()
        
        payload = {
            "model_json": {"framework": "pytorch"},
            "server_address": "localhost:8080",
            "client_id": "client_123"
        }
        
        response = self.client.post(
            '/api/v1/start-client',
            data=json.dumps(payload),
            content_type='application/json',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 403)
        
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['error'], 'User does not have training permissions')
    
    def test_api_request_logging(self):
        """Test that API requests are properly logged."""
        from users.models import APIRequest
        
        # Make a request
        response = self.client.get(
            '/api/v1/ping',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Check that request was logged
        log_entry = APIRequest.objects.filter(
            api_key=self.api_key,
            endpoint='/api/v1/ping',
            method='GET'
        ).first()
        
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.user, self.researcher_user)
        self.assertEqual(log_entry.status_code, 200)
        self.assertTrue(log_entry.is_successful)
        self.assertEqual(log_entry.ip_address, '127.0.0.1')
    
    def test_api_key_last_used_update(self):
        """Test that API key last_used fields are updated."""
        # Check initial state
        self.assertIsNone(self.api_key.last_used_at)
        self.assertIsNone(self.api_key.last_used_ip)
        
        # Make a request
        response = self.client.get(
            '/api/v1/ping',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Reload API key and check updates
        self.api_key.refresh_from_db()
        self.assertIsNotNone(self.api_key.last_used_at)
        self.assertEqual(self.api_key.last_used_ip, '127.0.0.1')
    
    def test_dataset_format_compatibility(self):
        """Test that dataset format matches client_api.py expectations."""
        response = self.client.get(
            '/api/v1/get-data-info',
            **self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content.decode('utf-8'))
        
        # Verify structure matches client_api.py format
        self.assertIsInstance(data['dataset_id'], list)
        self.assertIsInstance(data['dataset_name'], list)
        self.assertIsInstance(data['medical_domain'], list)
        self.assertIsInstance(data['patient_count'], list)
        self.assertIsInstance(data['data_type'], list)
        self.assertIsInstance(data['file_size'], list)
        self.assertIsInstance(data['description'], list)
        self.assertIsInstance(data['created_at'], list)
        
        # Verify all lists have same length
        lengths = [len(data[key]) for key in data.keys()]
        self.assertTrue(all(length == lengths[0] for length in lengths))
        
        # Verify data types
        if data['dataset_id']:  # If we have data
            self.assertIsInstance(data['dataset_id'][0], int)
            self.assertIsInstance(data['dataset_name'][0], str)
            self.assertIsInstance(data['medical_domain'][0], str)
            self.assertIsInstance(data['patient_count'][0], int)
            self.assertIsInstance(data['data_type'][0], str)
            self.assertIsInstance(data['file_size'][0], int)
            self.assertIsInstance(data['description'][0], str)
            self.assertIsInstance(data['created_at'][0], str)


class APIErrorHandlingTests(TestCase):
    """Test API error handling scenarios."""
    
    def test_invalid_endpoint_returns_401(self):
        """Test that invalid endpoints return 401 (auth required first)."""
        # API middleware checks auth first, so invalid endpoints return 401 without auth
        response = self.client.get('/api/v1/invalid-endpoint')
        self.assertEqual(response.status_code, 401)
    
    def test_wrong_http_method(self):
        """Test endpoints with wrong HTTP methods."""
        # API middleware checks authentication first, so wrong methods return 401 without auth
        # This is actually correct behavior for a secure API
        response = self.client.post('/api/v1/ping')
        self.assertEqual(response.status_code, 401)  # Unauthorized (auth required first)
        
        response = self.client.get('/api/v1/start-client')
        self.assertEqual(response.status_code, 401)  # Unauthorized (auth required first)