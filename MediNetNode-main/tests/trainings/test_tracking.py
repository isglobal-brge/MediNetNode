"""
Unit tests for training tracking functionality.
Tests integration with torch_client.py tracking functions.
"""
import uuid
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth import get_user_model
from trainings.models import TrainingSession, TrainingRound

User = get_user_model()


class TrainingTrackingTests(TestCase):
    """Test cases for training tracking functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.model_config = {
            'model': {
                'type': 'neural_network',
                'layers': [128, 64, 32, 1],
                'dataset': {
                    'selected_datasets': [{
                        'dataset_id': 1,
                        'dataset_name': 'Test Dataset'
                    }]
                },
                'training': {
                    'epochs': 5,
                    'learning_rate': 0.01
                }
            }
        }
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    @patch('api.federated.torch_client.psutil.Process')
    def test_create_training_session(self, mock_process):
        """Test create_training_session function."""
        from api.federated.torch_client import create_training_session
        
        # Mock psutil Process
        mock_process_instance = MagicMock()
        mock_process_instance.pid = 12345
        mock_process.return_value = mock_process_instance
        
        # Create training session
        session = create_training_session(
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            model_config=self.model_config,
            server_address='localhost:8080',
            client_id='test_client_001'
        )
        
        # Verify session was created
        self.assertIsNotNone(session)
        self.assertIsInstance(session, TrainingSession)
        self.assertEqual(session.user, self.user)
        self.assertEqual(session.dataset_id, 1)
        self.assertEqual(session.dataset_name, 'Test Dataset')
        self.assertEqual(session.client_id, 'test_client_001')
        self.assertEqual(session.server_address, 'localhost:8080')
        self.assertEqual(session.status, 'STARTING')
        self.assertEqual(session.process_id, 12345)
        self.assertEqual(session.model_config, self.model_config)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', False)
    def test_create_training_session_no_django(self):
        """Test create_training_session when Django unavailable."""
        from api.federated.torch_client import create_training_session
        
        session = create_training_session(
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            model_config=self.model_config,
            server_address='localhost:8080',
            client_id='test_client_001'
        )
        
        # Should return None when Django unavailable
        self.assertIsNone(session)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    @patch('api.federated.torch_client.TRAINING_SESSION')
    @patch('api.federated.torch_client.CURRENT_PROCESS')
    def test_update_training_progress(self, mock_process, mock_session):
        """Test update_training_progress function."""
        from api.federated.torch_client import update_training_progress
        
        # Create real training session
        training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            total_rounds=10
        )
        
        # Mock global variables
        mock_session_instance = MagicMock()
        mock_session_instance.session_id = training_session.session_id
        mock_session_instance.total_rounds = 10
        mock_session.return_value = training_session
        
        mock_process_instance = MagicMock()
        mock_process_instance.cpu_percent.return_value = 45.2
        mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024  # 512 MB
        mock_process.return_value = mock_process_instance
        
        # Patch the global variables
        with patch('api.federated.torch_client.TRAINING_SESSION', training_session), \
             patch('api.federated.torch_client.CURRENT_PROCESS', mock_process_instance):
            
            metrics = {
                'loss': 0.4,
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1': 0.85
            }
            
            update_training_progress(5, metrics)
        
        # Verify session was updated
        training_session.refresh_from_db()
        self.assertEqual(training_session.current_round, 5)
        self.assertEqual(training_session.status, 'ACTIVE')
        self.assertEqual(training_session.progress_percentage, 50.0)
        self.assertEqual(float(training_session.cpu_usage), 45.2)
        self.assertEqual(float(training_session.memory_usage), 512.0)
        
        # Verify round was created
        training_round = TrainingRound.objects.get(
            session=training_session,
            round_number=5
        )
        self.assertEqual(float(training_round.loss), 0.4)
        self.assertEqual(float(training_round.accuracy), 0.85)
        self.assertEqual(float(training_round.precision), 0.82)
        self.assertEqual(float(training_round.recall), 0.88)
        self.assertEqual(float(training_round.f1_score), 0.85)
        self.assertIsNotNone(training_round.completed_at)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    @patch('api.federated.torch_client.TRAINING_SESSION')
    def test_complete_training_session(self, mock_session):
        """Test complete_training_session function."""
        from api.federated.torch_client import complete_training_session
        
        # Create real training session
        training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            total_rounds=10,
            status='ACTIVE'
        )
        
        with patch('api.federated.torch_client.TRAINING_SESSION', training_session):
            final_metrics = {
                'accuracy': 0.92,
                'loss': 0.08,
                'precision': 0.90,
                'recall': 0.94,
                'f1': 0.92
            }
            
            complete_training_session(final_metrics)
        
        # Verify session was completed
        training_session.refresh_from_db()
        self.assertEqual(training_session.status, 'COMPLETED')
        self.assertIsNotNone(training_session.completed_at)
        self.assertEqual(training_session.progress_percentage, 100.0)
        self.assertEqual(float(training_session.final_accuracy), 0.92)
        self.assertEqual(float(training_session.final_loss), 0.08)
        self.assertEqual(float(training_session.final_precision), 0.90)
        self.assertEqual(float(training_session.final_recall), 0.94)
        self.assertEqual(float(training_session.final_f1), 0.92)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    @patch('api.federated.torch_client.TRAINING_SESSION')
    def test_fail_training_session(self, mock_session):
        """Test fail_training_session function."""
        from api.federated.torch_client import fail_training_session
        
        # Create real training session
        training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            total_rounds=10,
            status='ACTIVE'
        )
        
        with patch('api.federated.torch_client.TRAINING_SESSION', training_session):
            error_message = "CUDA out of memory"
            traceback = "Traceback (most recent call last):\n  RuntimeError: CUDA out of memory"
            
            fail_training_session(error_message, traceback)
        
        # Verify session was failed
        training_session.refresh_from_db()
        self.assertEqual(training_session.status, 'FAILED')
        self.assertIsNotNone(training_session.completed_at)
        self.assertEqual(training_session.error_message, error_message)
        self.assertEqual(training_session.error_traceback, traceback)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    def test_start_flower_client_tracking(self):
        """Test training tracking in start_flower_client function."""
        from api.federated.torch_client import start_flower_client
        
        # Mock the actual flower client starting
        with patch('api.federated.torch_client.start_client') as mock_start_client, \
             patch('api.federated.torch_client.complete_training_session') as mock_complete:
            
            # Mock successful completion
            mock_start_client.return_value = None
            
            # Call start_flower_client with user context
            start_flower_client(
                model_json=self.model_config,
                server_address='localhost:8080',
                client_id='test_client_001',
                user=self.user
            )
            
            # Verify training session was created
            training_session = TrainingSession.objects.get(client_id='test_client_001')
            self.assertEqual(training_session.user, self.user)
            self.assertEqual(training_session.dataset_id, 1)
            self.assertEqual(training_session.dataset_name, 'Test Dataset')
            
            # Verify completion was called
            mock_complete.assert_called_once()
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    def test_start_flower_client_failure(self):
        """Test training tracking when start_flower_client fails."""
        from api.federated.torch_client import start_flower_client
        
        # Mock the flower client failure
        with patch('api.federated.torch_client.start_client') as mock_start_client, \
             patch('api.federated.torch_client.fail_training_session') as mock_fail:
            
            # Mock failure
            mock_start_client.side_effect = RuntimeError("Connection failed")
            
            # Call start_flower_client with user context
            with self.assertRaises(RuntimeError):
                start_flower_client(
                    model_json=self.model_config,
                    server_address='localhost:8080',
                    client_id='test_client_001',
                    user=self.user
                )
            
            # Verify training session was created
            training_session = TrainingSession.objects.get(client_id='test_client_001')
            self.assertEqual(training_session.user, self.user)
            
            # Verify failure was called
            mock_fail.assert_called_once()
    
    def test_start_flower_client_no_user(self):
        """Test start_flower_client without user context."""
        from api.federated.torch_client import start_flower_client
        
        # Reset global training session to ensure clean state
        with patch('api.federated.torch_client.TRAINING_SESSION', None), \
             patch('api.federated.torch_client.start_client') as mock_start_client:
            mock_start_client.return_value = None
            
            # Call without user
            start_flower_client(
                model_json=self.model_config,
                server_address='localhost:8080',
                client_id='test_client_001',
                user=None
            )
            
            # Verify no training session was created
            self.assertEqual(TrainingSession.objects.count(), 0)
    
    @patch('api.federated.torch_client.DJANGO_AVAILABLE', True)
    def test_round_counter_tracking(self):
        """Test round counter tracking in FlowerClient.fit method."""
        # This would require more complex mocking of the FlowerClient class
        # For now, we test the basic logic
        
        from api.federated.torch_client import update_training_progress
        
        # Create training session
        training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            total_rounds=5
        )
        
        # Mock process for resource monitoring
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 45.0
        mock_process.memory_info.return_value.rss = 256 * 1024 * 1024  # 256 MB
        
        with patch('api.federated.torch_client.TRAINING_SESSION', training_session), \
             patch('api.federated.torch_client.CURRENT_PROCESS', mock_process):
            # Simulate multiple rounds
            for round_num in range(1, 6):
                metrics = {
                    'loss': 1.0 - (round_num * 0.15),  # Decreasing loss
                    'accuracy': 0.5 + (round_num * 0.08),  # Increasing accuracy
                    'f1': 0.4 + (round_num * 0.1)
                }
                
                update_training_progress(round_num, metrics)
            
            # Verify all rounds were created
            self.assertEqual(training_session.rounds.count(), 5)
            
            # Verify progress tracking
            training_session.refresh_from_db()
            self.assertEqual(training_session.current_round, 5)
            self.assertEqual(training_session.progress_percentage, 100.0)
            
            # Verify final round metrics
            final_round = training_session.rounds.get(round_number=5)
            self.assertAlmostEqual(float(final_round.loss), 0.25, places=7)
            self.assertAlmostEqual(float(final_round.accuracy), 0.9, places=7)
            self.assertAlmostEqual(float(final_round.f1_score), 0.9, places=7)