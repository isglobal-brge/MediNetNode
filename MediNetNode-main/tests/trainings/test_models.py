"""
Unit tests for training models.
Tests TrainingSession and TrainingRound functionality.
"""
import uuid
from decimal import Decimal
from datetime import datetime, timedelta
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.exceptions import ValidationError
from trainings.models import TrainingSession, TrainingRound

User = get_user_model()


class TrainingSessionModelTests(TestCase):
    """Test cases for TrainingSession model."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            model_config={
                'model': {
                    'type': 'neural_network',
                    'layers': [128, 64, 32, 1]
                },
                'training': {
                    'epochs': 10,
                    'learning_rate': 0.01
                }
            },
            server_address='localhost:8080',
            total_rounds=10
        )
    
    def test_session_creation(self):
        """Test basic training session creation."""
        session = self.training_session
        
        # Test basic fields
        self.assertEqual(session.client_id, 'test_client_001')
        self.assertEqual(session.user, self.user)
        self.assertEqual(session.dataset_id, 1)
        self.assertEqual(session.dataset_name, 'Test Dataset')
        self.assertEqual(session.server_address, 'localhost:8080')
        self.assertEqual(session.total_rounds, 10)
        
        # Test defaults
        self.assertEqual(session.status, 'STARTING')
        self.assertEqual(session.current_round, 0)
        self.assertEqual(session.progress_percentage, 0.0)
        
        # Test UUID generation
        self.assertIsInstance(session.session_id, uuid.UUID)
        
        # Test timestamps
        self.assertIsNotNone(session.started_at)
        self.assertIsNone(session.completed_at)
    
    def test_str_representation(self):
        """Test string representation of TrainingSession."""
        expected = f"Training {self.training_session.session_id} - {self.user.username} - STARTING"
        self.assertEqual(str(self.training_session), expected)
    
    def test_duration_property_completed(self):
        """Test duration property with completed session."""
        # Set completion time
        self.training_session.completed_at = timezone.now()
        self.training_session.save()
        
        duration = self.training_session.duration
        self.assertIsNotNone(duration)
        self.assertIsInstance(duration, timedelta)
    
    def test_duration_property_active(self):
        """Test duration property with active session."""
        duration = self.training_session.duration
        self.assertIsNotNone(duration)
        self.assertIsInstance(duration, timedelta)
    
    def test_duration_property_no_start(self):
        """Test duration property with no start time."""
        session = TrainingSession(
            client_id='test',
            user=self.user,
            dataset_id=1,
            dataset_name='Test'
        )
        # Don't save to avoid auto_now_add on started_at
        session.started_at = None
        self.assertIsNone(session.duration)
    
    def test_is_active_property(self):
        """Test is_active property."""
        # Test STARTING status
        self.training_session.status = 'STARTING'
        self.assertTrue(self.training_session.is_active)
        
        # Test ACTIVE status
        self.training_session.status = 'ACTIVE'
        self.assertTrue(self.training_session.is_active)
        
        # Test COMPLETED status
        self.training_session.status = 'COMPLETED'
        self.assertFalse(self.training_session.is_active)
        
        # Test FAILED status
        self.training_session.status = 'FAILED'
        self.assertFalse(self.training_session.is_active)
        
        # Test CANCELLED status
        self.training_session.status = 'CANCELLED'
        self.assertFalse(self.training_session.is_active)
    
    def test_is_finished_property(self):
        """Test is_finished property."""
        # Test STARTING status
        self.training_session.status = 'STARTING'
        self.assertFalse(self.training_session.is_finished)
        
        # Test ACTIVE status
        self.training_session.status = 'ACTIVE'
        self.assertFalse(self.training_session.is_finished)
        
        # Test COMPLETED status
        self.training_session.status = 'COMPLETED'
        self.assertTrue(self.training_session.is_finished)
        
        # Test FAILED status
        self.training_session.status = 'FAILED'
        self.assertTrue(self.training_session.is_finished)
        
        # Test CANCELLED status
        self.training_session.status = 'CANCELLED'
        self.assertTrue(self.training_session.is_finished)
    
    def test_update_progress(self):
        """Test update_progress method."""
        self.training_session.update_progress(5)
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.current_round, 5)
        self.assertEqual(self.training_session.progress_percentage, 50.0)  # 5/10 * 100
    
    def test_update_progress_with_new_total(self):
        """Test update_progress with new total rounds."""
        self.training_session.update_progress(3, total_rounds=15)
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.current_round, 3)
        self.assertEqual(self.training_session.total_rounds, 15)
        self.assertEqual(self.training_session.progress_percentage, 20.0)  # 3/15 * 100
    
    def test_mark_completed(self):
        """Test mark_completed method."""
        metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'precision': 0.94,
            'recall': 0.96,
            'f1': 0.95
        }
        
        self.training_session.mark_completed(**metrics)
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.status, 'COMPLETED')
        self.assertIsNotNone(self.training_session.completed_at)
        self.assertEqual(self.training_session.progress_percentage, 100.0)
        self.assertEqual(float(self.training_session.final_accuracy), 0.95)
        self.assertEqual(float(self.training_session.final_loss), 0.05)
        self.assertEqual(float(self.training_session.final_precision), 0.94)
        self.assertEqual(float(self.training_session.final_recall), 0.96)
        self.assertEqual(float(self.training_session.final_f1), 0.95)
    
    def test_mark_completed_partial_metrics(self):
        """Test mark_completed with partial metrics."""
        self.training_session.mark_completed(accuracy=0.85, loss=0.15)
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.status, 'COMPLETED')
        self.assertEqual(float(self.training_session.final_accuracy), 0.85)
        self.assertEqual(float(self.training_session.final_loss), 0.15)
        self.assertIsNone(self.training_session.final_precision)
    
    def test_mark_failed(self):
        """Test mark_failed method."""
        error_msg = "Training failed due to data corruption"
        traceback_msg = "Traceback (most recent call last):\n  File..."
        
        self.training_session.mark_failed(error_msg, traceback_msg)
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.status, 'FAILED')
        self.assertIsNotNone(self.training_session.completed_at)
        self.assertEqual(self.training_session.error_message, error_msg)
        self.assertEqual(self.training_session.error_traceback, traceback_msg)
    
    def test_cancel_training_active(self):
        """Test cancel_training method on active session."""
        self.training_session.status = 'ACTIVE'
        self.training_session.save()
        
        result = self.training_session.cancel_training()
        self.training_session.refresh_from_db()
        
        self.assertTrue(result)
        self.assertEqual(self.training_session.status, 'CANCELLED')
        self.assertIsNotNone(self.training_session.completed_at)
    
    def test_cancel_training_completed(self):
        """Test cancel_training method on completed session."""
        self.training_session.status = 'COMPLETED'
        self.training_session.save()
        
        result = self.training_session.cancel_training()
        
        self.assertFalse(result)
        self.assertEqual(self.training_session.status, 'COMPLETED')
    
    def test_validation_constraints(self):
        """Test model validation constraints."""
        # Test negative total_rounds
        with self.assertRaises(ValidationError):
            session = TrainingSession(
                client_id='test',
                user=self.user,
                dataset_id=1,
                dataset_name='Test',
                total_rounds=-1
            )
            session.full_clean()
        
        # Test total_rounds over maximum
        with self.assertRaises(ValidationError):
            session = TrainingSession(
                client_id='test',
                user=self.user,
                dataset_id=1,
                dataset_name='Test',
                total_rounds=1001
            )
            session.full_clean()
        
        # Test progress_percentage over 100
        with self.assertRaises(ValidationError):
            session = TrainingSession(
                client_id='test',
                user=self.user,
                dataset_id=1,
                dataset_name='Test',
                progress_percentage=101.0
            )
            session.full_clean()


class TrainingRoundModelTests(TestCase):
    """Test cases for TrainingRound model."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.training_session = TrainingSession.objects.create(
            client_id='test_client_001',
            user=self.user,
            dataset_id=1,
            dataset_name='Test Dataset',
            total_rounds=5
        )
        
        self.training_round = TrainingRound.objects.create(
            session=self.training_session,
            round_number=1,
            loss=0.5,
            accuracy=0.8,
            precision=0.75,
            recall=0.85,
            f1_score=0.79
        )
    
    def test_round_creation(self):
        """Test basic training round creation."""
        round_obj = self.training_round
        
        self.assertEqual(round_obj.session, self.training_session)
        self.assertEqual(round_obj.round_number, 1)
        self.assertEqual(float(round_obj.loss), 0.5)
        self.assertEqual(float(round_obj.accuracy), 0.8)
        self.assertEqual(float(round_obj.precision), 0.75)
        self.assertEqual(float(round_obj.recall), 0.85)
        self.assertEqual(float(round_obj.f1_score), 0.79)
        
        # Test timestamps
        self.assertIsNotNone(round_obj.started_at)
        self.assertIsNone(round_obj.completed_at)
    
    def test_str_representation(self):
        """Test string representation of TrainingRound."""
        expected = f"Round 1 - Session {self.training_session.session_id}"
        self.assertEqual(str(self.training_round), expected)
    
    def test_duration_property(self):
        """Test duration property."""
        # Not completed yet
        self.assertIsNone(self.training_round.duration)
        
        # Complete the round
        self.training_round.completed_at = timezone.now()
        duration = self.training_round.duration
        
        self.assertIsNotNone(duration)
        self.assertIsInstance(duration, timedelta)
    
    def test_is_completed_property(self):
        """Test is_completed property."""
        # Initially not completed
        self.assertFalse(self.training_round.is_completed)
        
        # Complete the round
        self.training_round.completed_at = timezone.now()
        self.assertTrue(self.training_round.is_completed)
    
    def test_complete_round(self):
        """Test complete_round method."""
        metrics = {
            'loss': 0.3,
            'accuracy': 0.9,
            'precision': 0.88,
            'recall': 0.92,
            'f1_score': 0.90,
            'custom_metric': 0.85
        }
        
        self.training_round.complete_round(**metrics)
        self.training_round.refresh_from_db()
        
        # Check completion timestamp
        self.assertIsNotNone(self.training_round.completed_at)
        
        # Check updated metrics
        self.assertEqual(float(self.training_round.loss), 0.3)
        self.assertEqual(float(self.training_round.accuracy), 0.9)
        self.assertEqual(float(self.training_round.precision), 0.88)
        self.assertEqual(float(self.training_round.recall), 0.92)
        self.assertEqual(float(self.training_round.f1_score), 0.90)
        
        # Check additional metrics stored in JSON
        self.assertIn('custom_metric', self.training_round.metrics)
        self.assertEqual(self.training_round.metrics['custom_metric'], 0.85)
        
        # Check that parent session progress was updated
        self.training_session.refresh_from_db()
        self.assertEqual(self.training_session.current_round, 1)
        self.assertEqual(self.training_session.progress_percentage, 20.0)  # 1/5 * 100
    
    def test_complete_round_partial_metrics(self):
        """Test complete_round with partial metrics."""
        self.training_round.complete_round(loss=0.25, accuracy=0.92)
        self.training_round.refresh_from_db()
        
        self.assertEqual(float(self.training_round.loss), 0.25)
        self.assertEqual(float(self.training_round.accuracy), 0.92)
        # Original values should remain
        self.assertEqual(float(self.training_round.precision), 0.75)
    
    def test_unique_constraint(self):
        """Test unique constraint on session + round_number."""
        with self.assertRaises(Exception):  # IntegrityError wrapped by Django
            TrainingRound.objects.create(
                session=self.training_session,
                round_number=1,  # Same round number
                loss=0.4
            )
    
    def test_round_number_validation(self):
        """Test round_number validation."""
        with self.assertRaises(ValidationError):
            round_obj = TrainingRound(
                session=self.training_session,
                round_number=0  # Should be >= 1
            )
            round_obj.full_clean()
    
    def test_ordering(self):
        """Test model ordering."""
        # Create additional rounds
        round2 = TrainingRound.objects.create(
            session=self.training_session,
            round_number=2,
            loss=0.4
        )
        round3 = TrainingRound.objects.create(
            session=self.training_session,
            round_number=3,
            loss=0.3
        )
        
        # Check ordering (session, round_number)
        rounds = list(TrainingRound.objects.all())
        self.assertEqual(rounds[0], self.training_round)  # round 1
        self.assertEqual(rounds[1], round2)  # round 2
        self.assertEqual(rounds[2], round3)  # round 3
    
    def test_cascade_delete(self):
        """Test cascade delete when session is deleted."""
        round_id = self.training_round.id
        
        # Delete the session
        self.training_session.delete()
        
        # Round should be deleted too
        self.assertFalse(TrainingRound.objects.filter(id=round_id).exists())


class TrainingModelIntegrationTests(TestCase):
    """Integration tests for training models working together."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.training_session = TrainingSession.objects.create(
            client_id='integration_test',
            user=self.user,
            dataset_id=1,
            dataset_name='Integration Test Dataset',
            total_rounds=3
        )
    
    def test_full_training_workflow(self):
        """Test complete training workflow."""
        # Start training
        self.assertEqual(self.training_session.status, 'STARTING')
        self.assertEqual(self.training_session.current_round, 0)
        
        # Simulate 3 training rounds
        for round_num in range(1, 4):
            # Create and complete round
            training_round = TrainingRound.objects.create(
                session=self.training_session,
                round_number=round_num,
                loss=0.8 - (round_num * 0.2),  # Decreasing loss
                accuracy=0.6 + (round_num * 0.1)  # Increasing accuracy
            )
            
            # Complete the round
            training_round.complete_round(
                loss=training_round.loss,
                accuracy=training_round.accuracy
            )
            
            # Check progress updates
            self.training_session.refresh_from_db()
            self.assertEqual(self.training_session.current_round, round_num)
            expected_progress = (round_num / 3) * 100
            self.assertEqual(self.training_session.progress_percentage, expected_progress)
        
        # Complete training
        final_metrics = {
            'accuracy': 0.9,
            'loss': 0.2,
            'precision': 0.88,
            'recall': 0.92,
            'f1': 0.90
        }
        self.training_session.mark_completed(**final_metrics)
        
        # Verify final state
        self.training_session.refresh_from_db()
        self.assertEqual(self.training_session.status, 'COMPLETED')
        self.assertEqual(self.training_session.progress_percentage, 100.0)
        self.assertEqual(float(self.training_session.final_accuracy), 0.9)
        
        # Verify rounds were created
        self.assertEqual(self.training_session.rounds.count(), 3)
        
        # Verify round metrics
        first_round = self.training_session.rounds.get(round_number=1)
        self.assertAlmostEqual(float(first_round.loss), 0.6, places=7)
        self.assertAlmostEqual(float(first_round.accuracy), 0.7, places=7)
    
    def test_training_failure_workflow(self):
        """Test training failure workflow."""
        # Start some rounds
        round1 = TrainingRound.objects.create(
            session=self.training_session,
            round_number=1,
            loss=0.6,
            accuracy=0.7
        )
        round1.complete_round(loss=0.6, accuracy=0.7)
        
        # Simulate failure during round 2
        error_msg = "Out of memory error during training"
        traceback_msg = "Traceback...\nRuntimeError: CUDA out of memory"
        
        self.training_session.mark_failed(error_msg, traceback_msg)
        
        # Verify failure state
        self.training_session.refresh_from_db()
        self.assertEqual(self.training_session.status, 'FAILED')
        self.assertIsNotNone(self.training_session.completed_at)
        self.assertEqual(self.training_session.error_message, error_msg)
        self.assertEqual(self.training_session.error_traceback, traceback_msg)
        
        # Verify partial rounds still exist
        self.assertEqual(self.training_session.rounds.count(), 1)