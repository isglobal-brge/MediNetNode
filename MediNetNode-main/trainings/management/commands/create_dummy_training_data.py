
import uuid
import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from trainings.models import TrainingSession, TrainingRound

User = get_user_model()

"""
Management command to create dummy training data for testing purposes.
"""

class Command(BaseCommand):
    help = 'Create dummy training data for testing the training monitoring system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--sessions',
            type=int,
            default=20,
            help='Number of training sessions to create (default: 20)'
        )
        parser.add_argument(
            '--active',
            type=int,
            default=3,
            help='Number of active sessions to create (default: 3)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing training data before creating new data'
        )
    
    def handle(self, *args, **options):
        if options['clear']:
            self.stdout.write('Clearing existing training data...')
            TrainingRound.objects.all().delete()
            TrainingSession.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('[OK] Cleared existing data'))
        
        # Get or create test users
        users = self.get_or_create_test_users()
        
        # Create training sessions
        sessions_count = options['sessions']
        active_count = options['active']
        
        self.stdout.write(f'Creating {sessions_count} training sessions ({active_count} active)...')
        
        # Create completed/failed sessions first
        completed_count = sessions_count - active_count
        for i in range(completed_count):
            session = self.create_completed_session(users, i)
            self.create_training_rounds(session)
        
        # Create active sessions
        for i in range(active_count):
            session = self.create_active_session(users, completed_count + i)
            self.create_training_rounds(session, partial=True)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'[OK] Successfully created {sessions_count} training sessions '
                f'({active_count} active, {completed_count} completed)'
            )
        )
    
    def get_or_create_test_users(self):
        """Get or create test users for training sessions."""
        users = []
        test_usernames = ['researcher1', 'researcher2', 'investigador1', 'admin']
        
        for username in test_usernames:
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    'email': f'{username}@medinet.com',
                    'is_staff': username == 'admin',
                    'is_superuser': username == 'admin'
                }
            )
            if created:
                user.set_password('password123')
                user.save()
                self.stdout.write(f'Created test user: {username}')
            users.append(user)
        
        return users
    
    def create_completed_session(self, users, index):
        """Create a completed training session with realistic data."""
        user = random.choice(users)
        
        # Random dataset info
        datasets = [
            {'id': 1, 'name': 'Heart Disease Dataset'},
            {'id': 2, 'name': 'Diabetes Prediction Dataset'},
            {'id': 3, 'name': 'Cancer Detection Dataset'},
            {'id': 4, 'name': 'COVID-19 Analysis Dataset'},
            {'id': 5, 'name': 'Mental Health Dataset'},
        ]
        dataset = random.choice(datasets)
        
        # Random timing (completed in last 30 days)
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        started_at = timezone.now() - timedelta(days=days_ago, hours=hours_ago)
        
        # Random duration (30 minutes to 4 hours)
        duration_minutes = random.randint(30, 240)
        completed_at = started_at + timedelta(minutes=duration_minutes)
        
        # Random status (mostly completed, some failed)
        status = random.choices(
            ['COMPLETED', 'FAILED', 'CANCELLED'],
            weights=[70, 20, 10],
            k=1
        )[0]
        
        # Model configuration
        model_config = {
            'model': {
                'type': 'neural_network',
                'layers': [random.randint(32, 256), random.randint(16, 128), random.randint(8, 64), 1],
                'activation': random.choice(['relu', 'tanh', 'sigmoid']),
                'dataset': {
                    'selected_datasets': [{
                        'dataset_id': dataset['id'],
                        'dataset_name': dataset['name']
                    }]
                },
                'training': {
                    'epochs': random.randint(5, 20),
                    'learning_rate': random.uniform(0.001, 0.1),
                    'batch_size': random.choice([16, 32, 64, 128]),
                    'optimizer': {
                        'type': random.choice(['Adam', 'SGD', 'RMSprop']),
                        'learning_rate': random.uniform(0.001, 0.1)
                    }
                }
            }
        }
        
        total_rounds = random.randint(5, 15)
        current_round = total_rounds if status == 'COMPLETED' else random.randint(1, total_rounds - 1)
        progress = (current_round / total_rounds) * 100 if status == 'COMPLETED' else random.uniform(10, 90)
        
        session = TrainingSession.objects.create(
            client_id=f'client_{index:03d}',
            user=user,
            dataset_id=dataset['id'],
            dataset_name=dataset['name'],
            model_config=model_config,
            server_address='localhost:8080',
            status=status,
            started_at=started_at,
            completed_at=completed_at if status in ['COMPLETED', 'FAILED', 'CANCELLED'] else None,
            current_round=current_round,
            total_rounds=total_rounds,
            progress_percentage=progress,
            cpu_usage=random.uniform(20, 90),
            memory_usage=random.uniform(100, 2048),
            process_id=random.randint(1000, 9999),
            # Final metrics for completed sessions
            final_accuracy=random.uniform(0.7, 0.98) if status == 'COMPLETED' else None,
            final_loss=random.uniform(0.02, 0.3) if status == 'COMPLETED' else None,
            final_precision=random.uniform(0.65, 0.95) if status == 'COMPLETED' else None,
            final_recall=random.uniform(0.7, 0.96) if status == 'COMPLETED' else None,
            final_f1=random.uniform(0.68, 0.94) if status == 'COMPLETED' else None,
            # Error info for failed sessions
            error_message='Training failed due to convergence issues' if status == 'FAILED' else None
        )
        
        return session
    
    def create_active_session(self, users, index):
        """Create an active training session."""
        user = random.choice(users)
        
        datasets = [
            {'id': 1, 'name': 'Heart Disease Dataset'},
            {'id': 2, 'name': 'Diabetes Prediction Dataset'},
            {'id': 3, 'name': 'Cancer Detection Dataset'},
        ]
        dataset = random.choice(datasets)
        
        # Started recently (last 2 hours)
        minutes_ago = random.randint(10, 120)
        started_at = timezone.now() - timedelta(minutes=minutes_ago)
        
        model_config = {
            'model': {
                'type': 'neural_network',
                'layers': [random.randint(64, 256), random.randint(32, 128), random.randint(16, 64), 1],
                'activation': 'relu',
                'dataset': {
                    'selected_datasets': [{
                        'dataset_id': dataset['id'],
                        'dataset_name': dataset['name']
                    }]
                },
                'training': {
                    'epochs': random.randint(10, 25),
                    'learning_rate': random.uniform(0.01, 0.1),
                    'batch_size': random.choice([32, 64, 128]),
                    'optimizer': {
                        'type': 'Adam',
                        'learning_rate': 0.01
                    }
                }
            }
        }
        
        total_rounds = random.randint(10, 20)
        current_round = random.randint(2, int(total_rounds * 0.7))  # Partially completed
        progress = (current_round / total_rounds) * 100
        
        status = random.choice(['STARTING', 'ACTIVE']) if random.random() > 0.2 else 'ACTIVE'
        
        session = TrainingSession.objects.create(
            client_id=f'active_client_{index:03d}',
            user=user,
            dataset_id=dataset['id'],
            dataset_name=dataset['name'],
            model_config=model_config,
            server_address='localhost:8080',
            status=status,
            started_at=started_at,
            current_round=current_round,
            total_rounds=total_rounds,
            progress_percentage=progress,
            cpu_usage=random.uniform(40, 95),
            memory_usage=random.uniform(200, 1536),
            process_id=random.randint(1000, 9999)
        )
        
        return session
    
    def create_training_rounds(self, session, partial=False):
        """Create training rounds for a session."""
        rounds_to_create = session.current_round
        
        if partial and session.status == 'ACTIVE':
            # For active sessions, create slightly fewer rounds
            rounds_to_create = max(1, session.current_round - 1)
        
        base_accuracy = random.uniform(0.5, 0.7)  # Starting accuracy
        base_loss = random.uniform(0.8, 1.2)      # Starting loss
        
        for round_num in range(1, rounds_to_create + 1):
            # Simulate improving metrics over rounds
            improvement_factor = round_num / session.total_rounds
            
            accuracy = base_accuracy + (random.uniform(0.2, 0.4) * improvement_factor)
            loss = base_loss - (random.uniform(0.3, 0.7) * improvement_factor)
            precision = accuracy + random.uniform(-0.05, 0.05)
            recall = accuracy + random.uniform(-0.08, 0.08)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Ensure realistic bounds
            accuracy = max(0.1, min(0.99, accuracy))
            loss = max(0.01, min(2.0, loss))
            precision = max(0.1, min(0.99, precision))
            recall = max(0.1, min(0.99, recall))
            f1 = max(0.1, min(0.99, f1))
            
            # Random round timing
            round_start = session.started_at + timedelta(minutes=random.randint(1, 10) * round_num)
            round_duration = random.randint(30, 180)  # 30s to 3min per round
            round_end = round_start + timedelta(seconds=round_duration)
            
            training_round = TrainingRound.objects.create(
                session=session,
                round_number=round_num,
                started_at=round_start,
                completed_at=round_end,
                loss=loss,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cpu_usage=random.uniform(30, 100),
                memory_usage=random.uniform(150, 1200),
                metrics={
                    'batch_size': random.choice([16, 32, 64]),
                    'learning_rate': random.uniform(0.001, 0.1),
                    'validation_accuracy': accuracy + random.uniform(-0.02, 0.02)
                }
            )
            
        self.stdout.write(f'Created {rounds_to_create} rounds for session {session.client_id}')