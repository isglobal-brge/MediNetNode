import json
import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

User = get_user_model()

"""
Training monitoring models.
All views are restricted to ADMIN and AUDITOR users only.
RESEARCHER users have NO ACCESS to web interfaces.
"""

class TrainingSession(models.Model):
    """Track federated learning training sessions."""
    
    STATUS_CHOICES = [
        ('STARTING', 'Starting'),
        ('ACTIVE', 'Training Active'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled'),
    ]
    
    # Identification
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client_id = models.CharField(max_length=100, null=True, blank=True, help_text="Flower client ID")
    
    # User & Dataset (cross-database reference)
    user = models.ForeignKey(User, on_delete=models.CASCADE, help_text="User who initiated training")
    dataset_id = models.IntegerField(help_text="Dataset ID from datasets_db")
    dataset_name = models.CharField(max_length=200, help_text="Dataset name for display")
    
    # Training Configuration
    model_config = models.JSONField(default=dict, help_text="Complete model configuration JSON")
    server_address = models.CharField(max_length=200, default="localhost:8080")
    total_rounds = models.PositiveIntegerField(default=10, validators=[MinValueValidator(1), MaxValueValidator(1000)])
    
    # Status Tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='STARTING')
    
    # Timing
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    estimated_duration = models.DurationField(null=True, blank=True)
    
    # Progress
    current_round = models.PositiveIntegerField(default=0)
    progress_percentage = models.FloatField(
        default=0.0, 
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    
    # Resources
    cpu_usage = models.FloatField(null=True, blank=True, help_text="CPU usage percentage")
    memory_usage = models.FloatField(null=True, blank=True, help_text="Memory usage in MB")
    process_id = models.PositiveIntegerField(null=True, blank=True, help_text="Training process PID")
    
    # Training Results
    final_accuracy = models.FloatField(null=True, blank=True)
    final_loss = models.FloatField(null=True, blank=True)
    final_precision = models.FloatField(null=True, blank=True)
    final_recall = models.FloatField(null=True, blank=True)
    final_f1 = models.FloatField(null=True, blank=True)
    
    # Error tracking
    error_message = models.TextField(null=True, blank=True)
    error_traceback = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['started_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"Training {self.session_id} - {self.user.username} - {self.status}"
    
    @property
    def duration(self):
        """Calculate training duration."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return timezone.now() - self.started_at
        return None
    
    @property
    def is_active(self):
        """Check if training is currently active."""
        return self.status in ['STARTING', 'ACTIVE']
    
    @property
    def is_finished(self):
        """Check if training is finished (completed, failed, or cancelled)."""
        return self.status in ['COMPLETED', 'FAILED', 'CANCELLED']
    
    @property
    def formatted_model_config(self):
        """Return pretty formatted JSON for model configuration."""
        if not self.model_config:
            return "No configuration available"
        
        try:
            if isinstance(self.model_config, str):
                # If it's a JSON string, parse and format
                data = json.loads(self.model_config)
            else:
                # If it's already a dict, use it directly
                data = self.model_config
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # If formatting fails, return as string
            return str(self.model_config)
    
    def update_progress(self, current_round, total_rounds=None):
        """Update training progress."""
        self.current_round = current_round
        if total_rounds:
            self.total_rounds = total_rounds
        if self.total_rounds > 0:
            self.progress_percentage = (current_round / self.total_rounds) * 100
        self.save(update_fields=['current_round', 'total_rounds', 'progress_percentage'])
    
    def mark_completed(self, accuracy=None, loss=None, precision=None, recall=None, f1=None):
        """Mark training as completed with final metrics."""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.progress_percentage = 100.0
        
        if accuracy is not None:
            self.final_accuracy = accuracy
        if loss is not None:
            self.final_loss = loss
        if precision is not None:
            self.final_precision = precision
        if recall is not None:
            self.final_recall = recall
        if f1 is not None:
            self.final_f1 = f1
            
        self.save()
    
    def mark_failed(self, error_message=None, traceback=None):
        """Mark training as failed with error details."""
        self.status = 'FAILED'
        self.completed_at = timezone.now()
        if error_message:
            self.error_message = error_message
        if traceback:
            self.error_traceback = traceback
        self.save()
    
    def cancel_training(self):
        """Cancel active training."""
        if self.is_active:
            self.status = 'CANCELLED'
            self.completed_at = timezone.now()
            self.save()
            return True
        return False


class TrainingRound(models.Model):
    """Track individual federated learning rounds."""
    
    session = models.ForeignKey(
        TrainingSession, 
        on_delete=models.CASCADE, 
        related_name='rounds'
    )
    round_number = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    
    # Timing
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Round Metrics
    loss = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Additional metrics (flexible JSON storage)
    metrics = models.JSONField(default=dict, help_text="Additional round-specific metrics")
    
    # Resource usage during this round
    cpu_usage = models.FloatField(null=True, blank=True)
    memory_usage = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['session', 'round_number']
        unique_together = [['session', 'round_number']]
        indexes = [
            models.Index(fields=['session', 'round_number']),
            models.Index(fields=['started_at']),
        ]
    
    def __str__(self):
        return f"Round {self.round_number} - Session {self.session.session_id}"
    
    @property
    def duration(self):
        """Calculate round duration."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_completed(self):
        """Check if round is completed."""
        return self.completed_at is not None
    
    def complete_round(self, loss=None, accuracy=None, precision=None, recall=None, f1_score=None, **kwargs):
        """Mark round as completed with metrics."""
        self.completed_at = timezone.now()
        
        if loss is not None:
            self.loss = loss
        if accuracy is not None:
            self.accuracy = accuracy
        if precision is not None:
            self.precision = precision
        if recall is not None:
            self.recall = recall
        if f1_score is not None:
            self.f1_score = f1_score
        
        # Store additional metrics
        if kwargs:
            self.metrics.update(kwargs)
        
        self.save()
        
        # Update parent session progress
        self.session.update_progress(self.round_number, self.session.total_rounds)