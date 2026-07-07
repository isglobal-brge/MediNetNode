from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import hashlib
import os
import uuid


class DeployedModelManager(models.Manager):
    """Custom manager for DeployedModel with permission-based filtering."""

    def accessible_by_user(self, user):
        """
        Get models accessible by a user based on their permissions.

        Args:
            user: CustomUser instance

        Returns:
            QuerySet of DeployedModel instances the user can access
        """
        # H9: no superuser shortcut — access is driven by the role's
        # inference.execute scope (the ADMIN role carries scope 'ALL').
        scope = user.get_permission_scope('inference.execute')

        if scope is None:
            return self.none()

        # Filter by approved and public models first
        qs = self.filter(status='approved', is_public=True)

        if scope == 'ALL':
            return qs

        if isinstance(scope, list):
            # User can access specific domains only
            return qs.filter(domain__in=scope)

        return self.none()


class DeployedModel(models.Model):
    """
    ONNX model ready for inference.

    Security-first design with ONNX format only, domain-based access control,
    and comprehensive audit trails.
    """

    STATUS_CHOICES = (
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('deprecated', 'Deprecated'),
        ('rejected', 'Rejected'),
    )

    SOURCE_CHOICES = (
        ('training', 'From Training'),
        ('upload', 'Manual Upload'),
    )

    DOMAIN_CHOICES = (
        ('cardiology', 'Cardiology'),
        ('neurology', 'Neurology'),
        ('oncology', 'Oncology'),
        ('radiology', 'Radiology'),
        ('pathology', 'Pathology'),
        ('general', 'General Medicine'),
    )

    # Identification
    name = models.CharField(
        max_length=200,
        help_text="Descriptive name for the model"
    )
    version = models.CharField(
        max_length=50,
        default='1.0.0',
        help_text="Semantic version (e.g., 1.0.0)"
    )
    description = models.TextField(
        help_text="Detailed description of model purpose and usage"
    )
    domain = models.CharField(
        max_length=50,
        choices=DOMAIN_CHOICES,
        help_text="Medical domain for this model"
    )

    # File Storage
    model_file = models.FileField(
        upload_to='inference/models/%Y/%m/',
        help_text="ONNX model file (.onnx)"
    )
    file_size = models.BigIntegerField(
        help_text="File size in bytes",
        null=True,
        blank=True
    )
    checksum = models.CharField(
        max_length=64,
        help_text="SHA256 checksum of model file",
        blank=True
    )

    # Model Schema
    input_schema = models.JSONField(
        help_text="Expected input format: {feature_names: [str], dtypes: {feature: dtype}, shape: [int]}"
    )
    output_schema = models.JSONField(
        help_text="Output format: {output_names: [str], dtypes: {output: dtype}, shape: [int]}"
    )

    # Visibility & Access
    is_public = models.BooleanField(
        default=False,
        help_text="Public models are accessible to all users with inference.execute permission"
    )

    # Status & Approval Workflow
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Approval status of the model"
    )

    # Audit Trail
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='uploaded_models',
        help_text="User who uploaded this model"
    )
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='approved_models',
        help_text="Admin who approved this model"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when model was approved"
    )

    # Source Tracking
    source = models.CharField(
        max_length=20,
        choices=SOURCE_CHOICES,
        default='upload',
        help_text="Origin of this model"
    )
    training_session_id = models.CharField(
        max_length=100,
        blank=True,
        help_text="Reference to training session if source=training"
    )

    # Model Metrics
    accuracy = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Model accuracy (0.0 to 1.0)"
    )
    validation_notes = models.TextField(
        blank=True,
        help_text="Validation results and notes from testing"
    )

    # Security Settings - Rate Limiting
    max_requests_per_minute = models.IntegerField(
        default=60,
        validators=[MinValueValidator(1)],
        help_text="Maximum requests per minute per user"
    )
    max_batch_size = models.IntegerField(
        default=1000,
        validators=[MinValueValidator(1)],
        help_text="Maximum number of rows per prediction request"
    )

    # Security Settings - Differential Privacy
    enable_differential_privacy = models.BooleanField(
        default=False,
        help_text="Add noise to predictions for privacy protection"
    )
    dp_epsilon = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.01)],
        help_text="Differential privacy epsilon parameter (lower = more privacy)"
    )
    dp_noise_scale = models.FloatField(
        default=0.1,
        validators=[MinValueValidator(0.0)],
        help_text="Scale of noise to add to predictions"
    )

    # Usage Statistics
    total_predictions = models.BigIntegerField(
        default=0,
        help_text="Total number of predictions made with this model"
    )
    last_prediction_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp of last prediction"
    )

    # Custom Manager
    objects = DeployedModelManager()

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['domain', 'status']),
            models.Index(fields=['uploaded_by', '-created_at']),
            models.Index(fields=['status', 'is_public']),
        ]
        unique_together = [['name', 'version']]

    def __str__(self):
        return f"{self.name} v{self.version} ({self.domain})"

    def save(self, *args, **kwargs):
        """Override save to compute checksum and file size."""
        if self.model_file and not self.checksum:
            self.model_file.seek(0)
            file_hash = hashlib.sha256()
            for chunk in self.model_file.chunks():
                file_hash.update(chunk)
            self.checksum = file_hash.hexdigest()
            self.model_file.seek(0)

        if self.model_file and not self.file_size:
            self.file_size = self.model_file.size

        super().save(*args, **kwargs)

    def approve(self, admin_user):
        """
        Approve this model for use.

        Args:
            admin_user: CustomUser with inference.approve permission
        """
        self.status = 'approved'
        self.approved_by = admin_user
        self.approved_at = timezone.now()
        self.save(update_fields=['status', 'approved_by', 'approved_at'])

    def reject(self, admin_user, reason=''):
        """
        Reject this model.

        Args:
            admin_user: CustomUser with inference.approve permission
            reason: Optional rejection reason
        """
        self.status = 'rejected'
        self.approved_by = admin_user
        if reason:
            self.validation_notes = f"REJECTED: {reason}\n\n{self.validation_notes}"
        self.save(update_fields=['status', 'approved_by', 'validation_notes'])

    def deprecate(self):
        """Mark this model as deprecated."""
        self.status = 'deprecated'
        self.save(update_fields=['status'])

    def increment_predictions(self):
        """Increment prediction counter."""
        self.total_predictions += 1
        self.last_prediction_at = timezone.now()
        self.save(update_fields=['total_predictions', 'last_prediction_at'])

    def get_file_path(self):
        """Get absolute path to model file."""
        if self.model_file:
            return self.model_file.path
        return None

    @property
    def file_size_formatted(self):
        """Return human-readable file size."""
        if not self.file_size:
            return "0 B"
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != 'B' else f"{size} B"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def accuracy_percent(self):
        """Return accuracy as percentage (0-100)."""
        if self.accuracy is not None:
            return self.accuracy * 100
        return None

    @property
    def rejection_reason(self):
        """Extract rejection reason from validation_notes."""
        if self.status == 'rejected' and self.validation_notes:
            if self.validation_notes.startswith('REJECTED:'):
                reason = self.validation_notes[9:].split('\n\n')[0].strip()
                return reason
        return None


class PredictionAudit(models.Model):
    """
    Audit log for all prediction requests.

    Security-focused design:
    - NO storing of actual prediction inputs or outputs (privacy)
    - Input hash for duplicate detection only
    - Comprehensive security metrics
    - Used for anti-reverse engineering pattern detection
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )

    # Who made the request
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='prediction_audits',
        help_text="User who made the prediction request"
    )
    api_key = models.ForeignKey(
        'users.APIKey',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='prediction_audits',
        help_text="API key used for the request (if via API)"
    )
    ip_address = models.GenericIPAddressField(
        help_text="IP address of the requester"
    )

    # What model was used
    model = models.ForeignKey(
        DeployedModel,
        on_delete=models.SET_NULL,
        null=True,
        related_name='prediction_audits',
        help_text="Model used for prediction"
    )
    model_name = models.CharField(
        max_length=200,
        help_text="Model name snapshot (in case model is deleted)"
    )
    model_version = models.CharField(
        max_length=50,
        help_text="Model version snapshot"
    )
    model_domain = models.CharField(
        max_length=50,
        help_text="Model domain snapshot"
    )

    # When
    timestamp = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="When the prediction was made"
    )

    # Metrics
    records_count = models.IntegerField(
        help_text="Number of records in the prediction batch"
    )
    execution_time_ms = models.IntegerField(
        help_text="Execution time in milliseconds"
    )

    # Security Monitoring
    rate_limit_remaining = models.IntegerField(
        help_text="Remaining requests in rate limit window"
    )
    suspicious_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Suspicion score from pattern detection (0.0 = normal, 1.0 = very suspicious)"
    )
    patterns_detected = models.JSONField(
        default=list,
        help_text="List of suspicious patterns detected (e.g., ['rapid_fire', 'exhaustive_search'])"
    )

    # Input Fingerprint (for duplicate detection, NOT storing actual data)
    input_hash = models.CharField(
        max_length=64,
        help_text="SHA256 hash of input data (for duplicate detection, privacy-preserving)"
    )

    # Response Status
    success = models.BooleanField(
        default=True,
        help_text="Whether the prediction succeeded"
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error message if prediction failed"
    )

    # Differential Privacy Applied
    dp_noise_applied = models.BooleanField(
        default=False,
        help_text="Whether differential privacy noise was applied to this prediction"
    )

    class Meta:
        db_table = 'prediction_audit'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['model', '-timestamp']),
            models.Index(fields=['-timestamp']),
            models.Index(fields=['suspicious_score', '-timestamp']),
            models.Index(fields=['ip_address', '-timestamp']),
            models.Index(fields=['input_hash']),
        ]

    def __str__(self):
        user_str = self.user.username if self.user else 'Unknown'
        return f"{user_str} - {self.model_name} v{self.model_version} @ {self.timestamp}"

    @staticmethod
    def compute_input_hash(input_data):
        """
        Compute SHA256 hash of input data for duplicate detection.

        Args:
            input_data: Input array/dataframe as string or bytes

        Returns:
            str: SHA256 hexdigest
        """
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        return hashlib.sha256(input_data).hexdigest()

    def mark_suspicious(self, patterns):
        """
        Mark this audit entry as suspicious with detected patterns.

        Args:
            patterns: List of pattern names detected
        """
        self.patterns_detected = patterns
        self.suspicious_score = min(len(patterns) * 0.2, 1.0)  # 0.2 per pattern, max 1.0
        self.save(update_fields=['patterns_detected', 'suspicious_score'])
