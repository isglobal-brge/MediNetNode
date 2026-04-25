import hashlib
import logging
import math
import os
from typing import ClassVar
from django.db import models
from django.db.models import F
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)

User = get_user_model()


class Dataset(models.Model):
    """Main dataset model with medical data management capabilities."""
    
    # Medical domain choices
    MEDICAL_DOMAINS = (
        ('cardiology', 'Cardiology'),
        ('neurology', 'Neurology'), 
        ('oncology', 'Oncology'),
        ('radiology', 'Radiology'),
        ('pathology', 'Pathology'),
        ('dermatology', 'Dermatology'),
        ('ophthalmology', 'Ophthalmology'),
        ('general', 'General Medicine'),
        ('other', 'Other'),
    )
    
    # Data type choices
    DATA_TYPES = (
        ('tabular', 'Tabular Data'),
        ('image', 'Image Data'),
        ('text', 'Text Data'),
        ('time_series', 'Time Series'),
        ('mixed', 'Mixed Data'),
    )
    
    # File format choices
    FILE_FORMATS = (
        ('csv', 'CSV'),
        ('json', 'JSON'),
        ('parquet', 'Parquet'),
        ('h5', 'HDF5'),
        ('npy', 'NumPy'),
        ('other', 'Other'),
    )
    
    # Basic fields
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField()
    file_path = models.CharField(max_length=500)
    # Store user ID instead of foreign key for cross-database compatibility
    uploaded_by_id = models.IntegerField(help_text="ID of user who uploaded the dataset")
    
    # Medical fields
    medical_domain = models.CharField(
        max_length=50,
        choices=MEDICAL_DOMAINS,
        default='general'
    )
    patient_count = models.PositiveIntegerField(null=True, blank=True)
    data_type = models.CharField(
        max_length=50,
        choices=DATA_TYPES,
        default='tabular'
    )
    anonymized = models.BooleanField(default=True)
    
    # Technical fields
    file_size = models.BigIntegerField(help_text="File size in bytes")
    file_format = models.CharField(
        max_length=50,
        choices=FILE_FORMATS,
        default='csv'
    )
    columns_count = models.PositiveIntegerField(null=True, blank=True)
    rows_count = models.PositiveIntegerField(null=True, blank=True)
    
    # Audit fields
    uploaded_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(null=True, blank=True)
    access_count = models.PositiveIntegerField(default=0)
    
    # Security and integrity fields
    checksum_sha256 = models.CharField(
        max_length=64,
        editable=False,
        null=True,
        blank=True,
        help_text='SHA-256 checksum for file integrity verification'
    )
    checksum_md5_deprecated = models.CharField(
        max_length=32,
        editable=False,
        null=True,
        blank=True,
        help_text='DEPRECATED: MD5 checksum (vulnerable). Use checksum_sha256 instead.'
    )
    is_active = models.BooleanField(default=True)
    
    # Federated learning fields
    target_column = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Column name that will be used as target for federated learning"
    )

    # Experimental split — populated only when split_ratio was given at upload
    experiment_file_path = models.CharField(max_length=500, null=True, blank=True)
    experiment_row_count = models.PositiveIntegerField(null=True, blank=True)
    experiment_split_ratio = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        
    def __str__(self) -> str:
        return f"{self.name} ({self.medical_domain})"
        
    def calculate_checksum(self) -> str:
        """
        Calculate SHA-256 checksum of the file.

        SHA-256 provides cryptographically secure file integrity verification,
        resistant to collision attacks that compromise MD5.

        Returns:
            64-character hexadecimal SHA-256 hash

        Raises:
            ValidationError: If file does not exist
        """
        if not os.path.exists(self.file_path):
            raise ValidationError(f"File not found: {self.file_path}")

        hash_sha256 = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def update_access_count(self):
        """Update access count and last accessed timestamp."""
        self.access_count += 1
        self.last_accessed = timezone.now()
        self.save(update_fields=['access_count', 'last_accessed'])
        
    def get_file_size_display(self) -> str:
        """Return human-readable file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"
        
    def clean(self):
        """Validate model fields."""
        super().clean()
        
        # Validate medical domain
        if self.medical_domain not in [choice[0] for choice in self.MEDICAL_DOMAINS]:
            raise ValidationError({'medical_domain': 'Invalid medical domain.'})
            
        # Validate file path exists
        if self.file_path and not os.path.exists(self.file_path):
            raise ValidationError({'file_path': 'File does not exist.'})
            
        # Validate patient count for certain domains
        if self.data_type in ['tabular', 'mixed'] and not self.patient_count:
            raise ValidationError({'patient_count': 'Patient count is required for tabular and mixed data.'})
    
    def save(self, *args, **kwargs):
        """Override save to automatically calculate checksum."""
        # Calculate file size if not provided
        if self.file_path and os.path.exists(self.file_path):
            if not self.file_size:
                self.file_size = os.path.getsize(self.file_path)

            # Calculate checksum if not provided or file changed
            if not self.checksum_sha256:
                self.checksum_sha256 = self.calculate_checksum()

        super().save(*args, **kwargs)


class DatasetAccess(models.Model):
    """Dataset access permissions for users."""
    
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='access_permissions'
    )
    # Store user IDs instead of foreign keys for cross-database compatibility
    user_id = models.IntegerField(help_text="ID of user in main database")
    assigned_by_id = models.IntegerField(help_text="ID of user who assigned access in main database")
    assigned_at = models.DateTimeField(auto_now_add=True)
    
    # Permission fields
    can_train = models.BooleanField(default=True)
    can_view_metadata = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ('dataset', 'user_id')
        ordering = ['-assigned_at']
        
    def __str__(self) -> str:
        return f"User {self.user_id} -> {self.dataset.name}"
    
    @property
    def user(self):
        """Get user object from main database."""
        from django.contrib.auth import get_user_model
        UserModel = get_user_model()
        try:
            return UserModel.objects.using('default').get(id=self.user_id)
        except UserModel.DoesNotExist:
            return None
    
    @property
    def assigned_by(self):
        """Get assigned_by user object from main database."""
        from django.contrib.auth import get_user_model
        UserModel = get_user_model()
        try:
            return UserModel.objects.using('default').get(id=self.assigned_by_id)
        except UserModel.DoesNotExist:
            return None


class DatasetMetadata(models.Model):
    """Statistical and quality metadata for datasets."""
    
    dataset = models.OneToOneField(
        Dataset,
        on_delete=models.CASCADE,
        related_name='metadata',
        primary_key=True
    )
    
    # Statistical data stored as JSON
    statistical_summary = models.JSONField(
        default=dict,
        help_text="Statistical summary (mean, std, min, max, etc.)"
    )
    missing_values = models.JSONField(
        default=dict,
        help_text="Missing values count per column"
    )
    data_distribution = models.JSONField(
        default=dict,
        help_text="Data distribution information per column"
    )
    
    # Quality metrics
    quality_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Overall dataset quality score (0.0-1.0)"
    )
    completeness_percentage = models.FloatField(
        null=True,
        blank=True,
        help_text="Data completeness percentage"
    )
    
    # Metadata generation timestamp
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        
    def __str__(self) -> str:
        return f"Metadata for {self.dataset.name}"
        
    def calculate_completeness(self) -> float:
        """Calculate data completeness percentage based on missing values."""
        if not self.missing_values or not self.dataset.rows_count:
            return 0.0
            
        total_cells = self.dataset.rows_count * (self.dataset.columns_count or 1)
        missing_cells = sum(self.missing_values.values())
        
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        return round(completeness, 2)
        
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score based on various metrics."""
        if not self.completeness_percentage:
            self.completeness_percentage = self.calculate_completeness()
        
        # Basic quality score based on completeness
        # Can be extended with more sophisticated metrics
        quality_factors = [
            self.completeness_percentage / 100,  # Normalize to 0-1
        ]
        
        # Add other quality factors as needed
        if self.statistical_summary:
            # Bonus for having statistical summary
            quality_factors.append(0.1)
            
        if self.data_distribution:
            # Bonus for having distribution data
            quality_factors.append(0.1)
        
        quality_score = sum(quality_factors) / len(quality_factors)
        return round(min(quality_score, 1.0), 3)  # Cap at 1.0
        
    def save(self, *args, **kwargs):
        """Override save to auto-calculate quality metrics."""
        if not self.completeness_percentage:
            self.completeness_percentage = self.calculate_completeness()

        if not self.quality_score:
            self.quality_score = self.calculate_quality_score()

        super().save(*args, **kwargs)


class DatasetPrivacyPolicy(models.Model):
    """Per-dataset differential-privacy budget enforced by the Node.

    The Node admin classifies dataset sensitivity; limits are derived
    automatically from literature-validated presets (see SENSITIVITY_DEFAULTS).
    The Hub (researcher side) cannot override these limits.
    """

    SENSITIVITY_CHOICES = [
        ('high',   'Alta — diagnóstico, salud mental, genómica'),
        ('medium', 'Media — riesgo cardiovascular, general'),
        ('low',    'Baja — estadísticas agregadas'),
    ]

    # Literature-backed presets (npj Digital Medicine 2025, OpenMined guide).
    # Keyed by sensitivity choice value for O(1) lookup.
    SENSITIVITY_DEFAULTS: ClassVar[dict[str, dict[str, float]]] = {
        'high':   {'max_epsilon_per_job': 0.5,  'lifetime_budget': 2.0},
        'medium': {'max_epsilon_per_job': 1.0,  'lifetime_budget': 5.0},
        'low':    {'max_epsilon_per_job': 3.0,  'lifetime_budget': 15.0},
    }

    dataset = models.OneToOneField(
        'Dataset',
        on_delete=models.CASCADE,
        related_name='privacy_policy',
    )
    sensitivity = models.CharField(
        max_length=10,
        choices=SENSITIVITY_CHOICES,
        default='medium',
    )
    max_epsilon_per_job = models.FloatField(
        help_text="Maximum ε allowed per training job (Node-enforced).",
    )
    lifetime_budget = models.FloatField(
        help_text="Total ε allowed across all jobs for this dataset.",
    )
    spent_epsilon = models.FloatField(
        default=0.0,
        help_text="Cumulative ε spent across all completed jobs.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Dataset Privacy Policy'
        verbose_name_plural = 'Dataset Privacy Policies'
        constraints = [
            models.CheckConstraint(
                check=models.Q(max_epsilon_per_job__gt=0.0),
                name='privacy_policy_max_eps_positive',
            ),
            models.CheckConstraint(
                check=models.Q(lifetime_budget__gt=0.0),
                name='privacy_policy_lifetime_positive',
            ),
            models.CheckConstraint(
                check=models.Q(spent_epsilon__gte=0.0),
                name='privacy_policy_spent_nonneg',
            ),
        ]

    def __str__(self) -> str:
        return (
            f"PrivacyPolicy({self.dataset.name}, "
            f"sensitivity={self.sensitivity}, "
            f"spent={self.spent_epsilon:.4f}/{self.lifetime_budget})"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def clean(self) -> None:
        super().clean()
        for field_name, value in [
            ('max_epsilon_per_job', self.max_epsilon_per_job),
            ('lifetime_budget', self.lifetime_budget),
        ]:
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0.0:
                raise ValidationError(
                    {field_name: f"Debe ser un número positivo finito (recibido: {value})."}
                )
        if (
            self.max_epsilon_per_job is not None
            and self.lifetime_budget is not None
            and self.max_epsilon_per_job > self.lifetime_budget
        ):
            raise ValidationError(
                'max_epsilon_per_job no puede superar lifetime_budget.'
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def save(self, *args, **kwargs) -> None:
        """Auto-populate budget limits from sensitivity preset when not set.

        Uses explicit `is None` (not falsy) so that an explicitly-set 0.0
        triggers clean() to raise ValidationError rather than being silently
        overwritten with a permissive preset default.
        """
        defaults = self.SENSITIVITY_DEFAULTS.get(self.sensitivity, self.SENSITIVITY_DEFAULTS['medium'])
        if self.max_epsilon_per_job is None:
            self.max_epsilon_per_job = defaults['max_epsilon_per_job']
        if self.lifetime_budget is None:
            self.lifetime_budget = defaults['lifetime_budget']
        self.clean()  # Enforce field-level constraints on every ORM save
        super().save(*args, **kwargs)

    # ------------------------------------------------------------------
    # Budget properties
    # ------------------------------------------------------------------

    @property
    def remaining_budget(self) -> float:
        """Remaining lifetime ε budget.

        Returns 0.0 when stored values are NaN/inf (corrupt DB row) so the
        system fails closed rather than silently accepting jobs against a
        corrupted budget. max(0.0, ...) also prevents negative results from
        float drift when spent slightly exceeds budget.
        """
        if (
            not math.isfinite(self.lifetime_budget)
            or not math.isfinite(self.spent_epsilon)
        ):
            return 0.0
        return max(0.0, self.lifetime_budget - self.spent_epsilon)

    # ------------------------------------------------------------------
    # Enforcement API
    # ------------------------------------------------------------------

    def can_accept_job(self, estimated_epsilon: float) -> tuple[bool, str]:
        """Return (True, 'ok') or (False, human-readable reason).

        Rejects non-positive or non-finite epsilon — these indicate a failed
        estimation (sentinel -1.0) or an adversarial input and must never be
        allowed to bypass budget checks by exploiting Python's NaN comparison
        rules (nan > x is always False, which would let NaN pass silently).

        Refreshes spent_epsilon from DB before checking to reduce the
        TOCTOU window between this call and record_spent().
        """
        # Reduce TOCTOU window: read the current spent_epsilon from DB.
        self.refresh_from_db(fields=['spent_epsilon'])

        if not math.isfinite(estimated_epsilon) or estimated_epsilon <= 0.0:
            return False, (
                f"ε estimado inválido ({estimated_epsilon}): debe ser un número "
                f"positivo y finito. La estimación falló o el valor es malformado."
            )
        # Guard against a corrupted stored per-job limit (e.g. NaN written directly
        # to the DB). nan > x is False, so a NaN cap would silently vanish.
        if not math.isfinite(self.max_epsilon_per_job) or self.max_epsilon_per_job <= 0.0:
            return False, (
                "Política de privacidad corrupta: límite por job inválido. "
                "Contacte al administrador del Node."
            )
        if estimated_epsilon > self.max_epsilon_per_job:
            return False, (
                f"Job ε estimado ({estimated_epsilon:.4f}) supera el máximo por job "
                f"({self.max_epsilon_per_job:.4f}) para datos de sensibilidad "
                f"'{self.sensitivity}'."
            )
        if estimated_epsilon > self.remaining_budget:
            return False, (
                f"Presupuesto agotado: quedan {self.remaining_budget:.4f} ε de "
                f"{self.lifetime_budget:.4f} para este dataset."
            )
        return True, "ok"

    def record_spent(self, actual_epsilon: float) -> None:
        """Atomically add actual_epsilon to spent_epsilon.

        Uses a conditional DB-level F() update: the row is only updated when
        the current spent_epsilon still has room for the new value. This closes
        the TOCTOU gap between can_accept_job() and record_spent() — if two
        concurrent callers both pass can_accept_job() and both call
        record_spent(), the second update's WHERE clause (spent <= budget -
        delta) fails and the budget overrun is logged instead of silently
        applied.

        Silently skips non-positive or non-finite values (e.g., the -1.0
        sentinel meaning 'epsilon measurement failed').
        """
        if not math.isfinite(actual_epsilon) or actual_epsilon <= 0.0:
            return
        delta = round(actual_epsilon, 6)
        updated = DatasetPrivacyPolicy.objects.filter(
            pk=self.pk,
            spent_epsilon__lte=F('lifetime_budget') - delta,
        ).update(
            spent_epsilon=F('spent_epsilon') + delta,
        )
        self.refresh_from_db(fields=['spent_epsilon'])
        if updated == 0:
            logger.error(
                "[DP] Budget overrun on DatasetPrivacyPolicy pk=%s: "
                "attempted to record ε=%.6f but budget was already exhausted.",
                self.pk, delta,
            )


class ResearcherEpsilonBudget(models.Model):
    """
    Presupuesto de epsilon DP por (dataset, researcher).

    researcher_id es IntegerField (no FK) porque User vive en 'default'
    y Dataset/Policy viven en 'datasets_db' — no se pueden hacer FK entre DBs distintas.
    """

    PERIOD_CHOICES = [
        ('annual', 'Anual'),
        ('monthly', 'Mensual'),
        ('never', 'Sin reset automático'),
    ]

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='researcher_budgets',
    )
    researcher_id = models.IntegerField(db_index=True)
    spent_epsilon = models.FloatField(default=0.0)
    lifetime_budget = models.FloatField()
    max_epsilon_per_job = models.FloatField()
    period = models.CharField(max_length=16, choices=PERIOD_CHOICES, default='annual')
    period_start = models.DateTimeField(default=timezone.now)
    last_reset = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [['dataset', 'researcher_id']]
        indexes = [
            models.Index(fields=['dataset', 'researcher_id']),
        ]

    @classmethod
    def get_or_create_for(cls, *, dataset, researcher_id, policy, period='annual'):
        obj, created = cls.objects.get_or_create(
            dataset=dataset,
            researcher_id=researcher_id,
            defaults={
                'lifetime_budget': policy.lifetime_budget,
                'max_epsilon_per_job': policy.max_epsilon_per_job,
                'period': period,
                'spent_epsilon': 0.0,
            },
        )
        return obj, created

    @property
    def remaining_budget(self) -> float:
        if not math.isfinite(self.spent_epsilon) or not math.isfinite(self.lifetime_budget):
            return 0.0
        return max(0.0, round(self.lifetime_budget - self.spent_epsilon, 6))

    def can_accept_job(self, estimated_epsilon: float):
        if not math.isfinite(estimated_epsilon) or estimated_epsilon <= 0:
            return False, "El epsilon estimado no es válido."

        if self.is_period_expired():
            self.reset_period()

        self.refresh_from_db()

        if not math.isfinite(self.max_epsilon_per_job) or estimated_epsilon > self.max_epsilon_per_job:
            return False, (
                f"El epsilon estimado ({estimated_epsilon:.4f}) supera el "
                f"máximo por job ({self.max_epsilon_per_job:.4f}) para este researcher."
            )

        if estimated_epsilon > self.remaining_budget:
            return False, (
                f"El epsilon estimado ({estimated_epsilon:.4f}) supera el "
                f"presupuesto restante del researcher ({self.remaining_budget:.4f})."
            )

        return True, ""

    def record_spent(self, actual_epsilon: float) -> None:
        import math as _math
        if not _math.isfinite(actual_epsilon) or actual_epsilon <= 0:
            return
        actual_epsilon = round(actual_epsilon, 6)
        ResearcherEpsilonBudget.objects.filter(
            pk=self.pk,
            spent_epsilon__lte=models.F('lifetime_budget'),
        ).update(spent_epsilon=models.F('spent_epsilon') + actual_epsilon)

    def is_period_expired(self) -> bool:
        if self.period == 'never':
            return False
        now = timezone.now()
        if self.period == 'annual':
            from dateutil.relativedelta import relativedelta
            return now >= self.period_start + relativedelta(years=1)
        if self.period == 'monthly':
            from dateutil.relativedelta import relativedelta
            return now >= self.period_start + relativedelta(months=1)
        return False

    def reset_period(self) -> None:
        ResearcherEpsilonBudget.objects.filter(pk=self.pk).update(
            spent_epsilon=0.0,
            period_start=timezone.now(),
            last_reset=timezone.now(),
        )
        self.refresh_from_db()
