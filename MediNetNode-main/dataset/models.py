import hashlib
import os
from django.db import models
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.contrib.auth import get_user_model

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
    checksum_md5 = models.CharField(max_length=32, editable=False)
    is_active = models.BooleanField(default=True)
    
    # Federated learning fields
    target_column = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Column name that will be used as target for federated learning"
    )
    
    class Meta:
        ordering = ['-uploaded_at']
        
    def __str__(self) -> str:
        return f"{self.name} ({self.medical_domain})"
        
    def calculate_checksum(self) -> str:
        """Calculate MD5 checksum of the file."""
        if not os.path.exists(self.file_path):
            raise ValidationError(f"File not found: {self.file_path}")
            
        hash_md5 = hashlib.md5()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
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
            if not self.checksum_md5:
                self.checksum_md5 = self.calculate_checksum()
        
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
