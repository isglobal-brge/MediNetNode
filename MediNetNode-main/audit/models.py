from django.db import models
from django.conf import settings
from django.utils import timezone
import json


class AuditEvent(models.Model):
    """Extended audit system with risk scoring and categorization."""
    
    # Event Categories
    CATEGORY_CHOICES = [
        ('AUTH', 'Authentication'),
        ('DATA_ACCESS', 'Data Access'),
        ('USER_MGMT', 'User Management'),
        ('DATASET_MGMT', 'Dataset Management'),
        ('TRAINING', 'Federated Training'),
        ('API', 'API Access'),
        ('SYSTEM', 'System Operations'),
    ]
    
    # Severity Levels
    SEVERITY_CHOICES = [
        ('INFO', 'Information'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('CRITICAL', 'Critical'),
        ('SECURITY', 'Security Event'),
    ]

    # Basic audit fields
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_events',
    )
    action = models.CharField(max_length=100)
    resource = models.CharField(max_length=200, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=True)
    details = models.JSONField(default=dict, blank=True)
    
    # Classification fields
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='SYSTEM')
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='INFO')
    risk_score = models.IntegerField(default=0, help_text="Risk score 0-100")
    
    # Context fields
    session_id = models.CharField(max_length=40, blank=True)
    user_agent = models.TextField(blank=True)
    request_size = models.IntegerField(null=True, blank=True, help_text="Request size in bytes")
    request_duration_ms = models.IntegerField(null=True, blank=True)
    
    # Review management
    requires_review = models.BooleanField(default=False)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_audit_events',
    )

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['action']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['category']),
            models.Index(fields=['severity']),
            models.Index(fields=['risk_score']),
            models.Index(fields=['requires_review']),
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['ip_address', 'timestamp']),
            models.Index(fields=['category', 'severity']),
        ]

    def __str__(self) -> str:
        status = 'OK' if self.success else 'FAIL'
        return f"{self.timestamp} - {self.category}:{self.action} ({status}) - Risk:{self.risk_score}"

    def mark_reviewed(self, reviewer):
        """Mark event as reviewed by specified user."""
        self.reviewed_at = timezone.now()
        self.reviewed_by = reviewer
        self.save(update_fields=['reviewed_at', 'reviewed_by'])


class DataAccessLog(models.Model):
    """Specific logging for dataset access events."""
    
    audit_event = models.OneToOneField(
        AuditEvent,
        on_delete=models.CASCADE,
        related_name='data_access_log'
    )
    
    # Medical context
    medical_domain = models.CharField(max_length=100, blank=True, help_text="Medical specialty domain")
    patient_count_accessed = models.IntegerField(default=0)
    data_sensitivity_level = models.IntegerField(
        default=1,
        choices=[(i, f"Level {i}") for i in range(1, 6)],
        help_text="Data sensitivity level 1-5"
    )
    
    # Access details
    columns_accessed = models.JSONField(default=list, blank=True)
    records_accessed = models.IntegerField(default=0)
    query_hash = models.CharField(max_length=64, blank=True, help_text="Hash of executed query")
    
    class Meta:
        indexes = [
            models.Index(fields=['medical_domain']),
            models.Index(fields=['data_sensitivity_level']),
            models.Index(fields=['patient_count_accessed']),
        ]

    def __str__(self) -> str:
        return f"DataAccess: {self.medical_domain} - {self.records_accessed} records (Level {self.data_sensitivity_level})"


class SecurityIncident(models.Model):
    """Security incidents automatically generated from critical audit events."""
    
    # Incident Types
    INCIDENT_TYPES = [
        ('UNAUTHORIZED_ACCESS', 'Unauthorized Access Attempt'),
        ('DATA_BREACH_ATTEMPT', 'Data Breach Attempt'),
        ('SUSPICIOUS_ACTIVITY', 'Suspicious Activity'),
        ('MULTIPLE_FAILED_AUTH', 'Multiple Failed Authentication'),
        ('PRIVILEGE_ESCALATION', 'Privilege Escalation Attempt'),
        ('ANOMALOUS_PATTERN', 'Anomalous Access Pattern'),
    ]
    
    # Incident States
    STATE_CHOICES = [
        ('OPEN', 'Open'),
        ('INVESTIGATING', 'Under Investigation'),
        ('RESOLVED', 'Resolved'),
        ('CLOSED', 'Closed'),
    ]
    
    # Severity levels
    SEVERITY_LEVELS = [
        (1, 'Low'),
        (2, 'Medium'), 
        (3, 'High'),
        (4, 'Critical'),
    ]

    # Incident details
    incident_type = models.CharField(max_length=30, choices=INCIDENT_TYPES)
    state = models.CharField(max_length=15, choices=STATE_CHOICES, default='OPEN')
    severity = models.IntegerField(choices=SEVERITY_LEVELS, default=2)
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_incidents',
    )
    
    # Context
    description = models.TextField()
    resolution_notes = models.TextField(blank=True)
    risk_score = models.IntegerField(default=0, help_text="Aggregated risk score from related events")
    
    # Relationships
    related_events = models.ManyToManyField(
        AuditEvent,
        related_name='security_incidents',
        help_text="Audit events that triggered this incident"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['incident_type']),
            models.Index(fields=['state']),
            models.Index(fields=['severity']),
            models.Index(fields=['created_at']),
            models.Index(fields=['assigned_to']),
        ]

    def __str__(self) -> str:
        return f"Incident-{self.id}: {self.incident_type} ({self.state}) - Severity {self.severity}"

    def update_risk_score(self):
        """Recalculate risk score based on related events."""
        if self.related_events.exists():
            self.risk_score = self.related_events.aggregate(
                max_risk=models.Max('risk_score')
            )['max_risk'] or 0
            self.save(update_fields=['risk_score'])


# Legacy model for backward compatibility - will be deprecated
class AuditLog(models.Model):
    """DEPRECATED: Use AuditEvent instead. Kept for backward compatibility."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_logs',
    )
    action = models.CharField(max_length=100)
    resource = models.CharField(max_length=200, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=True)
    details = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['action']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self) -> str:
        return f"{self.timestamp} - {self.action} ({'OK' if self.success else 'FAIL'})"


