from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from .models import AuditEvent, DataAccessLog, SecurityIncident, AuditLog


@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    """Admin interface for AuditEvent model."""
    
    list_display = [
        'timestamp', 'category', 'action', 'user', 'resource',
        'risk_score_display', 'severity', 'success', 'requires_review'
    ]
    list_filter = [
        'category', 'severity', 'success', 'requires_review',
        'timestamp', 'risk_score'
    ]
    search_fields = [
        'action', 'resource', 'user__username', 'ip_address',
        'details', 'session_id'
    ]
    readonly_fields = [
        'timestamp', 'risk_score', 'category', 'severity',
        'session_id', 'user_agent', 'request_size', 'request_duration_ms'
    ]
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'action', 'resource', 'success', 'timestamp')
        }),
        ('Classification', {
            'fields': ('category', 'severity', 'risk_score'),
            'classes': ('collapse',)
        }),
        ('Context', {
            'fields': ('ip_address', 'session_id', 'user_agent', 'request_size', 'request_duration_ms'),
            'classes': ('collapse',)
        }),
        ('Review Status', {
            'fields': ('requires_review', 'reviewed_at', 'reviewed_by'),
            'classes': ('collapse',)
        }),
        ('Details', {
            'fields': ('details',),
            'classes': ('collapse',)
        }),
    )
    
    def risk_score_display(self, obj):
        """Display risk score with color coding."""
        if obj.risk_score >= 80:
            color = 'red'
            icon = '⚠️'
        elif obj.risk_score >= 70:
            color = 'orange'
            icon = '⚡'
        elif obj.risk_score >= 30:
            color = 'blue'
            icon = 'ℹ️'
        else:
            color = 'green'
            icon = '✓'
        
        return format_html(
            f'<span style="color: {color};">{icon} {obj.risk_score}</span>'
        )
    risk_score_display.short_description = 'Risk Score'
    risk_score_display.admin_order_field = 'risk_score'
    
    def get_queryset(self, request):
        """Optimize queryset for admin display."""
        return super().get_queryset(request).select_related('user', 'reviewed_by')

    actions = ['mark_as_reviewed']
    
    def mark_as_reviewed(self, request, queryset):
        """Mark selected events as reviewed."""
        count = 0
        for event in queryset:
            if not event.reviewed_at:
                event.mark_reviewed(request.user)
                count += 1
        
        self.message_user(request, f'{count} events marked as reviewed.')
    mark_as_reviewed.short_description = "Mark selected events as reviewed"


@admin.register(DataAccessLog)
class DataAccessLogAdmin(admin.ModelAdmin):
    """Admin interface for DataAccessLog model."""
    
    list_display = [
        'audit_event', 'medical_domain', 'data_sensitivity_level',
        'records_accessed', 'patient_count_accessed'
    ]
    list_filter = [
        'medical_domain', 'data_sensitivity_level',
        'audit_event__timestamp'
    ]
    search_fields = [
        'medical_domain', 'query_hash', 'columns_accessed',
        'audit_event__user__username'
    ]
    readonly_fields = ['query_hash', 'audit_event']
    ordering = ['-audit_event__timestamp']
    
    fieldsets = (
        ('Audit Context', {
            'fields': ('audit_event',)
        }),
        ('Medical Information', {
            'fields': ('medical_domain', 'patient_count_accessed', 'data_sensitivity_level')
        }),
        ('Access Details', {
            'fields': ('records_accessed', 'columns_accessed', 'query_hash'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        """Optimize queryset for admin display."""
        return super().get_queryset(request).select_related('audit_event__user')


@admin.register(SecurityIncident)
class SecurityIncidentAdmin(admin.ModelAdmin):
    """Admin interface for SecurityIncident model."""
    
    list_display = [
        'id', 'incident_type', 'severity_display', 'state',
        'risk_score', 'assigned_to', 'created_at'
    ]
    list_filter = [
        'incident_type', 'state', 'severity', 'created_at', 'assigned_to'
    ]
    search_fields = [
        'description', 'resolution_notes', 'assigned_to__username'
    ]
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Incident Information', {
            'fields': ('incident_type', 'severity', 'state', 'description')
        }),
        ('Assignment', {
            'fields': ('assigned_to', 'risk_score')
        }),
        ('Resolution', {
            'fields': ('resolution_notes',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ['created_at', 'updated_at', 'risk_score']
    
    filter_horizontal = ['related_events']
    
    def severity_display(self, obj):
        """Display severity with color coding."""
        severity_colors = {
            1: ('green', 'Low'),
            2: ('blue', 'Medium'),
            3: ('orange', 'High'),
            4: ('red', 'Critical'),
        }
        color, label = severity_colors.get(obj.severity, ('gray', 'Unknown'))
        return format_html(
            f'<span style="color: {color}; font-weight: bold;">{label}</span>'
        )
    severity_display.short_description = 'Severity'
    severity_display.admin_order_field = 'severity'
    
    def get_queryset(self, request):
        """Optimize queryset for admin display."""
        return super().get_queryset(request).select_related('assigned_to')

    actions = ['assign_to_me', 'mark_investigating', 'mark_resolved']
    
    def assign_to_me(self, request, queryset):
        """Assign selected incidents to current user."""
        count = queryset.update(assigned_to=request.user)
        self.message_user(request, f'{count} incidents assigned to you.')
    assign_to_me.short_description = "Assign selected incidents to me"
    
    def mark_investigating(self, request, queryset):
        """Mark selected incidents as under investigation."""
        count = queryset.update(state='INVESTIGATING')
        self.message_user(request, f'{count} incidents marked as investigating.')
    mark_investigating.short_description = "Mark as investigating"
    
    def mark_resolved(self, request, queryset):
        """Mark selected incidents as resolved."""
        count = queryset.update(state='RESOLVED')
        self.message_user(request, f'{count} incidents marked as resolved.')
    mark_resolved.short_description = "Mark as resolved"


