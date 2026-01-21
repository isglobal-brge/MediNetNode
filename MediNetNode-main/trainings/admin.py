"""
Training monitoring admin interface.
All views are restricted to ADMIN and AUDITOR users only.
RESEARCHER users have NO ACCESS to web interfaces.
"""
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import TrainingSession, TrainingRound


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    """Admin interface for TrainingSession model."""
    
    list_display = [
        'session_id_short',
        'user',
        'dataset_name',
        'status_badge',
        'current_round',
        'progress_percentage',
        'started_at',
        'duration_display',
        'final_accuracy'
    ]
    
    list_filter = [
        'status',
        'started_at',
        'user',
        'dataset_name',
    ]
    
    search_fields = [
        'session_id',
        'client_id',
        'user__username',
        'dataset_name',
        'error_message',
    ]
    
    readonly_fields = [
        'session_id',
        'started_at',
        'completed_at',
        'duration_display',
        'progress_percentage',
    ]
    
    fieldsets = (
        ('Identification', {
            'fields': ('session_id', 'client_id', 'user')
        }),
        ('Dataset Information', {
            'fields': ('dataset_id', 'dataset_name')
        }),
        ('Training Configuration', {
            'fields': ('model_config', 'server_address', 'total_rounds'),
            'classes': ('collapse',)
        }),
        ('Status & Progress', {
            'fields': ('status', 'current_round', 'progress_percentage')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'estimated_duration', 'duration_display')
        }),
        ('Resources', {
            'fields': ('cpu_usage', 'memory_usage', 'process_id'),
            'classes': ('collapse',)
        }),
        ('Final Results', {
            'fields': ('final_accuracy', 'final_loss', 'final_precision', 'final_recall', 'final_f1'),
            'classes': ('collapse',)
        }),
        ('Error Information', {
            'fields': ('error_message', 'error_traceback'),
            'classes': ('collapse',)
        })
    )
    
    ordering = ['-started_at']
    
    def session_id_short(self, obj):
        """Display shortened session ID."""
        return str(obj.session_id)[:8] + '...'
    session_id_short.short_description = 'Session ID'
    
    def status_badge(self, obj):
        """Display status with color coding."""
        colors = {
            'STARTING': '#ffc107',    # Yellow
            'ACTIVE': '#28a745',      # Green
            'COMPLETED': '#007bff',   # Blue
            'FAILED': '#dc3545',      # Red
            'CANCELLED': '#6c757d',   # Gray
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">{}</span>',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def duration_display(self, obj):
        """Display formatted duration."""
        duration = obj.duration
        if duration:
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        return "-"
    duration_display.short_description = 'Duration'
    
    def has_add_permission(self, request):
        """Training sessions should only be created via API."""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """Only allow deletion of failed/cancelled sessions."""
        if obj and obj.status in ['FAILED', 'CANCELLED']:
            return super().has_delete_permission(request, obj)
        return False


@admin.register(TrainingRound)
class TrainingRoundAdmin(admin.ModelAdmin):
    """Admin interface for TrainingRound model."""
    
    list_display = [
        'session_link',
        'round_number',
        'loss',
        'accuracy',
        'f1_score',
        'started_at',
        'duration_display',
        'is_completed_badge'
    ]
    
    list_filter = [
        'started_at',
        'session__status',
        'session__user',
    ]
    
    search_fields = [
        'session__session_id',
        'session__user__username',
        'session__dataset_name',
    ]
    
    readonly_fields = [
        'started_at',
        'completed_at',
        'duration_display',
    ]
    
    fieldsets = (
        ('Round Information', {
            'fields': ('session', 'round_number')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'duration_display')
        }),
        ('Metrics', {
            'fields': ('loss', 'accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Additional Metrics', {
            'fields': ('metrics',),
            'classes': ('collapse',)
        }),
        ('Resources', {
            'fields': ('cpu_usage', 'memory_usage'),
            'classes': ('collapse',)
        })
    )
    
    ordering = ['-started_at', 'round_number']
    
    def session_link(self, obj):
        """Display link to parent training session."""
        url = reverse('admin:trainings_trainingsession_change', args=[obj.session.session_id])
        return format_html('<a href="{}">{}</a>', url, str(obj.session.session_id)[:8] + '...')
    session_link.short_description = 'Session'
    
    def duration_display(self, obj):
        """Display formatted duration."""
        duration = obj.duration
        if duration:
            total_seconds = int(duration.total_seconds())
            if total_seconds < 60:
                return f"{total_seconds}s"
            else:
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes}m {seconds}s"
        return "-"
    duration_display.short_description = 'Duration'
    
    def is_completed_badge(self, obj):
        """Display completion status badge."""
        if obj.is_completed:
            return format_html(
                '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">✓ Completed</span>'
            )
        else:
            return format_html(
                '<span style="background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 3px; font-size: 11px;">⏳ In Progress</span>'
            )
    is_completed_badge.short_description = 'Status'
    
    def has_add_permission(self, request):
        """Training rounds should only be created via API."""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """Only allow deletion if parent session allows it."""
        if obj and obj.session.status in ['FAILED', 'CANCELLED']:
            return super().has_delete_permission(request, obj)
        return False


# Custom admin site configuration
admin.site.site_header = "MediNet Training Monitoring"
admin.site.site_title = "Training Admin"
admin.site.index_title = "Training Management"
