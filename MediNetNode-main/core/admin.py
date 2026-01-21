from django.contrib import admin
from .models import SystemConfiguration


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    """
    Admin interface for SystemConfiguration.

    Single instance (singleton) - only one configuration can exist.
    """
    list_display = (
        'center_display_name',
        'center_id',
        'center_email',
        'setup_completed_at',
        'setup_completed_by'
    )

    fieldsets = (
        ('Center Information', {
            'fields': ('center_id', 'center_display_name', 'center_email')
        }),
        ('Extra Settings (JSON)', {
            'fields': ('extra_settings',),
            'classes': ('collapse',),
            'description': 'Additional configuration in JSON format for future features'
        }),
        ('Setup Metadata', {
            'fields': ('setup_completed_at', 'setup_completed_by', 'last_modified'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('setup_completed_at', 'setup_completed_by', 'last_modified')

    def has_add_permission(self, request):
        """
        Prevent adding more than one configuration (singleton pattern).
        """
        if SystemConfiguration.objects.exists():
            return False
        return super().has_add_permission(request)

    def has_delete_permission(self, request, obj=None):
        """
        Prevent deletion of system configuration.
        """
        return False
