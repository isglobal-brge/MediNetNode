from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import CustomUser, Role


@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        (
            'Security',
            {
                'fields': (
                    'role',
                    'created_by',
                    'is_active_session',
                    'last_activity',
                    'failed_login_attempts',
                    'account_locked_until',
                )
            },
        ),
    )
    list_display = (
        'username',
        'email',
        'role',
        'is_staff',
        'is_active',
        'last_login',
        'failed_login_attempts',
    )
    search_fields = ('username', 'email')
