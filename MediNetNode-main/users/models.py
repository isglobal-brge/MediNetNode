from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
import secrets
import string


class Role(models.Model):
    """User role with granular permissions stored in JSON."""

    ROLE_CHOICES = (
        ('ADMIN', 'ADMIN'),
        ('MEMBER', 'MEMBER'),
        ('RESEARCHER', 'RESEARCHER'),
        ('AUDITOR', 'AUDITOR'),
    )

    name = models.CharField(max_length=50, unique=True, choices=ROLE_CHOICES)
    permissions = models.JSONField(default=dict)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class CustomUser(AbstractUser):
    """Usuario personalizado con control de seguridad y rol."""

    role = models.ForeignKey(
        Role,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='users',
    )
    created_by = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_users',
    )
    is_active_session = models.BooleanField(default=False)
    last_activity = models.DateTimeField(null=True, blank=True)
    failed_login_attempts = models.PositiveIntegerField(default=0)
    account_locked_until = models.DateTimeField(null=True, blank=True)

    def increment_failed_attempts(self) -> None:
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        self.save(update_fields=['failed_login_attempts'])

    def reset_failed_attempts(self) -> None:
        self.failed_login_attempts = 0
        self.account_locked_until = None
        self.save(update_fields=['failed_login_attempts', 'account_locked_until'])

    def is_account_locked(self) -> bool:
        if self.account_locked_until:
            return timezone.now() < self.account_locked_until
        return False

    def has_permission(self, permission_key: str, domain: str = None) -> bool:
        """
        Check if user has a specific permission.

        Args:
            permission_key: The permission to check (e.g., 'inference.execute')
            domain: Optional domain to check against scope (e.g., 'cardiology')

        Returns:
            bool: True if user has permission

        Supports both simple boolean permissions and scope-based permissions:
        - Simple: 'api.access': True
        - Scope: 'inference.execute': {'scope': 'ALL'} or {'scope': ['cardiology', 'neurology']}
        """
        # Superusers have all permissions by definition
        if getattr(self, 'is_superuser', False):
            return True
        if not self.role or not self.role.permissions:
            return False

        permission_value = self.role.permissions.get(permission_key)

        # No permission found
        if permission_value is None:
            return False

        # Simple boolean permission (backward compatible)
        if isinstance(permission_value, bool):
            return permission_value

        # Scope-based permission
        if isinstance(permission_value, dict):
            scope = permission_value.get('scope')

            # If no scope defined in the permission, deny access
            if scope is None:
                return False

            # No domain check needed, just verify permission exists with valid scope
            if domain is None:
                # User has permission, scope will be checked when accessing specific resources
                return True

            # Check domain against scope
            if scope == 'ALL':
                return True

            if isinstance(scope, list):
                return domain in scope

            # Unknown scope type
            return False

        # Fallback: treat as truthy
        return bool(permission_value)

    def get_permission_scope(self, permission_key: str):
        """
        Get the scope of a permission.

        Args:
            permission_key: The permission to check

        Returns:
            str | list | None: 'ALL', list of domains, or None if no scope
        """
        if not self.role or not self.role.permissions:
            return None

        permission_value = self.role.permissions.get(permission_key)

        if isinstance(permission_value, dict):
            return permission_value.get('scope')

        return None

    def is_session_expired(self) -> bool:
        """Check if user's session has expired based on idle timeout."""
        from django.conf import settings
        
        if not self.last_activity:
            return True
        
        idle_timeout = getattr(settings, 'SESSION_IDLE_TIMEOUT', 1800)
        time_diff = timezone.now() - self.last_activity
        return time_diff.total_seconds() > idle_timeout

    def set_password(self, raw_password):
        """Override to save password history before changing password."""
        if self.pk and self.password:  # Only if user exists and has a current password
            # Save current password to history
            PasswordHistory.objects.create(
                user=self,
                password_hash=self.password
            )
            # Keep only last 5 passwords
            history_count = PasswordHistory.objects.filter(user=self).count()
            if history_count > 5:
                oldest_passwords = PasswordHistory.objects.filter(user=self).order_by('created_at')[:history_count-5]
                PasswordHistory.objects.filter(id__in=[p.id for p in oldest_passwords]).delete()
        
        super().set_password(raw_password)

    def check_password_history(self, raw_password):
        """Check if password was used in the last 5 passwords."""
        if not raw_password:
            return False
            
        # Check current password
        if check_password(raw_password, self.password):
            return True
            
        # Check last 5 passwords in history
        if not self.pk:
            return False
        for history in PasswordHistory.objects.filter(user=self).order_by('-created_at')[:5]:
            if check_password(raw_password, history.password_hash):
                return True
        
        return False


class PasswordHistory(models.Model):
    """Track user password history for security compliance."""
    
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name='password_history'
    )
    password_hash = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.created_at}"


class APIKey(models.Model):
    """API key for stateless authentication of RESEARCHER users."""
    
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name='api_keys',
        help_text="RESEARCHER user associated with this API key"
    )
    key = models.CharField(
        max_length=64, 
        unique=True,
        help_text="Unique API key for authentication"
    )
    name = models.CharField(
        max_length=100,
        help_text="Descriptive name for this API key"
    )
    ip_whitelist = models.JSONField(
        default=list,
        help_text="List of allowed IP addresses for this API key"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Expiration date for this API key"
    )
    last_used_at = models.DateTimeField(null=True, blank=True)
    last_used_ip = models.GenericIPAddressField(null=True, blank=True)
    
    # TODO: Add regeneration fields after migration is applied
    # regenerated_count = models.IntegerField(default=0, help_text="Number of times this key has been regenerated")
    # last_regenerated_at = models.DateTimeField(null=True, blank=True, help_text="Last time the key was regenerated")
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['key']),
            models.Index(fields=['user', 'is_active']),
        ]
    
    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_api_key()
        super().save(*args, **kwargs)
    
    @staticmethod
    def generate_api_key():
        """Generate a secure random API key."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(64))
    
    def is_expired(self):
        """Check if API key has expired."""
        if not self.expires_at:
            return False
        return timezone.now() > self.expires_at
    
    def is_ip_allowed(self, ip_address):
        """Check if IP address is in whitelist."""
        if not self.ip_whitelist:
            return False
        
        import ipaddress
        
        try:
            # Convert string IP to IP object
            client_ip = ipaddress.ip_address(ip_address)
            
            # Check each whitelist entry
            for allowed_ip in self.ip_whitelist:
                try:
                    # Handle CIDR notation (e.g., '0.0.0.0/0', '192.168.1.0/24')
                    if '/' in allowed_ip:
                        network = ipaddress.ip_network(allowed_ip, strict=False)
                        if client_ip in network:
                            return True
                    else:
                        # Handle single IP address
                        allowed = ipaddress.ip_address(allowed_ip)
                        if client_ip == allowed:
                            return True
                except (ipaddress.AddressValueError, ipaddress.NetmaskValueError):
                    # If IP parsing fails, fall back to string comparison
                    if ip_address == allowed_ip:
                        return True
            
            return False
            
        except ipaddress.AddressValueError:
            # If client IP parsing fails, fall back to string comparison
            return ip_address in self.ip_whitelist
    
    def update_last_used(self, ip_address):
        """Update last used timestamp and IP."""
        self.last_used_at = timezone.now()
        self.last_used_ip = ip_address
        self.save(update_fields=['last_used_at', 'last_used_ip'])
    
    def __str__(self):
        return f"API Key: {self.name} ({self.user.username})"


class APIRequest(models.Model):
    """Audit log for API requests made with API keys."""
    
    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.CASCADE,
        related_name='requests',
        null=True,
        blank=True
    )
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name='api_requests',
        null=True,
        blank=True
    )
    endpoint = models.CharField(max_length=200)
    method = models.CharField(max_length=10)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    status_code = models.PositiveIntegerField()
    response_time_ms = models.PositiveIntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Security fields
    is_successful = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['api_key', '-timestamp']),
            models.Index(fields=['ip_address', '-timestamp']),
            models.Index(fields=['endpoint', '-timestamp']),
        ]
    
    def __str__(self):
        user_str = self.user.username if self.user else 'Anonymous'
        return f"{user_str} - {self.method} {self.endpoint} ({self.status_code})"

