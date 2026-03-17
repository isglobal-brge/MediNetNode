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

            # No domain provided: just confirm the permission exists.
            # Callers that need domain filtering must supply a domain argument.
            if domain is None:
                return True

            # Check domain against scope
            if scope == 'ALL':
                return True

            if isinstance(scope, list):
                return domain in scope

            # Unknown scope type
            return False

        # Fail-closed: unexpected permission type — deny and log for investigation.
        import logging
        logging.getLogger('security').warning(
            f"Unexpected permission type for '{permission_key}': "
            f"{type(permission_value).__name__} (user={self.username})"
        )
        return False

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
    key_hash = models.CharField(
        max_length=128,
        unique=True,
        null=True,  # Temporarily nullable for migration
        blank=True,
        help_text="Hashed API key for secure authentication (stores hash, not plaintext)"
    )
    key_prefix = models.CharField(
        max_length=8,
        db_index=True,
        blank=True,
        default='',
        help_text=(
            "First 8 chars of raw key for indexed pre-filter before hash verification. "
            "Non-secret. '__LEGACY__' marks keys created before this field was added."
        )
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

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['key_hash']),
            models.Index(fields=['user', 'is_active']),
        ]

    def set_key(self, raw_key):
        """
        Hash and store the API key securely. Also stores key_prefix for indexed lookup.

        Args:
            raw_key (str): The plaintext API key to hash
        """
        from django.contrib.auth.hashers import make_password
        self.key_hash = make_password(raw_key)
        self.key_prefix = raw_key[:8]

    def check_key(self, raw_key):
        """
        Verify if a raw API key matches the stored hash.

        Args:
            raw_key (str): The plaintext API key to verify

        Returns:
            bool: True if the key matches, False otherwise
        """
        from django.contrib.auth.hashers import check_password
        # Guard: key_hash can be null on uninitialized instances
        if not self.key_hash:
            return False
        return check_password(raw_key, self.key_hash)

    def save(self, *args, **kwargs):
        # Only generate new key if this is a new instance and no key_hash set
        if not self.pk and not self.key_hash:
            raw_key = self.generate_api_key()
            self.set_key(raw_key)
            # Store the raw key temporarily so it can be shown to user once
            self._raw_key = raw_key
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
                except (ValueError, ipaddress.AddressValueError, ipaddress.NetmaskValueError):
                    # Fail-closed: deny this entry and log — do not fall back to string comparison
                    import logging
                    logging.getLogger('security').error(
                        f"Failed to parse IP/CIDR whitelist entry: {allowed_ip!r}"
                    )
                    continue

            return False

        except ipaddress.AddressValueError:
            # Fail-closed: deny if client IP cannot be parsed
            import logging
            logging.getLogger('security').error(
                f"Failed to parse client IP address: {ip_address!r}"
            )
            return False
    
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

