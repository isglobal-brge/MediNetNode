from __future__ import annotations

from typing import Optional

from django.conf import settings
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.hashers import check_password
from django.utils import timezone

from users.models import CustomUser
from audit.audit_logger import AuditLogger


class SecureLoginBackend(BaseBackend):
    """Authentication backend with account lockout and audit logging."""

    def authenticate(self, request, username: Optional[str] = None, password: Optional[str] = None, **kwargs):
        if username is None or password is None:
            return None

        try:
            user = CustomUser.objects.using('default').get(username=username)
        except CustomUser.DoesNotExist:
            self._log_attempt(None, 'LOGIN_FAIL', success=False, request=request, details={'reason': 'user_not_found', 'username': username})
            return None

        if user.is_account_locked():
            self._log_attempt(user, 'LOGIN_FAIL_LOCKED', success=False, request=request)
            return None

        if user.password and check_password(password, user.password):
            user.reset_failed_attempts()
            user.last_activity = timezone.now()
            user.is_active_session = True
            user.save(update_fields=['last_activity', 'is_active_session'])
            self._log_attempt(user, 'LOGIN_SUCCESS', success=True, request=request)
            return user

        # Failed password
        user.increment_failed_attempts()
        lock_after = getattr(settings, 'ACCOUNT_LOCKOUT_ATTEMPTS', 5)
        if (user.failed_login_attempts or 0) >= lock_after:
            duration_seconds = int(getattr(settings, 'ACCOUNT_LOCKOUT_DURATION', 300))
            user.account_locked_until = timezone.now() + timezone.timedelta(seconds=duration_seconds)
            user.save(update_fields=['account_locked_until'])
            self._log_attempt(user, 'ACCOUNT_LOCKED', success=False, request=request, details={'attempts': user.failed_login_attempts})
        else:
            self._log_attempt(user, 'LOGIN_FAIL', success=False, request=request, details={'attempts': user.failed_login_attempts})

        return None

    def get_user(self, user_id: int) -> Optional[CustomUser]:
        try:
            return CustomUser.objects.using('default').get(pk=user_id)
        except CustomUser.DoesNotExist:
            return None

    @staticmethod
    def _log_attempt(user: Optional[CustomUser], action: str, success: bool, request=None, details: Optional[dict] = None) -> None:
        ip = None
        if request is not None:
            xff = request.META.get('HTTP_X_FORWARDED_FOR')
            if xff:
                ip = xff.split(',')[0].strip()
            else:
                ip = request.META.get('REMOTE_ADDR')
        
        # Use new AuditLogger system
        AuditLogger.log_authentication(
            action=action,
            user=user,
            success=success,
            ip_address=ip,
            details=details or {}
        )


