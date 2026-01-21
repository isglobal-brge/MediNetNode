from __future__ import annotations

import re
from typing import Any

from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _


class MedicalPasswordValidator:
    """Enforces strong passwords and bans common medical-related terms.

    Rules:
    - Must include upper, lower, digit, and special character
    - Must not contain banned medical terms or project names
    """

    banned_terms = {
        'password', 'synapnetica', 'medinet', 'medical', 'cardio', 'neuro', 'oncologia', 'radiologia'
    }

    def validate(self, password: str, user: Any = None) -> None:
        p = password or ''
        if not re.search(r"[A-Z]", p):
            raise ValidationError(_("La contraseña debe incluir al menos una mayúscula."))
        if not re.search(r"[a-z]", p):
            raise ValidationError(_("La contraseña debe incluir al menos una minúscula."))
        if not re.search(r"\d", p):
            raise ValidationError(_("La contraseña debe incluir al menos un número."))
        if not re.search(r"[^\w\s]", p):
            raise ValidationError(_("La contraseña debe incluir al menos un carácter especial."))

        low = p.lower()
        for term in self.banned_terms:
            if term in low:
                raise ValidationError(_("La contraseña contiene términos prohibidos."))

    def get_help_text(self) -> str:
        return _("La contraseña debe incluir mayúsculas, minúsculas, números y un carácter especial, y no debe contener términos médicos comunes.")


class PasswordHistoryValidator:
    """Prevents reuse of the last 5 passwords."""

    def validate(self, password: str, user: Any = None) -> None:
        if not user or not hasattr(user, 'check_password_history'):
            return  # Skip for users without password history support
        
        if user.check_password_history(password):
            raise ValidationError(_("No puedes reutilizar ninguna de tus últimas 5 contraseñas."))

    def get_help_text(self) -> str:
        return _("No puedes reutilizar ninguna de tus últimas 5 contraseñas.")



