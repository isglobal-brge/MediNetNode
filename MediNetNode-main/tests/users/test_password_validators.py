from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.test import TestCase


class PasswordValidatorsTests(TestCase):
    def test_rejects_weak_passwords(self) -> None:
        with self.assertRaises(ValidationError):
            validate_password('password')  # common/weak
        with self.assertRaises(ValidationError):
            validate_password('Cardio1234')  # contains banned medical term
        with self.assertRaises(ValidationError):
            validate_password('alllowercase!1')  # missing upper
        with self.assertRaises(ValidationError):
            validate_password('ALLUPPERCASE!1')  # missing lower
        with self.assertRaises(ValidationError):
            validate_password('MissingSpecial123')  # missing special

    def test_accepts_strong_password(self) -> None:
        # Should not raise
        validate_password('Str0ng!Passw0rd')



