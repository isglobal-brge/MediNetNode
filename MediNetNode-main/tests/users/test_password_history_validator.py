"""
Tests for password history validation functionality.
"""
from django.test import TestCase
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import get_user_model

from users.models import Role, PasswordHistory
from users.validators import PasswordHistoryValidator, MedicalPasswordValidator

User = get_user_model()


class PasswordHistoryValidatorTests(TestCase):
    """Test password history validation."""
    
    def setUp(self):
        self.admin_role = Role.objects.get(name='ADMIN')
        self.validator = PasswordHistoryValidator()
        
        self.user = User.objects.create_user(
            username='history_test_user',
            password='InitialPass123!',
            role=self.admin_role
        )
    
    def test_password_history_validator_prevents_reuse(self):
        """Test: Password history validator prevents reuse of recent passwords"""
        # Change password to create history
        self.user.set_password('SecondPass123!')
        self.user.save()
        
        # Try to reuse initial password - should fail
        with self.assertRaises(ValidationError) as context:
            self.validator.validate('InitialPass123!', self.user)
        
        self.assertIn('últimas 5 contraseñas', str(context.exception))
    
    def test_password_history_validator_allows_new_password(self):
        """Test: Password history validator allows completely new passwords"""
        # This should not raise an exception
        try:
            self.validator.validate('CompletelyNewPass123!', self.user)
        except ValidationError:
            self.fail("Password history validator should allow new passwords")
    
    def test_password_history_validator_skips_users_without_history(self):
        """Test: Password history validator skips users without history support"""
        # Create a mock user object without check_password_history method
        class MockUser:
            pass
        
        mock_user = MockUser()
        
        # Should not raise an exception for users without history support
        try:
            self.validator.validate('AnyPassword123!', mock_user)
        except ValidationError:
            self.fail("Password history validator should skip users without history support")
    
    def test_password_history_validator_with_none_user(self):
        """Test: Password history validator handles None user gracefully"""
        # Should not raise an exception for None user
        try:
            self.validator.validate('AnyPassword123!', None)
        except ValidationError:
            self.fail("Password history validator should handle None user gracefully")
    
    def test_full_password_validation_with_history(self):
        """Test: Full password validation includes history check"""
        # Change password several times to build history
        passwords = [
            'FirstPass123!',
            'SecondPass123!',
            'ThirdPass123!',
            'FourthPass123!',
        ]
        
        for pwd in passwords:
            self.user.set_password(pwd)
            self.user.save()
        
        # Try to reuse an old password through Django's validation system
        with self.assertRaises(ValidationError) as context:
            validate_password('FirstPass123!', self.user)
        
        # Should contain our custom error message
        error_messages = [str(error) for error in context.exception.error_list]
        self.assertTrue(any('últimas 5 contraseñas' in msg for msg in error_messages))
    
    def test_get_help_text(self):
        """Test: Password history validator provides helpful error message"""
        help_text = self.validator.get_help_text()
        self.assertIn('últimas 5 contraseñas', help_text)


class MedicalPasswordValidatorTests(TestCase):
    """Test medical-specific password validation."""
    
    def setUp(self):
        self.validator = MedicalPasswordValidator()
    
    def test_medical_password_validator_blocks_medical_terms(self):
        """Test: Medical password validator blocks common medical terms"""
        medical_passwords = [
            'ValidPass123medinet!',   # Contains 'medinet' - should be blocked
            'ValidPass123Cardio!',    # Contains 'cardio' - should be blocked
            'ValidPass123Neuro!',     # Contains 'neuro' - should be blocked  
            'Synapnetica123!',        # Contains 'synapnetica' - should be blocked
        ]
        
        for pwd in medical_passwords:
            with self.assertRaises(ValidationError) as context:
                self.validator.validate(pwd)
            self.assertIn('términos prohibidos', str(context.exception))
    
    def test_medical_password_validator_requires_complexity(self):
        """Test: Medical password validator enforces complexity rules"""
        weak_passwords = [
            'password123!',      # No uppercase
            'PASSWORD123!',      # No lowercase  
            'Password!',         # No numbers
            'Password123',       # No special chars
        ]
        
        for pwd in weak_passwords:
            with self.assertRaises(ValidationError):
                self.validator.validate(pwd)
    
    def test_medical_password_validator_accepts_strong_passwords(self):
        """Test: Medical password validator accepts strong, compliant passwords"""
        strong_passwords = [
            'SecureKey123!',
            'MyStrong123$Code',
            'CompliantAuth456#',
        ]
        
        for pwd in strong_passwords:
            try:
                self.validator.validate(pwd)
            except ValidationError:
                self.fail(f"Medical password validator should accept strong password: {pwd}")
    
    def test_get_help_text_medical_validator(self):
        """Test: Medical password validator provides comprehensive help text"""
        help_text = self.validator.get_help_text()
        self.assertIn('mayúsculas', help_text)
        self.assertIn('minúsculas', help_text)
        self.assertIn('números', help_text)
        self.assertIn('carácter especial', help_text)
        self.assertIn('términos médicos', help_text)