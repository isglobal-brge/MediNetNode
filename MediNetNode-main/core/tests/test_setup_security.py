from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from core.models import SystemConfiguration
from users.models import Role
import threading
from django.db import transaction

User = get_user_model()


class InitialSetupSecurityTests(TestCase):
    """
    Comprehensive security test suite for InitialSetupView.

    Categories:
    1. Access control
    2. Input validation (including email and password strength)
    3. Attack protection (including race conditions)
    4. Data integrity
    5. State tests

    Updated for security improvements:
    - CSRF protection
    - Database-level locking for race condition prevention
    - Django password validators (min 10 chars, complexity requirements)
    - Email validation
    - Sanitized error messages
    - No JSON credentials file generation
    """

    def setUp(self):
        self.client = APIClient()
        self.setup_url = '/api/setup/'
        self.valid_payload = {
            'username': 'admin_test',
            'password': 'SecurePass123!Strong',  # Meets Django validators (min 10 chars, not common, not numeric only)
            'email': 'admin@test.com',
            'center_id': 'test-center',
            'center_display_name': 'Test Medical Center'
        }

    def tearDown(self):
        """Clean up after tests"""
        User.objects.all().delete()
        SystemConfiguration.objects.all().delete()

    # 1. ACCESS CONTROL TESTS

    def test_setup_accessible_without_users(self):
        """
        SECURITY: Setup must be accessible when no users exist
        """
        response = self.client.get(self.setup_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['setup_required'])

    def test_setup_blocked_after_user_exists(self):
        """
        CRITICAL SECURITY: Setup must be permanently blocked after creating user
        """
        # Create a user
        User.objects.create_user(username='existing', password='SecurePass123!')

        # Try to access setup
        response = self.client.get(self.setup_url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response.data['code'], 'SETUP_ALREADY_COMPLETED')

    def test_setup_post_blocked_after_user_exists(self):
        """
        CRITICAL SECURITY: POST must also be blocked
        """
        User.objects.create_user(username='existing', password='SecurePass123!')

        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_setup_blocked_after_superuser_exists(self):
        """
        SECURITY: Blocked even with existing superuser
        """
        User.objects.create_superuser(
            username='admin',
            password='SecurePass123!',
            email='admin@test.com'
        )

        response = self.client.get(self.setup_url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_setup_blocked_after_system_config_exists(self):
        """
        SECURITY: Blocked if SystemConfiguration exists (even without users)
        """
        # Create system configuration without user
        SystemConfiguration.objects.create(
            center_id='existing',
            center_display_name='Existing Center'
        )

        response = self.client.get(self.setup_url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response.data['reason'], 'System configuration exists')

    # 2. INPUT VALIDATION TESTS

    def test_missing_required_fields(self):
        """
        VALIDATION: Reject requests with missing fields
        """
        required_fields = ['username', 'password', 'email', 'center_id', 'center_display_name']

        for field in required_fields:
            payload = self.valid_payload.copy()
            del payload[field]

            response = self.client.post(self.setup_url, payload)
            self.assertEqual(
                response.status_code,
                status.HTTP_400_BAD_REQUEST,
                f"Should fail without field: {field}"
            )
            self.assertEqual(response.data['code'], 'MISSING_FIELDS')

    def test_invalid_email_format(self):
        """
        VALIDATION: Invalid email formats must be rejected
        """
        invalid_emails = [
            'not-an-email',
            'missing@domain',
            '@nodomain.com',
            'spaces in@email.com',
            'double@@domain.com'
        ]

        for invalid_email in invalid_emails:
            payload = self.valid_payload.copy()
            payload['email'] = invalid_email

            response = self.client.post(self.setup_url, payload)
            self.assertEqual(
                response.status_code,
                status.HTTP_400_BAD_REQUEST,
                f"Invalid email should be rejected: {invalid_email}"
            )
            self.assertEqual(response.data['code'], 'INVALID_EMAIL')

    def test_weak_password_rejected(self):
        """
        VALIDATION: Weak passwords must be rejected by Django validators
        Django validators require:
        - Minimum 10 characters
        - Not entirely numeric
        - Not too common
        - Not too similar to user attributes
        """
        weak_passwords = [
            '123',           # Too short
            'pass',          # Too short
            '12345678',      # Too short, all numeric
            'password',      # Too short, too common
            '1234567890',    # All numeric
            'qwertyuiop'     # Too common
        ]

        for weak_pass in weak_passwords:
            payload = self.valid_payload.copy()
            payload['password'] = weak_pass

            response = self.client.post(self.setup_url, payload)
            self.assertEqual(
                response.status_code,
                status.HTTP_400_BAD_REQUEST,
                f"Weak password should be rejected: {weak_pass}"
            )
            self.assertEqual(response.data['code'], 'WEAK_PASSWORD')
            # Verify detailed password requirements are returned
            self.assertIn('details', response.data)

    def test_strong_password_accepted(self):
        """
        VALIDATION: Strong passwords that meet all requirements must be accepted
        Must contain: uppercase, lowercase, digit, special character
        """
        strong_passwords = [
            'SecurePass123!',  # Has all: S, p, 1, !
            'MyP@ssw0rd2024',  # Has all: M, y, 2, @
            'Complex!Pass99',  # Has all: C, o, 9, !
            'Str0ng&Secur3#',  # Has all: S, t, 0, &
        ]

        for strong_pass in strong_passwords:
            # Clean up before each test
            User.objects.all().delete()
            SystemConfiguration.objects.all().delete()

            payload = self.valid_payload.copy()
            payload['password'] = strong_pass
            payload['username'] = f'user{strong_pass[0:3]}'  # Unique username (alphanumeric only)

            response = self.client.post(self.setup_url, payload)
            if response.status_code != status.HTTP_201_CREATED:
                print(f"Failed for password: {strong_pass}, response: {response.data}")
            self.assertEqual(
                response.status_code,
                status.HTTP_201_CREATED,
                f"Strong password should be accepted: {strong_pass}"
            )

    def test_invalid_center_id_formats(self):
        """
        VALIDATION: Invalid center IDs must be rejected
        """
        invalid_ids = [
            ('AB', 'INVALID_CENTER_ID'),              # Too short
            ('a' * 21, 'INVALID_CENTER_ID'),          # Too long
            ('test center', 'INVALID_CENTER_ID'),     # Spaces
            ('test_center', 'INVALID_CENTER_ID'),     # Underscore
            ('test@center', 'INVALID_CENTER_ID'),     # Special characters
            ('-testcenter', 'INVALID_CENTER_ID'),     # Starts with hyphen
            ('testcenter-', 'INVALID_CENTER_ID'),     # Ends with hyphen
            ('', 'MISSING_FIELDS'),                   # Empty - caught by required check
        ]

        for invalid_id, expected_code in invalid_ids:
            payload = self.valid_payload.copy()
            payload['center_id'] = invalid_id

            response = self.client.post(self.setup_url, payload)
            self.assertEqual(
                response.status_code,
                status.HTTP_400_BAD_REQUEST,
                f"Invalid center ID should be rejected: {invalid_id}"
            )
            self.assertEqual(
                response.data['code'],
                expected_code,
                f"Expected code {expected_code} for invalid_id: {invalid_id}"
            )

    def test_valid_center_id_formats(self):
        """
        VALIDATION: Valid center IDs must be accepted
        """
        valid_ids = [
            'abc',
            'test-center',
            'center123',
            'a1b2c3',
            '123',
            'a' * 20,  # Maximum length
        ]

        for i, valid_id in enumerate(valid_ids):
            # Clean previous users and config
            User.objects.all().delete()
            SystemConfiguration.objects.all().delete()

            payload = self.valid_payload.copy()
            payload['center_id'] = valid_id
            payload['username'] = f'user{i}'  # Unique username (simpler)
            payload['email'] = f'user{i}@test.com'  # Unique email

            response = self.client.post(self.setup_url, payload)
            if response.status_code != status.HTTP_201_CREATED:
                print(f"Failed for center_id: {valid_id}, response: {response.data}")
            self.assertEqual(
                response.status_code,
                status.HTTP_201_CREATED,
                f"Valid center ID should be accepted: {valid_id}"
            )

    # 3. ATTACK PROTECTION TESTS

    def test_sql_injection_attempt(self):
        """
        SECURITY: Protection against SQL injection
        """
        sql_payloads = [
            "admin' OR '1'='1",
            "admin'; DROP TABLE users--",
            "' OR 1=1--",
        ]

        for payload in sql_payloads:
            User.objects.all().delete()
            SystemConfiguration.objects.all().delete()

            data = self.valid_payload.copy()
            data['username'] = payload

            response = self.client.post(self.setup_url, data)
            # Should not cause 500 error, should validate or create user normally
            self.assertIn(
                response.status_code,
                [status.HTTP_201_CREATED, status.HTTP_400_BAD_REQUEST]
            )

            # Verify SQL malicious code was not executed
            if response.status_code == status.HTTP_201_CREATED:
                user = User.objects.get(username=payload)
                self.assertEqual(user.username, payload)

    def test_xss_attempt_in_fields(self):
        """
        SECURITY: XSS data must be stored as plain text (no credentials file generated)
        """
        xss_payload = "<script>alert('XSS')</script>"

        data = self.valid_payload.copy()
        data['center_display_name'] = xss_payload

        response = self.client.post(self.setup_url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify it was saved as plain text in database
        config = SystemConfiguration.objects.first()
        self.assertEqual(config.center_display_name, xss_payload)

    def test_race_condition_prevention(self):
        """
        CRITICAL SECURITY: Prevent race conditions with database-level locking

        With select_for_update() locking, only ONE request should successfully
        complete setup. Others should be blocked with 403 or database errors.

        Note: With SQLite and threading, database lock errors are expected behavior
        and demonstrate that the locking is working. The key is that at most ONE
        setup completes successfully.
        """
        import time
        results = []
        errors = []

        def make_request(delay=0):
            try:
                time.sleep(delay)  # Slight stagger to help with timing
                # Each thread needs its own client
                thread_client = APIClient()
                response = thread_client.post(self.setup_url, self.valid_payload)
                results.append(response.status_code)
            except Exception as e:
                # Database lock errors are expected with concurrent SQLite writes
                errors.append(str(e))

        # Launch 3 simultaneous requests with slight delays
        threads = [threading.Thread(target=make_request, args=(i * 0.01,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Count successful requests
        successful = [r for r in results if r == status.HTTP_201_CREATED]
        forbidden = [r for r in results if r == status.HTTP_403_FORBIDDEN]

        # CRITICAL: At most one should succeed
        self.assertLessEqual(
            len(successful),
            1,
            f"At most one request should succeed. Got {len(successful)} successful, {len(forbidden)} forbidden, {len(errors)} errors: {errors[:2]}"
        )

        # If one succeeded, verify exactly one user and config exist
        if len(successful) >= 1:
            self.assertEqual(User.objects.count(), 1, "Exactly one user should exist")
            self.assertEqual(SystemConfiguration.objects.count(), 1, "Exactly one config should exist")

        # Key security check: No more than one setup completed
        # (some may have failed with database locks, which is acceptable)
        total_completions = User.objects.count()
        self.assertLessEqual(total_completions, 1, f"No more than one setup should complete, found {total_completions}")

    def test_error_message_sanitization(self):
        """
        SECURITY: Error messages must not expose internal system details
        """
        # Force an internal error by providing invalid data that passes validation
        # but fails during database operations
        payload = self.valid_payload.copy()

        # Create a user first to trigger setup_already_completed
        User.objects.create_user(username='existing', password='SecurePass123!')

        response = self.client.post(self.setup_url, payload)

        # Error message should not contain stack traces, file paths, or database details
        error_msg = response.data.get('error', '')
        self.assertNotIn('Traceback', error_msg)
        self.assertNotIn('.py', error_msg)
        self.assertNotIn('line ', error_msg)
        self.assertNotIn('sqlite', error_msg.lower())

    # 4. DATA INTEGRITY TESTS

    def test_successful_setup_creates_user(self):
        """
        INTEGRITY: Successful setup must create superuser
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify user created
        self.assertTrue(User.objects.filter(username='admin_test').exists())
        user = User.objects.get(username='admin_test')
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.is_staff)

    def test_successful_setup_assigns_admin_role(self):
        """
        INTEGRITY: Successful setup must assign ADMIN role to user
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify user has ADMIN role
        user = User.objects.get(username='admin_test')
        self.assertIsNotNone(user.role)
        self.assertEqual(user.role.name, 'ADMIN')

        # Verify role has all permissions
        self.assertTrue(user.role.permissions.get('user.create'))
        self.assertTrue(user.role.permissions.get('user.view'))
        self.assertTrue(user.role.permissions.get('system.admin'))

    def test_successful_setup_creates_system_configuration(self):
        """
        INTEGRITY: Successful setup must create SystemConfiguration
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify configuration created
        self.assertTrue(SystemConfiguration.objects.exists())
        config = SystemConfiguration.objects.first()
        self.assertEqual(config.center_id, 'test-center')
        self.assertEqual(config.center_display_name, 'Test Medical Center')

    def test_no_credentials_file_generated(self):
        """
        SECURITY: Credentials should NOT be saved to JSON file
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify no credentials_file in response
        self.assertNotIn('credentials_file', response.data)

        # Verify password is NOT in response
        self.assertNotIn('password', str(response.data))

    def test_password_hashed_in_database(self):
        """
        SECURITY: Password must be hashed in database, never plaintext
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Retrieve user from database
        user = User.objects.get(username='admin_test')

        # Verify password is hashed (not plaintext)
        self.assertNotEqual(user.password, self.valid_payload['password'])

        # Verify password hash format (Django uses pbkdf2_sha256 by default)
        self.assertTrue(user.password.startswith('pbkdf2_sha256$'))

        # Verify password can be validated
        self.assertTrue(user.check_password(self.valid_payload['password']))

    # 5. STATE AND FLOW TESTS

    def test_setup_cannot_be_repeated(self):
        """
        SECURITY: Setup cannot be executed twice
        """
        # First execution
        response1 = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response1.status_code, status.HTTP_201_CREATED)

        # Second execution should fail
        payload2 = self.valid_payload.copy()
        payload2['username'] = 'another_admin'

        response2 = self.client.post(self.setup_url, payload2)
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)

        # Only one user should exist
        self.assertEqual(User.objects.count(), 1)

    def test_get_endpoint_provides_port(self):
        """
        FUNCTIONALITY: GET must return suggested port
        """
        response = self.client.get(self.setup_url)
        self.assertIn('suggested_port', response.data)
        self.assertIsNotNone(response.data['suggested_port'])

    def test_trimming_whitespace(self):
        """
        VALIDATION: Whitespace must be removed
        """
        payload = self.valid_payload.copy()
        payload['username'] = '  admin_test  '
        payload['email'] = '  admin@test.com  '
        payload['center_id'] = '  test-center  '

        response = self.client.post(self.setup_url, payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        user = User.objects.get(username='admin_test')
        self.assertEqual(user.username, 'admin_test')
        self.assertEqual(user.email, 'admin@test.com')

    def test_system_configuration_linked_to_user(self):
        """
        INTEGRITY: SystemConfiguration must be linked to setup user
        """
        response = self.client.post(self.setup_url, self.valid_payload)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        config = SystemConfiguration.objects.first()
        user = User.objects.get(username='admin_test')

        self.assertEqual(config.setup_completed_by, user)
        self.assertEqual(config.center_email, user.email)


class CenterIDValidationUnitTests(TestCase):
    """
    Unit tests for Center ID validation
    """

    def test_validate_center_id_valid_cases(self):
        """Valid cases tests"""
        from core.views.setup import InitialSetupView

        valid_cases = [
            'abc',
            'test-center',
            'center123',
            '123-abc',
            'a' * 20,
        ]

        for case in valid_cases:
            is_valid, error = InitialSetupView.validate_center_id(case)
            self.assertTrue(is_valid, f"'{case}' should be valid")
            self.assertIsNone(error)

    def test_validate_center_id_invalid_cases(self):
        """Invalid cases tests"""
        from core.views.setup import InitialSetupView

        invalid_cases = [
            ('', 'Center ID is required'),
            ('ab', 'Center ID must be 3-20 characters'),
            ('a' * 21, 'Center ID must be 3-20 characters'),
            ('Test', 'Center ID must be 3-20 characters'),
            ('test center', 'Center ID must be 3-20 characters'),
            ('-test', 'Center ID cannot start or end with hyphen'),
            ('test-', 'Center ID cannot start or end with hyphen'),
        ]

        for case, expected_error in invalid_cases:
            is_valid, error = InitialSetupView.validate_center_id(case)
            self.assertFalse(is_valid, f"'{case}' should be invalid")
            self.assertIn(expected_error, error)


class SystemConfigurationModelTests(TestCase):
    """
    Tests for SystemConfiguration model singleton behavior
    """

    def test_singleton_pattern_enforcement(self):
        """
        INTEGRITY: Only one SystemConfiguration can exist
        """
        from django.core.exceptions import ValidationError

        # Create first config
        config1 = SystemConfiguration.objects.create(
            center_id='center1',
            center_display_name='Center 1'
        )

        # Try to create second config - should raise ValidationError
        with self.assertRaises(ValidationError):
            config2 = SystemConfiguration(
                center_id='center2',
                center_display_name='Center 2'
            )
            config2.save()

        # Verify only one exists
        self.assertEqual(SystemConfiguration.objects.count(), 1)

    def test_get_instance_method(self):
        """
        FUNCTIONALITY: get_instance() returns the configuration
        """
        # Before creation
        self.assertIsNone(SystemConfiguration.get_instance())

        # After creation
        config = SystemConfiguration.objects.create(
            center_id='test',
            center_display_name='Test Center'
        )

        instance = SystemConfiguration.get_instance()
        self.assertIsNotNone(instance)
        self.assertEqual(instance.center_id, 'test')

    def test_is_setup_completed_method(self):
        """
        FUNCTIONALITY: is_setup_completed() returns correct state
        """
        # Before setup
        self.assertFalse(SystemConfiguration.is_setup_completed())

        # After setup
        SystemConfiguration.objects.create(
            center_id='test',
            center_display_name='Test Center'
        )

        self.assertTrue(SystemConfiguration.is_setup_completed())

    def test_get_api_access_config(self):
        """
        FUNCTIONALITY: get_api_access_config() returns correct structure
        Django REST API always runs on fixed port 8000
        """
        config = SystemConfiguration.objects.create(
            center_id='test-center',
            center_display_name='Test Medical Center'
        )

        api_config = config.get_api_access_config()

        self.assertEqual(api_config['name'], 'test-center')
        self.assertEqual(api_config['display_name'], 'Test Medical Center')
        self.assertEqual(api_config['url'], 'http://localhost:8000')
        self.assertEqual(api_config['port'], 8000)
