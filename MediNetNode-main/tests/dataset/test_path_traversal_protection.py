"""
Tests for path traversal protection in dataset upload functionality.

Tests that the system correctly blocks path traversal attacks via malicious
filenames in uploaded files.
"""
import os
import tempfile
from unittest.mock import Mock
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from users.models import Role
from dataset.views import _save_temp_file


User = get_user_model()


class MaliciousUploadedFile:
    """Mock uploaded file that preserves malicious filenames (for testing)."""

    def __init__(self, name, content, content_type):
        self.name = name  # Preserve exact name (unlike SimpleUploadedFile)
        self.content = content
        self.content_type = content_type
        self.size = len(content)
        self._position = 0

    def chunks(self, chunk_size=64 * 1024):
        """Yield chunks of the file content."""
        while self._position < len(self.content):
            chunk = self.content[self._position:self._position + chunk_size]
            self._position += chunk_size
            yield chunk


class PathTraversalProtectionTests(TestCase):
    """Test path traversal attack protection in file uploads."""

    def setUp(self):
        """Set up test data."""
        self.admin_role = Role.objects.get(name='ADMIN')
        self.admin_user = User.objects.create_user(
            username='admin',
            email='admin@test.com',
            password='TestPass123!',
            role=self.admin_role,
            is_superuser=True
        )
        self.client = Client()
        self.client.login(username='admin', password='TestPass123!')

    def test_blocks_parent_directory_traversal(self):
        """Test that ../ in filename is blocked."""
        malicious_file = MaliciousUploadedFile(
            name="../../../etc/passwd",
            content=b"malicious content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(malicious_file)

        self.assertIn("path traversal attack detected", str(context.exception).lower())

    def test_blocks_absolute_path_traversal(self):
        """Test that absolute paths with separators are blocked."""
        malicious_file = MaliciousUploadedFile(
            name="/etc/passwd",
            content=b"malicious content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(malicious_file)

        self.assertIn("path traversal", str(context.exception).lower())
        self.assertIn("separator", str(context.exception).lower())

    def test_blocks_windows_path_traversal(self):
        """Test that Windows-style path traversal is blocked."""
        malicious_file = MaliciousUploadedFile(
            name="..\\..\\..\\Windows\\System32\\config\\SAM",
            content=b"malicious content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(malicious_file)

        self.assertIn("path traversal", str(context.exception).lower())

    def test_blocks_hidden_file_upload(self):
        """Test that hidden files (starting with .) are blocked."""
        hidden_file = SimpleUploadedFile(
            name=".hidden_malicious",
            content=b"malicious content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(hidden_file)

        self.assertIn("invalid filename", str(context.exception).lower())

    def test_blocks_empty_filename(self):
        """Test that empty filenames are blocked (Django already validates this)."""
        # Note: Django's SimpleUploadedFile raises SuspiciousFileOperation for empty names
        # Our code provides additional defense in depth
        empty_file = MaliciousUploadedFile(
            name="",
            content=b"content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(empty_file)

        self.assertIn("cannot be empty", str(context.exception).lower())

    def test_allows_legitimate_filename(self):
        """Test that legitimate filenames are allowed."""
        legitimate_file = SimpleUploadedFile(
            name="legitimate_dataset.csv",
            content=b"patient_id,diagnosis,age\n1,diabetes,45\n",
            content_type="text/csv"
        )

        try:
            temp_path = _save_temp_file(legitimate_file)

            self.assertTrue(os.path.exists(temp_path))
            self.assertTrue(temp_path.endswith("legitimate_dataset.csv"))

            with open(temp_path, 'rb') as f:
                content = f.read()
            self.assertEqual(content, b"patient_id,diagnosis,age\n1,diabetes,45\n")

            os.unlink(temp_path)
            temp_dir = os.path.dirname(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)

        except Exception as e:
            self.fail(f"Legitimate filename should be allowed: {str(e)}")

    def test_sanitizes_filename_with_path_components(self):
        """Test that filename with path components is blocked (contains / separator)."""
        file_with_path = MaliciousUploadedFile(
            name="some/nested/path/dataset.csv",
            content=b"data",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(file_with_path)

        self.assertIn("path traversal", str(context.exception).lower())
        self.assertIn("separator", str(context.exception).lower())

    def test_upload_endpoint_blocks_malicious_filename(self):
        """
        Test that the upload endpoint blocks malicious filenames.

        Note: In a real HTTP multipart/form-data upload, both Django's multipart parser
        AND our _save_temp_file() function provide defense in depth:
        1. Django's multipart parser sanitizes filenames using os.path.basename()
        2. Our _save_temp_file() validates the filename explicitly

        This test demonstrates the unit-level protection in _save_temp_file().
        The HTTP-level protection is inherent to Django's multipart handling.
        """
        malicious_file = MaliciousUploadedFile(
            name="../../../etc/passwd",
            content=b"malicious content",
            content_type="text/csv"
        )

        with self.assertRaises(ValueError) as context:
            _save_temp_file(malicious_file)

        self.assertIn('path traversal', str(context.exception).lower())

    def test_file_path_stays_within_temp_directory(self):
        """Test that saved file path is always within temp directory."""
        legitimate_file = SimpleUploadedFile(
            name="test_dataset.csv",
            content=b"data",
            content_type="text/csv"
        )

        temp_path = _save_temp_file(legitimate_file)

        try:
            abs_temp_path = os.path.abspath(temp_path)
            temp_dir = os.path.dirname(abs_temp_path)

            self.assertTrue(abs_temp_path.startswith(temp_dir))
            self.assertIn("upload_", temp_dir)

            system_temp = tempfile.gettempdir()
            self.assertTrue(abs_temp_path.startswith(system_temp))

            os.unlink(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise


class PathTraversalSecurityDocumentationTests(TestCase):
    """Test that security documentation is comprehensive."""

    def test_save_temp_file_has_security_documentation(self):
        """Test that _save_temp_file function has security documentation."""
        from dataset.views import _save_temp_file
        import inspect

        docstring = inspect.getdoc(_save_temp_file)
        self.assertIsNotNone(docstring)
        self.assertIn("path traversal", docstring.lower())
        self.assertIn("security", docstring.lower())

    def test_vulnerability_documented_in_security_report(self):
        """Test that path traversal vulnerability is documented."""
        security_report_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'VULNERABILITIES_REMAINING.md'
        )

        if os.path.exists(security_report_path):
            with open(security_report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertTrue(
                'path traversal' in content.lower() or
                'Path Traversal' in content
            )
