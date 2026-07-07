"""
BaseUploader — abstract base class for all MediNet file uploaders.

Usage
-----
Subclass ``BaseUploader`` in any domain module, declare ``ALLOWED_EXTENSIONS``,
and implement ``upload()``.  Register the subclass with ``uploader_registry``
inside your AppConfig.ready() so the platform can dispatch to it automatically.

Example::

    # genetics/uploaders.py
    from medinet_core.uploaders import BaseUploader, uploader_registry

    class IDAUploader(BaseUploader):
        ALLOWED_EXTENSIONS = {'.idat': ['application/octet-stream']}

        def upload(self, file_path, **kwargs):
            self._validate_file(file_path)
            # ... genetics-specific processing ...

    uploader_registry.register(IDAUploader)
"""
import abc
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import magic as _magic_lib
    _MAGIC_AVAILABLE = True
except ImportError:
    _magic_lib = None
    _MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────────────

class UploadError(Exception):
    """Base exception for all upload failures."""


class SecurityValidationError(UploadError):
    """Raised when security or privacy validation fails."""


# ── Base class ───────────────────────────────────────────────────────────────

class BaseUploader(abc.ABC):
    """
    Abstract base for platform uploaders.

    Shared responsibilities:
      - File-extension + MIME-type validation
      - SHA-256 checksum calculation
      - Progress reporting
      - Quarantine directory management
      - PHI column pattern constants

    Domain-specific uploaders must declare ``ALLOWED_EXTENSIONS`` and implement
    ``upload()``.
    """

    # ── Subclass must override ─────────────────────────────────────────────
    #: Mapping of extension → list of acceptable MIME types.
    #: e.g. {'.csv': ['text/csv', 'text/plain']}
    ALLOWED_EXTENSIONS: Dict[str, list] = {}

    # ── Platform-wide constants ────────────────────────────────────────────
    #: Minimum rows required for k-anonymity compliance.
    MIN_K_ANONYMITY: int = 5

    #: Column name patterns that may contain Protected Health Information (PHI).
    #: Checked against lowercase column names using word-boundary regex.
    FORBIDDEN_PATTERNS = [
        r'\bid\b', r'\bpatient_id\b', r'\bmrn\b', r'\bmedical_record\b',
        r'\bssn\b', r'\bsocial_security\b', r'\bname\b', r'\bfirst_name\b',
        r'\blast_name\b', r'\bemail\b', r'\bphone\b', r'\baddress\b',
        r'\bzip\b', r'\bpostal\b', r'\bbirth_date\b', r'\bdob\b',
        r'\bdate_of_birth\b',
    ]

    def __init__(self, user, progress_callback=None):
        """
        Args:
            user: Authenticated Django user performing the upload.
            progress_callback: Optional ``callable(stage: str, message: str)``
                               for progress reporting.
        """
        self.user = user
        self.progress_callback = progress_callback
        self.temp_dir: Optional[str] = None
        self.quarantine_dir: str = self._get_quarantine_dir()

    # ── Abstract interface ─────────────────────────────────────────────────

    @abc.abstractmethod
    def upload(self, file_path: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Process and persist an uploaded file.

        Args:
            file_path: Absolute path to the temporary file to process.
            **kwargs: Domain-specific keyword arguments (name, description, …).

        Returns:
            Tuple of (domain_model_instance, upload_info_dict).

        Raises:
            UploadError: On any processing failure.
            SecurityValidationError: On security or PHI validation failure.
        """

    # ── Shared utilities ───────────────────────────────────────────────────

    def _validate_file(self, file_path: str) -> None:
        """
        Validate file existence, extension, and MIME type.

        Raises:
            SecurityValidationError: if the file fails any check.
        """
        if not os.path.exists(file_path):
            raise SecurityValidationError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise SecurityValidationError(
                f"File extension '{ext}' is not allowed. "
                f"Accepted: {list(self.ALLOWED_EXTENSIONS)}"
            )

        if _MAGIC_AVAILABLE:
            detected_mime = _magic_lib.from_file(file_path, mime=True)
            allowed_mimes = self.ALLOWED_EXTENSIONS[ext]
            if detected_mime not in allowed_mimes:
                raise SecurityValidationError(
                    f"File MIME type '{detected_mime}' does not match expected "
                    f"{allowed_mimes} for extension '{ext}'. "
                    "File may have been tampered with."
                )

    def _calculate_checksum(self, file_path: str) -> str:
        """Return the SHA-256 hex digest of the file at ``file_path``."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _update_progress(self, stage: str, message: str) -> None:
        """Invoke ``progress_callback`` if one was provided."""
        if self.progress_callback:
            try:
                self.progress_callback(stage, message)
            except Exception:
                pass  # Never let a broken callback abort an upload

    def _get_quarantine_dir(self) -> str:
        """Return (and create) the quarantine directory for rejected files."""
        from django.conf import settings
        quarantine = os.path.join(str(settings.MEDIA_ROOT), 'quarantine')
        os.makedirs(quarantine, exist_ok=True)
        return quarantine

    def _is_phi_column(self, column_name: str) -> bool:
        """Return True if the column name matches any known PHI pattern."""
        lower = column_name.lower()
        return any(re.search(pat, lower) for pat in self.FORBIDDEN_PATTERNS)

    def _make_temp_dir(self) -> str:
        """Create and record a temporary working directory."""
        self.temp_dir = tempfile.mkdtemp(prefix='medinet_upload_')
        return self.temp_dir
