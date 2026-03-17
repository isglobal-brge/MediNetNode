"""
Abstract base model for upload-tracking records across MediNet modules.

Why abstract?
  The platform uses two SQLite databases (default / datasets_db).
  Domain models (dataset.Dataset, genetics.EpiSignature) each live in their
  own DB, so a concrete cross-DB foreign key is not feasible.  An abstract
  model lets each domain inherit the standard field set and helper methods
  without creating a shared DB table or requiring a router change.

Usage in a domain model:
    from medinet_core.models import BaseUploadRecord

    class Dataset(BaseUploadRecord):
        # domain-specific fields ...

        class Meta(BaseUploadRecord.Meta):
            app_label = 'dataset'
"""
import hashlib
import os
from django.db import models
from django.core.exceptions import ValidationError


class BaseUploadRecord(models.Model):
    """
    Shared abstract base for file-upload records.

    Provides the minimal common interface that every upload-tracking model
    in the platform must implement:
      - file_path / file_size storage
      - SHA-256 checksum calculation and storage
      - uploaded_by_id as a plain integer (cross-DB safe)
      - soft-delete via is_active
      - utility methods shared across modules
    """

    # ── File fields ────────────────────────────────────────────────────────
    file_path = models.CharField(
        max_length=500,
        help_text="Absolute filesystem path to the stored file",
    )
    file_size = models.BigIntegerField(
        help_text="File size in bytes",
    )

    # ── Integrity ──────────────────────────────────────────────────────────
    checksum_sha256 = models.CharField(
        max_length=64,
        editable=False,
        null=True,
        blank=True,
        help_text="SHA-256 hex digest for file integrity verification",
    )

    # ── Ownership — plain int avoids cross-DB FK constraint ────────────────
    uploaded_by_id = models.IntegerField(
        help_text="PK of the CustomUser (users_logs.sqlite3) who uploaded this file",
    )

    # ── Audit ──────────────────────────────────────────────────────────────
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True
        ordering = ['-uploaded_at']

    # ── Helper methods ─────────────────────────────────────────────────────

    def calculate_checksum(self) -> str:
        """
        Compute and return the SHA-256 hex digest of the stored file.

        Raises:
            ValidationError: if the file does not exist at ``file_path``.
        """
        if not os.path.exists(self.file_path):
            raise ValidationError(f"File not found: {self.file_path}")

        sha256 = hashlib.sha256()
        with open(self.file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_file_size_display(self) -> str:
        """Return a human-readable file size string (B / KB / MB / GB / TB)."""
        size = float(self.file_size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def ensure_checksum(self) -> None:
        """
        Calculate and persist the SHA-256 checksum if it is not yet set.

        Intended to be called from ``save()`` overrides in concrete models:

            def save(self, *args, **kwargs):
                if self.file_path and os.path.exists(self.file_path):
                    self.ensure_checksum()
                super().save(*args, **kwargs)
        """
        if not self.checksum_sha256 and self.file_path and os.path.exists(self.file_path):
            self.checksum_sha256 = self.calculate_checksum()
