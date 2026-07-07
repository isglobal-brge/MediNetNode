"""
Uploader registry — maps file extensions to their uploader classes.

Domain modules register their uploaders during AppConfig.ready() so the
platform can dispatch uploads without hard-coding domain-specific imports.

Example (inside genetics/apps.py)::

    def ready(self):
        from medinet_core.uploaders import uploader_registry
        from genetics.uploaders import IDAUploader, CNVUploader
        uploader_registry.register(IDAUploader)
        uploader_registry.register(CNVUploader)

Then in a view::

    from medinet_core.uploaders import uploader_registry
    uploader_cls = uploader_registry.get('.idat')
    if uploader_cls is None:
        raise Http404('No uploader registered for this file type')
    uploader = uploader_cls(request.user)
    dataset, info = uploader.upload(tmp_path, name=name, ...)
"""
import logging
from typing import Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseUploader  # pragma: no cover

logger = logging.getLogger(__name__)


class UploaderRegistry:
    """
    Thread-safe registry that maps file extensions to uploader classes.

    Extensions must be lower-case and include the leading dot (e.g. '.csv').
    Registration happens once at startup (AppConfig.ready()); there is no
    thread-safety concern at lookup time.
    """

    def __init__(self):
        self._registry: Dict[str, Type['BaseUploader']] = {}

    def register(self, uploader_cls: Type['BaseUploader']) -> None:
        """
        Register an uploader class for all extensions in its ALLOWED_EXTENSIONS.

        Raises:
            ValueError: if ``uploader_cls`` declares no ALLOWED_EXTENSIONS.
        """
        extensions = getattr(uploader_cls, 'ALLOWED_EXTENSIONS', {})
        if not extensions:
            raise ValueError(
                f"{uploader_cls.__name__}.ALLOWED_EXTENSIONS is empty. "
                "Define at least one extension before registering."
            )

        for ext in extensions:
            ext = ext.lower()
            if ext in self._registry:
                existing = self._registry[ext].__name__
                logger.warning(
                    "UploaderRegistry: extension '%s' already claimed by %s; "
                    "overriding with %s.",
                    ext, existing, uploader_cls.__name__,
                )
            self._registry[ext] = uploader_cls
            logger.debug(
                "UploaderRegistry: registered %s for '%s'",
                uploader_cls.__name__, ext,
            )

    def get(self, extension: str) -> Optional[Type['BaseUploader']]:
        """
        Return the uploader class for *extension*, or None if not registered.

        Args:
            extension: Lower-case file extension including dot (e.g. '.csv').
        """
        return self._registry.get(extension.lower())

    def extensions(self) -> list:
        """Return a sorted list of all registered extensions."""
        return sorted(self._registry)

    def __repr__(self) -> str:  # pragma: no cover
        return f"UploaderRegistry({self._registry})"


# Module-level singleton — import this instance everywhere.
uploader_registry = UploaderRegistry()
