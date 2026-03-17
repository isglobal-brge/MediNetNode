"""
Role registry — maps role names to their permission definitions.

Domain modules register their roles during AppConfig.ready() so permission
setup commands and tests can discover all platform roles without
hard-coding them in the core app.

Example (inside genetics/apps.py)::

    def ready(self):
        from medinet_core.roles import role_registry, RoleDefinition
        role_registry.register(RoleDefinition(
            name='GENETICIST',
            display_name='Geneticist',
            permissions={
                'api.access': True,
                'dataset.view': {'scope': 'ALL'},
                'inference.execute': {'scope': 'ALL'},
                'genetics.view': True,
                'genetics.upload': True,
            },
        ))
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoleDefinition:
    """
    Descriptor for a platform role and its default permissions.

    Attributes:
        name: Canonical name stored in ``Role.name`` (e.g. 'ADMIN').
        permissions: Default permission map suitable for ``Role.permissions``.
        display_name: Human-readable label (defaults to ``name``).
        description: Optional explanation of the role's purpose.
    """
    name: str
    permissions: Dict[str, Any] = field(default_factory=dict)
    display_name: str = ''
    description: str = ''

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.capitalize()


class RoleRegistry:
    """
    Ordered registry of :class:`RoleDefinition` objects.

    Registration happens once at Django startup (AppConfig.ready()).
    """

    def __init__(self):
        self._registry: Dict[str, RoleDefinition] = {}

    def register(self, role_def: RoleDefinition) -> None:
        """
        Register *role_def*.

        Logs a warning if the name is already registered (the new definition
        wins) so that accidental double-registration is visible in logs.
        """
        if role_def.name in self._registry:
            logger.warning(
                "RoleRegistry: role '%s' already registered; overriding.",
                role_def.name,
            )
        self._registry[role_def.name] = role_def
        logger.debug("RoleRegistry: registered role '%s'.", role_def.name)

    def get(self, name: str) -> Optional[RoleDefinition]:
        """Return the :class:`RoleDefinition` for *name*, or None."""
        return self._registry.get(name)

    def all(self) -> List[RoleDefinition]:
        """Return all registered definitions in insertion order."""
        return list(self._registry.values())

    def names(self) -> List[str]:
        """Return all registered role names in insertion order."""
        return list(self._registry)

    def as_role_choices(self):
        """Return a tuple of ``(name, display_name)`` pairs for Django's choices."""
        return tuple((r.name, r.display_name) for r in self.all())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:  # pragma: no cover
        return f"RoleRegistry({self.names()})"


# Module-level singleton — import this everywhere.
role_registry = RoleRegistry()
