from django.db import models
from django.core.exceptions import ValidationError


class SystemConfiguration(models.Model):
    """
    Configuración global del sistema - Singleton pattern.

    Este modelo almacena configuraciones del centro médico y del sistema
    que pueden ser extendidas en el futuro sin modificar el código.

    Solo puede existir una instancia de este modelo.
    """

    # Identificación del Centro Médico
    center_id = models.CharField(
        max_length=20,
        unique=True,
        help_text="Identificador único del centro en la red MediNet (lowercase, alfanumérico con guiones)"
    )
    center_display_name = models.CharField(
        max_length=200,
        help_text="Nombre completo del centro médico para mostrar"
    )
    center_email = models.EmailField(
        blank=True,
        null=True,
        help_text="Email de contacto del centro médico"
    )

    # Configuración Extensible (JSON)
    extra_settings = models.JSONField(
        default=dict,
        blank=True,
        help_text="Configuraciones adicionales en formato JSON (extensible para futuras features)"
    )

    # Metadatos
    setup_completed_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Fecha y hora en que se completó el setup inicial"
    )
    setup_completed_by = models.ForeignKey(
        'users.CustomUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='system_setup',
        help_text="Usuario que completó el setup inicial"
    )
    last_modified = models.DateTimeField(
        auto_now=True,
        help_text="Última modificación de la configuración"
    )

    class Meta:
        verbose_name = "System Configuration"
        verbose_name_plural = "System Configuration"

    def save(self, *args, **kwargs):
        """
        Enforce singleton pattern: solo puede existir una configuración.
        """
        if not self.pk and SystemConfiguration.objects.exists():
            raise ValidationError(
                "Solo puede existir una configuración del sistema. "
                "Edite la configuración existente en lugar de crear una nueva."
            )

        super().save(*args, **kwargs)

    @classmethod
    def get_instance(cls):
        """
        Obtener la instancia única de configuración.
        Retorna None si no existe.
        """
        return cls.objects.first()

    @classmethod
    def is_setup_completed(cls):
        """
        Verificar si el setup inicial ya fue completado.
        """
        return cls.objects.exists()

    def get_api_access_config(self):
        """
        Generate api_access configuration for credentials JSON.

        Django REST API always runs on port 8000.
        """
        return {
            "name": self.center_id,
            "display_name": self.center_display_name,
            "url": "http://localhost:8000",  # Django REST API fixed URL
            "port": 8000  # Django REST API fixed port
        }

    def __str__(self):
        return f"MediNet Configuration - {self.center_display_name} ({self.center_id})"
