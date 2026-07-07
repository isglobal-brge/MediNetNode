from django.apps import AppConfig


class MediNetCoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'medinet_core'
    verbose_name = 'MediNet Core'

    def ready(self):
        from medinet_core.roles.base_roles import register_base_roles
        register_base_roles()
