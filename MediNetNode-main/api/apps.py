import sys
import threading

from django.apps import AppConfig


def _warm_ml_stack():
    # Pre-import opacus (pulls torch) so the first start-client request doesn't
    # pay a 30-60s cold import inside estimate_job_epsilon and time out the Hub.
    try:
        from opacus.accountants import RDPAccountant  # noqa: F401
    except Exception:
        pass


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    verbose_name = 'API for RESEARCHER Users'

    def ready(self):
        serving = any(cmd in ' '.join(sys.argv) for cmd in ('runserver', 'gunicorn', 'uvicorn', 'daphne'))
        if serving:
            threading.Thread(target=_warm_ml_stack, daemon=True).start()
