"""Lightweight launcher for the Flower client, kept free of top-level ML imports
so the heavy torch/flwr import happens in the spawned child, not in the request."""

import os

import django
from django.apps import apps

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.medinet")

if not apps.ready:
    django.setup()


class _Tee:
    """Duplicate a stream to the console and a per-session log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def run_flower_client(model_json, server_address="localhost:8080", client_id=None,
                      user_id=None, session_id=None, ca_cert=None):
    # user_id is passed as an int, not a User instance: a model pickled across
    # Windows 'spawn' unpickles before django.setup() and raises AppRegistryNotReady.
    import sys
    from django.conf import settings

    # All child output (prints + Flower logs) -> logs/flower_client_<session>.log
    log_dir = os.path.join(str(settings.BASE_DIR), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tag = session_id or client_id or 'unknown'
    log_file = open(os.path.join(log_dir, f'flower_client_{tag}.log'),
                    'a', encoding='utf-8', errors='backslashreplace', buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    from . import client

    user = None
    if user_id is not None:
        from django.contrib.auth import get_user_model

        try:
            user = get_user_model().objects.get(pk=user_id)
        except get_user_model().DoesNotExist:
            user = None

    client.start_flower_client(
        model_json,
        server_address=server_address,
        client_id=client_id,
        user=user,
        session_id=session_id,
        ca_cert=ca_cert,
    )
