"""
Lightweight process launcher for the Flower federated-learning client.

The heavy ML stack (torch / flwr / HuggingFace datasets) lives in ``client.py``
and must be imported *inside* the spawned child process, never in the Django
request path. Importing it in the ``/api/v2/start-client`` handler blocks the
HTTP response on a cold torch import (10-30 s), which trips the Hub's activation
read-timeout and makes the center report as "not activated".

Keeping the target here — a module with no top-level ML imports — means the
request handler only pays for spawning the process, and the expensive import
happens in the background child.

Django bootstrap (important for Windows ``spawn``):
    The multiprocessing machinery unpickles the ``Process`` object's target
    (this module) *before* its arguments. One of those arguments is a Django
    ``User`` model instance, whose unpickling calls ``apps.get_model()`` and
    therefore requires a ready app registry. By calling ``django.setup()`` at
    import time we guarantee the registry is populated before the ``user`` arg
    is unpickled — otherwise the child dies with ``AppRegistryNotReady``. In the
    parent (already-configured) Django process the guard makes this a no-op.
"""

import os

import django
from django.apps import apps

# Inherited from the parent process on spawn; keep a sensible default for safety.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.medinet")

if not apps.ready:
    django.setup()


def run_flower_client(model_json, server_address="localhost:8080", client_id=None,
                      user=None, session_id=None, ca_cert=None):
    """Child-process entry point: import the heavy client lazily, then run it."""
    from . import client

    client.start_flower_client(
        model_json,
        server_address=server_address,
        client_id=client_id,
        user=user,
        session_id=session_id,
        ca_cert=ca_cert,
    )
