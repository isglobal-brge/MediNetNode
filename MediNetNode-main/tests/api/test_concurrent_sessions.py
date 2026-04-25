import sys
sys.modules.setdefault('magic', None)

import pytest
import json
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from trainings.models import TrainingSession
from users.models import Role
import uuid

User = get_user_model()
MAX_CONCURRENT = 2  # must match the constant in views.py


def _make_researcher(username):
    role = Role.objects.get(name='RESEARCHER')
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_active_session(researcher, n=1):
    for i in range(n):
        TrainingSession.objects.create(
            session_id=uuid.uuid4(),
            client_id=f"c{i}",
            user=researcher,
            dataset_id=1,
            dataset_name="ds",
            model_config={},
            server_address="hub:8080",
            status='ACTIVE',
            total_rounds=3,
        )


@pytest.mark.django_db
class TestConcurrentSessionLimit:

    def setup_method(self):
        self.researcher = _make_researcher("cs_r1")
        self.factory = RequestFactory()

    def test_rejects_when_max_concurrent_reached(self):
        _make_active_session(self.researcher, n=MAX_CONCURRENT)
        from api.views import start_client
        request = self.factory.post(
            '/api/v1/start-client/',
            data=json.dumps({'model_json': {}, 'server_address': 'hub:8080'}),
            content_type='application/json',
        )
        request.api_user = self.researcher
        request.api_key = 'test-key'

        with patch('api.views.validate_training_config', return_value=(None, {})), \
             patch('api.views.validate_training_permissions', return_value=None):
            resp = start_client(request)

        assert resp.status_code == 429
        import json as _json
        body = _json.loads(resp.content)
        assert 'simultáne' in body['error'].lower() or 'concurrent' in body['error'].lower() or 'limit' in body['error'].lower()
