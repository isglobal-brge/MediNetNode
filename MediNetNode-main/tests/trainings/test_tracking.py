"""
Tests for the training-session tracking helpers in ``api.federated.utils``.

Replaces the old tests that targeted the removed ``api.federated.torch_client``
module. The tracking functions now live in ``api.federated.utils`` with
explicit-argument signatures (no module-level globals), so these tests exercise
that current API directly.
"""
import uuid
from unittest.mock import patch

from django.test import TestCase
from django.contrib.auth import get_user_model

from trainings.models import TrainingSession
from users.models import Role

User = get_user_model()


def _make_session(user, *, status='STARTING', total_rounds=4):
    return TrainingSession.objects.create(
        session_id=uuid.uuid4(),
        client_id="test-client",
        user=user,
        dataset_id=1,
        dataset_name="test dataset",
        model_config={},
        server_address="testserver:9090",
        status=status,
        total_rounds=total_rounds,
        process_id=1234,
    )


class TrainingTrackingTests(TestCase):
    # complete_/fail_training_session call _record_privacy_spend, which reads
    # the privacy policy / researcher budget from datasets_db.
    databases = {'default', 'datasets_db'}

    def setUp(self):
        role = Role.objects.get(name='RESEARCHER')
        self.user = User.objects.create_user(
            username='tracker', password='x', role=role
        )

    # ------------------------------------------------------------------ update
    def test_update_training_progress_activates_and_tracks_round(self):
        from api.federated.utils import update_training_progress
        session = _make_session(self.user, status='STARTING', total_rounds=4)

        update_training_progress(session, round_number=1, current_process=None)

        session.refresh_from_db()
        self.assertEqual(session.status, 'ACTIVE')
        self.assertEqual(session.current_round, 1)
        self.assertAlmostEqual(session.progress_percentage, 25.0)

    def test_update_training_progress_none_session_is_noop(self):
        from api.federated.utils import update_training_progress
        # Must not raise when there is no session to track.
        update_training_progress(None, round_number=1, current_process=None)

    # ---------------------------------------------------------------- complete
    def test_complete_training_session_with_metrics(self):
        from api.federated.utils import complete_training_session
        session = _make_session(self.user, status='ACTIVE')

        complete_training_session(session, final_metrics={'accuracy': 0.9, 'loss': 0.1})

        session.refresh_from_db()
        self.assertEqual(session.status, 'COMPLETED')

    def test_complete_training_session_without_metrics(self):
        from api.federated.utils import complete_training_session
        session = _make_session(self.user, status='ACTIVE')

        complete_training_session(session)

        session.refresh_from_db()
        self.assertEqual(session.status, 'COMPLETED')

    def test_complete_training_session_none_is_noop(self):
        from api.federated.utils import complete_training_session
        complete_training_session(None)  # must not raise

    # -------------------------------------------------------------------- fail
    def test_fail_training_session_marks_failed(self):
        from api.federated.utils import fail_training_session
        session = _make_session(self.user, status='ACTIVE')

        fail_training_session(session, error_message="boom")

        session.refresh_from_db()
        self.assertEqual(session.status, 'FAILED')

    def test_fail_training_session_none_is_noop(self):
        from api.federated.utils import fail_training_session
        fail_training_session(None, error_message="boom")  # must not raise

    # --------------------------------------------------- graceful degradation
    @patch('api.federated.utils.DJANGO_AVAILABLE', False)
    def test_helpers_noop_when_django_unavailable(self):
        from api.federated.utils import complete_training_session
        session = _make_session(self.user, status='ACTIVE')

        complete_training_session(session, final_metrics={'accuracy': 0.5})

        session.refresh_from_db()
        self.assertEqual(session.status, 'ACTIVE')  # unchanged: helper short-circuited
