"""
Integration tests for TrainingSession and TrainingRound lifecycle.

Validates session creation, status transitions, round tracking,
progress calculation, and completion/failure/cancellation flows.
"""
import pytest


@pytest.fixture
def training_session(db, integration_researcher_user):
    """Create a minimal TrainingSession in STARTING state."""
    from trainings.models import TrainingSession

    return TrainingSession.objects.create(
        client_id="test-client",
        user=integration_researcher_user,
        dataset_id=1,
        dataset_name="Test Dataset",
        model_config={"model_type": "dl"},
        server_address="192.168.1.100:8080",
        total_rounds=5,
        status="STARTING",
    )


@pytest.mark.django_db
class TestTrainingSessionInitialState:
    """Verify that a newly created session has the expected defaults."""

    def test_default_status_is_starting(self, training_session) -> None:
        assert training_session.status == "STARTING"

    def test_is_active_true_when_starting(self, training_session) -> None:
        assert training_session.is_active is True

    def test_is_finished_false_when_starting(self, training_session) -> None:
        assert training_session.is_finished is False

    def test_session_id_is_uuid(self, training_session) -> None:
        import uuid
        assert isinstance(training_session.session_id, uuid.UUID)

    def test_progress_percentage_defaults_to_zero(self, training_session) -> None:
        assert training_session.progress_percentage == 0.0

    def test_current_round_defaults_to_zero(self, training_session) -> None:
        assert training_session.current_round == 0


@pytest.mark.django_db
class TestTrainingSessionStatusTransitions:
    """Verify correct status transitions and property changes."""

    def test_mark_completed_sets_status(self, training_session) -> None:
        training_session.mark_completed(accuracy=0.9, loss=0.1)
        assert training_session.status == "COMPLETED"

    def test_mark_completed_sets_final_accuracy(self, training_session) -> None:
        training_session.mark_completed(accuracy=0.87)
        assert training_session.final_accuracy == pytest.approx(0.87)

    def test_mark_completed_sets_progress_100(self, training_session) -> None:
        training_session.mark_completed()
        assert training_session.progress_percentage == 100.0

    def test_mark_completed_is_finished(self, training_session) -> None:
        training_session.mark_completed()
        assert training_session.is_finished is True

    def test_mark_failed_sets_status(self, training_session) -> None:
        training_session.mark_failed(error_message="out of memory")
        assert training_session.status == "FAILED"

    def test_mark_failed_stores_error_message(self, training_session) -> None:
        training_session.mark_failed(error_message="OOM error")
        assert "OOM error" in training_session.error_message

    def test_mark_failed_is_finished(self, training_session) -> None:
        training_session.mark_failed()
        assert training_session.is_finished is True

    def test_cancel_active_session_succeeds(self, training_session) -> None:
        result = training_session.cancel_training()
        assert result is True
        assert training_session.status == "CANCELLED"

    def test_cancel_sets_is_finished(self, training_session) -> None:
        training_session.cancel_training()
        assert training_session.is_finished is True

    def test_cancel_completed_session_returns_false(self, training_session) -> None:
        training_session.mark_completed()
        result = training_session.cancel_training()
        assert result is False
        assert training_session.status == "COMPLETED"


@pytest.mark.django_db
class TestTrainingSessionProgressTracking:
    """Validate update_progress calculations."""

    def test_update_progress_sets_current_round(self, training_session) -> None:
        training_session.update_progress(3)
        training_session.refresh_from_db()
        assert training_session.current_round == 3

    def test_update_progress_calculates_percentage(self, training_session) -> None:
        # total_rounds=5, current_round=2 → 40%
        training_session.update_progress(2)
        training_session.refresh_from_db()
        assert training_session.progress_percentage == pytest.approx(40.0)

    def test_update_progress_persists_to_db(self, training_session) -> None:
        training_session.update_progress(5)
        fresh = type(training_session).objects.get(pk=training_session.pk)
        assert fresh.current_round == 5


@pytest.mark.django_db
class TestTrainingRoundLifecycle:
    """Validate round creation, unique_together, and complete_round()."""

    def test_round_can_be_created_for_session(self, training_session) -> None:
        from trainings.models import TrainingRound

        r = TrainingRound.objects.create(
            session=training_session,
            round_number=1,
        )
        assert r.round_number == 1

    def test_round_is_not_completed_initially(self, training_session) -> None:
        from trainings.models import TrainingRound

        r = TrainingRound.objects.create(
            session=training_session,
            round_number=1,
        )
        assert r.is_completed is False

    def test_complete_round_sets_completed_at(self, training_session) -> None:
        from trainings.models import TrainingRound

        r = TrainingRound.objects.create(
            session=training_session,
            round_number=1,
        )
        r.complete_round(loss=0.3, accuracy=0.85)
        assert r.is_completed is True

    def test_complete_round_stores_metrics(self, training_session) -> None:
        from trainings.models import TrainingRound

        r = TrainingRound.objects.create(
            session=training_session,
            round_number=1,
        )
        r.complete_round(loss=0.25, accuracy=0.9, f1_score=0.88)
        assert r.loss == pytest.approx(0.25)
        assert r.accuracy == pytest.approx(0.9)
        assert r.f1_score == pytest.approx(0.88)

    def test_complete_round_updates_session_progress(self, training_session) -> None:
        from trainings.models import TrainingRound

        r = TrainingRound.objects.create(
            session=training_session,
            round_number=3,
        )
        r.complete_round(loss=0.2)
        training_session.refresh_from_db()
        assert training_session.current_round == 3

    def test_duplicate_round_number_raises_integrity_error(self, training_session) -> None:
        from django.db import IntegrityError
        from trainings.models import TrainingRound

        TrainingRound.objects.create(session=training_session, round_number=1)
        with pytest.raises(IntegrityError):
            TrainingRound.objects.create(session=training_session, round_number=1)

    def test_multiple_rounds_ordered_by_round_number(self, training_session) -> None:
        from trainings.models import TrainingRound

        for n in [3, 1, 2]:
            TrainingRound.objects.create(session=training_session, round_number=n)

        rounds = list(training_session.rounds.values_list("round_number", flat=True))
        assert rounds == sorted(rounds)
