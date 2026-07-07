import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from trainings.models import BudgetResetRequest
from users.models import Role

User = get_user_model()


def _make_user(username, role_name):
    role = Role.objects.get(name=role_name)
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


@pytest.mark.django_db
class TestBudgetResetRequest:

    def setup_method(self):
        self.researcher = _make_user("brr_r1", "RESEARCHER")
        self.admin = _make_user("brr_a1", "ADMIN")

    def test_create_pending_request(self):
        req = BudgetResetRequest.objects.create(
            dataset_id=42,
            researcher_id=self.researcher.id,
            reason="Nuevo proyecto aprobado por comité de ética.",
        )
        assert req.status == 'pending'
        assert req.reviewed_by_id is None
        assert req.reviewed_at is None

    def test_approve_sets_status_and_reviewer(self):
        req = BudgetResetRequest.objects.create(
            dataset_id=42,
            researcher_id=self.researcher.id,
            reason="Motivo válido.",
        )
        req.approve(admin=self.admin, notes="Aprobado.")
        req.refresh_from_db()
        assert req.status == 'approved'
        assert req.reviewed_by_id == self.admin.id
        assert req.reviewed_at is not None
        assert req.review_notes == "Aprobado."

    def test_reject_sets_status_and_reviewer(self):
        req = BudgetResetRequest.objects.create(
            dataset_id=42,
            researcher_id=self.researcher.id,
            reason="Motivo.",
        )
        req.reject(admin=self.admin, notes="No procede.")
        req.refresh_from_db()
        assert req.status == 'rejected'
        assert req.reviewed_by_id == self.admin.id

    def test_cannot_approve_already_reviewed(self):
        req = BudgetResetRequest.objects.create(
            dataset_id=42,
            researcher_id=self.researcher.id,
            reason="x",
        )
        req.approve(admin=self.admin, notes="ok")
        with pytest.raises(ValueError, match="ya ha sido revisada"):
            req.approve(admin=self.admin, notes="ok")

    def test_only_one_pending_per_researcher_dataset(self):
        BudgetResetRequest.objects.create(
            dataset_id=42, researcher_id=self.researcher.id, reason="x"
        )
        with pytest.raises(Exception):
            BudgetResetRequest.objects.create(
                dataset_id=42, researcher_id=self.researcher.id, reason="y"
            )
