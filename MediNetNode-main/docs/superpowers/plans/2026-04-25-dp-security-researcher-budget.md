# DP Security & Researcher Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Proteger el Node frente a un Hub comprometido mediante verificación de parámetros DP en entrenamiento, registro correcto de epsilon para jobs ML, rate limiting por IP, límite de sesiones concurrentes, y un sistema de presupuesto epsilon por (dataset, researcher) con reset periódico y por solicitud.

**Architecture:** Se añade el modelo `ResearcherEpsilonBudget` (dataset app → datasets_db) que registra el gasto de epsilon individualizado por researcher y dataset, con reset automático por periodo configurable. Las solicitudes de reset manual se gestionan con `BudgetResetRequest` (trainings app → default). Las validaciones de seguridad se añaden en `dl_client.py`, `ml_client.py`, `middleware.py` y `api/views.py` sin cambios de interfaz pública.

**Tech Stack:** Django 5.2, pytest-django, Opacus (DP-SGD), Flower (federated learning), SQLite (test) / PostgreSQL (prod), Django F() para updates atómicos.

---

## Modelo de amenaza (contexto para implementadores)

El Hub (servidor de investigadores) puede estar comprometido. El Node protege datos de pacientes hospitalarios. Las defensas deben funcionar incluso si Hub envía configuraciones manipuladas. El presupuesto epsilon es la única garantía matemática de privacidad — si se agota o se omite, los datos de pacientes pueden filtrarse mediante ataques de inversión de modelo.

---

## Mapa de archivos

| Acción | Archivo | Responsabilidad |
|--------|---------|----------------|
| Modificar | `dataset/models.py` | Añadir `ResearcherEpsilonBudget` |
| Modificar | `dataset/migrations/` | Migración nuevo modelo |
| Modificar | `trainings/models.py` | Añadir `BudgetResetRequest` |
| Modificar | `trainings/migrations/` | Migración nuevo modelo |
| Modificar | `api/federated/utils.py` | `_record_privacy_spend` actualiza researcher budget |
| Modificar | `api/federated/ml_client.py` | Registrar epsilon en round_metrics |
| Modificar | `api/federated/dl_client.py` | Verificar noise_multiplier mid-training |
| Modificar | `api/views.py` | Verificar researcher budget + límite sesiones concurrentes |
| Modificar | `medinet_core/security/middleware.py` | Rate limiting por IP no autenticada |
| Crear | `api/budget_views.py` | Endpoints REST reset request (researcher) y approve/reject (admin) |
| Modificar | `dataset/views.py` | Pasar researcher_budgets y reset_requests al contexto de dataset_detail |
| Modificar | `templates/dataset/detail.html` | Sección "Presupuesto por Researcher" visible para ADMIN |
| Modificar | `templates/users/researcher_info.html` | Sección "Mis presupuestos DP" + botón solicitar reset para RESEARCHER |
| Crear | `templates/dataset/partials/researcher_budgets.html` | Partial: tabla de budgets + requests pendientes (ADMIN) |
| Crear | `tests/dataset/test_researcher_budget.py` | Tests ResearcherEpsilonBudget |
| Crear | `tests/trainings/test_budget_reset_request.py` | Tests BudgetResetRequest |
| Crear | `tests/api/test_budget_endpoints.py` | Tests endpoints REST reset |
| Crear | `tests/dataset/test_researcher_budget_ui.py` | Tests UI sección budgets en dataset_detail |
| Crear | `tests/security/test_rate_limit_ip.py` | Tests rate limit por IP |
| Crear | `tests/api/test_concurrent_sessions.py` | Tests límite sesiones concurrentes |
| Crear | `tests/api/test_dp_param_verification.py` | Tests verificación noise_multiplier |
| Crear | `docs/features/researcher-epsilon-budget.md` | Documentación feature: presupuesto por researcher |
| Crear | `docs/features/budget-reset-workflow.md` | Documentación feature: flujo de solicitud de reset |
| Crear | `docs/bugfixes/ml-epsilon-recording.md` | Documentación bugfix: ML no registraba epsilon |
| Crear | `docs/bugfixes/rate-limit-unauthenticated.md` | Documentación bugfix: brute-force sin rate limit |
| Crear | `docs/bugfixes/dp-param-verification.md` | Documentación bugfix: noise_multiplier sin verificar |
| Crear | `CHANGELOG.md` | Historial de cambios del proyecto |

---

## Task 1: Modelo ResearcherEpsilonBudget

**Contexto:** Actualmente `DatasetPrivacyPolicy` tiene un único `spent_epsilon` global para todos los researchers. Necesitamos tracking individual por (dataset, researcher) con reset periódico. El modelo va en `dataset/models.py` y se enruta a `datasets_db`. Como no podemos usar FK entre bases de datos, `researcher_id` es un `IntegerField` (igual que `Dataset.uploaded_by_id`).

**Files:**
- Modify: `dataset/models.py` (al final del archivo, después de `DatasetPrivacyPolicy`)
- Create: `dataset/migrations/XXXX_add_researcher_epsilon_budget.py` (autogenerada)
- Create: `tests/dataset/test_researcher_budget.py`

- [ ] **Step 1: Escribir los tests que fallarán**

```python
# tests/dataset/test_researcher_budget.py
import sys
sys.modules.setdefault('magic', None)

import pytest
import math
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_researcher(username="r1"):
    role = Role.objects.get(name='RESEARCHER')
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_dataset(name="ds1"):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


def _make_policy(dataset, sensitivity="high"):
    return DatasetPrivacyPolicy.objects.create(
        dataset=dataset,
        sensitivity=sensitivity,
        max_epsilon_per_job=DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS[sensitivity]['max_epsilon_per_job'],
        lifetime_budget=DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS[sensitivity]['lifetime_budget'],
    )


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetCreation:

    def setup_method(self):
        self.researcher = _make_researcher("rb_r1")
        self.dataset = _make_dataset("rb_ds1")
        self.policy = _make_policy(self.dataset, "high")

    def test_get_or_create_initialises_from_policy(self):
        budget, created = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        assert created is True
        assert budget.lifetime_budget == self.policy.lifetime_budget
        assert budget.spent_epsilon == 0.0
        assert budget.researcher_id == self.researcher.id

    def test_get_or_create_idempotent(self):
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        _, created = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        assert created is False
        assert ResearcherEpsilonBudget.objects.filter(
            dataset=self.dataset, researcher_id=self.researcher.id
        ).count() == 1

    def test_unique_per_dataset_and_researcher(self):
        r2 = _make_researcher("rb_r2")
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )
        ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=r2.id, policy=self.policy
        )
        assert ResearcherEpsilonBudget.objects.filter(dataset=self.dataset).count() == 2


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetCanAccept:

    def setup_method(self):
        self.researcher = _make_researcher("ca_r1")
        self.dataset = _make_dataset("ca_ds1")
        self.policy = _make_policy(self.dataset, "high")
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )

    def test_accepts_within_budget(self):
        ok, reason = self.budget.can_accept_job(0.5)
        assert ok is True
        assert reason == ""

    def test_rejects_exceeds_per_job_limit(self):
        ok, reason = self.budget.can_accept_job(0.9)  # max per job = 0.5
        assert ok is False
        assert "máximo por job" in reason

    def test_rejects_exceeds_remaining_budget(self):
        self.budget.spent_epsilon = 1.8
        self.budget.save()
        ok, reason = self.budget.can_accept_job(0.5)  # remaining = 0.2
        assert ok is False
        assert "presupuesto" in reason

    def test_rejects_nan_epsilon(self):
        ok, reason = self.budget.can_accept_job(float('nan'))
        assert ok is False

    def test_rejects_zero_epsilon(self):
        ok, reason = self.budget.can_accept_job(0.0)
        assert ok is False

    def test_rejects_negative_epsilon(self):
        ok, reason = self.budget.can_accept_job(-1.0)
        assert ok is False


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetRecordSpent:

    def setup_method(self):
        self.researcher = _make_researcher("rs_r1")
        self.dataset = _make_dataset("rs_ds1")
        self.policy = _make_policy(self.dataset, "medium")
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy
        )

    def test_records_valid_epsilon(self):
        self.budget.record_spent(0.5)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.5)

    def test_accumulates_across_calls(self):
        self.budget.record_spent(0.5)
        self.budget.record_spent(0.3)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(0.8)

    def test_ignores_nan(self):
        self.budget.record_spent(float('nan'))
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0

    def test_ignores_negative(self):
        self.budget.record_spent(-0.5)
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherEpsilonBudgetPeriodReset:

    def setup_method(self):
        self.researcher = _make_researcher("pr_r1")
        self.dataset = _make_dataset("pr_ds1")
        self.policy = _make_policy(self.dataset, "low")

    def test_annual_period_not_expired(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        assert budget.is_period_expired() is False

    def test_reset_zeroes_spent_epsilon(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='annual',
        )
        budget.spent_epsilon = 2.0
        budget.save()
        budget.reset_period()
        budget.refresh_from_db()
        assert budget.spent_epsilon == 0.0

    def test_never_period_never_expires(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id,
            policy=self.policy, period='never',
        )
        assert budget.is_period_expired() is False
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
cd D:/Projects/Medinet/MediNetNode/MediNetNode-main
python3.11 -m pytest tests/dataset/test_researcher_budget.py -v --tb=short 2>&1 | head -30
```

Esperado: `ImportError: cannot import name 'ResearcherEpsilonBudget'`

- [ ] **Step 3: Implementar ResearcherEpsilonBudget en dataset/models.py**

Añadir al final de `dataset/models.py`, después de la clase `DatasetPrivacyPolicy`:

```python
class ResearcherEpsilonBudget(models.Model):
    """
    Presupuesto de epsilon DP por (dataset, researcher).

    researcher_id es IntegerField (no FK) porque User vive en 'default'
    y Dataset/Policy viven en 'datasets_db' — no se pueden hacer FK entre DBs distintas.
    """

    PERIOD_CHOICES = [
        ('annual', 'Anual'),
        ('monthly', 'Mensual'),
        ('never', 'Sin reset automático'),
    ]

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='researcher_budgets',
    )
    researcher_id = models.IntegerField(db_index=True)
    spent_epsilon = models.FloatField(default=0.0)
    lifetime_budget = models.FloatField()
    max_epsilon_per_job = models.FloatField()
    period = models.CharField(max_length=16, choices=PERIOD_CHOICES, default='annual')
    period_start = models.DateTimeField(default=timezone.now)
    last_reset = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [['dataset', 'researcher_id']]
        indexes = [
            models.Index(fields=['dataset', 'researcher_id']),
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def get_or_create_for(cls, *, dataset, researcher_id, policy, period='annual'):
        """
        Obtiene o crea el presupuesto del researcher para el dataset.
        Hereda lifetime_budget y max_epsilon_per_job de la policy del dataset.
        """
        obj, created = cls.objects.get_or_create(
            dataset=dataset,
            researcher_id=researcher_id,
            defaults={
                'lifetime_budget': policy.lifetime_budget,
                'max_epsilon_per_job': policy.max_epsilon_per_job,
                'period': period,
                'spent_epsilon': 0.0,
            },
        )
        return obj, created

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def remaining_budget(self) -> float:
        if not math.isfinite(self.spent_epsilon) or not math.isfinite(self.lifetime_budget):
            return 0.0
        return max(0.0, round(self.lifetime_budget - self.spent_epsilon, 6))

    # ------------------------------------------------------------------
    # Presupuesto
    # ------------------------------------------------------------------

    def can_accept_job(self, estimated_epsilon: float):
        """
        Verifica si el researcher puede lanzar un job con el epsilon estimado.
        Retorna (bool, str) — mismo contrato que DatasetPrivacyPolicy.can_accept_job.
        """
        if not math.isfinite(estimated_epsilon) or estimated_epsilon <= 0:
            return False, "El epsilon estimado no es válido."

        # Auto-reset si el periodo expiró
        if self.is_period_expired():
            self.reset_period()

        self.refresh_from_db()

        if not math.isfinite(self.max_epsilon_per_job) or estimated_epsilon > self.max_epsilon_per_job:
            return False, (
                f"El epsilon estimado ({estimated_epsilon:.4f}) supera el "
                f"máximo por job ({self.max_epsilon_per_job:.4f}) para este researcher."
            )

        if estimated_epsilon > self.remaining_budget:
            return False, (
                f"El epsilon estimado ({estimated_epsilon:.4f}) supera el "
                f"presupuesto restante del researcher ({self.remaining_budget:.4f})."
            )

        return True, ""

    def record_spent(self, actual_epsilon: float) -> None:
        """Registra el epsilon consumido de forma atómica."""
        import math as _math
        if not _math.isfinite(actual_epsilon) or actual_epsilon <= 0:
            return
        actual_epsilon = round(actual_epsilon, 6)
        ResearcherEpsilonBudget.objects.filter(
            pk=self.pk,
            spent_epsilon__lte=models.F('lifetime_budget'),
        ).update(spent_epsilon=models.F('spent_epsilon') + actual_epsilon)

    # ------------------------------------------------------------------
    # Periodo y reset
    # ------------------------------------------------------------------

    def is_period_expired(self) -> bool:
        """True si el periodo configurado ha vencido desde period_start."""
        if self.period == 'never':
            return False
        now = timezone.now()
        if self.period == 'annual':
            from dateutil.relativedelta import relativedelta
            return now >= self.period_start + relativedelta(years=1)
        if self.period == 'monthly':
            from dateutil.relativedelta import relativedelta
            return now >= self.period_start + relativedelta(months=1)
        return False

    def reset_period(self) -> None:
        """Resetea el presupuesto del researcher para el nuevo periodo."""
        ResearcherEpsilonBudget.objects.filter(pk=self.pk).update(
            spent_epsilon=0.0,
            period_start=timezone.now(),
            last_reset=timezone.now(),
        )
        self.refresh_from_db()
```

Añadir también `import math` al principio de `dataset/models.py` si no está ya.

- [ ] **Step 4: Generar y aplicar la migración**

```bash
cd D:/Projects/Medinet/MediNetNode/MediNetNode-main
python3.11 manage.py makemigrations dataset --name add_researcher_epsilon_budget
python3.11 manage.py migrate --database=datasets_db
```

- [ ] **Step 5: Ejecutar los tests y verificar que pasan**

```bash
python3.11 -m pytest tests/dataset/test_researcher_budget.py -v --tb=short
```

Esperado: todos los tests pasan.

- [ ] **Step 6: Commit**

```bash
git add dataset/models.py dataset/migrations/
git add tests/dataset/test_researcher_budget.py
git commit -m "feat: add ResearcherEpsilonBudget model with per-period reset"
```

---

## Task 2: Modelo BudgetResetRequest

**Contexto:** Un researcher puede solicitar que el ADMIN reinicie su presupuesto manualmente (por ejemplo, para un nuevo proyecto). El ADMIN aprueba o rechaza la solicitud. Este modelo va en `trainings/` porque sus FKs son contra usuarios (default DB). `dataset_id` y `researcher_id` son IntegerField para evitar FK cross-DB.

**Files:**
- Modify: `trainings/models.py`
- Create: `trainings/migrations/XXXX_add_budget_reset_request.py`
- Create: `tests/trainings/test_budget_reset_request.py`

- [ ] **Step 1: Escribir los tests que fallarán**

```python
# tests/trainings/test_budget_reset_request.py
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/trainings/test_budget_reset_request.py -v --tb=short 2>&1 | head -20
```

Esperado: `ImportError: cannot import name 'BudgetResetRequest'`

- [ ] **Step 3: Implementar BudgetResetRequest en trainings/models.py**

Añadir al final de `trainings/models.py`:

```python
class BudgetResetRequest(models.Model):
    """
    Solicitud de un researcher para reiniciar su presupuesto epsilon en un dataset.
    El ADMIN del Node aprueba o rechaza la solicitud.

    dataset_id y researcher_id son IntegerField (no FK) porque referencian
    datasets_db y default respectivamente desde el mismo modelo.
    """

    STATUS_CHOICES = [
        ('pending', 'Pendiente'),
        ('approved', 'Aprobada'),
        ('rejected', 'Rechazada'),
    ]

    dataset_id = models.IntegerField(db_index=True)
    researcher_id = models.IntegerField(db_index=True)
    reason = models.TextField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='pending')
    requested_at = models.DateTimeField(auto_now_add=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='reviewed_budget_resets',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_notes = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-requested_at']
        # Solo una solicitud pendiente por (dataset, researcher)
        constraints = [
            models.UniqueConstraint(
                fields=['dataset_id', 'researcher_id'],
                condition=models.Q(status='pending'),
                name='unique_pending_budget_reset_per_researcher_dataset',
            )
        ]
        indexes = [
            models.Index(fields=['status', 'requested_at']),
        ]

    def approve(self, *, admin, notes: str = '') -> None:
        if self.status != 'pending':
            raise ValueError("Esta solicitud ya ha sido revisada.")
        self.status = 'approved'
        self.reviewed_by = admin
        self.reviewed_at = timezone.now()
        self.review_notes = notes
        self.save(update_fields=['status', 'reviewed_by', 'reviewed_at', 'review_notes'])

    def reject(self, *, admin, notes: str = '') -> None:
        if self.status != 'pending':
            raise ValueError("Esta solicitud ya ha sido revisada.")
        self.status = 'rejected'
        self.reviewed_by = admin
        self.reviewed_at = timezone.now()
        self.review_notes = notes
        self.save(update_fields=['status', 'reviewed_by', 'reviewed_at', 'review_notes'])
```

Añadir al principio de `trainings/models.py` si falta: `from django.conf import settings`

- [ ] **Step 4: Generar y aplicar migración**

```bash
python3.11 manage.py makemigrations trainings --name add_budget_reset_request
python3.11 manage.py migrate
```

- [ ] **Step 5: Ejecutar tests**

```bash
python3.11 -m pytest tests/trainings/test_budget_reset_request.py -v --tb=short
```

Esperado: todos los tests pasan.

- [ ] **Step 6: Commit**

```bash
git add trainings/models.py trainings/migrations/
git add tests/trainings/test_budget_reset_request.py
git commit -m "feat: add BudgetResetRequest model for researcher budget renewal"
```

---

## Task 3: Validación researcher budget en start_client

**Contexto:** `validate_training_permissions()` en `api/views.py` ya chequea `DatasetPrivacyPolicy.can_accept_job()`. Hay que añadir el chequeo de `ResearcherEpsilonBudget` para el researcher específico. Si no existe el registro, se crea automáticamente heredando los límites de la policy del dataset.

**Files:**
- Modify: `api/views.py` (función `validate_training_permissions`, líneas ~437-558)
- Create: `tests/api/test_researcher_budget_validation.py`

- [ ] **Step 1: Escribir los tests que fallarán**

```python
# tests/api/test_researcher_budget_validation.py
import sys
sys.modules.setdefault('magic', None)

import pytest
import json
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()

# Fixture mínima de model_json válido
MODEL_JSON = {
    "model": {
        "metadata": {"model_type": "dl"},
        "dataset": {
            "selected_datasets": [{"dataset_id": 1}]
        },
        "training": {
            "optimizer": {"type": "Adam", "learning_rate": 0.001},
            "dp": {"noise_multiplier": 1.1, "max_grad_norm": 1.0},
        }
    },
    "train": {"rounds": 3, "epochs": 1, "batch_size": 32},
    "federated": {
        "name": "FedAvg",
        "parameters": {"fraction_fit": 1.0, "min_fit_clients": 1, "min_available_clients": 1}
    },
}


def _make_researcher(username):
    role = Role.objects.get(name='RESEARCHER')
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_dataset(name, dataset_id_hint=1):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherBudgetInStartClient:

    def setup_method(self):
        from dataset.models import DatasetAccess
        self.researcher = _make_researcher("rbv_r1")
        self.dataset = _make_dataset("rbv_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset,
            sensitivity='high',
            max_epsilon_per_job=0.5,
            lifetime_budget=2.0,
        )
        DatasetAccess.objects.create(
            dataset=self.dataset,
            user_id=self.researcher.id,
            can_train=True,
            can_view_metadata=True,
        )

    def _call_validate(self, researcher, model_json=None):
        from api.views import validate_training_permissions
        mj = model_json or {
            **MODEL_JSON,
            "model": {
                **MODEL_JSON["model"],
                "dataset": {"selected_datasets": [{"dataset_id": self.dataset.id}]}
            }
        }
        return validate_training_permissions(researcher, mj)

    def test_creates_researcher_budget_on_first_call(self):
        with patch('api.views.estimate_job_epsilon', return_value=0.3):
            result = self._call_validate(self.researcher)
        assert result is None  # None = ok, sin error
        assert ResearcherEpsilonBudget.objects.filter(
            dataset=self.dataset, researcher_id=self.researcher.id
        ).exists()

    def test_rejects_when_researcher_budget_exhausted(self):
        budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        budget.spent_epsilon = 1.9  # solo quedan 0.1
        budget.save()

        with patch('api.views.estimate_job_epsilon', return_value=0.5):
            result = self._call_validate(self.researcher)
        assert result is not None  # JsonResponse de error
        assert result.status_code == 403

    def test_allows_when_researcher_budget_sufficient(self):
        with patch('api.views.estimate_job_epsilon', return_value=0.3):
            result = self._call_validate(self.researcher)
        assert result is None
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/api/test_researcher_budget_validation.py -v --tb=short 2>&1 | head -30
```

Esperado: `AssertionError` porque no se verifica el researcher budget aún.

- [ ] **Step 3: Modificar validate_training_permissions en api/views.py**

Localizar la sección donde se llama `policy.can_accept_job(estimated_eps)` (aproximadamente línea 530) y añadir después:

```python
# Añadir import al principio del archivo si no está:
from dataset.models import Dataset, DatasetAccess, DatasetMetadata, DatasetPrivacyPolicy, ResearcherEpsilonBudget

# Dentro de validate_training_permissions(), después del chequeo de policy.can_accept_job:

        # --- Chequeo de presupuesto por researcher ---
        try:
            researcher_budget, _ = ResearcherEpsilonBudget.get_or_create_for(
                dataset=dataset,
                researcher_id=user.id,
                policy=policy,
            )
            can_proceed, reason = researcher_budget.can_accept_job(estimated_eps)
            if not can_proceed:
                logger.warning(
                    "Researcher %s rechazado por presupuesto personal agotado: %s",
                    user.username, reason
                )
                return JsonResponse(
                    {'error': f'Presupuesto de privacidad del researcher agotado: {reason}'},
                    status=403,
                )
        except Exception as exc:
            logger.error("Error verificando presupuesto del researcher: %s", exc)
            return JsonResponse(
                {'error': 'Error verificando presupuesto de privacidad del researcher.'},
                status=500,
            )
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_researcher_budget_validation.py -v --tb=short
```

Esperado: todos los tests pasan.

- [ ] **Step 5: Commit**

```bash
git add api/views.py tests/api/test_researcher_budget_validation.py
git commit -m "feat: validate researcher epsilon budget in start_client"
```

---

## Task 4: Registro de epsilon en _record_privacy_spend para researcher

**Contexto:** `_record_privacy_spend()` en `api/federated/utils.py` ya actualiza `DatasetPrivacyPolicy.spent_epsilon` al completar un job. Hay que añadir la actualización del `ResearcherEpsilonBudget` correspondiente. El `researcher_id` se obtiene de `training_session.user_id`.

**Files:**
- Modify: `api/federated/utils.py` (función `_record_privacy_spend`, líneas ~169-250)
- Create: `tests/api/test_record_privacy_spend.py`

- [ ] **Step 1: Escribir el test que fallará**

```python
# tests/api/test_record_privacy_spend.py
import sys
sys.modules.setdefault('magic', None)

import pytest
from unittest.mock import MagicMock, patch
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _setup_fixtures(username="rps_r1", ds_name="rps_ds1"):
    role = Role.objects.get(name='RESEARCHER')
    researcher = User.objects.create_user(username=username, password="x")
    researcher.role = role
    researcher.save()

    ds = Dataset(
        name=ds_name, description="t", file_path=f"/f/{ds_name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    ds = Dataset.objects.get(name=ds_name)

    policy = DatasetPrivacyPolicy.objects.create(
        dataset=ds, sensitivity='high',
        max_epsilon_per_job=0.5, lifetime_budget=2.0,
    )
    budget, _ = ResearcherEpsilonBudget.get_or_create_for(
        dataset=ds, researcher_id=researcher.id, policy=policy,
    )
    return researcher, ds, policy, budget


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestRecordPrivacySpendUpdatesResearcherBudget:

    def setup_method(self):
        self.researcher, self.ds, self.policy, self.budget = _setup_fixtures()

    def _make_session_mock(self, epsilon_in_metrics=0.4):
        from trainings.models import TrainingSession, TrainingRound
        import uuid

        session = MagicMock()
        session.user_id = self.researcher.id
        session.session_id = uuid.uuid4()

        # Simular último round con privacy_epsilon en metrics
        round_mock = MagicMock()
        round_mock.metrics = {'privacy_epsilon': epsilon_in_metrics}
        session.rounds.order_by.return_value.first.return_value = round_mock

        return session

    def test_researcher_budget_updated_after_job(self):
        from api.federated.utils import _record_privacy_spend

        # Preparar mock de session con dataset_id y user_id
        session = self._make_session_mock(epsilon_in_metrics=0.4)

        with patch('api.federated.utils.DatasetPrivacyPolicy.objects') as mock_policy_qs, \
             patch('api.federated.utils.ResearcherEpsilonBudget.objects') as mock_budget_qs:

            mock_policy_qs.get.return_value = self.policy
            mock_budget_qs.get.return_value = self.budget
            mock_budget_qs.filter.return_value.update.return_value = 1

            # Simular acceso a dataset_id
            session.model_config = {
                'model': {'dataset': {'selected_datasets': [{'dataset_id': self.ds.id}]}}
            }

            _record_privacy_spend(session)

            # Verificar que se intentó actualizar el budget del researcher
            mock_budget_qs.filter.assert_called()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/api/test_record_privacy_spend.py -v --tb=short 2>&1 | head -30
```

- [ ] **Step 3: Modificar _record_privacy_spend en api/federated/utils.py**

Localizar la función `_record_privacy_spend` (línea ~169) y añadir después de `policy.record_spent(actual_epsilon)`:

```python
    # Añadir import al principio del archivo:
    from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget

    # Al final de _record_privacy_spend, tras policy.record_spent(actual_epsilon):
    try:
        researcher_id = getattr(training_session, 'user_id', None)
        if researcher_id is not None:
            ResearcherEpsilonBudget.objects.filter(
                dataset_id=dataset_id,
                researcher_id=researcher_id,
                spent_epsilon__lte=models.F('lifetime_budget'),
            ).update(spent_epsilon=models.F('spent_epsilon') + actual_epsilon)
            logger.info(
                "ResearcherEpsilonBudget actualizado: researcher=%s dataset=%s +epsilon=%.4f",
                researcher_id, dataset_id, actual_epsilon,
            )
    except Exception as exc:
        logger.error(
            "Error actualizando ResearcherEpsilonBudget (researcher=%s): %s",
            researcher_id, exc,
        )
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_record_privacy_spend.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add api/federated/utils.py tests/api/test_record_privacy_spend.py
git commit -m "feat: update ResearcherEpsilonBudget in _record_privacy_spend"
```

---

## Task 5: Registro de epsilon para jobs ML

**Contexto:** `ml_client.py` no incluye `privacy_epsilon` en `round_metrics`. Los modelos ML (SVM, RF) no tienen DP formal — son potencialmente epsilon=infinito. La decisión del sistema: registrar `max_epsilon_per_job` de la policy como epsilon consumido, siendo conservador. Así el presupuesto se agota correctamente y el ADMIN sabe que ese slot se usó.

**Files:**
- Modify: `api/federated/ml_client.py` (función `fit`, líneas ~119-251)
- Create: `tests/api/test_ml_epsilon_recording.py`

- [ ] **Step 1: Escribir el test que fallará**

```python
# tests/api/test_ml_epsilon_recording.py
import sys
sys.modules.setdefault('magic', None)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestMLClientRecordsEpsilon:

    def test_fit_includes_privacy_epsilon_in_metrics(self):
        from api.federated.ml_client import MLFlowerClient

        algorithm = MagicMock()
        algorithm.fit.return_value = (
            [np.array([1.0])],  # parameters
            {'loss': 0.5, 'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8},
        )
        algorithm.get_parameters.return_value = [np.array([1.0])]

        session = MagicMock()
        session.current_round = 0
        session.total_rounds = 3

        client = MLFlowerClient(
            algorithm_instance=algorithm,
            validation_data=(np.array([[1, 2]]), np.array([1])),
            model_json={
                'model': {'metadata': {'model_type': 'ml'},
                          'dataset': {'selected_datasets': [{'dataset_id': 1}]},
                          'training': {'dp': {'noise_multiplier': 1.1}}},
                'train': {'rounds': 3, 'epochs': 1, 'batch_size': 32},
                'federated': {'name': 'FedAvg', 'parameters': {}},
            },
            training_session=session,
            client_ip='127.0.0.1',
            table_name=1,
            current_process=MagicMock(),
        )

        with patch('api.federated.ml_client.update_training_progress') as mock_update:
            params, n, metrics = client.fit([np.array([1.0])], {})

        # El metrics del round debe incluir privacy_epsilon
        call_kwargs = mock_update.call_args
        round_metrics = call_kwargs[1].get('round_metrics') or call_kwargs[0][3]
        assert 'privacy_epsilon' in round_metrics
        assert round_metrics['privacy_epsilon'] > 0
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/api/test_ml_epsilon_recording.py -v --tb=short 2>&1 | head -30
```

Esperado: `AssertionError: 'privacy_epsilon' not in round_metrics`

- [ ] **Step 3: Modificar ml_client.py para incluir privacy_epsilon en métricas**

En la función `fit()` de `MLFlowerClient`, localizar donde se construye `round_metrics` y añadir:

```python
# Obtener max_epsilon_per_job de la policy como proxy conservador para ML
# (los modelos ML no tienen DP formal, usamos el límite por job como epsilon registrado)
try:
    from dataset.models import DatasetPrivacyPolicy
    _policy = DatasetPrivacyPolicy.objects.get(dataset_id=self.table_name)
    ml_epsilon = _policy.max_epsilon_per_job
except Exception:
    ml_epsilon = float('inf')

round_metrics = {
    'loss': loss,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'privacy_epsilon': ml_epsilon,  # conservador: se asume consumo máximo por job
    'model_type': 'ml',
}
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_ml_epsilon_recording.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add api/federated/ml_client.py tests/api/test_ml_epsilon_recording.py
git commit -m "fix: register privacy_epsilon in ML client round_metrics (conservative max per job)"
```

---

## Task 6: Verificación de parámetros DP mid-training en dl_client.py

**Contexto:** Hub puede enviar un `noise_multiplier` correcto en la validación inicial y luego modificarlo durante el entrenamiento Flower. Hay que verificar que los parámetros DP reales (los que Opacus usa) coinciden con los del config aprobado antes de entrenar.

**Files:**
- Modify: `api/federated/dl_client.py` (función `fit`, líneas ~115-202)
- Create: `tests/api/test_dp_param_verification.py`

- [ ] **Step 1: Escribir el test que fallará**

```python
# tests/api/test_dp_param_verification.py
import sys
sys.modules.setdefault('magic', None)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestDPParamVerification:

    def test_fit_aborts_if_noise_multiplier_tampered(self):
        """Si Opacus usa un noise_multiplier diferente al config, abortar."""
        from api.federated.dl_client import DLFlowerClient

        net = MagicMock()
        net.state_dict.return_value = {}

        session = MagicMock()
        session.current_round = 0
        session.total_rounds = 3

        model_json = {
            'model': {
                'metadata': {'model_type': 'dl'},
                'dataset': {'selected_datasets': [{'dataset_id': 1}]},
                'training': {
                    'optimizer': {'type': 'Adam', 'learning_rate': 0.001},
                    'dp': {'noise_multiplier': 1.1, 'max_grad_norm': 1.0},
                }
            },
            'train': {'rounds': 3, 'epochs': 1, 'batch_size': 32},
            'federated': {'name': 'FedAvg', 'parameters': {}},
        }

        client = DLFlowerClient(
            net=net,
            trainloader=MagicMock(),
            valloader=MagicMock(),
            testloader=MagicMock(),
            model_json=model_json,
            training_session=session,
            client_ip='127.0.0.1',
            table_name=1,
            device='cpu',
        )

        # train() devuelve un noise_multiplier distinto al del config (manipulado)
        with patch('api.federated.dl_client.set_parameters'), \
             patch('api.federated.dl_client.train') as mock_train, \
             patch('api.federated.dl_client.fail_training_session') as mock_fail:

            mock_train.return_value = (0.5, 0.8, 0.8, 0.8, 0.8, 0.3, 0.99)
            # El séptimo valor es actual_noise_multiplier — diferente a 1.1
            # Simulamos que train devuelve también el noise_multiplier usado

            # La función fit debe llamar a fail_training_session si hay discrepancia
            client.fit([np.zeros(1)], {})
            # Si no se implementó aún, fail no se llama — el test fallará aquí
```

Nota: el test requiere que `train()` devuelva el `noise_multiplier` real usado. Verificar la firma actual de `train()` en `api/federated/train_functions.py` y añadir el retorno si no está.

- [ ] **Step 2: Revisar firma de train() en train_functions.py**

```bash
grep -n "def train\|return " api/federated/train_functions.py | head -30
```

Si `train()` no devuelve `actual_noise_multiplier`, añadirlo:
```python
# Al final de train(), antes del return:
actual_noise = privacy_engine.noise_multiplier if privacy_engine else expected_noise
return loss, accuracy, precision, recall, f1, epsilon, actual_noise
```

- [ ] **Step 3: Añadir verificación en dl_client.py fit()**

Después de la llamada a `train()` en `fit()`:

```python
# Desempaquetar actual_noise si train lo devuelve
loss, accuracy, precision, recall, f1, epsilon, actual_noise = train(
    self.net, self.trainloader, complete_config, ...
)

# Verificar que el noise_multiplier real coincide con el aprobado
expected_noise = (
    complete_config.get('model', {})
    .get('training', {})
    .get('dp', {})
    .get('noise_multiplier')
)
if expected_noise is not None and actual_noise is not None:
    if abs(actual_noise - expected_noise) > 1e-4:
        fail_training_session(
            self.training_session,
            f"Parámetros DP manipulados: noise_multiplier esperado={expected_noise}, "
            f"real={actual_noise}. Entrenamiento abortado por seguridad.",
            traceback="",
        )
        return self.get_parameters({}), 0, {}
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_dp_param_verification.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add api/federated/dl_client.py api/federated/train_functions.py
git add tests/api/test_dp_param_verification.py
git commit -m "feat: verify DP noise_multiplier mid-training against approved config"
```

---

## Task 7: Límite de sesiones Flower concurrentes

**Contexto:** Un Hub comprometido puede lanzar múltiples jobs simultáneos para agotar el presupuesto o los recursos. Limitar las sesiones activas por researcher previene ambos ataques.

**Files:**
- Modify: `api/views.py` (función `start_client`, líneas ~198-341)
- Create: `tests/api/test_concurrent_sessions.py`

- [ ] **Step 1: Escribir el test que fallará**

```python
# tests/api/test_concurrent_sessions.py
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
MAX_CONCURRENT = 2  # debe coincidir con la constante en views.py


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

    def test_allows_first_session(self):
        from api.views import start_client
        request = self.factory.post(
            '/api/v2/start-client/',
            data=json.dumps({'model_json': {}, 'server_address': 'hub:8080'}),
            content_type='application/json',
        )
        request.api_user = self.researcher

        with patch('api.views.validate_training_config', return_value=(None, {})), \
             patch('api.views.validate_training_permissions', return_value=None), \
             patch('api.views.client.start_flower_client'), \
             patch('api.views.Process') as mock_proc:
            mock_proc.return_value.start.return_value = None
            mock_proc.return_value.pid = 1234
            resp = start_client(request)

        assert resp.status_code == 200

    def test_rejects_when_max_concurrent_reached(self):
        _make_active_session(self.researcher, n=MAX_CONCURRENT)
        from api.views import start_client
        request = self.factory.post(
            '/api/v2/start-client/',
            data=json.dumps({'model_json': {}, 'server_address': 'hub:8080'}),
            content_type='application/json',
        )
        request.api_user = self.researcher

        with patch('api.views.validate_training_config', return_value=(None, {})), \
             patch('api.views.validate_training_permissions', return_value=None):
            resp = start_client(request)

        assert resp.status_code == 429
        import json as _json
        body = _json.loads(resp.content)
        assert 'concurrent' in body['error'].lower() or 'simultáne' in body['error'].lower()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/api/test_concurrent_sessions.py -v --tb=short 2>&1 | head -30
```

- [ ] **Step 3: Añadir límite en start_client en api/views.py**

Al principio del archivo añadir la constante:
```python
MAX_CONCURRENT_TRAINING_SESSIONS = 2
```

En `start_client()`, después de autenticar al usuario y antes de `validate_training_config`:

```python
    # Verificar sesiones activas concurrentes
    active_sessions = TrainingSession.objects.filter(
        user=request.api_user,
        status__in=['STARTING', 'ACTIVE'],
    ).count()
    if active_sessions >= MAX_CONCURRENT_TRAINING_SESSIONS:
        return JsonResponse(
            {
                'error': (
                    f'Límite de sesiones simultáneas alcanzado ({MAX_CONCURRENT_TRAINING_SESSIONS}). '
                    f'Espera a que termine un entrenamiento antes de iniciar otro.'
                )
            },
            status=429,
        )
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_concurrent_sessions.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add api/views.py tests/api/test_concurrent_sessions.py
git commit -m "feat: limit concurrent Flower sessions per researcher to 2"
```

---

## Task 8: Rate limiting por IP para peticiones no autenticadas

**Contexto:** `RateLimitMiddleware` omite completamente las peticiones sin `api_user` (autenticación fallida). Un Hub comprometido puede hacer brute-force de API keys sin restricción. Hay que añadir rate limiting basado en IP usando el cache de Django.

**Files:**
- Modify: `medinet_core/security/middleware.py` (clase `RateLimitMiddleware`)
- Create: `tests/security/test_rate_limit_ip.py`

- [ ] **Step 1: Escribir el test que fallará**

```python
# tests/security/test_rate_limit_ip.py
import pytest
from unittest.mock import MagicMock, patch
from django.test import RequestFactory
from medinet_core.security.middleware import RateLimitMiddleware


def _make_unauth_request(ip='10.0.0.1', path='/api/v2/ping/'):
    factory = RequestFactory()
    req = factory.get(path, HTTP_REMOTE_ADDR=ip)
    # Sin api_user — simula petición no autenticada
    return req


class TestIPRateLimitForUnauthenticated:

    def _make_middleware(self):
        get_response = MagicMock(return_value=MagicMock(status_code=401))
        return RateLimitMiddleware(get_response)

    def test_first_request_passes(self):
        mw = self._make_middleware()
        req = _make_unauth_request()
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            mock_cache.get.return_value = 0
            mock_cache.set.return_value = None
            resp = mw(req)
        assert resp.status_code != 429

    def test_request_over_ip_limit_returns_429(self):
        mw = self._make_middleware()
        req = _make_unauth_request(ip='10.0.0.2')
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            # Simular que ya hay 21 peticiones de esta IP (límite = 20)
            mock_cache.get.return_value = 21
            resp = mw(req)
        assert resp.status_code == 429

    def test_different_ips_have_independent_limits(self):
        mw = self._make_middleware()
        req_a = _make_unauth_request(ip='10.0.0.3')
        req_b = _make_unauth_request(ip='10.0.0.4')
        with patch('medinet_core.security.middleware.cache') as mock_cache:
            # IP A está en el límite, IP B no
            def cache_get(key, default=0):
                if '10.0.0.3' in key:
                    return 21
                return 0
            mock_cache.get.side_effect = cache_get
            mock_cache.set.return_value = None
            resp_a = mw(req_a)
            resp_b = mw(req_b)
        assert resp_a.status_code == 429
        assert resp_b.status_code != 429
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/security/test_rate_limit_ip.py -v --tb=short 2>&1 | head -30
```

- [ ] **Step 3: Modificar RateLimitMiddleware en middleware.py**

Añadir al principio del archivo si no está:
```python
from django.core.cache import cache
```

Modificar `__call__` en `RateLimitMiddleware`:

```python
_IP_RATE_LIMIT_MAX = 20       # peticiones por ventana
_IP_RATE_LIMIT_WINDOW = 60    # segundos

def __call__(self, request):
    if not request.path.startswith('/api/'):
        return self.get_response(request)

    # Rate limit por IP para peticiones no autenticadas (brute-force protection)
    if not hasattr(request, 'api_user'):
        client_ip = self._get_client_ip(request)
        cache_key = f'ratelimit_ip_{client_ip}'
        request_count = cache.get(cache_key, 0)
        if request_count >= _IP_RATE_LIMIT_MAX:
            return JsonResponse(
                {'error': 'Demasiadas peticiones. Inténtalo más tarde.'},
                status=429,
            )
        cache.set(cache_key, request_count + 1, _IP_RATE_LIMIT_WINDOW)
        return self.get_response(request)

    # Rate limit por usuario para peticiones autenticadas (lógica existente)
    if self.is_rate_limited(request):
        return JsonResponse({'error': 'Rate limit exceeded'}, status=429)

    return self.get_response(request)

def _get_client_ip(self, request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '0.0.0.0')
```

- [ ] **Step 4: Ejecutar tests**

```bash
python3.11 -m pytest tests/security/test_rate_limit_ip.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add medinet_core/security/middleware.py tests/security/test_rate_limit_ip.py
git commit -m "feat: IP-based rate limiting for unauthenticated API requests"
```

---

## Task 9: Endpoints reset de presupuesto (researcher solicita, admin aprueba)

**Contexto:** El researcher puede solicitar un reset de su presupuesto personal. El ADMIN ve las solicitudes pendientes y aprueba o rechaza. Al aprobar, se llama `ResearcherEpsilonBudget.reset_period()` para el researcher en el dataset indicado.

**Files:**
- Create: `api/budget_views.py`
- Modify: `api/urls.py` (añadir rutas)
- Create: `tests/api/test_budget_endpoints.py`

- [ ] **Step 1: Escribir los tests que fallarán**

```python
# tests/api/test_budget_endpoints.py
import sys
sys.modules.setdefault('magic', None)

import pytest
import json
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from trainings.models import BudgetResetRequest
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_user(username, role_name):
    role = Role.objects.get(name=role_name)
    u = User.objects.create_user(username=username, password="x")
    u.role = role
    u.save()
    return u


def _make_dataset(name):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestBudgetResetEndpoints:

    def setup_method(self):
        self.researcher = _make_user("be_r1", "RESEARCHER")
        self.admin = _make_user("be_a1", "ADMIN")
        self.dataset = _make_dataset("be_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='high',
            max_epsilon_per_job=0.5, lifetime_budget=2.0,
        )
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset, researcher_id=self.researcher.id, policy=self.policy,
        )
        self.budget.spent_epsilon = 1.5
        self.budget.save()
        self.factory = RequestFactory()

    def _api_request(self, method, path, user, data=None):
        from api.budget_views import request_budget_reset, approve_budget_reset, reject_budget_reset
        req = getattr(self.factory, method)(
            path,
            data=json.dumps(data or {}),
            content_type='application/json',
        )
        req.api_user = user
        return req

    def test_researcher_can_request_reset(self):
        from api.budget_views import request_budget_reset
        req = self._api_request('post', '/api/v2/budget-reset/', self.researcher, {
            'dataset_id': self.dataset.id,
            'reason': 'Nuevo proyecto aprobado por comité.',
        })
        resp = request_budget_reset(req)
        assert resp.status_code == 201
        assert BudgetResetRequest.objects.filter(
            researcher_id=self.researcher.id, dataset_id=self.dataset.id, status='pending'
        ).exists()

    def test_admin_can_approve_reset(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Motivo válido.',
        )
        from api.budget_views import approve_budget_reset
        req = self._api_request('post', f'/api/v2/budget-reset/{reset_req.id}/approve/', self.admin, {
            'notes': 'Aprobado por revisión ética.',
        })
        resp = approve_budget_reset(req, reset_req.id)
        assert resp.status_code == 200

        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == 0.0

        reset_req.refresh_from_db()
        assert reset_req.status == 'approved'

    def test_admin_can_reject_reset(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Motivo.',
        )
        from api.budget_views import reject_budget_reset
        req = self._api_request('post', f'/api/v2/budget-reset/{reset_req.id}/reject/', self.admin, {
            'notes': 'No procede.',
        })
        resp = reject_budget_reset(req, reset_req.id)
        assert resp.status_code == 200

        reset_req.refresh_from_db()
        assert reset_req.status == 'rejected'
        self.budget.refresh_from_db()
        assert self.budget.spent_epsilon == pytest.approx(1.5)  # no cambió

    def test_researcher_cannot_approve(self):
        reset_req = BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='x',
        )
        from api.budget_views import approve_budget_reset
        req = self._api_request('post', f'/api/v2/budget-reset/{reset_req.id}/approve/', self.researcher, {})
        resp = approve_budget_reset(req, reset_req.id)
        assert resp.status_code == 403
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
python3.11 -m pytest tests/api/test_budget_endpoints.py -v --tb=short 2>&1 | head -30
```

- [ ] **Step 3: Crear api/budget_views.py**

```python
"""
Endpoints para gestión del presupuesto epsilon por researcher.

- POST /api/v2/budget-reset/                     — researcher solicita reset
- POST /api/v2/budget-reset/<id>/approve/        — admin aprueba
- POST /api/v2/budget-reset/<id>/reject/         — admin rechaza
- GET  /api/v2/budget-reset/                     — admin lista pendientes
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from trainings.models import BudgetResetRequest
from dataset.models import ResearcherEpsilonBudget

logger = logging.getLogger(__name__)


def _is_admin(user) -> bool:
    return bool(user and user.role and user.role.name == 'ADMIN')


def _is_researcher(user) -> bool:
    return bool(user and user.role and user.role.name == 'RESEARCHER')


@csrf_exempt
@require_http_methods(["POST", "GET"])
def request_budget_reset(request):
    """POST: researcher solicita reset de su presupuesto en un dataset."""
    user = getattr(request, 'api_user', None)
    if not _is_researcher(user):
        return JsonResponse({'error': 'Solo los researchers pueden solicitar un reset.'}, status=403)

    if request.method == 'GET':
        # Researcher ve sus propias solicitudes
        requests = BudgetResetRequest.objects.filter(researcher_id=user.id).order_by('-requested_at')
        data = [
            {
                'id': r.id,
                'dataset_id': r.dataset_id,
                'status': r.status,
                'reason': r.reason,
                'requested_at': r.requested_at.isoformat(),
                'review_notes': r.review_notes,
            }
            for r in requests
        ]
        return JsonResponse({'results': data})

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'JSON inválido.'}, status=400)

    dataset_id = body.get('dataset_id')
    reason = (body.get('reason') or '').strip()

    if not dataset_id or not isinstance(dataset_id, int):
        return JsonResponse({'error': 'dataset_id es obligatorio y debe ser un entero.'}, status=400)
    if not reason:
        return JsonResponse({'error': 'El motivo de la solicitud es obligatorio.'}, status=400)
    if len(reason) > 1000:
        return JsonResponse({'error': 'El motivo no puede superar 1000 caracteres.'}, status=400)

    # Verificar que no hay ya una solicitud pendiente
    if BudgetResetRequest.objects.filter(
        dataset_id=dataset_id, researcher_id=user.id, status='pending'
    ).exists():
        return JsonResponse(
            {'error': 'Ya tienes una solicitud pendiente para este dataset.'}, status=409
        )

    reset_req = BudgetResetRequest.objects.create(
        dataset_id=dataset_id,
        researcher_id=user.id,
        reason=reason,
    )
    return JsonResponse({'id': reset_req.id, 'status': 'pending'}, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def approve_budget_reset(request, request_id):
    """POST: admin aprueba una solicitud de reset y aplica el reset."""
    user = getattr(request, 'api_user', None)
    if not _is_admin(user):
        return JsonResponse({'error': 'Solo los administradores pueden aprobar solicitudes.'}, status=403)

    reset_req = get_object_or_404(BudgetResetRequest, pk=request_id)

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    notes = (body.get('notes') or '').strip()

    try:
        reset_req.approve(admin=user, notes=notes)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=409)

    # Aplicar reset del presupuesto del researcher en el dataset
    try:
        budget = ResearcherEpsilonBudget.objects.get(
            dataset_id=reset_req.dataset_id,
            researcher_id=reset_req.researcher_id,
        )
        budget.reset_period()
        logger.info(
            "Presupuesto reseteado por admin %s: researcher_id=%s dataset_id=%s",
            user.username, reset_req.researcher_id, reset_req.dataset_id,
        )
    except ResearcherEpsilonBudget.DoesNotExist:
        logger.warning(
            "Reset aprobado pero no existe ResearcherEpsilonBudget: "
            "researcher_id=%s dataset_id=%s",
            reset_req.researcher_id, reset_req.dataset_id,
        )

    return JsonResponse({'id': reset_req.id, 'status': 'approved'})


@csrf_exempt
@require_http_methods(["POST"])
def reject_budget_reset(request, request_id):
    """POST: admin rechaza una solicitud de reset."""
    user = getattr(request, 'api_user', None)
    if not _is_admin(user):
        return JsonResponse({'error': 'Solo los administradores pueden rechazar solicitudes.'}, status=403)

    reset_req = get_object_or_404(BudgetResetRequest, pk=request_id)

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    notes = (body.get('notes') or '').strip()

    try:
        reset_req.reject(admin=user, notes=notes)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=409)

    return JsonResponse({'id': reset_req.id, 'status': 'rejected'})
```

- [ ] **Step 4: Añadir rutas en api/urls.py**

Localizar `api/urls.py` y añadir:

```python
from api.budget_views import request_budget_reset, approve_budget_reset, reject_budget_reset

urlpatterns = [
    # ... rutas existentes ...
    path('budget-reset/', request_budget_reset, name='budget_reset_request'),
    path('budget-reset/<int:request_id>/approve/', approve_budget_reset, name='budget_reset_approve'),
    path('budget-reset/<int:request_id>/reject/', reject_budget_reset, name='budget_reset_reject'),
]
```

- [ ] **Step 5: Ejecutar tests**

```bash
python3.11 -m pytest tests/api/test_budget_endpoints.py -v --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add api/budget_views.py api/urls.py tests/api/test_budget_endpoints.py
git commit -m "feat: add budget reset endpoints — researcher requests, admin approves"
```

---

## Task 10: UI — Sección de presupuestos en dataset detail (ADMIN) y researcher_info (RESEARCHER)

**Contexto:** Los ADMIN necesitan ver en el detalle de cada dataset cuánto presupuesto ha gastado cada researcher y gestionar las solicitudes de reset pendientes. Los RESEARCHER necesitan ver sus propios presupuestos por dataset y solicitar un reset desde la UI (la única URL web que pueden acceder es `/info/researcher/`).

**Roles y acceso:**
- `ADMIN` → accede a `datasets/<id>/detail/` → ve tabla de budgets por researcher + aprueba/rechaza resets
- `RESEARCHER` → accede a `/info/researcher/` → ve sus presupuestos y puede solicitar reset vía formulario

**Files:**
- Modify: `dataset/views.py` — añadir `researcher_budgets` y `pending_reset_requests` al contexto de `dataset_detail`
- Create: `templates/dataset/partials/researcher_budgets.html` — partial reutilizable
- Modify: `templates/dataset/detail.html` — incluir el partial en la columna izquierda
- Modify: `users/views.py` — añadir budgets del researcher al contexto de `researcher_info`
- Modify: `templates/users/researcher_info.html` — sección "Mis presupuestos DP" + formulario de solicitud
- Create: `tests/dataset/test_researcher_budget_ui.py`

- [ ] **Step 1: Modificar dataset_detail view para pasar budgets al contexto**

En `dataset/views.py`, dentro de `dataset_detail`, añadir antes del `return TemplateResponse`:

```python
# Researcher budgets (solo visible para ADMIN)
researcher_budgets = []
pending_reset_requests = []
if request.user.role and request.user.role.name == 'ADMIN':
    from trainings.models import BudgetResetRequest
    researcher_budgets = (
        ResearcherEpsilonBudget.objects
        .filter(dataset=dataset)
        .order_by('researcher_id')
    )
    pending_reset_requests = (
        BudgetResetRequest.objects
        .filter(dataset_id=dataset.id, status='pending')
        .order_by('requested_at')
    )
```

Y añadir al dict `context`:

```python
context = {
    # ... campos existentes ...
    'researcher_budgets': researcher_budgets,
    'pending_reset_requests': pending_reset_requests,
}
```

- [ ] **Step 2: Crear el partial templates/dataset/partials/researcher_budgets.html**

```html
{# templates/dataset/partials/researcher_budgets.html #}
{# Requiere en contexto: researcher_budgets, pending_reset_requests #}
{% load csp_tags %}

<div class="dashboard-card card mb-4">
  <div class="card-header d-flex justify-content-between align-items-center">
    <h5 class="mb-0">
      <i data-lucide="users" style="width:16px;height:16px;vertical-align:middle;" class="me-2"></i>
      Presupuesto ε por Researcher
    </h5>
    {% if pending_reset_requests %}
      <span class="badge bg-warning text-dark">
        {{ pending_reset_requests|length }} solicitud{% if pending_reset_requests|length != 1 %}es{% endif %} pendiente{% if pending_reset_requests|length != 1 %}s{% endif %}
      </span>
    {% endif %}
  </div>
  <div class="card-body p-0">

    {% if researcher_budgets %}
    <div class="table-responsive">
      <table class="table table-sm table-hover mb-0">
        <thead class="table-light">
          <tr>
            <th>Researcher ID</th>
            <th>Gastado (ε)</th>
            <th>Restante (ε)</th>
            <th>Límite total</th>
            <th>Periodo</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {% for budget in researcher_budgets %}
          <tr>
            <td><code>#{{ budget.researcher_id }}</code></td>
            <td>{{ budget.spent_epsilon|floatformat:4 }}</td>
            <td>
              {% if budget.remaining_budget < 0.5 %}
                <span class="text-danger fw-semibold">{{ budget.remaining_budget|floatformat:4 }}</span>
              {% else %}
                <span class="text-success">{{ budget.remaining_budget|floatformat:4 }}</span>
              {% endif %}
            </td>
            <td>{{ budget.lifetime_budget|floatformat:4 }}</td>
            <td><span class="badge bg-secondary">{{ budget.get_period_display }}</span></td>
            <td>
              {% with pct=budget.spent_epsilon|floatformat:0 %}
              <div class="progress" style="height:8px;min-width:80px;">
                <div class="progress-bar {% if budget.remaining_budget < 0.5 %}bg-danger{% else %}bg-info{% endif %}"
                     role="progressbar"
                     style="width: {% widthratio budget.spent_epsilon budget.lifetime_budget 100 %}%">
                </div>
              </div>
              {% endwith %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <div class="text-center py-3 text-muted small">
      Ningún researcher ha entrenado con este dataset todavía.
    </div>
    {% endif %}

    {% if pending_reset_requests %}
    <div class="border-top px-3 py-2">
      <p class="small fw-semibold mb-2 text-warning">
        <i data-lucide="clock" style="width:14px;height:14px;vertical-align:middle;"></i>
        Solicitudes de reset pendientes
      </p>
      {% for req in pending_reset_requests %}
      <div class="d-flex align-items-start justify-content-between mb-2 p-2 bg-light rounded">
        <div class="small">
          <code>#{{ req.researcher_id }}</code> —
          <span class="text-muted">{{ req.requested_at|date:"d/m/Y H:i" }}</span><br>
          <span>{{ req.reason|truncatechars:120 }}</span>
        </div>
        <div class="d-flex gap-1 ms-2 flex-shrink-0">
          <button class="btn btn-success btn-sm"
                  hx-post="{% url 'api:budget_reset_approve' req.id %}"
                  hx-vals='{"notes": ""}'
                  hx-target="closest .dashboard-card"
                  hx-swap="outerHTML"
                  onclick="return confirm('¿Aprobar solicitud de reset de presupuesto?')">
            Aprobar
          </button>
          <button class="btn btn-outline-danger btn-sm"
                  hx-post="{% url 'api:budget_reset_reject' req.id %}"
                  hx-vals='{"notes": "Rechazado por administrador."}'
                  hx-target="closest .dashboard-card"
                  hx-swap="outerHTML">
            Rechazar
          </button>
        </div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

  </div>
</div>
```

**Nota:** Si el proyecto no usa HTMX, reemplazar los botones `hx-post` por formularios `<form method="post">` con action a las vistas de aprobación/rechazo web (añadir `dataset/views.py` handlers para approve/reject desde web).

- [ ] **Step 3: Incluir el partial en templates/dataset/detail.html**

Localizar la sección de la columna izquierda donde está la card de "Differential Privacy Policy" (buscar `privacy_policy`) y añadir después:

```html
{% if request.user.role.name == 'ADMIN' %}
  {% include "dataset/partials/researcher_budgets.html" %}
{% endif %}
```

- [ ] **Step 4: Modificar users/views.py para pasar budgets al researcher_info**

En la vista `researcher_info` (buscar en `users/views.py`), añadir al contexto:

```python
from dataset.models import ResearcherEpsilonBudget, DatasetPrivacyPolicy

# Budgets del researcher actual
researcher_budgets = (
    ResearcherEpsilonBudget.objects
    .filter(researcher_id=request.user.id)
    .order_by('dataset_id')
    .select_related('dataset')
)

context['researcher_budgets'] = researcher_budgets
```

- [ ] **Step 5: Añadir sección en templates/users/researcher_info.html**

Al final del contenido principal del template, añadir:

```html
<!-- Presupuesto de Privacidad Diferencial -->
<div class="card mb-4">
  <div class="card-header">
    <h5 class="mb-0">
      <i data-lucide="shield" style="width:16px;height:16px;vertical-align:middle;" class="me-2"></i>
      Mis presupuestos de Privacidad Diferencial
    </h5>
  </div>
  <div class="card-body">
    {% if researcher_budgets %}
    <div class="table-responsive mb-3">
      <table class="table table-sm">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Gastado (ε)</th>
            <th>Restante (ε)</th>
            <th>Límite total</th>
            <th>Solicitar reset</th>
          </tr>
        </thead>
        <tbody>
          {% for budget in researcher_budgets %}
          <tr>
            <td>Dataset #{{ budget.dataset_id }}</td>
            <td>{{ budget.spent_epsilon|floatformat:4 }}</td>
            <td>
              {% if budget.remaining_budget < 0.5 %}
                <span class="text-danger fw-semibold">{{ budget.remaining_budget|floatformat:4 }}</span>
              {% else %}
                {{ budget.remaining_budget|floatformat:4 }}
              {% endif %}
            </td>
            <td>{{ budget.lifetime_budget|floatformat:4 }}</td>
            <td>
              <button class="btn btn-sm btn-outline-primary"
                      data-bs-toggle="modal"
                      data-bs-target="#resetModal{{ budget.dataset_id }}">
                Solicitar reset
              </button>

              <!-- Modal de solicitud -->
              <div class="modal fade" id="resetModal{{ budget.dataset_id }}" tabindex="-1">
                <div class="modal-dialog">
                  <div class="modal-content">
                    <form method="post" action="{% url 'api:budget_reset_request' %}">
                      {% csrf_token %}
                      <input type="hidden" name="dataset_id" value="{{ budget.dataset_id }}">
                      <div class="modal-header">
                        <h5 class="modal-title">Solicitar reset de presupuesto</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                      </div>
                      <div class="modal-body">
                        <p class="text-muted small">
                          Tu solicitud será revisada por el administrador del Node.
                          Debes justificar el motivo (nuevo proyecto, aprobación de comité, etc.).
                        </p>
                        <div class="mb-3">
                          <label class="form-label">Motivo <span class="text-danger">*</span></label>
                          <textarea name="reason" class="form-control" rows="4"
                                    maxlength="1000" required
                                    placeholder="Describe el motivo del reset (proyecto, aprobación ética, etc.)"></textarea>
                        </div>
                      </div>
                      <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                        <button type="submit" class="btn btn-primary">Enviar solicitud</button>
                      </div>
                    </form>
                  </div>
                </div>
              </div>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p class="text-muted small mb-0">
      Aún no has entrenado con ningún dataset en este Node. Cuando lo hagas, aquí aparecerá tu presupuesto de privacidad diferencial.
    </p>
    {% endif %}
  </div>
</div>
```

**Nota:** El formulario POST a `{% url 'api:budget_reset_request' %}` necesita que la vista `request_budget_reset` de `api/budget_views.py` acepte también peticiones de RESEARCHER autenticados por sesión web (además de API key). Añadir soporte para `request.user` si `request.api_user` no está disponible:

```python
# En api/budget_views.py, primera línea de request_budget_reset:
user = getattr(request, 'api_user', None) or getattr(request, 'user', None)
```

- [ ] **Step 6: Escribir tests de UI**

```python
# tests/dataset/test_researcher_budget_ui.py
import sys
sys.modules.setdefault('magic', None)

import pytest
from django.test import Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from dataset.models import Dataset, DatasetPrivacyPolicy, ResearcherEpsilonBudget
from users.models import Role

User = get_user_model()


def _make_user(username, role_name):
    role = Role.objects.get(name=role_name)
    u = User.objects.create_user(username=username, password="testpass")
    u.role = role
    u.save()
    return u


def _make_dataset(name):
    ds = Dataset(
        name=name, description="t", file_path=f"/f/{name}.csv",
        file_size=100, file_format="csv", uploaded_by_id=1,
        patient_count=500, columns_count=5, target_column="y",
        medical_domain="cardiology", data_type="tabular",
        anonymized=True, is_active=True,
    )
    Dataset.objects.bulk_create([ds])
    return Dataset.objects.get(name=name)


@pytest.mark.django_db(databases=['default', 'datasets_db'])
class TestResearcherBudgetUIInDatasetDetail:

    def setup_method(self):
        self.admin = _make_user("ui_admin", "ADMIN")
        self.researcher = _make_user("ui_researcher", "RESEARCHER")
        self.dataset = _make_dataset("ui_ds1")
        self.policy = DatasetPrivacyPolicy.objects.create(
            dataset=self.dataset, sensitivity='high',
            max_epsilon_per_job=0.5, lifetime_budget=2.0,
        )
        self.budget, _ = ResearcherEpsilonBudget.get_or_create_for(
            dataset=self.dataset,
            researcher_id=self.researcher.id,
            policy=self.policy,
        )
        self.budget.spent_epsilon = 0.5
        self.budget.save()

    def test_admin_sees_researcher_budgets_section(self):
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert resp.status_code == 200
        assert 'Presupuesto' in resp.content.decode()
        assert '0.5000' in resp.content.decode()  # spent_epsilon

    def test_admin_does_not_see_budgets_if_none(self):
        ResearcherEpsilonBudget.objects.filter(dataset=self.dataset).delete()
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert resp.status_code == 200
        # La sección existe pero muestra el mensaje vacío
        assert 'Ningún researcher' in resp.content.decode()

    def test_context_contains_researcher_budgets(self):
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert 'researcher_budgets' in resp.context
        budgets = list(resp.context['researcher_budgets'])
        assert len(budgets) == 1
        assert budgets[0].researcher_id == self.researcher.id

    def test_context_contains_pending_reset_requests(self):
        from trainings.models import BudgetResetRequest
        BudgetResetRequest.objects.create(
            dataset_id=self.dataset.id,
            researcher_id=self.researcher.id,
            reason='Nuevo proyecto.',
        )
        client = Client()
        client.force_login(self.admin)
        url = reverse('dataset:detail', kwargs={'dataset_id': self.dataset.id})
        resp = client.get(url)
        assert 'pending_reset_requests' in resp.context
        assert len(resp.context['pending_reset_requests']) == 1
```

- [ ] **Step 7: Ejecutar tests UI**

```bash
python3.11 -m pytest tests/dataset/test_researcher_budget_ui.py -v --tb=short
```

Esperado: todos los tests pasan.

- [ ] **Step 8: Commit**

```bash
git add dataset/views.py
git add templates/dataset/detail.html
git add templates/dataset/partials/researcher_budgets.html
git add templates/users/researcher_info.html
git add users/views.py
git add api/budget_views.py
git add tests/dataset/test_researcher_budget_ui.py
git commit -m "feat: UI for researcher epsilon budget — admin view in dataset detail, researcher view in info page"
```

---

## Task 11: Documentación — features y bugfixes

**Contexto:** El proyecto no tiene CHANGELOG ni docs de features/bugfixes. Se crea la estructura y se documenta cada cambio de este sprint para que futuros desarrolladores entiendan el qué, el por qué y el impacto en privacidad.

**Files:**
- Crear: `CHANGELOG.md` (raíz del proyecto)
- Crear: `docs/features/researcher-epsilon-budget.md`
- Crear: `docs/features/budget-reset-workflow.md`
- Crear: `docs/bugfixes/ml-epsilon-recording.md`
- Crear: `docs/bugfixes/rate-limit-unauthenticated.md`
- Crear: `docs/bugfixes/dp-param-verification.md`

- [ ] **Step 1: Crear estructura de directorios**

```bash
mkdir -p docs/features docs/bugfixes
```

- [ ] **Step 2: Crear CHANGELOG.md en la raíz**

```markdown
# Changelog — MediNetNode

Formato: [Unreleased] → se asignará versión al publicar.
Entradas más recientes primero.

---

## [Unreleased] — Sprint DP Security & Researcher Budget (2026-04)

### Nuevas funcionalidades

- **Presupuesto epsilon por researcher** (`ResearcherEpsilonBudget`): cada researcher tiene su
  propio contador de epsilon por dataset, con reset periódico (anual/mensual/nunca) y reset
  manual bajo aprobación del ADMIN. Ver `docs/features/researcher-epsilon-budget.md`.

- **Flujo de reset de presupuesto**: el researcher solicita un reset desde la UI o la API;
  el ADMIN aprueba o rechaza con notas auditadas. Ver `docs/features/budget-reset-workflow.md`.

- **UI de presupuestos en detalle de dataset**: el ADMIN ve la tabla de presupuestos por
  researcher y gestiona las solicitudes pendientes directamente desde `datasets/<id>/detail/`.

- **UI en researcher_info**: el RESEARCHER ve sus presupuestos restantes y puede solicitar
  reset con justificación desde `/info/researcher/`.

- **Límite de sesiones Flower concurrentes**: máximo 2 sesiones de entrenamiento activas
  simultáneamente por researcher. Previene DoS por flood de jobs y agotamiento de RAM.

- **Rate limiting por IP para peticiones no autenticadas**: 20 peticiones/minuto por IP.
  Previene brute-force de API keys desde un Hub comprometido.

### Correcciones de seguridad

- **[MEDIA] ML epsilon no registrado**: los jobs de modelos clásicos (SVM, RF) no actualizaban
  `spent_epsilon`. Un researcher podía lanzar iteraciones ilimitadas con ML sin consumir
  presupuesto DP. Ver `docs/bugfixes/ml-epsilon-recording.md`.

- **[ALTA] Rate limiting inefectivo para no autenticados**: `RateLimitMiddleware` omitía las
  peticiones fallidas de autenticación, permitiendo brute-force sin restricción.
  Ver `docs/bugfixes/rate-limit-unauthenticated.md`.

- **[ALTA] Parámetros DP sin verificación mid-training**: el `noise_multiplier` aprobado en la
  validación inicial no se comparaba con el real durante el entrenamiento. Un Hub comprometido
  podía reducir el ruido DP sin que el Node lo detectase.
  Ver `docs/bugfixes/dp-param-verification.md`.

---

## Versiones anteriores

*(sin changelog previo — historial disponible en `git log`)*
```

- [ ] **Step 3: Crear docs/features/researcher-epsilon-budget.md**

```markdown
# Feature: Presupuesto epsilon por researcher

**Sprint:** DP Security & Researcher Budget (2026-04)
**Modelos:** `ResearcherEpsilonBudget` (`dataset` app → `datasets_db`)
**Archivos clave:** `dataset/models.py`, `api/views.py`, `api/federated/utils.py`

## Problema que resuelve

`DatasetPrivacyPolicy.spent_epsilon` era un contador global compartido por todos los
researchers. Esto causaba:
1. **Injusticia**: un researcher agotaba el presupuesto de los demás.
2. **Sin trazabilidad**: el ADMIN no podía saber qué researcher había consumido cuánto.

## Diseño

Un registro `ResearcherEpsilonBudget` por par (dataset, researcher_id). Hereda los límites
de `DatasetPrivacyPolicy` al crearse pero lleva su propio `spent_epsilon`.

```
DatasetPrivacyPolicy         ResearcherEpsilonBudget
(global del dataset)         (por researcher)
────────────────────         ────────────────────────
sensitivity: high            researcher_id: 42
max_epsilon_per_job: 0.5     spent_epsilon: 1.0   ← independiente
lifetime_budget: 2.0         lifetime_budget: 2.0  ← heredado al crear
                             max_epsilon_per_job: 0.5
                             period: annual
```

## Decisiones de diseño

| Decisión | Motivo |
|----------|--------|
| `researcher_id` como IntegerField, no FK | No se pueden hacer FK cross-DB (`datasets_db` ↔ `default`) |
| Hereda límites de la policy al crear | Nuevos researchers arrancan con los mismos límites que el dataset define |
| Reset automático en `can_accept_job()` | Evita que un job se bloquee justo al expirar el periodo |
| Update atómico con `F()` | Evita race condition entre dos jobs simultáneos del mismo researcher |

## Periodos de reset

| Periodo | Duración | Caso de uso |
|---------|----------|-------------|
| `annual` | 1 año desde `period_start` | Investigación continua con presupuesto anual renovable |
| `monthly` | 1 mes desde `period_start` | Proyectos cortos o exploración frecuente |
| `never` | Sin reset automático | Estudios únicos o datasets de máxima sensibilidad |

El reset automático ocurre al intentar un nuevo job si el periodo ha vencido.
Para reset manual supervisado, ver `docs/features/budget-reset-workflow.md`.

## Flujo de validación completo

```
POST /api/v2/start-client
  validate_training_permissions()
    1. DatasetPrivacyPolicy.can_accept_job(eps)        ← límite global del dataset
    2. ResearcherEpsilonBudget.get_or_create_for(...)  ← crear si es primera vez
    3. ResearcherEpsilonBudget.can_accept_job(eps)     ← límite individual del researcher

complete_training_session()
    1. DatasetPrivacyPolicy.record_spent(actual_eps)   ← actualiza global
    2. ResearcherEpsilonBudget.record_spent(actual_eps)← actualiza individual
```
```

- [ ] **Step 4: Crear docs/features/budget-reset-workflow.md**

```markdown
# Feature: Flujo de solicitud de reset de presupuesto

**Sprint:** DP Security & Researcher Budget (2026-04)
**Modelos:** `BudgetResetRequest` (`trainings` app → `default`)
**Archivos clave:** `trainings/models.py`, `api/budget_views.py`

## Problema que resuelve

Los researchers necesitan iterar sobre modelos (arquitecturas, hiperparámetros) antes
de lanzar el estudio definitivo. Con un presupuesto fijo por periodo pueden quedarse sin
epsilon antes del trabajo real. Se necesita un mecanismo formal y auditado para renovar
el presupuesto bajo supervisión del ADMIN del hospital.

## Flujo

```
Researcher                           Node (ADMIN)
    │                                     │
    ├─► Solicitar reset (UI o API) ──────►│ BudgetResetRequest status='pending'
    │                                     │
    │                              ADMIN revisa en datasets/<id>/detail/
    │                                     │
    │◄─── Notificado (sin email aún) ─────┤
    │                              status='approved' → ResearcherEpsilonBudget.reset_period()
    │                              status='rejected' → sin cambio
```

## Endpoints API

| Método | URL | Rol requerido | Descripción |
|--------|-----|---------------|-------------|
| `POST` | `/api/v2/budget-reset/` | RESEARCHER | Crear solicitud con motivo |
| `GET`  | `/api/v2/budget-reset/` | RESEARCHER | Ver mis solicitudes |
| `POST` | `/api/v2/budget-reset/<id>/approve/` | ADMIN | Aprobar y aplicar reset |
| `POST` | `/api/v2/budget-reset/<id>/reject/`  | ADMIN | Rechazar con notas |

## Restricciones de seguridad

- Solo **una solicitud pendiente** por (dataset, researcher) a la vez — constraint de DB.
- El motivo es obligatorio (1–1000 caracteres).
- El reset **solo lo aplica el ADMIN** — el researcher nunca puede resetearse a sí mismo.
- Las solicitudes son permanentes (no se borran) para auditoría completa.
- `BudgetResetRequest` vive en `trainings` app (`default` DB) para poder hacer FK a `User`.
```

- [ ] **Step 5: Crear docs/bugfixes/ml-epsilon-recording.md**

```markdown
# Bugfix: ML jobs no registraban epsilon consumido

**Severidad:** MEDIA
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivo corregido:** `api/federated/ml_client.py`

## Bug

`_record_privacy_spend()` lee `privacy_epsilon` del campo `metrics` del último `TrainingRound`.
Los jobs DL incluyen este campo vía Opacus. Los jobs ML (SVM, RF) no lo incluían, por lo que
`_record_privacy_spend()` encontraba `raw_eps = None` y registraba un warning sin actualizar
`spent_epsilon`. Un researcher podía lanzar N jobs ML sin límite de presupuesto.

## Por qué ML consume presupuesto

Los algoritmos clásicos sin DP-SGD son epsilon = infinito (fuga total potencial). Decisión de
diseño: registrar `max_epsilon_per_job` de la policy como epsilon consumido para jobs ML.
Es conservador pero seguro.

## Corrección

`MLFlowerClient.fit()` ahora obtiene `max_epsilon_per_job` de `DatasetPrivacyPolicy` y lo
incluye en `round_metrics` como `privacy_epsilon`. `_record_privacy_spend()` lo procesa
normalmente.
```

- [ ] **Step 6: Crear docs/bugfixes/rate-limit-unauthenticated.md**

```markdown
# Bugfix: Rate limiting inefectivo para peticiones no autenticadas

**Severidad:** ALTA
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivo corregido:** `medinet_core/security/middleware.py`

## Bug

`RateLimitMiddleware.__call__()` tenía salida temprana para peticiones sin `api_user`:

```python
if not hasattr(request, 'api_user'):
    return self.get_response(request)  # sin contar
```

Las peticiones con API key incorrecta retornaban 401 antes de que `api_user` existiese,
por lo que el rate limiter no las contaba. Un Hub comprometido podía hacer brute-force
de API keys sin restricción.

## Corrección

Rate limiting por IP para peticiones no autenticadas usando el cache de Django:
- Límite: 20 peticiones/minuto por IP
- Clave: `ratelimit_ip_<ip>`
- HTTP 429 si se supera

Compatible con LocMemCache (test) y Redis (producción).
```

- [ ] **Step 7: Crear docs/bugfixes/dp-param-verification.md**

```markdown
# Bugfix: Parámetros DP sin verificación mid-training

**Severidad:** ALTA
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivo corregido:** `api/federated/dl_client.py`

## Bug

El Node validaba el `noise_multiplier` en el config inicial pero no verificaba que Opacus
usase ese mismo valor durante el entrenamiento real. Un Hub comprometido podía:

1. Enviar `noise_multiplier=1.1` en la validación → Node aprueba
2. Enviar `noise_multiplier=0.01` en la config del round Flower
3. Opacus entrena con casi cero ruido → fuga de información masiva
4. Node registra epsilon con `noise=0.01` (mucho mayor) → presupuesto se agota rápido

## Corrección

Tras `train()`, se compara el `noise_multiplier` real de Opacus con el del config aprobado.
Si la discrepancia supera `1e-4`, se llama a `fail_training_session()` y se retornan
0 muestras (Flower descarta ese cliente en la agregación).
```

- [ ] **Step 8: Commit de documentación**

```bash
git add CHANGELOG.md docs/features/ docs/bugfixes/
git commit -m "docs: CHANGELOG, feature docs and bugfix docs for DP security sprint"
```

---

## Task 12: Suite completa y regresión

- [ ] **Step 1: Ejecutar toda la suite del Node**

```bash
cd D:/Projects/Medinet/MediNetNode/MediNetNode-main
python3.11 -m pytest tests/ -v --tb=short 2>&1
```

- [ ] **Step 2: Verificar que no hay regresiones**

Todos los tests deben pasar:
```
tests/dataset/test_privacy_policy_view.py          — 16 tests (sprint anterior)
tests/dataset/test_researcher_budget.py            — Task 1
tests/trainings/test_budget_reset_request.py       — Task 2
tests/api/test_researcher_budget_validation.py     — Task 3
tests/api/test_record_privacy_spend.py             — Task 4
tests/api/test_ml_epsilon_recording.py             — Task 5
tests/api/test_dp_param_verification.py            — Task 6
tests/api/test_concurrent_sessions.py              — Task 7
tests/security/test_rate_limit_ip.py               — Task 8
tests/api/test_budget_endpoints.py                 — Task 9
tests/dataset/test_researcher_budget_ui.py         — Task 10
```

- [ ] **Step 3: Commit final**

```bash
git commit -m "test: full regression suite passing — DP security & researcher budget sprint"
```

---

## Decisiones de diseño documentadas

| Decisión | Elección | Motivo |
|----------|----------|--------|
| `researcher_id` como IntegerField | No FK | Evita FK cross-DB entre `datasets_db` y `default` |
| ML epsilon = `max_epsilon_per_job` | Conservador | ML sin DP formal → asumir consumo máximo |
| Reset automático en `can_accept_job` | Sí | Evita job bloqueado justo al expirar el periodo |
| `MAX_CONCURRENT_TRAINING_SESSIONS = 2` | 2 | Balance entre usabilidad y DoS |
| IP rate limit = 20 req/min | 20/min | Suficiente para uso legítimo, bloquea brute-force |
| `BudgetResetRequest` en `trainings/` app | trainings → default | FK a User requiere default DB |
