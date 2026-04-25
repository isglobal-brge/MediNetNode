# MediNetNode v2 — What's New

**Sprint:** DP Security & Researcher Budget (2026-04)

---

## Resumen ejecutivo

Esta versión refuerza la postura de seguridad del Node frente a un Hub potencialmente comprometido, e introduce un sistema de presupuesto de Privacidad Diferencial (DP) individualizado por researcher. Los datos de pacientes están ahora protegidos por múltiples capas de defensa que funcionan de forma autónoma, sin confiar en el Hub para hacer cumplir los límites de privacidad.

---

## Nuevas funcionalidades

### Presupuesto epsilon por researcher (`ResearcherEpsilonBudget`)

Antes, `DatasetPrivacyPolicy` tenía un único contador de epsilon compartido por todos los researchers. Ahora cada researcher tiene su propio presupuesto individualizado por dataset.

**Qué cambia para el ADMIN:**
- En `datasets/<id>/detail/` aparece una nueva sección "Presupuesto ε por Researcher" con tabla de consumo individual.
- Puede ver qué researcher ha gastado cuánto y cuánto le queda.
- Recibe notificaciones de solicitudes de reset pendientes directamente en la misma página.

**Qué cambia para el RESEARCHER:**
- En `/info/researcher/` aparece la sección "Mis presupuestos de Privacidad Diferencial".
- Puede ver su epsilon gastado y restante por cada dataset.
- Puede solicitar un reset de presupuesto con justificación escrita.

**Periodos de reset automático configurables:**
| Periodo | Duración |
|---------|----------|
| `annual` (por defecto) | 1 año desde el inicio del periodo |
| `monthly` | 1 mes |
| `never` | Sin reset automático |

---

### Flujo de reset de presupuesto supervisado (`BudgetResetRequest`)

Los researchers pueden solicitar un reset de su presupuesto cuando lo agoten (p.ej., para un nuevo proyecto o tras aprobación de comité de ética). El ADMIN aprueba o rechaza con notas auditadas.

**Endpoints REST:**

| Método | URL | Rol | Descripción |
|--------|-----|-----|-------------|
| `POST` | `/api/v1/budget-reset/` | RESEARCHER | Solicitar reset |
| `GET` | `/api/v1/budget-reset/` | RESEARCHER | Ver mis solicitudes |
| `POST` | `/api/v1/budget-reset/<id>/approve/` | ADMIN | Aprobar + aplicar reset |
| `POST` | `/api/v1/budget-reset/<id>/reject/` | ADMIN | Rechazar con notas |

**Restricciones:** solo una solicitud pendiente por (dataset, researcher) a la vez. El researcher nunca puede resetearse a sí mismo.

---

### Límite de sesiones Flower concurrentes

Máximo **2 sesiones de entrenamiento activas** simultáneamente por researcher. Un Hub comprometido no puede lanzar un flood de jobs para agotar el presupuesto o la RAM del hospital.

- Respuesta: HTTP 429 con mensaje descriptivo.
- Configurable via la constante `MAX_CONCURRENT_TRAINING_SESSIONS` en `api/views.py`.

---

### Rate limiting por IP para peticiones no autenticadas

`RateLimitMiddleware` ahora aplica un límite de **20 peticiones/minuto por IP** para peticiones sin API key válida. Previene brute-force de credenciales desde un Hub comprometido.

- Ventana: 60 segundos (via Django cache framework).
- Respuesta: HTTP 429.
- Independiente del rate limiting por usuario para peticiones autenticadas.

---

## Correcciones de seguridad

### [ALTA] Parámetros DP sin verificación mid-training

El `noise_multiplier` se validaba antes de conectar al servidor Flower, pero no se comprobaba que Opacus usara ese mismo valor durante el entrenamiento. Un Hub comprometido podía reducir el ruido DP en silencio.

**Fix:** `train()` devuelve ahora el `noise_multiplier` real de Opacus. Si difiere del aprobado en más de 1e-4, el entrenamiento se aborta con `fail_training_session()` y se devuelven parámetros vacíos al servidor Flower.

---

### [ALTA] Rate limiting inefectivo para peticiones fallidas

`RateLimitMiddleware` omitía completamente las peticiones sin `api_user`, permitiendo brute-force sin restricción alguna.

**Fix:** Nuevo bloque de rate limiting por IP al inicio de `__call__()` para cualquier petición a `/api/` sin usuario autenticado.

---

### [MEDIA] ML jobs no registraban epsilon consumido

Los jobs con modelos clásicos (SVM, Random Forest) no incluían `privacy_epsilon` en `round_metrics`. El presupuesto DP no se actualizaba para esos entrenamientos.

**Fix:** `MLFlowerClient.fit()` ahora incluye `privacy_epsilon = max_epsilon_per_job` de la policy del dataset (enfoque conservador: sin DP formal, se registra el máximo permitido por job).

---

## Resumen de archivos nuevos y modificados

| Archivo | Cambio |
|---------|--------|
| `dataset/models.py` | + `ResearcherEpsilonBudget` |
| `dataset/migrations/0005_*` | Migración nueva tabla |
| `trainings/models.py` | + `BudgetResetRequest` |
| `trainings/migrations/0002_*` | Migración nueva tabla |
| `api/views.py` | + validación researcher budget, + límite sesiones concurrentes |
| `api/budget_views.py` | Nuevo — endpoints reset budget |
| `api/urls.py` | + 3 rutas budget-reset |
| `api/federated/utils.py` | + actualización `ResearcherEpsilonBudget` en `_record_privacy_spend` |
| `api/federated/ml_client.py` | + `privacy_epsilon` en `round_metrics` |
| `api/federated/dl_client.py` | + verificación `noise_multiplier` mid-training |
| `api/federated/train_functions.py` | `train()` retorna 7 valores (+ `actual_noise`) |
| `medinet_core/security/middleware.py` | + rate limiting por IP no autenticada |
| `dataset/views.py` | + `researcher_budgets` y `pending_reset_requests` al contexto |
| `templates/dataset/detail.html` | + sección presupuestos (ADMIN) |
| `templates/dataset/partials/researcher_budgets.html` | Nuevo partial |
| `templates/users/researcher_info.html` | + sección "Mis presupuestos DP" |
| `users/views.py` | + `researcher_budgets` al contexto de `researcher_info` |
| `CHANGELOG.md` | Nuevo |
| `docs/features/researcher-epsilon-budget.md` | Nuevo |
| `docs/features/budget-reset-workflow.md` | Nuevo |
| `docs/bugfixes/ml-epsilon-recording.md` | Nuevo |
| `docs/bugfixes/rate-limit-unauthenticated.md` | Nuevo |
| `docs/bugfixes/dp-param-verification.md` | Nuevo |

---

## Tests añadidos en este sprint

| Archivo de test | Tests |
|----------------|-------|
| `tests/dataset/test_researcher_budget.py` | 16 |
| `tests/dataset/test_researcher_budget_ui.py` | 3 |
| `tests/trainings/test_budget_reset_request.py` | 5 |
| `tests/api/test_researcher_budget_validation.py` | 3 |
| `tests/api/test_record_privacy_spend.py` | 1 |
| `tests/api/test_ml_epsilon_recording.py` | 1 |
| `tests/api/test_dp_param_verification.py` | 1 |
| `tests/api/test_concurrent_sessions.py` | 1 |
| `tests/api/test_budget_endpoints.py` | 4 |
| `tests/security/test_rate_limit_ip.py` | 3 |
| `tests/dataset/test_privacy_policy_view.py` | 16 (Tarea 6 anterior) |
| **Total** | **54** |
