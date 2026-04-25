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
POST /api/v1/start-client
  validate_training_permissions()
    1. DatasetPrivacyPolicy.can_accept_job(eps)        ← límite global del dataset
    2. ResearcherEpsilonBudget.get_or_create_for(...)  ← crear si es primera vez
    3. ResearcherEpsilonBudget.can_accept_job(eps)     ← límite individual del researcher

complete_training_session()
    1. DatasetPrivacyPolicy.record_spent(actual_eps)   ← actualiza global
    2. ResearcherEpsilonBudget.record_spent(actual_eps)← actualiza individual
```
