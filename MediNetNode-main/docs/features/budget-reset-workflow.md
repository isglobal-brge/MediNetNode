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
| `POST` | `/api/v1/budget-reset/` | RESEARCHER | Crear solicitud con motivo |
| `GET`  | `/api/v1/budget-reset/` | RESEARCHER | Ver mis solicitudes |
| `POST` | `/api/v1/budget-reset/<id>/approve/` | ADMIN | Aprobar y aplicar reset |
| `POST` | `/api/v1/budget-reset/<id>/reject/`  | ADMIN | Rechazar con notas |

## Restricciones de seguridad

- Solo **una solicitud pendiente** por (dataset, researcher) a la vez — constraint de DB.
- El motivo es obligatorio (1–1000 caracteres).
- El reset **solo lo aplica el ADMIN** — el researcher nunca puede resetearse a sí mismo.
- Las solicitudes son permanentes (no se borran) para auditoría completa.
- `BudgetResetRequest` vive en `trainings` app (`default` DB) para poder hacer FK a `User`.
