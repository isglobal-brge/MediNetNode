# Diseño — Presupuesto de privacidad por-researcher como capa de enforcement (H1 + H2 + H4)

- **Fecha:** 2026-06-28
- **Componente:** MediNetNode (`MediNetNode/MediNetNode-main`)
- **Origen:** Análisis forense del sistema de budget policy. Hallazgos H1 (reset inefectivo), H2 (fail-open de contabilidad DP) y H4 (aprobación de reset no atómica).

---

## 1. Problema

El sistema de presupuesto de privacidad (epsilon / ε) tiene **dos contadores** que miden casi lo mismo, con tamaños idénticos:

| Capa | Modelo | Rol actual | Problema |
|------|--------|-----------|----------|
| Global del dataset | `DatasetPrivacyPolicy.spent_epsilon` | Techo compartido por todos los researchers; **bloquea** | Crea desigualdad: el primer researcher que gasta agota el bote para todos |
| Por-researcher | `ResearcherEpsilonBudget.spent_epsilon` | Cuota individual; **bloquea** | `lifetime_budget` se copia idéntico al global (`dataset/models.py:580`) → la capa es inerte |

**H1 — Reset inefectivo:** el workflow de reset (`api/budget_views.py:101-124`) solo pone a cero la cuota del researcher, pero el techo global del dataset no se resetea (ni tiene método para hacerlo). Como el global bloquea primero y ambos son iguales, tras un reset el researcher sigue bloqueado. Un dataset agotado queda bloqueado para siempre. El test `tests/api/test_budget_endpoints.py:74` no lo detecta porque solo verifica `researcher.spent == 0`, nunca que el researcher pueda volver a entrenar.

**H2 — Fail-open de contabilidad DP:** `record_spent` descarta el registro entero cuando el gasto real superaría el presupuesto (`dataset/models.py:526-538`). Como el job ya entrenó y ya filtró ese ε real, descartar el registro hace que `spent` subestime la fuga real → futuros jobs se aprueban contra presupuesto fantasma. Varios tests consagran este comportamiento como correcto.

**H4 — Aprobación de reset no atómica:** `approve_budget_reset` persiste `status=approved` y *luego* aplica el reset; si el budget no existe, la solicitud queda aprobada sin efecto.

## 2. Decisión de semántica

El número configurado en el dataset (vía preset de sensibilidad) representa **la cuota de CADA investigador**, no un bote global compartido. Decisión del owner del producto: priorizar la equidad entre investigadores sobre el tope agregado formal de DP.

- **`ResearcherEpsilonBudget` = capa que MANDA.** Acepta/bloquea jobs y es lo que se resetea. Su `lifetime_budget` es la cuota individual (seguir copiándola desde la política es correcto bajo esta semántica).
- **`DatasetPrivacyPolicy` = plantilla de configuración + agregado de auditoría.** Define límites (`max_epsilon_per_job`, `lifetime_budget`) y acumula el total gastado por todos para visibilidad del admin, pero **ya no bloquea**.

### Matiz DP aceptado conscientemente

La garantía formal de DP es sobre los datos del paciente: la fuga real total = suma de ε de todos los investigadores. Bajo este modelo, el tope formal del dataset crece con el número de investigadores. Se acepta este tradeoff; el contador global se conserva como **auditoría/monitoreo** (no como muro de bloqueo) para mantener visibilidad de la fuga total real.

## 3. Cambios de diseño

### 3.1 Enforcement — `api/views.py` (≈ líneas 566-643)

- **Mantener** fail-closed: política ausente → `403`; error de sistema → `503`.
- **Eliminar** la llamada bloqueante `policy.can_accept_job(...)`.
- **Único portón:** `researcher_budget.can_accept_job(estimated_eps)`, obtenido vía `ResearcherEpsilonBudget.get_or_create_for(...)`. Esa función ya valida ε inválido (NaN/inf/≤0 → bloquea), el límite por-job y la cuota restante del researcher. No se pierde ninguna protección.
- La estimación fallida (`estimate_job_epsilon` → `inf`) sigue bloqueando porque `can_accept_job(inf)` falla el chequeo `isfinite`.

`DatasetPrivacyPolicy.can_accept_job` se conserva como método (uso de auditoría / defensa en profundidad) pero deja de invocarse como portón.

### 3.2 Recording — `api/federated/utils.py` (`_record_privacy_spend`, ≈ 172-290) + modelos

- **`DatasetPrivacyPolicy.record_spent` (auditoría):** acumular **siempre** el gasto real con incremento atómico incondicional (`F('spent_epsilon') + delta`). Quitar el `WHERE` condicional que descarta: el agregado legítimamente supera el número-plantilla, y descartarlo falsearía la auditoría.
- **`ResearcherEpsilonBudget.record_spent` (cuota real):** acumular también con la **verdad**. Si un job filtró más de lo estimado y cruza la cuota, se registra igual → `remaining_budget` se clampa a 0 y el **siguiente** job se bloquea en el portón. La protección contra carreras la sigue dando el incremento atómico `F()+delta` (no se pierden escrituras); se elimina únicamente el *descarte* (corrige H2).
- Se mantiene el ignorar valores no-positivos / no-finitos (centinela -1.0, NaN, inf).
- Se mantiene el salto de sesiones experimentales (`use_experiment=True`).

### 3.3 Reset — `api/budget_views.py`

- Sin cambios en la lógica de qué se resetea (solo la cuota del researcher): ahora es efectiva porque el researcher es quien bloquea.
- **H4:** envolver `reset_req.approve(...)` + `budget.reset_period()` en una transacción atómica (`transaction.atomic`) para que no quede una solicitud aprobada sin reset aplicado. Si no existe `ResearcherEpsilonBudget`, mantener el warning actual (no es error fatal).

## 4. Impacto en datos / migraciones

- Ninguna migración de esquema necesaria. Los campos existentes se reutilizan; solo cambia la semántica de quién bloquea y cómo se acumula.
- No hay backfill: los datos existentes siguen siendo válidos (el global pasa a ser interpretado como auditoría).

## 5. Tests

### Reescribir (codifican el comportamiento erróneo H2)
- `dataset/tests/test_privacy_policy.py`: `test_record_spent_skipped_when_budget_already_exhausted`, `test_record_spent_exactly_one_delta_over_budget_is_skipped`, y los relacionados de overrun → ahora deben afirmar **acumulación veraz** (el gasto se registra aunque cruce el límite) y bloqueo en el *siguiente* `can_accept_job`.
- `tests/dataset/test_researcher_budget.py`: `test_rejects_overrun`, `test_concurrent_record_spent_no_overrun` → afirmar acumulación veraz; la concurrencia ya no pierde escrituras pero **sí** acumula ambas.
- `tests/api/test_record_privacy_spend.py`: `test_researcher_budget_not_overrun_on_record` → afirmar que el gasto real se registra.

### Añadir
- **Test que esconde H1 (clave):** researcher agota su cuota → admin aprueba reset → el researcher **vuelve a entrenar con éxito** (no recibe 403). En `tests/api/test_budget_endpoints.py` o integración.
- **Enforcement a nivel vista (zona oscura):** cuota del researcher agotada → `403` con `budget_exhausted`; verificar que el contador global de auditoría **no** bloquea aunque esté "alto".
- **Auditoría veraz:** tras varios jobs de varios researchers, `policy.spent_epsilon` = suma real (puede superar el número-plantilla).
- **H4:** si `approve` falla a mitad, ni el status ni el reset quedan aplicados (atomicidad).

### Ajustar
- `api/tests/test_privacy_budget_enforcement.py`: los tests que esperaban `403` desde `policy.can_accept_job` (inf/NaN/negativo) deben pasar ahora por `researcher_budget.can_accept_job` (mismo resultado 403, distinta ruta). Asegurar que exista/cree el `ResearcherEpsilonBudget`.

## 6. Criterios de aceptación

1. Un researcher con cuota agotada que recibe un reset aprobado puede volver a entrenar.
2. Otro researcher con cuota disponible nunca es bloqueado por el gasto de un tercero.
3. `policy.spent_epsilon` refleja la suma real de todos los gastos (auditoría veraz), sin descartes.
4. La cuota del researcher acumula la fuga real; cruzar el límite bloquea el siguiente job, no descarta el registro del anterior.
5. Política ausente → 403; error de sistema → 503; ε inválido → 403 (fail-closed intacto).
6. `approve` es atómico (status + reset, todo o nada).

## 7. Fuera de alcance

- Tope global opcional de respaldo activable por el admin (descartado en esta iteración).
- Modelado del número de rondas federadas en `estimate_job_epsilon` (H3) — se aborda por separado.
- Validación de existencia/acceso de dataset en `request_budget_reset` (L1) — se aborda por separado.
