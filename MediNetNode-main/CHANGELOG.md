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
