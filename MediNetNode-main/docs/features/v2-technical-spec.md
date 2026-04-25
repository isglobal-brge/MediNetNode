# MediNetNode v2 — Documento Técnico

**Sprint:** DP Security & Researcher Budget (2026-04)
**Fecha:** 2026-04-25

---

## 1. Modelo de amenaza

El Node asume que el Hub (servidor de investigadores en MediNetHub) puede estar comprometido en cualquier momento. Las defensas de v2 funcionan de forma autónoma en el Node — no se confía en el Hub para hacer cumplir límites de privacidad, pausar entrenamientos o reportar parámetros DP reales.

**Vectores de ataque mitigados:**

| Vector | Mitigación |
|--------|-----------|
| Hub envía `noise_multiplier` alto en validación, bajo en training | Verificación mid-training en `dl_client.py` |
| Hub lanza flood de jobs simultáneos | Límite de 2 sesiones concurrentes por researcher |
| Hub hace brute-force de API keys | Rate limiting por IP (20 req/min) |
| Hub usa ML jobs para evadir el budget DP | ML epsilon registrado como `max_epsilon_per_job` |
| Researcher agota budget compartido de otros researchers | Budget individualizado por (dataset, researcher) |

---

## 2. Arquitectura del sistema de presupuesto DP

### 2.1 Doble capa de tracking

```
DatasetPrivacyPolicy                    ResearcherEpsilonBudget
(global por dataset)                    (por researcher + dataset)
─────────────────────                   ───────────────────────────
spent_epsilon: float      ←──┐          spent_epsilon: float      ←──┐
lifetime_budget: float       │          lifetime_budget: float       │
max_epsilon_per_job: float   │          max_epsilon_per_job: float   │
                             │          period: annual/monthly/never  │
                        _record_privacy_spend()                  _record_privacy_spend()
                             │                                        │
                             └──── actualización atómica F() ────────┘
```

Las dos capas se actualizan atómicamente al completar cada job. Un researcher que agote su budget personal es bloqueado aunque el budget global del dataset tenga epsilon disponible.

### 2.2 Routing de bases de datos

La arquitectura usa dos bases de datos separadas:

| App | DB | Motivo |
|-----|-----|--------|
| `dataset` | `datasets_db` | Datos de datasets y políticas |
| `users`, `trainings`, `audit` | `default` | Usuarios, sesiones, auditoría |

**Consecuencia:** No se pueden crear ForeignKey entre tablas de diferentes DBs.

**Solución aplicada:**
- `ResearcherEpsilonBudget.researcher_id` → `IntegerField` (no FK a `User`)
- `BudgetResetRequest.dataset_id` → `IntegerField` (no FK a `Dataset`)
- Mismo patrón que `Dataset.uploaded_by_id` ya existente en el proyecto

### 2.3 Consistencia concurrente

Los updates de epsilon usan `django.db.models.F()` para evitar race conditions cuando dos jobs del mismo researcher completan simultáneamente:

```python
ResearcherEpsilonBudget.objects.filter(
    pk=self.pk,
    spent_epsilon__lte=F('lifetime_budget'),  # guardia: no superar el límite
).update(spent_epsilon=F('spent_epsilon') + actual_epsilon)
```

La condición `spent_epsilon__lte=F('lifetime_budget')` actúa como guard atómica — si el budget ya está agotado, el UPDATE no modifica ninguna fila (0 rows affected) sin necesidad de SELECT previo ni locks explícitos.

---

## 3. Flujo de validación de un job de entrenamiento

```
POST /api/v1/start-client/
│
├── [NUEVO] Límite sesiones concurrentes
│   TrainingSession.objects.filter(user=researcher, status__in=['STARTING','ACTIVE']).count()
│   Si >= 2 → HTTP 429
│
├── validate_training_config() — validación del JSON
│
├── validate_training_permissions(researcher, model_json)
│   ├── DatasetAccess.can_train → si no → HTTP 403
│   ├── DatasetPrivacyPolicy.can_accept_job(eps) → si no → HTTP 403
│   └── [NUEVO] ResearcherEpsilonBudget.can_accept_job(eps) → si no → HTTP 403
│       └── get_or_create_for() — crea el registro si es la primera vez
│
├── Crear TrainingSession (status=STARTING)
│
└── Lanzar proceso Flower en segundo plano
    └── DLFlowerClient.fit() / MLFlowerClient.fit()
        ├── [NUEVO DL] Verificar noise_multiplier real vs aprobado
        │   Si |actual - expected| > 1e-4 → fail_training_session()
        └── [NUEVO ML] Incluir privacy_epsilon=max_epsilon_per_job en round_metrics

POST /api/v1/complete-training/ (interno, al finalizar)
└── _record_privacy_spend(session)
    ├── DatasetPrivacyPolicy.record_spent(actual_epsilon)
    └── [NUEVO] ResearcherEpsilonBudget.objects.filter(...).update(F('spent_epsilon') + eps)
```

---

## 4. Modelos de datos

### 4.1 `ResearcherEpsilonBudget` (`dataset` app → `datasets_db`)

```python
class ResearcherEpsilonBudget(models.Model):
    dataset          = ForeignKey(Dataset, CASCADE)
    researcher_id    = IntegerField(db_index=True)   # no FK (cross-DB)
    spent_epsilon    = FloatField(default=0.0)
    lifetime_budget  = FloatField()
    max_epsilon_per_job = FloatField()
    period           = CharField(choices=['annual','monthly','never'], default='annual')
    period_start     = DateTimeField(default=timezone.now)
    last_reset       = DateTimeField(null=True)

    class Meta:
        unique_together = [['dataset', 'researcher_id']]
```

**Métodos clave:**

| Método | Descripción |
|--------|-------------|
| `get_or_create_for(dataset, researcher_id, policy, period)` | Factoría — hereda límites de la policy al crear |
| `can_accept_job(estimated_epsilon)` → `(bool, str)` | Valida epsilon estimado; auto-reset si periodo vencido |
| `record_spent(actual_epsilon)` | Update atómico con F() |
| `is_period_expired()` | Compara `period_start + delta` con `timezone.now()` |
| `reset_period()` | Update atómico: spent=0, period_start=now |
| `remaining_budget` (property) | `max(0, lifetime_budget - spent_epsilon)` |

### 4.2 `BudgetResetRequest` (`trainings` app → `default`)

```python
class BudgetResetRequest(models.Model):
    dataset_id      = IntegerField(db_index=True)    # no FK (cross-DB)
    researcher_id   = IntegerField(db_index=True)    # no FK (cross-DB)
    reason          = TextField()
    status          = CharField(choices=['pending','approved','rejected'], default='pending')
    requested_at    = DateTimeField(auto_now_add=True)
    reviewed_by     = ForeignKey(User, null=True, SET_NULL)
    reviewed_at     = DateTimeField(null=True)
    review_notes    = TextField(blank=True)

    class Meta:
        constraints = [UniqueConstraint(
            fields=['dataset_id','researcher_id'],
            condition=Q(status='pending'),
            name='unique_pending_budget_reset_per_researcher_dataset',
        )]
```

**Métodos clave:**

| Método | Descripción |
|--------|-------------|
| `approve(admin, notes)` | Marca approved, registra reviewer; lanza ValueError si ya revisado |
| `reject(admin, notes)` | Marca rejected, registra reviewer; lanza ValueError si ya revisado |

---

## 5. Endpoints REST nuevos

### `POST /api/v1/budget-reset/`

**Rol requerido:** RESEARCHER

**Body:**
```json
{"dataset_id": 42, "reason": "Nuevo proyecto aprobado por comité de ética."}
```

**Respuestas:**
- `201 Created` — solicitud creada
- `400 Bad Request` — `dataset_id` o `reason` inválidos
- `403 Forbidden` — usuario no es RESEARCHER
- `409 Conflict` — ya existe solicitud pendiente para ese dataset

---

### `POST /api/v1/budget-reset/<id>/approve/`

**Rol requerido:** ADMIN

**Body:** `{"notes": "Aprobado."}`

**Efecto:** Marca la solicitud como `approved`, llama a `ResearcherEpsilonBudget.reset_period()` (zeroes `spent_epsilon`).

**Respuestas:** `200 OK` / `403 Forbidden` / `409 Conflict` (ya revisada)

---

### `POST /api/v1/budget-reset/<id>/reject/`

**Rol requerido:** ADMIN

**Body:** `{"notes": "No procede."}`

**Efecto:** Marca la solicitud como `rejected`. El presupuesto no cambia.

---

## 6. Seguridad: detalles de implementación

### 6.1 Verificación mid-training de parámetros DP

`api/federated/train_functions.py::train()` devuelve ahora 7 valores:
```python
return loss, accuracy, precision, recall, f1, epsilon, actual_noise_multiplier
```

Donde `actual_noise_multiplier = privacy_engine.noise_multiplier` si Opacus está activo.

En `api/federated/dl_client.py::DLFlowerClient.fit()`:
```python
if abs(actual_noise - expected_noise) > 1e-4:
    fail_training_session(session, "DP parameters tampered: ...")
    return self.get_parameters({}), 0, {}
```

El servidor Flower descarta los parámetros vacíos de ese round y puede detectar al cliente defectuoso.

### 6.2 Rate limiting por IP

`medinet_core/security/middleware.py::RateLimitMiddleware`:

```
Petición a /api/*
    │
    ├─ ¿tiene api_user?
    │   NO → rate limit por IP (20 req / 60s, Django cache)
    │         cache_key = f'ratelimit_ip_{client_ip}'
    │         Si count >= 20 → HTTP 429
    │
    └─ SÍ → rate limit por usuario (lógica existente)
```

El `client_ip` se extrae de `HTTP_X_FORWARDED_FOR` (primer valor) o `REMOTE_ADDR`.

### 6.3 ML epsilon proxy

Los modelos clásicos (SVM, RF) no tienen DP formal. Epsilon real = ∞. Solución conservadora:

```python
try:
    policy = DatasetPrivacyPolicy.objects.get(dataset_id=self.table_name)
    ml_epsilon = policy.max_epsilon_per_job
except Exception:
    ml_epsilon = float('inf')
```

`float('inf')` pasa la validación `> 0` y agota el budget al registrarse.

---

## 7. UI: cambios de plantillas

### ADMIN — `datasets/<id>/detail/`

Se incluye el partial `templates/dataset/partials/researcher_budgets.html` condicionalmente:

```html
{% if request.user.role.name == 'ADMIN' %}
  {% include "dataset/partials/researcher_budgets.html" %}
{% endif %}
```

El partial muestra:
- Tabla: researcher_id, gastado, restante (rojo si < 0.5), límite, periodo, barra de progreso
- Sub-sección de solicitudes de reset pendientes con botones Aprobar / Rechazar

### RESEARCHER — `/info/researcher/`

Nueva sección en `templates/users/researcher_info.html`:
- Tabla: dataset_id, gastado, restante, límite total
- Botón "Solicitar reset" por dataset → abre modal con textarea para justificación
- Formulario POST a `/api/v1/budget-reset/` con `dataset_id` y `reason`

---

## 8. Migraciones

| Migración | App | Tabla |
|-----------|-----|-------|
| `dataset/migrations/0005_add_researcher_epsilon_budget.py` | dataset | `dataset_researcherepsilonbudget` |
| `trainings/migrations/0002_add_budget_reset_request.py` | trainings | `trainings_budgetresetrequest` |

**Aplicar en producción:**
```bash
python manage.py migrate --database=datasets_db  # ResearcherEpsilonBudget
python manage.py migrate                          # BudgetResetRequest
```

---

## 9. Algoritmos ML federados

### 9.1 FedSVM (`FedSVMStrategy` — MediNetHub)

Variante OptMD del SVM federado. Protocolo de ronda:

1. Hub envía vectores soporte de las otras instituciones a cada cliente.
2. Cliente retraina su SVM local con los SVs recibidos.
3. Cliente devuelve sus nuevos vectores soporte al Hub.
4. Hub detecta convergencia cuando `Δ(SVs) < server_eps`.

No hay parámetros de modelo en el sentido DL — los "parámetros" son arrays de vectores soporte serializados como `np.ndarray`.

**Configuración (enviada al Node como `model_json`):**
```json
{
  "algorithm": {
    "ml_algorithm": {
      "type": "svm",
      "hyperparameters": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "server_eps": 0.01
      }
    }
  }
}
```

### 9.2 FedDP Random Forest (`FedDPRandomForestStrategy` — MediNetHub)

Protocolo de ronda:

1. Hub serializa el bosque global actual (lista de `tree_state` dicts) y lo envía a cada cliente.
2. Cliente entrena N árboles locales con DP (mecanismo de Laplace en umbrales de split).
3. Cliente devuelve árboles en el mismo formato `tree_state`.
4. Hub agrega árboles, filtra duplicados por SHA y actualiza `global_forest`.

**Formato `tree_state`:**
```python
{
    'tree_structure': {  # árbol recursivo
        'type': 'split',
        'feature': int,
        'threshold': float,
        'left': {...},   # nodo hijo izquierdo
        'right': {...},  # nodo hijo derecho
    },  # o {'type': 'leaf', 'label': int}
    'n_classes': int,
    'max_depth': int,
    'feature_bounds': [[min, max], ...],
    'epsilon': float,   # epsilon gastado por este árbol
}
```

**Predicción del bosque (`predict()`):**

Implementada en `FedDPRandomForestStrategy.predict()` mediante travesía recursiva de cada árbol en `global_forest`:

```python
def _predict_tree(tree_structure, x):
    node = tree_structure
    while node['type'] != 'leaf':
        node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
    return int(node['label'])
```

Votación por mayoría (`hard`) sobre todos los árboles del bosque. No requiere reconstruir `SecureDPTree` — opera directamente sobre el `tree_structure` serializado.

### 9.3 Selector de estrategia (`server.py::get_strategy()`)

```python
if model_type == 'ml':
    ml_algorithm = model_json.get('algorithm', {}).get('ml_algorithm', {}).get('type')
    if ml_algorithm == 'random_forest':
        return FedDPRandomForestStrategy(server_manager, ...)
    elif ml_algorithm == 'svm':
        return FedSVMStrategy(server_manager, ...)
else:
    return FedAvgWithDP(server_manager, ...)  # DL por defecto
```

---

## 10. Tests (MediNetNode)

Todos los tests nuevos usan `pytest-django` con el mismo patrón que el resto del proyecto. Tests que necesitan ambas DBs usan `@pytest.mark.django_db(databases=['default', 'datasets_db'])`.

El `sys.modules.setdefault('magic', None)` al inicio de los tests que importan `dataset.views` o `dataset.uploader` es necesario en Windows para evitar que `libmagic.dll` cuelgue el proceso.

**Cobertura añadida:** 54 tests nuevos, todos pasan en el entorno de desarrollo (Windows 10, Python 3.11, SQLite in-memory).
