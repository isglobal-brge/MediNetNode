# Bugfix: noise_multiplier no verificado mid-training

**Severidad:** Alta
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivos afectados:** `api/federated/dl_client.py`, `api/federated/train_functions.py`

## Descripción

El `noise_multiplier` era validado en `validate_training_permissions()` antes de conectar al
servidor Flower, pero no se verificaba que Opacus usara el mismo valor real durante el
entrenamiento. Un Hub comprometido podía modificar la configuración entre la validación y el
entrenamiento real, reduciendo el ruido DP sin que el Node lo detectara.

## Raíz del problema

`train()` en `train_functions.py` no devolvía el `noise_multiplier` realmente usado por
Opacus. `DLFlowerClient.fit()` no comparaba el valor aprobado con el valor real.

## Fix

### train_functions.py

`train()` ahora retorna 7 valores: `(loss, accuracy, precision, recall, f1, epsilon, actual_noise)`.
El séptimo valor es `privacy_engine.noise_multiplier` si Opacus está activo, o el valor
esperado del config para entrenamientos sin DP (sin discrepancia detectable).

### dl_client.py

En `fit()`, después de llamar a `train()`:

```python
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
            f"DP parameters tampered: expected noise_multiplier={expected_noise}, "
            f"actual={actual_noise}. Training aborted for security.",
        )
        return self.get_parameters({}), 0, {}
```

Si hay discrepancia > 1e-4, el entrenamiento se aborta con `fail_training_session()` y
se devuelven parámetros vacíos al servidor Flower (que descartará ese round).
