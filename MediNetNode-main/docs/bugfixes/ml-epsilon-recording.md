# Bugfix: ML jobs no registraban epsilon consumido

**Severidad:** Media
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivo afectado:** `api/federated/ml_client.py`

## Descripción

Los jobs de entrenamiento con modelos de ML clásico (SVM, Random Forest, etc.) no incluían
`privacy_epsilon` en sus `round_metrics`. Como consecuencia, `_record_privacy_spend()` en
`api/federated/utils.py` no actualizaba `DatasetPrivacyPolicy.spent_epsilon` ni
`ResearcherEpsilonBudget.spent_epsilon` para esos jobs.

Un researcher podía lanzar iteraciones ilimitadas de modelos ML sin consumir presupuesto DP.

## Raíz del problema

`MLFlowerClient.fit()` construía `round_metrics` sin el campo `privacy_epsilon`, que `_record_privacy_spend()`
requiere para actualizar el contador.

## Decisión de diseño

Los modelos ML (SVM, RF) no tienen privacidad diferencial formal (epsilon = infinito matemáticamente).
La solución conservadora: registrar `max_epsilon_per_job` de la `DatasetPrivacyPolicy` como epsilon
consumido. Esto:
1. Agota el presupuesto del researcher de forma conservadora.
2. Informa al ADMIN que el slot fue usado por un job ML.
3. Incentiva a los researchers a usar DP-SGD cuando el presupuesto es limitado.

## Fix

En `MLFlowerClient.fit()`, antes de construir `round_metrics`:

```python
try:
    from dataset.models import DatasetPrivacyPolicy
    _policy = DatasetPrivacyPolicy.objects.get(dataset_id=self.table_name)
    ml_epsilon = _policy.max_epsilon_per_job
except Exception:
    ml_epsilon = float('inf')

round_metrics = {
    ...
    'privacy_epsilon': ml_epsilon,
    'model_type': 'ml',
}
```
