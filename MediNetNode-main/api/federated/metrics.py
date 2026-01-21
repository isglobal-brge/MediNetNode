import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

# --- Métricas personalizadas para segmentación ---
def iou(preds, targets):
    """
    Intersection over Union (IoU) para segmentación.
    
    Args:
        preds: Predicciones binarias (numpy array)
        targets: Ground truth binario (numpy array)
    
    Returns:
        float: IoU score
    """
    # Convertir a arrays numpy si no lo son
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Asegurar que sean binarios
    preds = (preds > 0.5).astype(np.float32)
    targets = (targets > 0.5).astype(np.float32)
    
    # Calcular intersección y unión
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets) - intersection
    
    # Evitar división por cero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def dice_score(preds, targets):
    """
    Dice Score (F1-score para segmentación).
    
    Args:
        preds: Predicciones binarias (numpy array)
        targets: Ground truth binario (numpy array)
    
    Returns:
        float: Dice score
    """
    # Convertir a arrays numpy si no lo son
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Asegurar que sean binarios
    preds = (preds > 0.5).astype(np.float32)
    targets = (targets > 0.5).astype(np.float32)
    
    # Calcular intersección
    intersection = np.sum(preds * targets)
    
    # Calcular suma total
    total = np.sum(preds) + np.sum(targets)
    
    # Evitar división por cero
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total

def pixel_accuracy(preds, targets):
    """
    Pixel Accuracy para segmentación.
    
    Args:
        preds: Predicciones binarias (numpy array)
        targets: Ground truth binario (numpy array)
    
    Returns:
        float: Pixel accuracy
    """
    # Convertir a arrays numpy si no lo son
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Asegurar que sean binarios
    preds = (preds > 0.5).astype(np.float32)
    targets = (targets > 0.5).astype(np.float32)
    
    # Calcular píxeles correctos
    correct_pixels = np.sum(preds == targets)
    total_pixels = preds.size
    
    return correct_pixels / total_pixels if total_pixels > 0 else 0.0

def sensitivity(preds, targets):
    """
    Sensitivity (Recall) para segmentación.
    
    Args:
        preds: Predicciones binarias (numpy array)
        targets: Ground truth binario (numpy array)
    
    Returns:
        float: Sensitivity score
    """
    # Convertir a arrays numpy si no lo son
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Asegurar que sean binarios
    preds = (preds > 0.5).astype(np.float32)
    targets = (targets > 0.5).astype(np.float32)
    
    # Calcular true positives y false negatives
    true_positives = np.sum(preds * targets)
    false_negatives = np.sum(targets) - true_positives
    
    # Evitar división por cero
    if (true_positives + false_negatives) == 0:
        return 1.0 if true_positives == 0 else 0.0
    
    return true_positives / (true_positives + false_negatives)

def specificity(preds, targets):
    """
    Specificity para segmentación.
    
    Args:
        preds: Predicciones binarias (numpy array)
        targets: Ground truth binario (numpy array)
    
    Returns:
        float: Specificity score
    """
    # Convertir a arrays numpy si no lo son
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Asegurar que sean binarios
    preds = (preds > 0.5).astype(np.float32)
    targets = (targets > 0.5).astype(np.float32)
    
    # Calcular true negatives y false positives
    true_negatives = np.sum((1 - preds) * (1 - targets))
    false_positives = np.sum(preds) - np.sum(preds * targets)
    
    # Evitar división por cero
    if (true_negatives + false_positives) == 0:
        return 1.0 if true_negatives == 0 else 0.0
    
    return true_negatives / (true_negatives + false_positives)

METRICS = {
    "binary_classification": {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    },
    "regression": {
        "mae": mean_absolute_error,
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score,
    },
    "segmentation": {
        "iou": iou,
        "dice_score": dice_score,
        "pixel_accuracy": pixel_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    },
}

def compute_metrics(preds, targets, metrics_list, problem_type):
    """
    Calcula múltiples métricas para un tipo de problema específico.
    
    Args:
        preds: Predicciones del modelo
        targets: Valores verdaderos
        metrics_list: Lista de nombres de métricas a calcular
        problem_type: Tipo de problema ('binary_classification', 'regression', 'segmentation')
    
    Returns:
        dict: Diccionario con los resultados de las métricas
    """
    results = {}
    available_metrics = METRICS.get(problem_type, {})
    
    for metric in metrics_list:
        func = available_metrics.get(metric)
        if func:
            try:
                # Para sklearn metrics, el orden es (y_true, y_pred)
                # Para métricas de segmentación personalizadas, el orden es (preds, targets)
                if problem_type in ["binary_classification", "regression"]:
                    results[metric] = func(targets, preds)
                else:
                    results[metric] = func(preds, targets)
            except Exception as e:
                results[metric] = f"error: {str(e)}"
        else:
            results[metric] = "not_applicable"
    
    return results 