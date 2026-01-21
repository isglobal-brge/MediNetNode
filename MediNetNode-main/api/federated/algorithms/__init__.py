"""
Federated ML Algorithms Registry.

This module provides a registry pattern for federated machine learning algorithms.
Each algorithm implements the FederatedMLAlgorithm interface and can be dynamically
loaded based on configuration.

Usage:
    from .algorithms import get_algorithm

    AlgorithmClass = get_algorithm('fedsvm')
    algorithm = AlgorithmClass(X_train, y_train, config)
"""

from typing import Dict, Type
from .base import FederatedMLAlgorithm
from .fedsvm import FedSVMAlgorithm
from .feddprf import FedDPRandomForestAlgorithm

# [INIT] ALGORITHM REGISTRY: Maps algorithm names to implementation classes
ALGORITHM_REGISTRY: Dict[str, Type[FederatedMLAlgorithm]] = {
    'fedsvm': FedSVMAlgorithm,
    'fed_svm': FedSVMAlgorithm,
    'svm': FedSVMAlgorithm,

    # DP Random Forest
    'dp_random_forest': FedDPRandomForestAlgorithm,
    'dp_rf': FedDPRandomForestAlgorithm,
    'feddprf': FedDPRandomForestAlgorithm,

    # Future algorithms (uncomment when implemented):
    # 'fed_knn': FedKNNAlgorithm,
    # 'fed_naive_bayes': FedNaiveBayesAlgorithm,
    # 'fed_logistic_regression': FedLogisticRegressionAlgorithm,
}


def get_algorithm(algorithm_name: str) -> Type[FederatedMLAlgorithm]:
    """
    Get algorithm class from registry.

    Args:
        algorithm_name: Name of the algorithm (case-insensitive).
                       Valid names: 'fedsvm', 'fed_svm', 'svm'

    Returns:
        Algorithm class that can be instantiated

    Raises:
        ValueError: If algorithm not found in registry

    Example:
        >>> AlgorithmClass = get_algorithm('fedsvm')
        >>> algorithm = AlgorithmClass(X_train, y_train, config)
    """
    algorithm_name = algorithm_name.lower().strip()

    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ', '.join(sorted(ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found in registry.\n"
            f"Available algorithms: {available}"
        )

    return ALGORITHM_REGISTRY[algorithm_name]


def list_algorithms() -> list:
    """
    List all available algorithms in the registry.

    Returns:
        List of algorithm names
    """
    return sorted(ALGORITHM_REGISTRY.keys())


def register_algorithm(name: str, algorithm_class: Type[FederatedMLAlgorithm]) -> None:
    """
    Register a new algorithm in the registry (for plugins/extensions).

    Args:
        name: Algorithm name (will be lowercased)
        algorithm_class: Class implementing FederatedMLAlgorithm

    Raises:
        ValueError: If algorithm_class doesn't inherit from FederatedMLAlgorithm

    Example:
        >>> class MyCustomAlgorithm(FederatedMLAlgorithm):
        ...     pass
        >>> register_algorithm('my_algorithm', MyCustomAlgorithm)
    """
    if not issubclass(algorithm_class, FederatedMLAlgorithm):
        raise ValueError(
            f"Algorithm class must inherit from FederatedMLAlgorithm, "
            f"got {algorithm_class.__name__}"
        )

    name = name.lower().strip()

    if name in ALGORITHM_REGISTRY:
        print(f"[WARNING] Warning: Overwriting existing algorithm '{name}'")

    ALGORITHM_REGISTRY[name] = algorithm_class
    print(f"[OK] Registered algorithm: '{name}' -> {algorithm_class.__name__}")


__all__ = [
    'FederatedMLAlgorithm',
    'get_algorithm',
    'list_algorithms',
    'register_algorithm',
    'ALGORITHM_REGISTRY'
]
