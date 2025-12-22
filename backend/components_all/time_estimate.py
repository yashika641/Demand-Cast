import numpy as np

def estimate_training_time(
    n_rows: int,
    n_features: int,
    model_type: str,
    complexity: float = 1.0,
    hardware: str = "cpu",
    fe_overhead: float = 0.2,   # +20% time for feature engineering
    batch_size: int = 32,
    epochs: int = 10
):
    """
    Estimate model training time based on dataset complexity and model choice.

    Args:
        n_rows (int): Number of samples.
        n_features (int): Number of features.
        model_type (str): Model name: "xgboost", "lightgbm", "random_forest",
                          "lstm", "transformer", "prophet".
        complexity (float): Factor 0.5–3.0 adjusting for hyperparameters.
        hardware (str): "cpu" or "gpu".
        fe_overhead (float): Fractional overhead for preprocessing.
        batch_size (int): (Deep learning models only)
        epochs (int): (Deep learning models only)

    Returns:
        float: Estimated training time in seconds.
    """

    # 🔹 Base computational cost factor
    data_complexity = (n_rows * np.log1p(n_features)) / 1e6  # normalized

    # 🔹 Model-specific multipliers (relative cost)
    model_cost_map = {
        "xgboost": 4.0,
        "lightgbm": 3.0,
        "random_forest": 2.5,
        "logistic_regression": 0.5,
        "linear_regression": 0.3,
        "svm": 5.0,
        "catboost": 5.5,
        "lstm": 10.0,
        "transformer": 20.0,
        "prophet": 1.0,
    }

    model_type = model_type.lower()
    model_multiplier = model_cost_map.get(model_type, 3.0)

    # 🔹 Hardware multipliers
    hardware_speed = {"cpu": 1.0, "gpu": 0.4}  # GPU is roughly 2.5× faster

    hw_multiplier = hardware_speed.get(hardware, 1.0)

    # 🔹 Special handling for deep learning models
    if model_type in ["lstm", "transformer"]:
        steps_per_epoch = n_rows / batch_size
        deep_learning_cost = steps_per_epoch * epochs * 0.002  # normalized
    else:
        deep_learning_cost = 0

    # 🔹 Base time estimation
    time_seconds = (
        data_complexity * model_multiplier * complexity * hw_multiplier
    )

    # Add deep learning cost if applicable
    time_seconds += deep_learning_cost

    # 🔹 Feature engineering overhead
    time_seconds *= (1 + fe_overhead)

    return round(time_seconds, 2)
