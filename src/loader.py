import joblib
import numpy as np
import warnings
import tensorflow as tf
from keras.models import load_model
from keras.regularizers import Regularizer

from src.config import (
    HEATING_FILE, COOLING_FILE, CONVERTOR_FILE,
    SCALER_THERMAL_FILE, SCALER_CONV_FILE,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="1/1", category=UserWarning)

# Custom regularizer
class CustomRegularizer(Regularizer):
    """L2 regularizer that penalises all features except the first (R_wall)."""

    def __init__(self, strength: float = 0.015):
        self.strength = strength

    def __call__(self, x):
        penalty = tf.reduce_sum(tf.square(x[:, 1:]))
        return self.strength * penalty

    def get_config(self) -> dict:
        return {"strength": self.strength}

# Public API
def load_all_models():
    custom_objs = {"CustomRegularizer": CustomRegularizer}

    scaler_thermal   = joblib.load(SCALER_THERMAL_FILE)
    scaler_convertor = joblib.load(SCALER_CONV_FILE)
    convertor  = load_model(CONVERTOR_FILE)
    heat_model = load_model(HEATING_FILE, custom_objects=custom_objs)
    cool_model = load_model(COOLING_FILE, custom_objects=custom_objs)

    return heat_model, cool_model, convertor, scaler_thermal, scaler_convertor

def compute_coefficients(
    A1: float,
    A2: float,
    A6: float,
    convertor,
    scaler_convertor,
) -> tuple[float, float]:
  
    a = np.array([[A1 / 730, A2 / 910, A6, 1]])
    b = np.array([[A1 / 730, A2 / 910, A6, 2]])

    cool_coef = float(convertor.predict(scaler_convertor.transform(a), verbose=0).item())
    heat_coef = float(convertor.predict(scaler_convertor.transform(b), verbose=0).item())

    return cool_coef, heat_coef
