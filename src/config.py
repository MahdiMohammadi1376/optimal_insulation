import numpy as np
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

HEATING_FILE        = os.path.join(MODEL_DIR, "heatingKfold2.h5")
COOLING_FILE        = os.path.join(MODEL_DIR, "coolingKfold2.h5")
CONVERTOR_FILE      = os.path.join(MODEL_DIR, "coefKfold9.h5")
SCALER_THERMAL_FILE = os.path.join(MODEL_DIR, "scaler_thermal.pkl")
SCALER_CONV_FILE    = os.path.join(MODEL_DIR, "scaler_convertor.pkl")

CACHE_FILE = os.path.join(RESULT_DIR, "eval_cache.pkl")

MATERIALS = {
    1: {"name": "Rock Wool",  "lambda": 0.042, "density": 50, "co2_ef":  1.045, "cost": 600_000},
    2: {"name": "XPS",        "lambda": 0.041, "density": 40, "co2_ef": 11.42,  "cost": 500_000},
    3: {"name": "EPS",        "lambda": 0.047, "density": 15, "co2_ef": 10.54,  "cost": 450_000},
    4: {"name": "Glass Wool", "lambda": 0.044, "density": 20, "co2_ef":  1.533, "cost": 550_000},
}

# CO2 emission factors
EF_GAS  = 0.20   # kg CO₂ / kWh  (natural gas)
EF_ELEC = 0.55   # kg CO₂ / kWh  (electricity)

# NSGA-II hyperparameters
POP_SIZE      = 50
N_GENERATIONS = 50
RANDOM_SEED   = 42
REF_POINT     = np.array([2.5e10, 2e6])
