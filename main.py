"""
main.py
-------
Entry point for the Building Insulation Optimizer.

Run:
    python main.py
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="1/1", category=UserWarning)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.indicators.hv import Hypervolume

from src.config import (
    MATERIALS, POP_SIZE, N_GENERATIONS, RANDOM_SEED, REF_POINT, RESULT_DIR
)
from src.loader import load_all_models, compute_coefficients
from src.evaluate import load_cache, save_cache
from src.problem import InsulationProblem
from src.postprocess import find_knee_point, save_results, plot_pareto

import os
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8")

print("""
╔══════════════════════════════════════════════════════╗
║       Building Insulation Optimizer  v1.0            ║
╚══════════════════════════════════════════════════════╝

Geometry inputs
  A1 : Gross floor area of the conditioned zone (m²)
  A2 : External wall area of the residential floor (m²)
  A6 : Window-to-wall ratio  (e.g. 0.30 for 30 %)

Material / HVAC inputs
  R_wall      : Thermal resistance of the external wall WITHOUT insulation (m²K/W)
  U_glass     : Window assembly U-value (W/m²K)
  SHGC        : Solar heat gain coefficient of the window (0–1)
  COP_cooling : COP of the cooling system
  COP_heating : COP of the heating system

Price inputs
  Insulation  : Cost per m² per cm of thickness (local currency)
  Gas price   : Natural gas cost per kWh
  Electricity : Electricity cost per kWh

Fixed assumptions
  Heating setpoint : 20 °C
  Cooling setpoint : 25 °C
  R_roof = R_floor : 2.50 m²K/W
""")

input("Press Enter when you are ready to continue … ")

A1, A2, A6 = map(float, input(
    "\nGeometry — enter A1, A2, A6 separated by commas: "
).split(","))

r_wall, u_glass, shgc, copc, coph = map(float, input(
    "Materials — enter R_wall, U_glass, SHGC, COP_cooling, COP_heating separated by commas: "
).split(","))

price_insulation_square_meter, price_gas, price_elec = map(float, input(
    "Prices — enter Insulation, Gas, Electricity separated by commas: "
).split(","))

cooling_fuel_type = input("Cooling fuel type (gas / electricity): ").strip().lower()
heating_fuel_type = input("Heating fuel type (gas / electricity): ").strip().lower()
useful_life       = float(input("Useful life of the building (years): "))

material_choice = int(input(
    "Optimise for: 1=Rock Wool  2=XPS  3=EPS  4=Glass Wool  0=All  → "
))

# ---------------------------------------------------------------------------
# Derived geometry  (reference building: 730 m² floor, 910 m² wall)
# ---------------------------------------------------------------------------
A4  = 910 * A6          # reference window area
A5  = 910 - A4          # reference wall area (no windows)
A44 = A2  * A6          # user window area
A55 = A2  - A44         # user opaque wall area

# ---------------------------------------------------------------------------
# Load models & compute coefficients
# ---------------------------------------------------------------------------
print("\nLoading ANN models …")
heat_model, cool_model, convertor, scaler_thermal, scaler_convertor = load_all_models()
cool_coef, heat_coef = compute_coefficients(A1, A2, A6, convertor, scaler_convertor)
print(f"  cool_coef = {cool_coef:.4f}   heat_coef = {heat_coef:.4f}")

# ---------------------------------------------------------------------------
# Shared keyword arguments for evaluate_solution
# ---------------------------------------------------------------------------
cache = load_cache()

eval_kwargs = dict(
    r_wall                     = r_wall,
    shgc                       = shgc,
    u_glass                    = u_glass,
    copc                       = copc,
    coph                       = coph,
    A4                         = A4,
    A5                         = A5,
    A55                        = A55,
    cool_coef                  = cool_coef,
    heat_coef                  = heat_coef,
    useful_life                = useful_life,
    cooling_fuel_type          = cooling_fuel_type,
    heating_fuel_type          = heating_fuel_type,
    price_gas                  = price_gas,
    price_elec                 = price_elec,
    price_insulation_square_meter = price_insulation_square_meter,
    cool_model                 = cool_model,
    heat_model                 = heat_model,
    scaler_thermal             = scaler_thermal,
    cache                      = cache,
)

# ---------------------------------------------------------------------------
# Run NSGA-II
# ---------------------------------------------------------------------------
print("\nRunning NSGA-II optimisation …")
np.random.seed(RANDOM_SEED)

problem    = InsulationProblem(material_choice, eval_kwargs)
algorithm  = NSGA2(pop_size=POP_SIZE)
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8, cvtol=1e-6, ftol=0.005, period=20, n_max_gen=N_GENERATIONS
)

res = minimize(problem, algorithm, termination, seed=RANDOM_SEED, verbose=True)
save_cache(cache)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
hv_value  = Hypervolume(ref_point=REF_POINT).do(res.F)
knee_idx  = find_knee_point(res.F)
knee_F    = res.F[knee_idx]
knee_vars = res.X[knee_idx]

mat_name = (
    MATERIALS[int(knee_vars[1])]["name"]
    if material_choice == 0
    else MATERIALS[material_choice]["name"]
)

print(f"\nHypervolume indicator : {hv_value:.4e}")
print(f"Knee-point objectives : LCC = {knee_F[0]:,.2f}   LCCO₂ = {knee_F[1]:,.2f} kg")
print(f"Knee-point solution   : {knee_vars[0]:.2f} cm of {mat_name}")

csv_path = save_results(res, material_choice, knee_idx)
print(f"\nPareto front saved to : {csv_path}")

plot_pareto(res, material_choice, knee_idx, hv_value)
