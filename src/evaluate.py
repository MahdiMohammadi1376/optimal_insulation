"""
evaluate.py
-----------
Core objective-function evaluation: given an insulation thickness and material
ID, returns the Life Cycle Cost (LCC) and Life Cycle CO₂ (LCCO₂) together
with the building's annual heating and cooling loads.

Results are cached to disk so repeated evaluations of the same (thickness,
material) pair are instant.
"""

import os
import pickle
import numpy as np

from src.config import MATERIALS, EF_GAS, EF_ELEC, CACHE_FILE, RESULT_DIR


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    """Load the evaluation cache from disk, or return an empty dict."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """Persist the evaluation cache to disk."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def evaluate_solution(
    x: np.ndarray,
    *,
    r_wall: float,
    shgc: float,
    u_glass: float,
    copc: float,
    coph: float,
    A4: float,
    A5: float,
    A55: float,
    cool_coef: float,
    heat_coef: float,
    useful_life: float,
    cooling_fuel_type: str,
    heating_fuel_type: str,
    price_gas: float,
    price_elec: float,
    price_insulation_square_meter: float,
    material_choice: int,
    cool_model,
    heat_model,
    scaler_thermal,
    cache: dict,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Evaluate LCC and LCCO₂ for one candidate solution.

    Parameters
    ----------
    x : array of length 2 (thickness_cm, mat_id) when material_choice == 0,
        or length 1 (thickness_cm) when a specific material is pre-selected.

    Returns
    -------
    objectives : np.ndarray([LCC, LCCO₂])
    loads      : (cooling_load_MWh, heating_load_MWh)
    """
    PENALTY = np.array([1e12, 1e12])

    thickness = x[0]
    mat_id = int(x[1]) if material_choice == 0 else material_choice

    if mat_id not in MATERIALS:
        return PENALTY, (0.0, 0.0)

    mat = MATERIALS[mat_id]
    r_insulation = 0.01 / mat["lambda"]          # R per cm of insulation
    cost_per_cm_m2 = (
        mat["cost"] if material_choice == 0 else price_insulation_square_meter
    )

    # --- Cache lookup ---
    key = (round(thickness, 4), mat_id)
    if key in cache:
        return cache[key]

    # --- ANN prediction ---
    effective_r_wall = r_wall + thickness * r_insulation
    x_input = np.array([[
        effective_r_wall,
        np.log(effective_r_wall),
        shgc,
        A4,
        u_glass,
        A5,
    ]])
    x_scaled = scaler_thermal.transform(x_input)
    cool_load = float(cool_coef * cool_model.predict(x_scaled, verbose=0).item())
    heat_load = float(heat_coef * heat_model.predict(x_scaled, verbose=0).item())

    # --- Energy breakdown (kWh/year) ---
    gas_cool = elec_cool = gas_heat = elec_heat = 0.0
    if cooling_fuel_type == "gas":
        gas_cool  = 1_000 * cool_load / copc
    else:
        elec_cool = 1_000 * cool_load / copc

    if heating_fuel_type == "gas":
        gas_heat  = 1_000 * heat_load / coph
    else:
        elec_heat = 1_000 * heat_load / coph

    # --- Life Cycle Cost ---
    insulation_cost = thickness * cost_per_cm_m2 * A55
    op_cost = (
        (gas_cool + gas_heat)   * price_gas  * useful_life
        + (elec_cool + elec_heat) * price_elec * useful_life
    )
    lcc = insulation_cost + op_cost

    # --- Life Cycle CO₂ ---
    volume_m3    = thickness * 0.01 * A55
    embodied_co2 = volume_m3 * mat["density"] * mat["co2_ef"]
    eol_co2      = embodied_co2 * 0.10
    op_co2       = (
        (gas_cool + gas_heat)   * EF_GAS
        + (elec_cool + elec_heat) * EF_ELEC
    ) * useful_life
    lcco2 = embodied_co2 + eol_co2 + op_co2

    result = (np.array([lcc, lcco2]), (cool_load, heat_load))
    cache[key] = result
    return result
