import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MATERIALS, RESULT_DIR

# Knee point
def find_knee_point(F: np.ndarray) -> int:
    F_min = np.min(F, axis=0)
    F_max = np.max(F, axis=0)
    F_norm = (F - F_min) / (F_max - F_min + 1e-10)
    distances = np.linalg.norm(F_norm, axis=1)
    return int(np.argmin(distances))

# Export
def save_results(
    res,
    material_choice: int,
    knee_idx: int,
) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    loads = res.opt.get("loads")

    # Build DataFrame
    if material_choice == 0:
        df = pd.DataFrame(
            np.hstack((res.X, res.F, loads)),
            columns=["thickness_cm", "material_id", "LCC", "LCCO2_kg",
                     "cooling_load_MWh", "heating_load_MWh"],
        )
        df["material_name"] = [MATERIALS[int(x[1])]["name"] for x in res.X]
    else:
        df = pd.DataFrame(
            np.column_stack((res.X, res.F, loads)),
            columns=["thickness_cm", "LCC", "LCCO2_kg",
                     "cooling_load_MWh", "heating_load_MWh"],
        )
        df["material_name"] = MATERIALS[material_choice]["name"]

    csv_path = os.path.join(RESULT_DIR, "pareto_insulation.csv")
    df.to_csv(csv_path, index=False)

    # Knee-point summary
    knee_F    = res.F[knee_idx]
    knee_vars = res.X[knee_idx]
    mat_name  = (
        MATERIALS[int(knee_vars[1])]["name"]
        if material_choice == 0
        else MATERIALS[material_choice]["name"]
    )
    txt_path = os.path.join(RESULT_DIR, "knee_point.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            f"Index     : {knee_idx}\n"
            f"LCC       : {knee_F[0]:,.2f}\n"
            f"LCCO2     : {knee_F[1]:,.2f} kg\n"
            f"Thickness : {knee_vars[0]:.2f} cm\n"
            f"Material  : {mat_name}\n"
        )

    return csv_path

# Plot
def plot_pareto(res, material_choice: int, knee_idx: int, hv_value: float) -> None:
    """
    Draw and save the Pareto front scatter plot, colour-coded by material.
    """
    COLOR_MAP = {1: "green", 2: "black", 3: "blue", 4: "red"}

    mat_ids = (
        [int(x[1]) for x in res.X]
        if material_choice == 0
        else [material_choice] * len(res.X)
    )
    unique_mats = sorted(set(mat_ids))

    fig, ax = plt.subplots(figsize=(10, 8))

    for mat_id in unique_mats:
        mask = np.array([mid == mat_id for mid in mat_ids])
        ax.scatter(
            res.F[mask, 0], res.F[mask, 1],
            c=COLOR_MAP[mat_id],
            label=MATERIALS[mat_id]["name"],
            alpha=0.6, s=80, marker="o",
        )

    # Knee point
    knee_mat_id = int(res.X[knee_idx, 1]) if material_choice == 0 else material_choice
    ax.scatter(
        res.F[knee_idx, 0], res.F[knee_idx, 1],
        c=COLOR_MAP[knee_mat_id],
        marker="*", s=400, edgecolors="gold", linewidths=1.5,
        label=f"Knee Point ({MATERIALS[knee_mat_id]['name']})",
        zorder=5,
    )

    ax.set_xlabel("Life Cycle Cost", fontsize=22)
    ax.set_ylabel("Life Cycle CO₂ (kg CO₂-eq)", fontsize=22)
    ax.set_title(
        f"Pareto Front: LCC vs. LCCO₂  (HV={hv_value:.2e}, "
        f"Materials: {len(unique_mats)}/4)",
        fontsize=18,
    )
    ax.legend(fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "pareto_lcc_lcco2.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {out_path}")
