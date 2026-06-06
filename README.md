# 🏠 Building Insulation Optimizer

A multi-objective optimization framework for determining optimal insulation thickness and material selection in residential buildings, minimizing **Life Cycle Cost (LCC)** and **Life Cycle CO₂ emissions (LCCO₂)** simultaneously using NSGA-II.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Inputs & Parameters](#inputs--parameters)
- [Outputs](#outputs)
- [ANN Models](#ann-models)
- [Optimization Algorithm](#optimization-algorithm)
- [Assumptions](#assumptions)
- [Dependencies](#dependencies)

---

## Overview

This tool helps engineers, architects, and researchers find the optimal insulation configuration for a residential building by simultaneously minimizing:

- **Life Cycle Cost (LCC):** Insulation installation cost + operational energy cost over the building's useful life
- **Life Cycle CO₂ (LCCO₂):** Embodied carbon from insulation + end-of-life emissions + operational carbon from energy use

The tool defaults to four insulation materials (Rock Wool, XPS, EPS, Glass Wool), and users can define their own insulation material with required properties in the "config.py" file. It can optimize for a single material or all four simultaneously, producing a **Pareto front** that represents the trade-off between cost and emissions.

---

## Methodology

The workflow consists of three stages:

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│  ANN-based Load Prediction      │
│  • Heating Load Model (Keras)   │
│  • Cooling Load Model (Keras)   │
│  • Coefficient Converter (Keras)│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Multi-Objective Optimization   │
│  • NSGA-II (pymoo)              │
│  • Decision variables:          │
│    - Insulation thickness (cm)  │
│    - Material type (1-n)        │
│  • Objectives: LCC, LCCO₂       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Results & Visualization        │
│  • Pareto front plot            │
│  • Knee point selection         │
│  • CSV export                   │
└─────────────────────────────────┘
```

### Load Prediction Pipeline

1. **Heating & Cooling ANN models** were trained on EnergyPlus simulation data for a reference building geometry. They predict loads as a function of wall thermal resistance (R-value), window U-value, SHGC, and window/wall areas.
2. A **Coefficient Convertor model** scales these predictions to match the user's actual building dimensions (gross floor area, external wall area, WWR), enabling generalization beyond the training geometry.

---

## Project Structure

```
insulation-optimizer/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py                  # Entry point: user prompts + NSGA-II execution
├── config.py                # Hyperparameters, constants, material properties
│
├── src/
│   ├── __init__.py
│   ├── ann_loader.py        # Load ANN models & scalers, compute conversion coefficients
│   ├── evaluate.py          # Objective function with caching
│   ├── problem.py           # pymoo Problem definition (InsulationProblem)
│   └── postprocess.py       # Knee point, Pareto plot, CSV export
│
├── models/                  # Trained Keras models & scalers
│   ├── heatingKfold2.h5
│   ├── coolingKfold2.h5
│   ├── coefKfold9.h5
│   ├── scaler_thermal.pkl
│   └── scaler_convertor.pkl
│
├── results/                 # Auto-generated output directory
│   ├── pareto_insulation.csv
│   ├── pareto_lcc_lcco2.png
│   ├── knee_point.txt
│   └── eval_cache.pkl
│
└── notebooks/
    └── exploration.ipynb    # Optional: demo and result visualization
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MahdiMohammadi1376/insulation-optimizer.git
cd insulation-optimizer
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the optimizer from the project root:

```bash
python main.py
```

The script will prompt you step-by-step for building geometry, material properties, energy prices, and optimization preferences. Follow the on-screen instructions.

**Example session:**

```
Enter geometry information in order 'A1', 'A2', 'A6': 2700, 1880, 0.30
Enter material information in order 'R_wall', 'U_glass', 'SHGC', 'COP cooling', 'COP heating': 1.5, 2.5, 0.35, 6.0, 0.8
Enter prices in order 'Insulation per square meter', 'Gas', 'Electricity': 0, 2000, 6000
Cooling fuel consumption (Gas or Electricity): electricity
Heating fuel consumption (Gas or Electricity): gas
Useful life of the building (years): 30
Optimize for specific material (1=Rock Wool, 2=XPS, 3=EPS, 4=Glass Wool, 0=All): 0
```

---

## Inputs & Parameters

### Geometry

| Parameter | Description                             | Unit           |
| --------- | --------------------------------------- | -------------- |
| `A1`      | Gross floor area of conditioned zone    | m²             |
| `A2`      | External wall area of residential floor | m²             |
| `A6`      | Window-to-wall ratio                    | fraction (0–1) |

### Thermal & HVAC Properties

| Parameter     | Description                                              | Unit  |
| ------------- | -------------------------------------------------------- | ----- |
| `R_wall`      | Thermal resistance of external wall (without insulation) | m²K/W |
| `U_glass`     | Window assembly thermal transmittance                    | W/m²K |
| `SHGC`        | Solar heat gain coefficient of window                    | —     |
| `COP_cooling` | Coefficient of Performance, cooling system               | —     |
| `COP_heating` | Coefficient of Performance, heating system               | —     |

### Economic Inputs

| Parameter           | Description                               | Unit               |
| ------------------- | ----------------------------------------- | ------------------ |
| `Insulation price`  | Cost per m² per cm of thickness           | local currency     |
| `Gas price`         | Natural gas energy cost                   | local currency/kWh |
| `Electricity price` | Electricity energy cost                   | local currency/kWh |
| `Useful life`       | Building service life for LCC calculation | years              |

### Material Options

| ID  | Material   | λ (W/mK) | Density (kg/m³) | CO₂-EF (kg CO₂/kg) |
| --- | ---------- | -------- | --------------- | ------------------ |
| 1   | Rock Wool  | 0.042    | 50              | 1.045              |
| 2   | XPS        | 0.041    | 40              | 11.42              |
| 3   | EPS        | 0.047    | 15              | 10.54              |
| 4   | Glass Wool | 0.044    | 20              | 1.533              |

---

## Outputs

All outputs are saved to the `results/` directory:

| File                    | Description                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `pareto_insulation.csv` | All Pareto-optimal solutions with thickness, material, LCC, LCCO₂, and energy loads |
| `pareto_lcc_lcco2.png`  | Scatter plot of the Pareto front, color-coded by material                           |
| `knee_point.txt`        | The knee-point solution (best compromise between LCC and LCCO₂)                     |
| `eval_cache.pkl`        | Cached ANN evaluations to speed up re-runs                                          |

### Example Pareto Front

The plot shows the trade-off between Life Cycle Cost and Life Cycle CO₂, with each material in a distinct color and the knee point marked with a star (⭐).

---

## ANN Models

### Training Data

The heating and cooling load models were trained on EnergyPlus simulation results generated by varying:

- Wall R-value
- Window U-value and SHGC
- Window and wall areas

The **coefficient convertor** model maps loads from the reference simulation geometry to user-specified building dimensions.

### Architecture

All models were built with Keras/TensorFlow. The heating and cooling models use a custom `CustomRegularizer` that penalizes non-R-value features, improving physical interpretability.

### Retraining

To retrain the models on your own EnergyPlus dataset, modify the training scripts (not included in this release) and replace the `.h5` and `.pkl` files in `models/`.

---

## Optimization Algorithm

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is used via the [pymoo](https://pymoo.org/) library.

| Hyperparameter                    | Value             |
| --------------------------------- | ----------------- |
| Population size                   | 150               |
| Max generations                   | 100               |
| Random seed                       | 42                |
| Convergence tolerance (`ftol`)    | 0.005             |
| Reference point (for Hypervolume) | [2.5×10¹⁰, 2×10⁶] |

**Decision variables:**

- Insulation thickness: 0–10 cm (continuous)
- Material ID: 1–4 (integer, only when optimizing all materials)

**Knee point selection:** The solution closest to the origin in normalized objective space is selected as the recommended balanced compromise.

---

## Assumptions

- Heating setpoint: **20°C**
- Cooling setpoint: **25°C**
- R-value for roof and floor: **2.50 m²K/W** (fixed)
- End-of-life CO₂ taken as **10%** of embodied CO₂
- CO₂ emission factors: Natural gas = **0.20 kg CO₂/kWh**, Electricity = **0.55 kg CO₂/kWh**
- Insulation cost is linear with thickness and wall area

---

## Dependencies

Install all with:

```bash
pip install -r requirements.txt
````

---
