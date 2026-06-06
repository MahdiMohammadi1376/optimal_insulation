"""
problem.py
----------
pymoo Problem subclass that wraps evaluate_solution() for use with NSGA-2.
"""

import numpy as np
from pymoo.core.problem import Problem

from src.evaluate import evaluate_solution


class InsulationProblem(Problem):
    """
    Two-objective minimization problem:
        f1 = Life Cycle Cost   (LCC)
        f2 = Life Cycle CO2   (LCCO2)

    Decision variables
    ------------------
    When material_choice == 0  →  [thickness_cm, mat_id]   (2 variables)
    Otherwise                  →  [thickness_cm]             (1 variable)
    """

    def __init__(self, material_choice: int, eval_kwargs: dict):
        """
        Parameters
        ----------
        material_choice : 0 = optimise all materials simultaneously;
                          1-4 = fix to that material ID.
        eval_kwargs     : keyword arguments forwarded to evaluate_solution().
        """
        self.material_choice = material_choice
        self.eval_kwargs = eval_kwargs

        if material_choice == 0:
            n_var = 2
            xl = [0.0, 1.0]
            xu = [10.0, 4.999]        # mat_id will be int-cast → 1..4
        else:
            n_var = 1
            xl = [0.0]
            xu = [10.0]

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    # ------------------------------------------------------------------
    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        objectives = []
        loads_list = []

        for x in X:
            obj, loads = evaluate_solution(
                x,
                material_choice=self.material_choice,
                **self.eval_kwargs,
            )
            objectives.append(obj)
            loads_list.append(loads)

        out["F"]     = np.row_stack(objectives)
        out["loads"] = np.array(loads_list)
