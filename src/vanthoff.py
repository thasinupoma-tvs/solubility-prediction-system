import numpy as np
import pandas as pd
from scipy.stats import linregress


R = 8.314462618  # J/(mol·K)


def calculate_vanthoff_parameters(group):
    """
    Calculate van't Hoff parameters for one
    solute-solvent pair.
    """

    # Need at least 3 temperature points
    if group["Temperature_K"].nunique() < 3:
        return None

    # Remove missing values
    data = group[
        ["Temperature_K", "LogS(mol/L)"]
    ].dropna()

    if len(data) < 3:
        return None

    T = data["Temperature_K"].values
    logS = data["LogS(mol/L)"].values

    # x = 1/T
    x = 1.0 / T

    # y = log10(S)
    y = logS

    # Linear regression
    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r2 = result.rvalue ** 2

    # log10(S) =
    # -DeltaH / (2.303 R) * (1/T) + C

    delta_H = -slope * 2.303 * R

    return {
        "n_temperatures": len(data),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "delta_H_J_mol": delta_H,
        "delta_H_kJ_mol": delta_H / 1000.0,
    }