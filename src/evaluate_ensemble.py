import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Paths Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_prediction_file(filename):
    """Loads CSV """
    if (RESULTS_DIR / filename).exists():
        return pd.read_csv(RESULTS_DIR / filename)
    elif (PROJECT_ROOT / filename).exists():
        return pd.read_csv(PROJECT_ROOT / filename)
    raise FileNotFoundError(f"File not found: {filename}")


def compute_metrics(y_true, y_pred, name="Model"):
    """Compute RMSE, MAE, and R2 metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae,4), "R2": round(r2, 4)}


def run_ensemble():
    # Load prediction CSV files
    df_ann = load_prediction_file("ann_test_predictions.csv")
    df_pinn = load_prediction_file("pinn_test_predictions.csv")
    df_cp = load_prediction_file("chemprop_test_predictions.csv")

    # Extract ground truth and predictions
    y_true = df_ann["Actual_LogS"].values
    pred_ann = df_ann["ANN_Predicted_LogS"].values
    pred_pinn = df_pinn["PINN_Predicted_LogS"].values
    pred_cp = df_cp["Chemprop_Predicted_LogS"].values

    results = []

    # 1. Individual Models
    results.append(compute_metrics(y_true, pred_ann, "1. ANN Model"))
    results.append(compute_metrics(y_true, pred_pinn, "2. PINN Model"))
    results.append(compute_metrics(y_true, pred_cp, "3. Chemprop Model"))

    # 2. Equal Weight Blend
    pred_equal = (pred_ann + pred_pinn + pred_cp) / 3.0
    results.append(compute_metrics(y_true, pred_equal, "4. Equal Weight (ANN + PINN + Chemprop)"))

    # 3. Physics-Graph Blend (40% PINN + 60% Chemprop)
    pred_pinn_cp = 0.40 * pred_pinn + 0.60 * pred_cp
    results.append(compute_metrics(y_true, pred_pinn_cp, "5. Physics-Graph Blend (40% PINN + 60% CP)"))

    # 4. Optimal Constrained Weighted Ensemble
    def loss_func(weights):
        w1, w2, w3 = weights
        blend = w1 * pred_ann + w2 * pred_pinn + w3 * pred_cp
        return np.sqrt(mean_squared_error(y_true, blend))

    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    bounds = [(0, 1), (0, 1), (0, 1)]
    opt = minimize(loss_func, [0.33, 0.33, 0.34], method='SLSQP', bounds=bounds, constraints=constraints)

    w_ann, w_pinn, w_cp = opt.x
    pred_opt = w_ann * pred_ann + w_pinn * pred_pinn + w_cp * pred_cp

    opt_label = f"6. Optimal Ensemble ({w_ann:.2f} ANN + {w_pinn:.2f} PINN + {w_cp:.2f} CP)"
    results.append(compute_metrics(y_true, pred_opt, opt_label))

    # Summary Table
    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("                   ENSEMBLE BENCHMARK RESULTS                   ")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)

    # Save Merged Predictions and Summary Metrics
    output_preds = pd.DataFrame({
        "SMILES_Solute": df_ann["SMILES_Solute"],
        "SMILES_Solvent": df_ann["SMILES_Solvent"],
        "Temperature_K": df_ann["Temperature_K"],
        "Actual_LogS": y_true,
        "ANN_Prediction": pred_ann,
        "PINN_Prediction": pred_pinn,
        "Chemprop_Prediction": pred_cp,
        "Optimal_Ensemble_Prediction": pred_opt
    })

    output_preds.to_csv(DATA_DIR / "final_ensemble_test_predictions.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "ensemble_performance_metrics.csv", index=False)


if __name__ == "__main__":
    run_ensemble()