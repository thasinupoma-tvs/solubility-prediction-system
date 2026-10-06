import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(r"C:\solubility prediction system")

ann_file = BASE_DIR / "results" / "ann_test_predictions.csv"
pinn_file = BASE_DIR / "results" / "pinn_test_predictions.csv"
chemprop_file = BASE_DIR / "results" / "chemprop_test_predictions.csv"


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("LOADING PREDICTION FILES")
print("=" * 70)

ann = pd.read_csv(ann_file)
pinn = pd.read_csv(pinn_file)
chemprop = pd.read_csv(chemprop_file)

print(f"ANN rows      : {len(ann)}")
print(f"PINN rows     : {len(pinn)}")
print(f"Chemprop rows : {len(chemprop)}")


# ============================================================
# CHECK ROW COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ROW COUNTS")
print("=" * 70)

assert len(ann) == len(pinn) == len(chemprop), \
    "Row counts are different!"

print(f"✓ All three files contain {len(ann)} rows.")


# ============================================================
# CHECK SAMPLE ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING SAMPLE ALIGNMENT")
print("=" * 70)

columns = [
    "SMILES_Solute",
    "SMILES_Solvent",
    "Temperature_K"
]

for col in columns:
    assert ann[col].equals(pinn[col]), \
        f"ANN and PINN mismatch in {col}"

    assert ann[col].equals(chemprop[col]), \
        f"ANN and Chemprop mismatch in {col}"

    print(f"✓ {col} matches for all rows.")


# ============================================================
# CHECK ACTUAL LOGS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ACTUAL LOGS")
print("=" * 70)

assert np.allclose(
    ann["Actual_LogS"],
    pinn["Actual_LogS"],
    atol=1e-6
)

assert np.allclose(
    ann["Actual_LogS"],
    chemprop["Actual_LogS"],
    atol=1e-6
)

print("✓ Actual_LogS values match across all files.")


# ============================================================
# CHECK PREDICTION VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING PREDICTION VALUES")
print("=" * 70)

prediction_columns = {
    "ANN": "ANN_Predicted_LogS",
    "PINN": "PINN_Predicted_LogS",
    "Chemprop": "Chemprop_Predicted_LogS"
}

for name, column in prediction_columns.items():

    predictions = pd.to_numeric(
        {"ANN": ann, "PINN": pinn, "Chemprop": chemprop}[name][column],
        errors="coerce"
    )

    nan_count = predictions.isna().sum()
    inf_count = np.isinf(predictions).sum()

    print(
        f"{name}: NaN={nan_count}, Inf={inf_count}"
    )

    assert nan_count == 0, \
        f"{name} contains NaN values"

    assert inf_count == 0, \
        f"{name} contains Inf values"


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL ALIGNMENT STATUS")
print("=" * 70)

print("✓ ANN, PINN and Chemprop are aligned row-by-row.")
print("✓ Same number of test samples.")
print("✓ Same solute, solvent and temperature.")
print("✓ Same Actual_LogS values.")
print("✓ No NaN or Inf predictions.")
print("\n✓ READY FOR ENSEMBLE.")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)