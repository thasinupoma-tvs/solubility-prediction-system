import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


from src.chemprop_training import (
    train_and_evaluate_chemprop
)


DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CHEMPROP_MODEL_DIR = MODEL_DIR / "chemprop_model"


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 101

random.seed(SEED)
np.random.seed(SEED)

print(f"Random seed: {SEED}")


# ============================================================
# Data paths
# ============================================================

TRAIN_FILE = DATA_DIR / "train_raw.csv"
TEST_FILE = DATA_DIR / "test_raw.csv"


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Loading Chemprop training data")
print("=" * 60)

print(f"Training file: {TRAIN_FILE}")
print(f"Test file    : {TEST_FILE}")

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

print(f"\nTraining samples: {len(df_train)}")
print(f"Test samples    : {len(df_test)}")


# ============================================================
#  Chemprop configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 20


# ============================================================
# Print final configuration
# ============================================================

print("\n" + "=" * 60)
print("Final Chemprop Configuration")
print("=" * 60)

print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print("Input         : Solute + Solvent molecular graphs + Temperature")


# ============================================================
# Train final Chemprop
# ============================================================

print("\n" + "=" * 60)
print("Starting Final Chemprop Training")
print("=" * 60)

result = train_and_evaluate_chemprop(
    train_data=df_train,
    test_data=df_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_model=True
)


# ============================================================
# Print final performance
# ============================================================

print("\n" + "=" * 60)
print("Final Chemprop Performance")
print("=" * 60)

print(f"RMSE           : {result['rmse']:.4f}")
print(f"MAE            : {result['mae']:.4f}")
print(f"R²             : {result['r2']:.4f}")
print(f"Epochs trained : {result['epochs_trained']}")


# ============================================================
# Reproduce Chemprop test filtering
# ============================================================

print("\n" + "=" * 60)
print("Preparing Test Identifiers")
print("=" * 60)


def is_valid_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        return mol is not None

    except Exception:

        return False


# Same filtering used inside chemprop_training.py

chemprop_test = df_test[
    [
        "SMILES_Solute",
        "SMILES_Solvent",
        "Temperature_K",
        "LogS(mol/L)"
    ]
].copy()


# Remove missing values

chemprop_test = chemprop_test.dropna(
    subset=[
        "SMILES_Solute",
        "SMILES_Solvent",
        "LogS(mol/L)"
    ]
)


# Remove invalid solute SMILES

chemprop_test = chemprop_test[
    chemprop_test["SMILES_Solute"].apply(
        is_valid_smiles
    )
]


# Remove invalid solvent SMILES

chemprop_test = chemprop_test[
    chemprop_test["SMILES_Solvent"].apply(
        is_valid_smiles
    )
]


chemprop_test = chemprop_test.reset_index(
    drop=True
)


# ============================================================
# Check prediction alignment
# ============================================================

y_pred = np.asarray(
    result["y_pred"]
).reshape(-1)

y_test = np.asarray(
    result["y_test"]
).reshape(-1)


print(f"Filtered Chemprop test samples: {len(chemprop_test)}")
print(f"Chemprop predictions          : {len(y_pred)}")


if len(chemprop_test) != len(y_pred):

    raise ValueError(
        "Mismatch between filtered Chemprop test samples "
        "and predictions.\n"
        f"Filtered test samples: {len(chemprop_test)}\n"
        f"Predictions: {len(y_pred)}"
    )


# ============================================================
# Save Chemprop predictions with identifiers
# ============================================================

print("\n" + "=" * 60)
print("Saving Chemprop Predictions")
print("=" * 60)


chemprop_predictions_df = chemprop_test[
    [
        "SMILES_Solute",
        "SMILES_Solvent",
        "Temperature_K"
    ]
].copy()


chemprop_predictions_df["Actual_LogS"] = y_test

chemprop_predictions_df["Chemprop_Predicted_LogS"] = y_pred


predictions_file = (
    RESULTS_DIR /
    "chemprop_test_predictions.csv"
)


chemprop_predictions_df.to_csv(
    predictions_file,
    index=False
)


print(
    f"[OK] Chemprop predictions saved to:"
)

print(
    f"     {predictions_file}"
)

print(
    f"Rows saved: {len(chemprop_predictions_df)}"
)


# ============================================================
# Save Chemprop metrics
# ============================================================

print("\n" + "=" * 60)
print("Saving Chemprop Performance Metrics")
print("=" * 60)


metrics_df = pd.DataFrame([{

    "Model": "Chemprop",

    "Feature_Set":
        "SMILES_Solute_Solvent_Temp",

    "Batch_Size":
        BATCH_SIZE,

    "Max_Epochs":
        EPOCHS,

    "RMSE":
        result["rmse"],

    "MAE":
        result["mae"],

    "R2":
        result["r2"],

    "Epochs_Trained":
        result["epochs_trained"]

}])


metrics_file = (
    RESULTS_DIR /
    "chemprop_final_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


print(
    f"[OK] Chemprop metrics saved to:"
)

print(
    f"     {metrics_file}"
)


# ============================================================
# Check saved files
# ============================================================

print("\n" + "=" * 60)
print("Checking Saved Files")
print("=" * 60)


saved_files = [

    CHEMPROP_MODEL_DIR /
    "model_0" /
    "best.pt",

    MODEL_DIR /
    "chemprop_scaler.pkl",

    RESULTS_DIR /
    "chemprop_test_predictions.csv",

    RESULTS_DIR /
    "chemprop_final_metrics.csv"

]


for file_path in saved_files:

    if file_path.exists():

        print(
            f"[OK] {file_path}"
        )

    else:

        print(
            f"[WARNING] Missing: {file_path}"
        )


print("\n" + "=" * 60)
print("Final Chemprop training completed.")
print("=" * 60)