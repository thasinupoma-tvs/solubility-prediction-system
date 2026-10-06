import sys
import torch
import random
import numpy as np
from pathlib import Path
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed: {SEED}")


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Import model and PINN training function
# ============================================================

from src.ann_model import Solubility_Predictor
from src.pinn_training import train_and_evaluate_PINN


# ============================================================
# Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Data paths
# ============================================================

TRAIN_FILE = (
    DATA_DIR /
    "train_combined_features.csv"
)

TEST_FILE = (
    DATA_DIR /
    "test_combined_features.csv"
)


# ============================================================
# Physics lookup table
# ============================================================

PHYSICS_LOOKUP_FILE = (
    RESULTS_DIR /
    "physics_lookup.csv"
)


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Loading PINN training data")
print("=" * 60)

print(f"Training file : {TRAIN_FILE}")
print(f"Test file     : {TEST_FILE}")
print(f"Physics lookup: {PHYSICS_LOOKUP_FILE}")


df_train = pd.read_csv(
    TRAIN_FILE
)

df_test = pd.read_csv(
    TEST_FILE
)

physics_lookup = pd.read_csv(
    PHYSICS_LOOKUP_FILE
)


print(
    f"\nTraining samples: {len(df_train)}"
)

print(
    f"Test samples    : {len(df_test)}"
)

print(
    f"Physics groups  : {len(physics_lookup)}"
)


# ============================================================
# Best PINN configuration
# ============================================================

BEST_LEARNING_RATE = 0.0005
BEST_BATCH_SIZE = 64
BEST_EPOCHS = 300


# ============================================================
# Physics loss weight
# ============================================================

LAMBDA_PHYS = 0.001


# ============================================================
# Final configuration
# ============================================================

print("\n" + "=" * 60)
print("Final PINN Configuration")
print("=" * 60)

print(
    f"Learning rate : {BEST_LEARNING_RATE}"
)

print(
    f"Batch size    : {BEST_BATCH_SIZE}"
)

print(
    f"Epochs        : {BEST_EPOCHS}"
)

print(
    f"Physics weight: {LAMBDA_PHYS}"
)

print(
    "Feature set   : Combined + Temperature"
)

print(
    "Base model    : Solubility_Predictor"
)


# ============================================================
# Start PINN training
# ============================================================

print("\n" + "=" * 60)
print("Starting Final PINN Training")
print("=" * 60)


result = train_and_evaluate_PINN(

    train_data=df_train,

    test_data=df_test,

    physics_lookup=physics_lookup,

    model_class=Solubility_Predictor,

    learning_rate=BEST_LEARNING_RATE,

    batch_size=BEST_BATCH_SIZE,

    num_epochs=BEST_EPOCHS,

    lambda_phys=LAMBDA_PHYS,

    save_model=True,

    model_filename="best_pinn_model.pth"
)


# ============================================================
# Final performance
# ============================================================

print("\n" + "=" * 60)
print("Final PINN Performance")
print("=" * 60)

print(
    f"RMSE                : "
    f"{result['rmse']:.4f}"
)

print(
    f"MAE                 : "
    f"{result['mae']:.4f}"
)

print(
    f"R²                  : "
    f"{result['r2']:.4f}"
)

print(
    f"Best validation loss: "
    f"{result['best_val_loss']:.6f}"
)

print(
    f"Epochs trained      : "
    f"{result['epochs_trained']}"
)


# ============================================================
# Prepare prediction arrays
# ============================================================

y_test = np.asarray(
    result["y_test"]
).reshape(-1)

y_pred = np.asarray(
    result["y_pred"]
).reshape(-1)


# ============================================================
# Check prediction length
# ============================================================

if len(df_test) != len(y_test):

    raise ValueError(
        "Mismatch between test data and PINN "
        "actual values.\n"
        f"Test rows: {len(df_test)}\n"
        f"Actual values: {len(y_test)}"
    )


if len(df_test) != len(y_pred):

    raise ValueError(
        "Mismatch between test data and PINN "
        "predictions.\n"
        f"Test rows: {len(df_test)}\n"
        f"Predictions: {len(y_pred)}"
    )


# ============================================================
# Save PINN predictions with sample identifiers
# ============================================================

print("\n" + "=" * 60)
print("Saving PINN Predictions")
print("=" * 60)


pinn_predictions_df = df_test[
    [
        "SMILES_Solute",
        "SMILES_Solvent",
        "Temperature_K"
    ]
].copy()


pinn_predictions_df[
    "Actual_LogS"
] = y_test


pinn_predictions_df[
    "PINN_Predicted_LogS"
] = y_pred


# ============================================================
# Save prediction file
# ============================================================

predictions_file = (
    RESULTS_DIR /
    "pinn_test_predictions.csv"
)


pinn_predictions_df.to_csv(
    predictions_file,
    index=False
)


print(
    "[OK] PINN predictions saved to:"
)

print(
    f"     {predictions_file}"
)

print(
    f"Rows saved: "
    f"{len(pinn_predictions_df)}"
)


# ============================================================
# Save final PINN metrics
# ============================================================

print("\n" + "=" * 60)
print("Saving PINN Performance Metrics")
print("=" * 60)


metrics_df = pd.DataFrame([{

    "Model": "PINN",

    "Feature_Set":
        "Combined_with_Temp",

    "Learning_Rate":
        BEST_LEARNING_RATE,

    "Batch_Size":
        BEST_BATCH_SIZE,

    "Max_Epochs":
        BEST_EPOCHS,

    "Lambda_Phys":
        LAMBDA_PHYS,

    "RMSE":
        result["rmse"],

    "MAE":
        result["mae"],

    "R2":
        result["r2"],

    "Best_Val_Loss":
        result["best_val_loss"],

    "Epochs_Trained":
        result["epochs_trained"]

}])


metrics_file = (
    RESULTS_DIR /
    "pinn_final_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


print(
    "[OK] PINN metrics saved to:"
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

    MODEL_DIR /
    "best_pinn_model.pth",

    MODEL_DIR /
    "pinn_scaler.pkl",

    MODEL_DIR /
    "pinn_feature_columns.pkl",

    MODEL_DIR /
    "pinn_train_medians.pkl",

    RESULTS_DIR /
    "pinn_test_predictions.csv",

    RESULTS_DIR /
    "pinn_final_metrics.csv"

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
print("Final PINN training completed.")
print("=" * 60)