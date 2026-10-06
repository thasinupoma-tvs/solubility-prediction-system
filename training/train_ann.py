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

from src.ann_model import Solubility_Predictor
from src.ann_training import train_and_evaluate_ANN


DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

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

TRAIN_FILE = DATA_DIR / "train_combined_features.csv"
TEST_FILE = DATA_DIR / "test_combined_features.csv"


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Loading ANN training data")
print("=" * 60)

print(f"Training file: {TRAIN_FILE}")
print(f"Test file    : {TEST_FILE}")

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

print(f"\nTraining samples: {len(df_train)}")
print(f"Test samples    : {len(df_test)}")

print(f"Training features/columns: {df_train.shape[1]}")
print(f"Test features/columns    : {df_test.shape[1]}")


# ============================================================
# Best ANN configuration
# ============================================================

BEST_LEARNING_RATE = 0.0005
BEST_BATCH_SIZE = 64
BEST_EPOCHS = 300


# ============================================================
# Print final configuration
# ============================================================

print("\n" + "=" * 60)
print("Final ANN Configuration")
print("=" * 60)

print(f"Learning rate : {BEST_LEARNING_RATE}")
print(f"Batch size    : {BEST_BATCH_SIZE}")
print(f"Epochs        : {BEST_EPOCHS}")
print("Feature set   : Combined + Temperature")


# ============================================================
# Train final ANN
# ============================================================

print("\n" + "=" * 60)
print("Starting Final ANN Training")
print("=" * 60)

result = train_and_evaluate_ANN(
    train_data=df_train,
    test_data=df_test,
    model_class=Solubility_Predictor,
    learning_rate=BEST_LEARNING_RATE,
    batch_size=BEST_BATCH_SIZE,
    num_epochs=BEST_EPOCHS,
    save_model=True,
    model_filename="best_ann_model.pth"
)


# ============================================================
# Print final performance
# ============================================================

print("\n" + "=" * 60)
print("Final ANN Performance")
print("=" * 60)

print(f"RMSE                : {result['rmse']:.4f}")
print(f"MAE                 : {result['mae']:.4f}")
print(f"R²                  : {result['r2']:.4f}")
print(f"Best validation loss: {result['best_val_loss']:.6f}")
print(f"Epochs trained      : {result['epochs_trained']}")


# ============================================================
# Save ANN predictions with sample identifiers
# ============================================================

print("\n" + "=" * 60)
print("Saving ANN Predictions")
print("=" * 60)

ann_predictions_df = df_test[
    [
        "SMILES_Solute",
        "SMILES_Solvent",
        "Temperature_K"
    ]
].copy()

ann_predictions_df["Actual_LogS"] = (
    np.asarray(result["y_test"]).reshape(-1)
)

ann_predictions_df["ANN_Predicted_LogS"] = (
    np.asarray(result["y_pred"]).reshape(-1)
)


# Check row count before saving
if len(ann_predictions_df) != len(result["y_pred"]):
    raise ValueError(
        "Mismatch between test samples and ANN predictions: "
        f"{len(ann_predictions_df)} samples vs "
        f"{len(result['y_pred'])} predictions."
    )


predictions_file = RESULTS_DIR / "ann_test_predictions.csv"

ann_predictions_df.to_csv(
    predictions_file,
    index=False
)

print(f"[OK] ANN predictions saved to:")
print(f"     {predictions_file}")
print(f"Rows saved: {len(ann_predictions_df)}")


# ============================================================
# Save final ANN metrics
# ============================================================

print("\n" + "=" * 60)
print("Saving ANN Performance Metrics")
print("=" * 60)

metrics_df = pd.DataFrame([{
    "Model": "ANN",
    "Feature_Set": "Combined_with_Temp",
    "Learning_Rate": BEST_LEARNING_RATE,
    "Batch_Size": BEST_BATCH_SIZE,
    "Max_Epochs": BEST_EPOCHS,
    "RMSE": result["rmse"],
    "MAE": result["mae"],
    "R2": result["r2"],
    "Best_Val_Loss": result["best_val_loss"],
    "Epochs_Trained": result["epochs_trained"]
}])

metrics_file = RESULTS_DIR / "ann_final_metrics.csv"

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(f"[OK] ANN metrics saved to:")
print(f"     {metrics_file}")


# ============================================================
# Check saved files
# ============================================================

print("\n" + "=" * 60)
print("Checking Saved Files")
print("=" * 60)

saved_files = [
    MODEL_DIR / "best_ann_model.pth",
    MODEL_DIR / "scaler.pkl",
    MODEL_DIR / "feature_columns.pkl",
    MODEL_DIR / "train_medians.pkl",
    RESULTS_DIR / "ann_test_predictions.csv",
    RESULTS_DIR / "ann_final_metrics.csv"
]

for file_path in saved_files:

    if file_path.exists():
        print(f"[OK] {file_path}")

    else:
        print(f"[WARNING] Missing: {file_path}")


print("\n" + "=" * 60)
print("Final ANN training completed.")
print("=" * 60)