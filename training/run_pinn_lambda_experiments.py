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

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# Imports
# ============================================================

from src.ann_model import Solubility_Predictor

from src.pinn_training import (
    train_and_evaluate_PINN
)


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

PHYSICS_LOOKUP_FILE = (
    RESULTS_DIR /
    "physics_lookup.csv"
)


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Loading PINN experiment data")
print("=" * 60)

df_train = pd.read_csv(TRAIN_FILE)

df_test = pd.read_csv(TEST_FILE)

physics_lookup = pd.read_csv(
    PHYSICS_LOOKUP_FILE
)


print(
    f"Training samples : {len(df_train)}"
)

print(
    f"Test samples     : {len(df_test)}"
)

print(
    f"Physics groups   : {len(physics_lookup)}"
)


# ============================================================
# Fixed PINN configuration
# ============================================================

LEARNING_RATE = 0.0005

BATCH_SIZE = 64

EPOCHS = 300


# ============================================================
# Physics weights to test
# ============================================================

LAMBDA_VALUES = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
]


# ============================================================
# Store results
# ============================================================

results = []


# ============================================================
# Run experiments
# ============================================================

for lambda_phys in LAMBDA_VALUES:

    print("\n" + "=" * 60)

    print(
        f"Running PINN with "
        f"lambda = {lambda_phys}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Train PINN
    # --------------------------------------------------------

    result = train_and_evaluate_PINN(

        train_data=df_train,

        test_data=df_test,

        physics_lookup=physics_lookup,

        model_class=Solubility_Predictor,

        learning_rate=LEARNING_RATE,

        batch_size=BATCH_SIZE,

        num_epochs=EPOCHS,

        lambda_phys=lambda_phys,

        save_model=False
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "lambda_phys": lambda_phys,

        "RMSE": result["rmse"],

        "MAE": result["mae"],

        "R2": result["r2"],

        "best_val_loss":
            result["best_val_loss"],

        "epochs_trained":
            result["epochs_trained"]

    })


# ============================================================
# Create results DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Save results
# ============================================================

output_file = (
    RESULTS_DIR /
    "pinn_lambda_experiments.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)

print("PINN Lambda Experiment Results")

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


print("\nResults saved to:")

print(output_file)