import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Path setup matching 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHEMPROP_MODEL_DIR = MODEL_DIR / "chemprop_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHEMPROP_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def to_chemprop_format(df: pd.DataFrame):
    """Extract and standardize columns for Chemprop multi-molecule input."""
    cp_df = df[["SMILES_Solute", "SMILES_Solvent", "Temperature_K", "LogS(mol/L)"]].copy()
    cp_df.columns = ["solute", "solvent", "temperature", "target"]
    return cp_df


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Filter missing targets and invalid SMILES strings."""
    def is_valid(smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None
        except Exception:
            return False

    df = df.dropna(subset=["solute", "solvent", "target"])
    df = df[df["solute"].apply(is_valid)]
    df = df[df["solvent"].apply(is_valid)]
    return df.reset_index(drop=True)


def prepare_chemprop_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    save_artifacts: bool = False
):
    """Formats datasets, scales temperature (fit on train only), and saves CSVs."""
    train_df = filter_valid_smiles(to_chemprop_format(train_data))
    test_df = filter_valid_smiles(to_chemprop_format(test_data))

    # Fit scaler ONLY on training temperature
    scaler = StandardScaler()
    train_df["temperature"] = scaler.fit_transform(train_df[["temperature"]])
    test_df["temperature"] = scaler.transform(test_df[["temperature"]])

    if save_artifacts:
        scaler_path = MODEL_DIR / "chemprop_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Temperature scaler saved to: {scaler_path}")

        train_cp_path = DATA_DIR / "train_chemprop.csv"
        test_cp_path = DATA_DIR / "test_chemprop.csv"

        train_df.to_csv(train_cp_path, index=False)
        test_df.to_csv(test_cp_path, index=False)

        print(f"[INFO] Chemprop datasets saved to:\n  - {train_cp_path}\n  - {test_cp_path}")

    return train_df, test_df


def train_and_evaluate_chemprop(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 64,
    save_model: bool = False
):
    """Executes Chemprop model training, prediction, and metric computation."""
    #  Prepare Data
    train_df, test_df = prepare_chemprop_data(
        train_data,
        test_data,
        save_artifacts=save_model
    )

    train_cp_path = DATA_DIR / "train_chemprop.csv"
    test_cp_path = DATA_DIR / "test_chemprop.csv"
    pred_path = DATA_DIR / "chemprop_predictions_raw.csv"

    # Ensure formatted files exist on disk for CLI invocation
    if not train_cp_path.exists() or not test_cp_path.exists():
        train_df.to_csv(train_cp_path, index=False)
        test_df.to_csv(test_cp_path, index=False)

    #  Train Chemprop via Subprocess
    print("\n" + "=" * 60)
    print("Executing Chemprop Training")
    print("=" * 60)

    train_cmd = [
        "chemprop", "train",
        "-i", str(train_cp_path),
        "-s", "solute", "solvent",
        "--target-columns", "target",
        "-t", "regression",
        "--descriptors-columns", "temperature",
        "--epochs", str(epochs),
        "-b", str(batch_size),
        "-o", str(CHEMPROP_MODEL_DIR)
    ]

    subprocess.run(train_cmd, check=True)

    #  Predict on Test Data
    print("\n" + "=" * 60)
    print("Executing Chemprop Prediction")
    print("=" * 60)

    best_model_pt = CHEMPROP_MODEL_DIR / "model_0" / "best.pt"

    predict_cmd = [
        "chemprop", "predict",
        "-i", str(test_cp_path),
        "-o", str(pred_path),
        "--model-paths", str(best_model_pt),
        "-s", "solute", "solvent",
        "--descriptors-columns", "temperature"
    ]

    subprocess.run(predict_cmd, check=True)

    #  Load Predictions and Evaluate
    pred_df = pd.read_csv(pred_path)
    y_true = test_df["target"].values
    y_pred = pred_df["target"].values

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nFinal Chemprop Performance")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print("-" * 40)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "epochs_trained": epochs,
        "y_test": y_true,
        "y_pred": y_pred
    }