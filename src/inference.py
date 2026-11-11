import os
import argparse
import tempfile
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors

# Existing ANN/PINN model architecture
from ann_model import Solubility_Predictor

warnings.filterwarnings("ignore")


# ============================================================
# Project Paths Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CP_MODEL_PT = (
    MODELS_DIR / "chemprop_model" / "model_0" / "best.pt"
)

def get_device():
    """
    Test whether CUDA can actually execute tensor operations.
    Fall back to CPU if CUDA initialization or execution fails.
    """
    if torch.cuda.is_available():
        try:
            test_tensor = torch.randn(100, 100, device="cuda")
            _ = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()

            print("CUDA is working. Using GPU.")
            return torch.device("cuda")

        except Exception as exc:
            print(f"CUDA initialization/test failed: {exc}")
            print("Falling back to CPU.")

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print("Using CPU for inference.")
    return torch.device("cpu")


# ============================================================
#  Calculate and return nine RDKit molecular properties
# ============================================================

def calculate_molecular_properties(smiles):
    """
   
    Calculates nine RDKit descriptors from a SMILES string.
    These properties are returned to the user.
    MACCS fingerprints are NOT included in this output.
    """

    # Validate that the SMILES is a non-empty string
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    # Convert SMILES into an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)

    # Reject invalid SMILES instead of continuing
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    #  Calculate the nine user-facing properties
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "HeavyAtomCount": int(Descriptors.HeavyAtomCount(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HDonors": int(Descriptors.NumHDonors(mol)),
        "HAcceptors": int(Descriptors.NumHAcceptors(mol)),
        "RotatableBonds": int(Descriptors.NumRotatableBonds(mol)),
        "RingCount": int(Descriptors.RingCount(mol)),
        "FractionCSP3": float(Descriptors.FractionCSP3(mol)),
    }


# ============================================================
# RDKit descriptor wrapper
# ============================================================

def extract_rdkit_descriptors(smiles):
    """
    Returns the same nine descriptors used by the model pipeline.
    """
    # the descriptor calculations.
    return calculate_molecular_properties(smiles)


# ============================================================
# MACCS fingerprint extraction
# Still used internally by ANN/PINN.
# Fingerprints are NOT returned to the user.
# ============================================================

def extract_maccs_fingerprints(smiles):

    # Import MACCSkeys locally
    from rdkit.Chem import MACCSkeys

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)

    # Reject invalid SMILES
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Generate 166 MACCS bits
    fp = list(MACCSkeys.GenMACCSKeys(mol))[1:167]

    # Return fingerprints for internal model features
    return {
        f"MACCS_{i}": int(bit)
        for i, bit in enumerate(fp)
    }


# ============================================================
# Generate model features
# This function still generates 351 features for ANN/PINN.
# ============================================================

def generate_features(solute_smiles, solvent_smiles, temperature):

    # Solute RDKit descriptors
    solute_rdkit = {
        f"Solute_{k}": v
        for k, v in extract_rdkit_descriptors(
            solute_smiles
        ).items()
    }

    # Solvent RDKit descriptors
    solvent_rdkit = {
        f"Solv_{k}": v
        for k, v in extract_rdkit_descriptors(
            solvent_smiles
        ).items()
    }

    # Solute MACCS fingerprints: internal model features only
    solute_maccs = {
        f"Solute_{k}": v
        for k, v in extract_maccs_fingerprints(
            solute_smiles
        ).items()
    }

    # Solvent MACCS fingerprints: internal model features only
    solvent_maccs = {
        f"Solv_{k}": v
        for k, v in extract_maccs_fingerprints(
            solvent_smiles
        ).items()
    }

    # Add temperature and all model features
    features = {"Temperature_K": temperature}
    features.update(solute_rdkit)
    features.update(solute_maccs)
    features.update(solvent_rdkit)
    features.update(solvent_maccs)

    return pd.DataFrame([features])


# ============================================================
# ANN prediction moved into a helper function
# ============================================================

def predict_ann(df_features, device):

    #  Load ANN preprocessing artifacts
    ann_medians = joblib.load(
        MODELS_DIR / "train_medians.pkl"
    )

    ann_scaler = joblib.load(
        MODELS_DIR / "scaler.pkl"
    )

    ann_columns = joblib.load(
        MODELS_DIR / "feature_columns.pkl"
    )

    # Fill missing values using training medians
    df_ann = df_features.fillna(ann_medians)

    # Ensure expected columns exist
    for col in ann_columns:
        if col not in df_ann.columns:
            df_ann[col] = 0.0

    # Apply the saved scaler
    X_ann = ann_scaler.transform(
        df_ann[ann_columns]
    )

    # Convert input into a PyTorch tensor
    X_ann_tensor = torch.tensor(
        X_ann, dtype=torch.float32
    ).to(device)

    # Reconstruct the trained ANN architecture
    ann_model = Solubility_Predictor(
        len(ann_columns)
    ).to(device)

    # Load trained ANN weights
    ann_model.load_state_dict(
        torch.load(
            MODELS_DIR / "best_ann_model.pth",
            map_location=device,
            weights_only=True,
        )
    )

    # Set inference mode
    ann_model.eval()

    # Predict without calculating gradients
    with torch.no_grad():
        prediction = ann_model(
            X_ann_tensor
        ).item()

    return float(prediction)


# ============================================================
# PINN prediction moved into a helper function
# Prediction approach and trained weights remain unchanged.
# ============================================================

def predict_pinn(df_features, device):

    # Load PINN-specific preprocessing artifacts
    pinn_medians = joblib.load(
        MODELS_DIR / "pinn_train_medians.pkl"
    )

    pinn_scaler = joblib.load(
        MODELS_DIR / "pinn_scaler.pkl"
    )

    pinn_columns = joblib.load(
        MODELS_DIR / "pinn_feature_columns.pkl"
    )

    # Fill missing values using PINN training medians
    df_pinn = df_features.fillna(pinn_medians)

    # Ensure expected columns exist
    for col in pinn_columns:
        if col not in df_pinn.columns:
            df_pinn[col] = 0.0

    # Apply the saved PINN scaler
    X_pinn = pinn_scaler.transform(
        df_pinn[pinn_columns]
    )

    # Convert to a PyTorch tensor
    X_pinn_tensor = torch.tensor(
        X_pinn, dtype=torch.float32
    ).to(device)

    #  Reconstruct the trained PINN architecture
    pinn_model = Solubility_Predictor(
        len(pinn_columns)
    ).to(device)

    # Load trained PINN weights
    pinn_model.load_state_dict(
        torch.load(
            MODELS_DIR / "best_pinn_model.pth",
            map_location=device,
            weights_only=True,
        )
    )

    pinn_model.eval()

    # Run inference
    with torch.no_grad():
        prediction = pinn_model(
            X_pinn_tensor
        ).item()

    return float(prediction)


def predict_chemprop(
    solute_smiles,
    solvent_smiles,
    temperature,
):

    #  Load the saved temperature scaler
    cp_scaler = joblib.load(
        MODELS_DIR / "chemprop_scaler.pkl"
    )

    # Scale temperature using training parameters
    scaled_temperature = float(
        cp_scaler.transform([[temperature]])[0][0]
    )

    # Create temporary input/output CSV files
    with tempfile.TemporaryDirectory() as tmpdir:

        input_csv = Path(tmpdir) / "cp_input.csv"
        output_csv = Path(tmpdir) / "cp_output.csv"

        # Build the Chemprop input DataFrame
        df_cp_input = pd.DataFrame({
            "solute": [solute_smiles],
            "solvent": [solvent_smiles],
            "temperature": [scaled_temperature],
        })

        df_cp_input.to_csv(
            input_csv, index=False
        )

        # UNCHANGED: Build the Chemprop prediction command
        cmd = [
            "chemprop",
            "predict",
            "-i", str(input_csv),
            "-o", str(output_csv),
            "--model-paths", str(CP_MODEL_PT),
            "-s", "solute", "solvent",
            "--descriptors-columns", "temperature",
        ]

        # New
        env = os.environ.copy()

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        except subprocess.CalledProcessError as exc:

            error_message = (
                (exc.stderr or "") + "\n" + (exc.stdout or "")
            )

            if any(
                keyword in error_message.upper()
                for keyword in ["CUDA", "CUBLAS", "GPU"]
            ):
                print("Chemprop GPU execution failed.")
                print("Retrying Chemprop inference using CPU.")

                cpu_env = os.environ.copy()
                cpu_env["CUDA_VISIBLE_DEVICES"] = ""

                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=cpu_env,
                    )

                except subprocess.CalledProcessError as cpu_exc:
                    raise RuntimeError(
                        "Chemprop CPU inference also failed:\n"
                        + (cpu_exc.stderr or "")
                    ) from cpu_exc

            else:
                raise RuntimeError(
                    f"Chemprop inference failed:\n{error_message}"
                ) from exc

        #  Verify that Chemprop created an output file
        if not output_csv.exists():
            raise RuntimeError(
                "Chemprop did not produce an output file."
            )

        df_cp_out = pd.read_csv(output_csv)

        #  Check for empty output
        if df_cp_out.empty:
            raise RuntimeError(
                "Chemprop returned an empty prediction."
            )

        # Read the prediction from the CSV
        if "target" in df_cp_out.columns:
            prediction = df_cp_out["target"].iloc[0]
        else:
            prediction = df_cp_out.iloc[0, -1]

        # NEW: Reject a missing prediction
        if pd.isna(prediction):
            raise RuntimeError(
                "Chemprop returned a missing prediction."
            )

        return float(prediction)


# ============================================================
# Adds solute and solvent properties to the returned output.
# ============================================================

def predict_solubility(
    solute_smiles,
    solvent_smiles,
    temperature,
):

    # Validate temperature
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Temperature must be a positive finite value in Kelvin."
        )

    # Calculate properties to return to the user
    solute_properties = calculate_molecular_properties(
        solute_smiles
    )

    solvent_properties = calculate_molecular_properties(
        solvent_smiles
    )

    # Generate the full model feature set
    df_features = generate_features(
        solute_smiles,
        solvent_smiles,
        temperature,
    )

    # Select GPU if available, otherwise CPU
    device = get_device()
  
    # The model predictions themselves use the same approach.
    ann_pred = predict_ann(
        df_features, device
    )

    pinn_pred = predict_pinn(
        df_features, device
    )

    cp_pred = predict_chemprop(
        solute_smiles,
        solvent_smiles,
        temperature,
    )

    # Original ensemble weights
    final_logs = (
        0.32 * ann_pred
        + 0.14 * pinn_pred
        + 0.54 * cp_pred
    )


    # Previously, only Ensemble_LogS was returned.
    # Now the result also contains input and RDKit properties.
    # Fingerprints and other internal features are not returned.
    return {
        "input": {
            "solute_smiles": solute_smiles,
            "solvent_smiles": solvent_smiles,
            "temperature_K": float(temperature),
        },

        "solute_properties": solute_properties,

        "solvent_properties": solvent_properties,

        "ensemble_prediction": {
            "LogS": round(float(final_logs), 4),
            "unit": "log10(mol/L)",
        },
    }


# ========================================================================
# Displays LogS plus the new molecular properties of solvent and solute.
# ========================================================================

if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Predict solubility and calculate RDKit molecular "
            "properties for a solute-solvent pair."
        )
    )

    # Required solute input
    parser.add_argument(
        "--solute",
        type=str,
        required=True,
        help="Solute SMILES, e.g. CCO",
    )

    # Required solvent input
    parser.add_argument(
        "--solvent",
        type=str,
        required=True,
        help="Solvent SMILES, e.g. O",
    )

    #  Required temperature input
    parser.add_argument(
        "--temp",
        type=float,
        required=True,
        help="Temperature in Kelvin, e.g. 298.15",
    )

    args = parser.parse_args()

    # Handle selected input and inference errors
    try:
        results = predict_solubility(
            args.solute,
            args.solvent,
            args.temp,
        )

        #  Print the predicted LogS
        print("\n--- SOLUBILITY PREDICTION ---")
        print(
            f"Predicted LogS: "
            f"{results['ensemble_prediction']['LogS']}"
        )
        print(
            f"Unit: "
            f"{results['ensemble_prediction']['unit']}"
        )

        #  Print all nine solute properties
        print("\n--- SOLUTE PROPERTIES ---")
        for name, value in results["solute_properties"].items():
            print(f"{name}: {value}")

        # Print all nine solvent properties
        print("\n--- SOLVENT PROPERTIES ---")
        for name, value in results["solvent_properties"].items():
            print(f"{name}: {value}")

    # Report selected errors instead of a traceback
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
