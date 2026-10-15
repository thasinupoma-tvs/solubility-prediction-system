import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import tempfile
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
import warnings
from ann_model import Solubility_Predictor
warnings.filterwarnings("ignore")

# ============================================================
#  Project Paths Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CP_MODEL_PT = MODELS_DIR / "chemprop_model" / "model_0" / "best.pt"

# ============================================================
#  Feature Extraction Logic (Tabular Models)
# ============================================================
def extract_rdkit_descriptors(smiles):
    empty_dict = {
        'MolWt': np.nan, 'HeavyAtomCount': np.nan, 'LogP': np.nan,
        'TPSA': np.nan, 'HDonors': np.nan, 'HAcceptors': np.nan,
        'RotatableBonds': np.nan, 'RingCount': np.nan, 'FractionCSP3': np.nan
    }
    mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
    if mol is None: return empty_dict

    return {
        'MolWt': Descriptors.MolWt(mol), 'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'LogP': Descriptors.MolLogP(mol), 'TPSA': Descriptors.TPSA(mol),
        'HDonors': Descriptors.NumHDonors(mol), 'HAcceptors': Descriptors.NumHAcceptors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol), 'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
    }

def extract_maccs_fingerprints(smiles):
    empty_dict = {f'MACCS_{i}': np.nan for i in range(166)}
    mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
    if mol is None: return empty_dict
    fp = list(MACCSkeys.GenMACCSKeys(mol))[1:167]
    return {f'MACCS_{i}': bit for i, bit in enumerate(fp)}

def generate_features(solute_smiles, solvent_smiles, temperature):
    solute_rdkit = {f"Solute_{k}": v for k, v in extract_rdkit_descriptors(solute_smiles).items()}
    solvent_rdkit = {f"Solv_{k}": v for k, v in extract_rdkit_descriptors(solvent_smiles).items()}
    solute_maccs = {f"Solute_{k}": v for k, v in extract_maccs_fingerprints(solute_smiles).items()}
    solvent_maccs = {f"Solv_{k}": v for k, v in extract_maccs_fingerprints(solvent_smiles).items()}

    features = {"Temperature_K": temperature}
    features.update(solute_rdkit)
    features.update(solute_maccs)
    features.update(solvent_rdkit)
    features.update(solvent_maccs)
    return pd.DataFrame([features])

# ============================================================
# Prediction Pipeline
# ============================================================
def predict_solubility(solute_smiles, solvent_smiles, temperature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    #  Tabular Feature Generation
    # ---------------------------------------------------------
    df_features = generate_features(solute_smiles, solvent_smiles, temperature)
    
    # ---------------------------------------------------------
    #  ANN Prediction
    # ---------------------------------------------------------
    ann_medians = joblib.load(MODELS_DIR / "train_medians.pkl")
    ann_scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    ann_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    
    df_ann = df_features.fillna(ann_medians)
    for col in ann_columns:
        if col not in df_ann.columns: df_ann[col] = 0.0
            
    X_ann = ann_scaler.transform(df_ann[ann_columns])
    X_ann_tensor = torch.tensor(X_ann, dtype=torch.float32).to(device)
    
    ann_model = Solubility_Predictor(len(ann_columns)).to(device)
    ann_model.load_state_dict(torch.load(MODELS_DIR / "best_ann_model.pth", map_location=device, weights_only=True))
    ann_model.eval()
    with torch.no_grad():
        ann_pred = ann_model(X_ann_tensor).item()

    # ---------------------------------------------------------
    #  PINN Prediction
    # ---------------------------------------------------------
    pinn_medians = joblib.load(MODELS_DIR / "pinn_train_medians.pkl")
    pinn_scaler = joblib.load(MODELS_DIR / "pinn_scaler.pkl")
    pinn_columns = joblib.load(MODELS_DIR / "pinn_feature_columns.pkl")
    
    df_pinn = df_features.fillna(pinn_medians)
    for col in pinn_columns:
        if col not in df_pinn.columns: df_pinn[col] = 0.0
            
    X_pinn = pinn_scaler.transform(df_pinn[pinn_columns])
    X_pinn_tensor = torch.tensor(X_pinn, dtype=torch.float32).to(device)
    
    pinn_model = Solubility_Predictor(len(pinn_columns)).to(device)
    pinn_model.load_state_dict(torch.load(MODELS_DIR / "best_pinn_model.pth", map_location=device, weights_only=True))
    pinn_model.eval()
    with torch.no_grad():
        pinn_pred = pinn_model(X_pinn_tensor).item()

    # ---------------------------------------------------------
    #  Chemprop Prediction (via Subprocess CLI)
    # ---------------------------------------------------------
    cp_scaler = joblib.load(MODELS_DIR / "chemprop_scaler.pkl")
    scaled_temperature = cp_scaler.transform([[temperature]])[0][0]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "cp_input.csv")
        output_csv = os.path.join(tmpdir, "cp_output.csv")
        
        # Create a single-row CSV formatted exactly as Chemprop expects
        df_cp_input = pd.DataFrame({
            "solute": [solute_smiles],
            "solvent": [solvent_smiles],
            "temperature": [scaled_temperature]
        })
        df_cp_input.to_csv(input_csv, index=False)
        
        # Execute the Chemprop CLI command silently
        cmd = [
            "chemprop", "predict",
            "-i", input_csv,
            "-o", output_csv,
            "--model-paths", str(CP_MODEL_PT),
            "-s", "solute", "solvent",
            "--descriptors-columns", "temperature"
        ]
        
        
    
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            df_cp_out = pd.read_csv(output_csv)
            if "target" in df_cp_out.columns:
                cp_pred = df_cp_out["target"].iloc[0]
            else:
                cp_pred = df_cp_out.iloc[0, -1] 
        except Exception as e:
            print(f"[ERROR] Chemprop inference failed: {e}")
            cp_pred = 0.0

    # ---------------------------------------------------------
    #  Final Late-Fusion Ensemble Math
    # ---------------------------------------------------------
    final_logs = (0.32 * ann_pred) + (0.14 * pinn_pred) + (0.54 * float(cp_pred))
    
    return {
  
        "Ensemble_LogS": round(final_logs, 4)
    }

# ============================================================
#  Command Line Interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict solubility (LogS) for a solute-solvent pair.")
    parser.add_argument("--solute", type=str, required=True, help="SMILES of the Solute (e.g., CCO)")
    parser.add_argument("--solvent", type=str, required=True, help="SMILES of the Solvent (e.g., O)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature in Kelvin (e.g., 298.15)")
    
    args = parser.parse_args()
    
    print("\n[AI Solubility Inference Engine]")
    print(f"Solute:  {args.solute}")
    print(f"Solvent: {args.solvent}")
    print(f"Temp:    {args.temp} K\n")
    
    print("Generating features and executing late-fusion ensemble...\n")
    
    results = predict_solubility(args.solute, args.solvent, args.temp)
    
    print("--- PREDICTION RESULTS ---")

    print(f"=============================")
    print(f"⭐ OPTIMAL ENSEMBLE: {results['Ensemble_LogS']} mol/L")
    print(f"=============================\n")