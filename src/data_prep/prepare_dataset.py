import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Project Paths Setup
# ============================================================
# Since this script is in src/data_prep/, we go up 2 levels to reach the root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "full_dataset" / "BigSolDBv2.0.csv"

# ============================================================
# 2. Leakage-Free Data Splitting (GroupShuffleSplit)
# ============================================================
def split_raw_data():
    print("=" * 60)
    print("Step 1: Leakage-Free Data Splitting (GroupShuffleSplit)")
    print("=" * 60)
    
    # Load raw data
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    data = pd.read_csv(RAW_DATA_PATH)
    data = data.dropna(subset=['LogS(mol/L)'])

    # Create the compound groups (Solute + Solvent pair)
    # This ensures no solute-solvent pair in train ever leaks into test
    groups = data['SMILES_Solute'].astype(str) + '_' + data['SMILES_Solvent'].astype(str)

    # Perform the Group Shuffle Split (80% Train, 20% Test)
    print("Executing 80/20 GroupShuffleSplit...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X=data, y=data['LogS(mol/L)'], groups=groups))

    train_raw = data.iloc[train_idx].reset_index(drop=True)
    test_raw = data.iloc[test_idx].reset_index(drop=True)

    # Save the RAW splits to CSV
    train_raw_path = DATA_DIR / "train_raw.csv"
    test_raw_path = DATA_DIR / "test_raw.csv"
    
    train_raw.to_csv(train_raw_path, index=False)
    test_raw.to_csv(test_raw_path, index=False)

    print(f"[OK] Saved train_raw.csv ({len(train_raw)} rows)")
    print(f"[OK] Saved test_raw.csv ({len(test_raw)} rows)\n")
    
    return train_raw, test_raw

# ============================================================
# 3. Feature Extraction Functions
# ============================================================
def extract_rdkit_descriptors(smiles):
    """Extract 9 key physicochemical descriptors using RDKit."""
    empty_dict = {
        'MolWt': np.nan, 'HeavyAtomCount': np.nan, 'LogP': np.nan,
        'TPSA': np.nan, 'HDonors': np.nan, 'HAcceptors': np.nan,
        'RotatableBonds': np.nan, 'RingCount': np.nan, 'FractionCSP3': np.nan
    }
    
    if pd.isna(smiles) or str(smiles).strip() in ["-", "", "NA", "N/A", "na", "NaN"]:
        return empty_dict

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty_dict

    return {
        'MolWt': Descriptors.MolWt(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HDonors': Descriptors.NumHDonors(mol),
        'HAcceptors': Descriptors.NumHAcceptors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
    }

def extract_maccs_fingerprints(smiles):
    """Generate 166-bit MACCS structural keys."""
    if pd.isna(smiles) or str(smiles).strip() in ["-", "", "NA", "N/A", "na", "NaN"]:
        return {f'MACCS_{i}': np.nan for i in range(166)}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {f'MACCS_{i}': np.nan for i in range(166)}

    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_array = list(fp)[1:167] # Bit 0 is a dummy
    return {f'MACCS_{i}': bit for i, bit in enumerate(fp_array)}

def apply_descriptors(df, smiles_column, prefix, extraction_type):
    """Apply feature extraction to an entire DataFrame column."""
    if extraction_type == 'rdkit':
        return df[smiles_column].apply(lambda x: pd.Series(extract_rdkit_descriptors(x))).add_prefix(prefix + '_')
    elif extraction_type == 'maccs':
        return df[smiles_column].apply(lambda x: pd.Series(extract_maccs_fingerprints(x))).add_prefix(prefix + '_')

# ============================================================
# 4.  Build Combined Feature Sets
# ============================================================
def build_feature_matrices(train_raw, test_raw):
    print("=" * 60)
    print("Step 2: Feature Extraction (RDKit + MACCS)")
    print("=" * 60)
    
    print("Extracting RDKit Descriptors for Train set...")
    train_solute_rdkit = apply_descriptors(train_raw, 'SMILES_Solute', 'Solute', 'rdkit')
    train_solvent_rdkit = apply_descriptors(train_raw, 'SMILES_Solvent', 'Solv', 'rdkit')

    print("Extracting RDKit Descriptors for Test set...")
    test_solute_rdkit = apply_descriptors(test_raw, 'SMILES_Solute', 'Solute', 'rdkit')
    test_solvent_rdkit = apply_descriptors(test_raw, 'SMILES_Solvent', 'Solv', 'rdkit')

    print("Extracting MACCS Fingerprints for Train set...")
    train_solute_maccs = apply_descriptors(train_raw, 'SMILES_Solute', 'Solute', 'maccs')
    train_solvent_maccs = apply_descriptors(train_raw, 'SMILES_Solvent', 'Solv', 'maccs')

    print("Extracting MACCS Fingerprints for Test set...")
    test_solute_maccs = apply_descriptors(test_raw, 'SMILES_Solute', 'Solute', 'maccs')
    test_solvent_maccs = apply_descriptors(test_raw, 'SMILES_Solvent', 'Solv', 'maccs')

    print("\nConcatenating Master Feature Matrices...")
    df_train_combine = pd.concat([
        train_raw, train_solute_rdkit, train_solute_maccs, train_solvent_rdkit, train_solvent_maccs
    ], axis=1)

    df_test_combine = pd.concat([
        test_raw, test_solute_rdkit, test_solute_maccs, test_solvent_rdkit, test_solvent_maccs
    ], axis=1)

    # Save Combined Features
    train_feat_path = DATA_DIR / "train_combined_features.csv"
    test_feat_path = DATA_DIR / "test_combined_features.csv"
    
    df_train_combine.to_csv(train_feat_path, index=False)
    df_test_combine.to_csv(test_feat_path, index=False)
    
    print(f"[OK] Saved train_combined_features.csv ({df_train_combine.shape[1]} columns)")
    print(f"[OK] Saved test_combined_features.csv ({df_test_combine.shape[1]} columns)")
    print("\nData Preparation Complete!")

if __name__ == "__main__":
    train_raw, test_raw = split_raw_data()
    build_feature_matrices(train_raw, test_raw)