# 🧪 Leakage-Free Solubility Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A machine learning pipeline for predicting the equilibrium solubility (LogS in mol/L) of unseen solute–solvent chemical systems across varying temperatures.

This repository implements a **Late-Fusion Optimal Ensemble** combining a descriptor-based Artificial Neural Network (ANN), a thermodynamically regularized Physics-Informed Neural Network (PINN), and a Directed Message-Passing Neural Network (Chemprop). It supports leakage-free model evaluation, thermodynamic analysis, model training, ensemble optimization, and command-line inference for new chemical systems.

---

## 📌 Project Highlights

- **Leakage-Free Evaluation:** Uses `GroupShuffleSplit` (random_state=42) with the solute–solvent SMILES pair as the grouping key to evaluate generalization to unseen chemical systems.
- **Aleatoric Uncertainty Analysis:** Quantifies experimental measurement variability across 3,234 duplicate experimental systems, estimating an intrinsic RMSE noise floor of approximately **0.20 LogS**.
- **Thermodynamic Regularization:** Incorporates the van 't Hoff relationship as a soft physics-informed loss constraint, using an optimal physics loss weight of `λ_phys = 0.010`.
- **Molecular Feature Engineering:** Combines 18 RDKit molecular descriptors, 332 MACCS fingerprints, and temperature features for a total of 351 input features.
- **Optimal Late-Fusion Ensemble:** Uses SciPy SLSQP optimization to combine ANN, PINN, and Chemprop predictions through optimized non-negative weights.
- **API & Command-Line Inference:** Provides Python backend integration and command-line execution (`src/inference.py`) to predict LogS and output JSON-ready results for a user-specified solute, solvent, and temperature.

---

## 📊 Model Architectures & Performance

Both the Artificial Neural Network (ANN) and Physics-Informed Neural Network (PINN) utilize a 4-layer deep neural network designed with batch normalization, LeakyReLU activation, and dropout regularization. The final prediction is generated using the following weighted ensemble:

**`LogS = 0.32 × ANN + 0.14 × PINN + 0.54 × Chemprop`**

| Model / Strategy | RMSE (LogS) | MAE (LogS) | R² Score |
| :--- | :---: | :---: | :---: |
| 1. Baseline ANN (Tabular + Temperature) | 0.5035 | 0.3271 | 0.8291 |
| 2. PINN (Tabular + Temperature + van 't Hoff) | 0.5054 | 0.3261 | 0.8278 |
| 3. Chemprop (Molecular Graph + Temperature) | 0.4874 | 0.3076 | 0.8399 |
| 4. Equal-Weight Blend (ANN + PINN + Chemprop) | 0.4617 | 0.2936 | 0.8563 |
| 5. Physics–Graph Blend (40% PINN + 60% Chemprop) | 0.4576 | 0.2893 | 0.8589 |
| **6. Optimal Ensemble (32% ANN + 14% PINN + 54% Chemprop)** | **0.4549** | **0.2878** | **0.8605** |

*Reported benchmarks were evaluated on a 20% holdout test set from BigSolDB 2.0.*

---

## 📂 Repository Structure

```text
.
├── data/                                      # DATASETS & FEATURES
│   ├── full_dataset/                          # BigSolDB 2.0 source dataset
│   ├── train_raw.csv / test_raw.csv           # Leakage-free data splits
│   ├── train_combined_features.csv            # Training features
│   ├── test_combined_features.csv             # Test features
│   ├── train_chemprop.csv / test_chemprop.csv  # Chemprop-formatted data
│   └── final_ensemble_test_predictions.csv    # Ensemble test predictions
│
├── docs/                                      # DOCUMENTATION
│   └── EcoPlast_Task3.3_Solubility_Prediction_Documentation_Final.pdf
│
├── models/                                    # TRAINED MODELS & PREPROCESSORS
│   ├── chemprop_model/
│   │   └── model_0/
│   │       └── best.pt                        # Chemprop checkpoint
│   ├── best_ann_model.pth                     # ANN weights
│   ├── best_pinn_model.pth                    # PINN weights
│   ├── scaler.pkl                             # ANN scaler
│   ├── pinn_scaler.pkl                        # PINN scaler
│   ├── chemprop_scaler.pkl                    # Chemprop temperature scaler
│   ├── train_medians.pkl                      # ANN imputation values
│   ├── pinn_train_medians.pkl                 # PINN imputation values
│   ├── feature_columns.pkl                    # ANN feature ordering
│   └── pinn_feature_columns.pkl               # PINN feature ordering
│
├── results/                                   # METRICS, PLOTS & LOGS
│   ├── aleatoric_heterogeneity_profile.png
│   ├── aleatoric_uncertainty_summary.csv
│   ├── ann_final_experiment_results.xlsx
│   ├── physics_lookup.csv
│   ├── pinn_lambda_experiments.csv
│   ├── vanthoff_analysis_classified.csv
│   └── ensemble_performance_metrics.csv
│
├── src/                                       # CORE PYTHON MODULES
│   ├── data_prep/
│   │   └── prepare_dataset.py                  # Data splitting & feature extraction
│   ├── aleatoric_analysis.py                   # Experimental noise analysis
│   ├── check_vantthof_quality.py               # Thermodynamic validation
│   ├── physics_lookup.py                      # Enthalpy extraction
│   ├── vanthoff.py                            # Thermodynamic equations
│   ├── ann_model.py                           # ANN/PINN architecture
│   ├── ann_training.py                        # ANN training utilities
│   ├── pinn_training.py                       # PINN training utilities
│   ├── chemprop_training.py                   # Chemprop training utilities
│   ├── evaluate_ensemble.py                   # SLSQP ensemble optimization
│   └── inference.py                           # Trained-model inference pipeline
│
├── training/                                  # TRAINING EXECUTION SCRIPTS
│   ├── run_ann_experiments.py                 # ANN hyperparameter experiments
│   ├── run_pinn_lambda_experiments.py          # PINN loss-weight experiments
│   ├── train_ann.py                           # Final ANN training
│   ├── train_pinn.py                          # Final PINN training
│   └── train_chemprop.py                      # Final Chemprop training
│
├── requirements.txt                           # Full development/training dependencies
└── requirements-cpu.txt                        # CPU-only inference dependencies
```

*The structure above summarizes the main project files. Actual filenames and generated artifacts may vary with the local repository.*

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/solubility-prediction-system.git
cd solubility-prediction-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Choose the dependency file according to your intended use.

#### Option A: Full environment for development and training

Use this option if you intend to run data preparation, model training, experiments, or other development workflows.

```bash
pip install -r requirements.txt
```

This is the full project environment. It may include GPU-enabled PyTorch dependencies, depending on the versions specified in `requirements.txt`.

#### Option B: CPU-only environment for inference

Use this option if you only need to run the trained models for prediction, such as when integrating the inference pipeline into a backend. **A GPU is not required.**

The `requirements-cpu.txt` file installs CPU-only PyTorch and the dependencies needed by the inference pipeline. It is configured for Python 3.11 and supports Windows/Linux.

```bash
pip install -r requirements-cpu.txt
```

The CPU-only dependency file uses `torch==2.11.0+cpu`; it does not install the CUDA-enabled PyTorch build.

---

## 🚀 Execution Pipeline

### 1. Data Preparation
Prepare the leakage-free training and test splits and extract the molecular features:
```bash
python src/data_prep/prepare_dataset.py
```

### 2. Data Analysis & Physics Validation
Run the experimental uncertainty analysis and validate the thermodynamic assumptions:
```bash
python src/aleatoric_analysis.py
python src/check_vantthof_quality.py
python src/physics_lookup.py
```

### 3. Model Training
Train the individual models using the prepared data:
```bash
python training/train_ann.py
python training/train_pinn.py
python training/train_chemprop.py
```

### 4. Ensemble Evaluation
Evaluate the individual models and optimize the late-fusion ensemble weights:
```bash
python src/evaluate_ensemble.py
```

---

## 🔌 Model Deployment and Integration

The system provides flexible inference capabilities for integration into existing backend systems or direct terminal usage.

### Python Integration (REST API Backend)
The `predict_solubility()` function can be imported into a Python backend and returns a JSON-ready dictionary.

```python
from src.inference import predict_solubility
import json

# Input from the frontend application
prediction_results = predict_solubility(
    solute_smiles="O=[N+]([O-])c1cccc2ccccc12",
    solvent_smiles="CCO",
    temperature=120
)
```

**Example JSON Response:**
```json
{
  "input": {
    "solute_smiles": "CCO",
    "solvent_smiles": "O",
    "temperature_K": 298.15
  },
  "solute_properties": {
    "MolWt": 46.069,
    "HeavyAtomCount": 3,
    "LogP": -0.0014,
    "TPSA": 20.23,
    "HDonors": 1,
    "HAcceptors": 1,
    "RotatableBonds": 0,
    "RingCount": 0,
    "FractionCSP3": 1.0
  },
  "solvent_properties": {
    "MolWt": 18.015,
    "HeavyAtomCount": 1,
    "LogP": -0.8247,
    "TPSA": 31.5,
    "HDonors": 2,
    "HAcceptors": 1,
    "RotatableBonds": 0,
    "RingCount": 0,
    "FractionCSP3": 0.0
  },
  "ensemble_prediction": {
    "LogS": -4.7516,
    "unit": "log10(mol/L)"
  }
}
```

### Command-Line Integration
The prediction script can be executed directly from the command line for automated workflows.

```bash
python src/inference.py --solute "O=[N+]([O-])c1cccc2ccccc12" --solvent "CCO" --temp 120
```

**Example terminal output:**
```text
SOLUBILITY PREDICTION
Predicted LogS: -4.7516
Unit: log10(mol/L)

SOLUTE PROPERTIES
MolWt: 173.171
HeavyAtomCount: 13
LogP: 2.748
TPSA: 43.14
HDonors: 0
HAcceptors: 2
RotatableBonds: 1
RingCount: 2
FractionCSP3: 0.0

SOLVENT PROPERTIES
MolWt: 46.069
HeavyAtomCount: 3
LogP: -0.0014
TPSA: 20.23
HDonors: 1
HAcceptors: 1
RotatableBonds: 0
RingCount: 0
FractionCSP3: 1.0
```

---

## 📈 Displaying Descriptor Effects to the User

Along with the predicted LogS, solute and solvent properties can be presented with their general effect on solubility. 

| Molecular Property | Property Value | Solubility | Condition / Note |
| :--- | :---: | :---: | :--- |
| Molecular Weight | ⬆ | ⬇ | Especially in water |
| Heavy Atom Count | ⬆ | ⬇ | Larger molecule |
| LogP | ⬆ | ⬇ | Water solubility |
| TPSA | ⬆ | ⬆ | In polar solvents |
| HDonors | ⬆ | ⬆ | In compatible solvents |
| HAcceptors | ⬆ | ⬆ | In compatible solvents |
| Rotatable Bonds | ⬆ | ⬆ | May improve solvent interactions |
| Ring Count | ⬆ | ⬇ | More rigidity; may reduce solubility |
| FractionCSP3 | ⬆ | ↕ | May influence solubility |

*(⬆ = increases, ⬇ = decreases, ↕ = effect depends on the solute-solvent system)*
