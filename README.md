# 🧪 Leakage-Free Solubility Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A machine learning pipeline for predicting the equilibrium solubility (LogS in mol/L) of unseen solute–solvent chemical systems across varying temperatures.

This repository implements a **Late-Fusion Optimal Ensemble** combining a descriptor-based Artificial Neural Network (ANN), a thermodynamically regularized Physics-Informed Neural Network (PINN), and a Directed Message-Passing Neural Network (Chemprop). It supports leakage-free model evaluation, thermodynamic analysis, model training, ensemble optimization, and command-line inference for new chemical systems.

---

## 📌 Project Highlights

- **Leakage-Free Evaluation:** Uses `GroupShuffleSplit` with the solute–solvent SMILES pair as the grouping key to evaluate generalization to unseen chemical systems.
- **Aleatoric Uncertainty Analysis:** Quantifies experimental measurement variability across 3,234 duplicate experimental systems, estimating an intrinsic RMSE noise floor of approximately **0.20 LogS**.
- **Thermodynamic Regularization:** Incorporates the van 't Hoff relationship as a soft physics-informed loss constraint, using an optimal physics loss weight of `λ_phys = 0.010`.
- **Molecular Feature Engineering:** Combines RDKit molecular descriptors, MACCS fingerprints, temperature features, and molecular graph representations.
- **Optimal Late-Fusion Ensemble:** Uses SciPy SLSQP optimization to combine ANN, PINN, and Chemprop predictions through optimized non-negative weights.
- **Command-Line Inference:** Provides `src/inference.py` to predict LogS for a user-specified solute, solvent, and temperature using the trained models without retraining.

---

## 📊 Model Architectures & Performance

The final prediction is generated using the following weighted ensemble:

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
│   └── Leakage-Free Solubility Prediction from Solute–Solvent Molecular Structures Using Deep Learning and Molecular Graph Learning.docx
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
└── requirements.txt                           # Python dependencies
```

*The structure above summarizes the main project files. Actual filenames and generated artifacts may vary with the local repository.*

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/solubility-prediction-system.git
cd solubility-prediction-system
```

Replace the example repository URL with your actual GitHub repository URL.

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

```bash
pip install -r requirements.txt
```

Ensure that the environment includes the required packages and compatible versions of PyTorch, RDKit, pandas, NumPy, scikit-learn, joblib, SciPy, and Chemprop.

The Chemprop command-line interface must be available in the active environment. Verify the installation with:

```bash
chemprop --help
```

The trained model checkpoints and preprocessing files must also be available under `models/` to run inference.

---

## 🚀 Execution Pipeline

### 1. Data Preparation

Prepare the leakage-free training and test splits and extract the molecular features:

```bash
python src/data_prep/prepare_dataset.py
```

This step prepares the dataset and generates the feature files used by the downstream training scripts.

### 2. Data Analysis & Physics Validation

Run the experimental uncertainty analysis and validate the thermodynamic assumptions:

```bash
python src/aleatoric_analysis.py
python src/check_vantthof_quality.py
python src/physics_lookup.py
```

These scripts analyze experimental variability, assess the quality of the van 't Hoff relationship, and generate thermodynamic information for physics-informed learning.

### 3. Model Training

Train the individual models using the prepared data:

```bash
python training/train_ann.py
python training/train_pinn.py
python training/train_chemprop.py
```

For hyperparameter experiments and physics-loss-weight sweeps, use the corresponding `run_*` scripts in the `training/` directory.

Training is not required when using the existing compatible model checkpoints for inference.

### 4. Ensemble Evaluation

Evaluate the individual models and optimize the late-fusion ensemble weights:

```bash
python src/evaluate_ensemble.py
```

The evaluation pipeline calculates ensemble predictions and performance metrics. Results are written to:

```text
results/ensemble_performance_metrics.csv
```

### 5. Solubility Prediction (Inference)

Use the inference pipeline to predict the LogS of a new solute–solvent system at a specified temperature.

**Command format:**

```bash
python src/inference.py --solute "<SOLUTE_SMILES>" --solvent "<SOLVENT_SMILES>" --temp <TEMPERATURE_K>
```

**Example:**

```bash
python src/inference.py --solute CCO --solvent O --temp 298.15
```

This example uses ethanol as the solute, water as the solvent, and a temperature of 298.15 K.

#### How the inference pipeline works

1. **Input:** Receives the solute SMILES, solvent SMILES, and temperature in Kelvin.
2. **Feature extraction:** Generates nine RDKit molecular descriptors and 166 MACCS fingerprint features for each molecule, along with temperature.
3. **ANN prediction:** Loads the trained ANN checkpoint, applies the saved preprocessing pipeline, and predicts LogS.
4. **PINN prediction:** Loads the trained PINN checkpoint and its corresponding preprocessing files to predict LogS.
5. **Chemprop prediction:** Writes the input to a temporary CSV file and executes the Chemprop prediction CLI using the trained molecular graph model and scaled temperature.
6. **Late-fusion ensemble:** Combines the three predictions using the weights 0.32 for ANN, 0.14 for PINN, and 0.54 for Chemprop.
7. **Output:** Returns the final ensemble LogS prediction rounded to four decimal places.

#### Example output format

```text
[AI Solubility Inference Engine]
Solute:  CCO
Solvent: O
Temp:    298.15 K

Generating features and executing late-fusion ensemble...

--- PREDICTION RESULTS ---
=============================
OPTIMAL ENSEMBLE LogS: -0.896
=============================
```

*The numerical prediction above is illustrative, not a verified output from the trained models.*

**Inference requirements:**

- The trained ANN and PINN checkpoints must exist in `models/`.
- The corresponding scalers, median-imputation files, and feature-column metadata must be available.
- The Chemprop checkpoint must be present at `models/chemprop_model/model_0/best.pt`.
- The inference environment must have compatible RDKit, PyTorch, scikit-learn, and Chemprop installations.
- Input SMILES strings must represent valid molecular structures, and the temperature must be supplied in Kelvin.

The inference script uses the existing trained models; it does not repeat dataset preparation, model training, or ensemble-weight optimization.

---

## 📄 Documentation & Methodology

Comprehensive technical details are available in the `docs/` folder, including:

- Leakage-free dataset splitting and molecular feature engineering.
- ANN architecture and hyperparameter experiments.
- Experimental uncertainty and the estimated intrinsic noise floor.
- Thermodynamic validation and van 't Hoff regularization.
- Chemprop molecular graph learning.
- Optimal late-fusion ensemble evaluation and performance metrics.

Refer to the project documentation for the complete methodology and experimental results.

---

## 🔬 Technologies Used

- **Python:** Core implementation.
- **RDKit:** Molecular parsing, descriptors, and MACCS fingerprints.
- **PyTorch:** ANN and PINN architectures and inference.
- **Chemprop:** Directed message-passing neural network for molecular graph learning.
- **Pandas & NumPy:** Data preparation and numerical processing.
- **Scikit-learn:** Data splitting, preprocessing, and evaluation.
- **SciPy:** Constrained optimization of ensemble weights.
- **Joblib:** Saving and loading preprocessing objects.

---
