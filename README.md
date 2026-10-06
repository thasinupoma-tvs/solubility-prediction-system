# 🧪 Leakage-Free Solubility Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A research-grade machine learning pipeline for predicting the equilibrium solubility (LogS in mol/L) of unseen solute–solvent chemical systems across varied temperatures. 

This repository implements a **Late-Fusion Optimal Ensemble** combining standard descriptor-based Artificial Neural Networks (ANN), Physics-Informed Neural Networks (PINN) regularized by thermodynamics, and Directed Message-Passing Neural Networks (Chemprop). The system operates exceptionally close to the intrinsic experimental noise floor of the dataset.

---

## 📌 Project Highlights

* **Leakage-Free Evaluation:** Utilizes a strict `GroupShuffleSplit` (Group = Solute SMILES + Solvent SMILES) to guarantee out-of-sample generalization to entirely novel chemical systems.
* **Aleatoric Uncertainty Baseline:** Quantified inter-laboratory measurement variance across 3,234 duplicate experimental systems to establish an intrinsic dataset noise floor of **RMSE ≈ 0.20 LogS**.
* **Thermodynamic Regularization:** Integrates the van 't Hoff equation as a soft physical loss constraint ($\lambda_{phys} = 0.010$) to ensure physically consistent, monotonic temperature-dependent predictions.
* **Late-Fusion Ensemble:** Leverages SciPy SLSQP optimization to mathematically blend the complementary strengths of 2D tabular features (RDKit + MACCS) and 3D graph representations.

---

## 📊 Model Architectures & Performance

The final prediction is driven by an optimally weighted late-fusion ensemble:  
**`LogS = 0.32 * ANN + 0.14 * PINN + 0.54 * Chemprop`**

| Model / Strategy | RMSE (LogS) | MAE (LogS) | R² Score |
| :--- | :---: | :---: | :---: |
| 1. Baseline ANN (Tabular + Temp) | 0.5035 | 0.3271 | 0.8291 |
| 2. PINN (Tabular + Temp + van 't Hoff) | 0.5054 | 0.3261 | 0.8278 |
| 3. Chemprop (Molecular Graph + Temp) | 0.4874 | 0.3076 | 0.8399 |
| 4. Equal Weight Blend (ANN + PINN + CP) | 0.4617 | 0.2936 | 0.8563 |
| 5. Physics-Graph Blend (40% PINN + 60% CP) | 0.4576 | 0.2893 | 0.8589 |
| **6. Optimal Ensemble (32% ANN + 14% PINN + 54% CP)** | **0.4549** | **0.2878** | **0.8605** |

*(Benchmarks evaluated on a strict 20% holdout test set from BigSolDB 2.0)*

---

## 📂 Repository Structure

```text
.
├── data/                               # DATASETS & FEATURES
│   ├── full_dataset/                   # Central source of truth (BigSolDB 2.0)
│   ├── train_raw.csv / test_raw.csv    # Leakage-free GroupShuffleSplits
│   ├── train_combined_features.csv     # Master training features (MACCS + RDKit)
│   ├── test_combined_features.csv      # Master test features (MACCS + RDKit)
│   ├── train_chemprop.csv / test_chemprop.csv  # Chemprop formatted data
│   └── final_ensemble_test_predictions.csv     # Master ensemble output
│
├── docs/                               # DOCUMENTATION
│   └── Leakage-Free Solubility Prediction from Solute–Solvent Molecular Structures Using Deep Learning and Molecular Graph Learning.docx
│
├── models/                             # TRAINED WEIGHTS & SCALERS
│   ├── chemprop_model/                 # GNN checkpoints (.pt)
│   ├── best_ann_model.pth              # Optimal ANN PyTorch weights
│   ├── best_pinn_model.pth             # Optimal PINN PyTorch weights
│   ├── scaler.pkl / pinn_scaler.pkl / chemprop_scaler.pkl  # Normalization states
│   ├── train_medians.pkl / pinn_train_medians.pkl          # Imputation states
│   └── feature_columns.pkl / pinn_feature_columns.pkl      # Feature ordering metadata
│
├── results/                            # METRICS, PLOTS & LOGS
│   ├── aleatoric_heterogeneity_profile.png         # Noise distribution plot
│   ├── aleatoric_uncertainty_summary.csv           # Intrinsic RMSE noise floor
│   ├── ann_final_experiment_results.xlsx           # 125-run hyperparameter grid search
│   ├── physics_lookup.csv                          # Thermodynamic baseline constraints
│   ├── pinn_lambda_experiments.csv                 # Physical loss weighting sweeps
│   ├── vanthoff_analysis_classified.csv            # Physics validation metrics
│   └── ensemble_performance_metrics.csv            # Final 32/14/54 blend metrics
│
├── src/                                # CORE PYTHON MODULES
│   ├── data_prep/                      
│   │   └── prepare_dataset.py          # Data splitting & feature extraction script
│   ├── aleatoric_analysis.py           # Dataset noise quantification logic
│   ├── check_vantthof_quality.py       # Thermodynamic feasibility validation
│   ├── physics_lookup.py               # Enthalpy extraction logic
│   ├── vanthoff.py                     # Physical equation definitions
│   ├── ann_model.py                    # PyTorch ANN Architecture class
│   ├── ann_training.py / pinn_training.py / chemprop_training.py # Core compilers
│   └── evaluate_ensemble.py            # Master SLSQP Late-Fusion weighting logic
│
├── training/                           # EXPERIMENT EXECUTION RUNNERS
│   ├── run_ann_experiments.py          # Executes full 125-run ANN grid
│   ├── run_pinn_lambda_experiments.py  # Executes physics loss hyperparameter sweeps
│   ├── train_ann.py                    # Trains final standard ANN
│   ├── train_pinn.py                   # Trains final PINN
│   └── train_chemprop.py               # Trains final chemprop
│
└── requirements.txt                    # Python environment dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/solubility-prediction-system.git
   cd solubility-prediction-system
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution Pipeline

### 1. Data Preparation
To run the leakage-free `GroupShuffleSplit` and extract the RDKit/MACCS features from scratch:
```bash
python src/data_prep/prepare_dataset.py
```

### 2. Data Analysis & Physics Validation
Calculate the dataset's intrinsic aleatoric uncertainty and validate the thermodynamic baseline assumptions:
```bash
python src/aleatoric_analysis.py
python src/check_vantthof_quality.py
python src/physics_lookup.py
```

### 3. Model Training
Train the individual architectures using the standardized features. For hyperparameter sweeping, utilize the scripts prefixed with `run_*` inside the `training/` folder. For final production models:
```bash
python training/train_ann.py
python training/train_pinn.py
python training/train_chemprop.py
```

### 4. Ensemble Evaluation
Calculate the optimal SLSQP late-fusion weights and output merged final predictions:
```bash
python src/evaluate_ensemble.py
```
*Results populate in `results/ensemble_performance_metrics.csv`*

---

## 📄 Documentation & Methodology
Comprehensive technical methodologies—including explicit deep learning network layouts, multi-seed validation protocols, thermodynamic derivation proofs, and hyperparameter tables—can be found in the `docs/` folder.

