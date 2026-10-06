# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.ann_model import Solubility_Predictor as ANNModel
from src.ann_training import train_and_evaluate_ANN


# ============================================================
# RANDOM SEEDS
# ============================================================

SEEDS = [
    42,
    7,
    99,
    123,
    2024
]


# ============================================================
# TRAINING CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "Config_1": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 300
    },

    "Config_2": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 300
    },

    "Config_3": {
        "learning_rate": 0.0005,
        "batch_size": 64,
        "num_epochs": 300
    },

    "Config_4": {
        "learning_rate": 0.0001,
        "batch_size": 64,
        "num_epochs": 300
    },

    "Config_5": {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "num_epochs": 300
    }
}


# ============================================================
# LOADING INDIVIDUAL AND COMBINED FEATURES
# ============================================================

print("Loading pre-split raw data frameworks...")

train_raw = pd.read_csv(
    DATA_DIR / "train_raw.csv"
)

test_raw = pd.read_csv(
    DATA_DIR / "test_raw.csv"
)


print("Loading pre-extracted specific features...")


# ============================================================
# RDKit FEATURES
# ============================================================

train_solute_rdkit = pd.read_csv(
    DATA_DIR / "train_solute_rdkit.csv"
)

train_solvent_rdkit = pd.read_csv(
    DATA_DIR / "train_solvent_rdkit.csv"
)

test_solute_rdkit = pd.read_csv(
    DATA_DIR / "test_solute_rdkit.csv"
)

test_solvent_rdkit = pd.read_csv(
    DATA_DIR / "test_solvent_rdkit.csv"
)


# ============================================================
# MACCS FEATURES
# ============================================================

train_solute_maccs = pd.read_csv(
    DATA_DIR / "train_solute_maccs.csv"
)

train_solvent_maccs = pd.read_csv(
    DATA_DIR / "train_solvent_maccs.csv"
)

test_solute_maccs = pd.read_csv(
    DATA_DIR / "test_solute_maccs.csv"
)

test_solvent_maccs = pd.read_csv(
    DATA_DIR / "test_solvent_maccs.csv"
)


# ============================================================
# CREATE MACCS FEATURE DATASETS
# ============================================================

print("Creating MACCS feature datasets...")

df_train_maccs = pd.concat(
    [
        train_raw,
        train_solute_maccs,
        train_solvent_maccs
    ],
    axis=1
)

df_test_maccs = pd.concat(
    [
        test_raw,
        test_solute_maccs,
        test_solvent_maccs
    ],
    axis=1
)


# ============================================================
# CREATE RDKit FEATURE DATASETS
# ============================================================

print("Creating RDKit feature datasets...")

df_train_rdkit = pd.concat(
    [
        train_raw,
        train_solute_rdkit,
        train_solvent_rdkit
    ],
    axis=1
)

df_test_rdkit = pd.concat(
    [
        test_raw,
        test_solute_rdkit,
        test_solvent_rdkit
    ],
    axis=1
)


# ============================================================
# LOAD COMBINED FEATURES
# ============================================================

print("Loading combined feature datasets...")

df_train_combine = pd.read_csv(
    DATA_DIR / "train_combined_features.csv"
)

df_test_combine = pd.read_csv(
    DATA_DIR / "test_combined_features.csv"
)


# ============================================================
# FEATURE DATASETS FOR EXPERIMENTS
# ============================================================

FEATURE_DATA = {

    # --------------------------------------------------------
    # Experiment 1
    # MACCS + Temperature
    # --------------------------------------------------------

    "Exp1_MACCS_with_Temp": (
        df_train_maccs,
        df_test_maccs
    ),


    # --------------------------------------------------------
    # Experiment 2
    # MACCS without Temperature
    # --------------------------------------------------------

    "Exp2_MACCS_no_Temp": (
        df_train_maccs.drop(
            columns=["Temperature_K"],
            errors="ignore"
        ),

        df_test_maccs.drop(
            columns=["Temperature_K"],
            errors="ignore"
        )
    ),


    # --------------------------------------------------------
    # Experiment 3
    # RDKit + Temperature
    # --------------------------------------------------------

    "Exp3_RDKit_with_Temp": (
        df_train_rdkit,
        df_test_rdkit
    ),


    # --------------------------------------------------------
    # Experiment 4
    # RDKit without Temperature
    # --------------------------------------------------------

    "Exp4_RDKit_no_Temp": (
        df_train_rdkit.drop(
            columns=["Temperature_K"],
            errors="ignore"
        ),

        df_test_rdkit.drop(
            columns=["Temperature_K"],
            errors="ignore"
        )
    ),


    # --------------------------------------------------------
    # Experiment 5
    # Combined features + Temperature
    # --------------------------------------------------------

    "Exp5_Combined_with_Temp": (
        df_train_combine,
        df_test_combine
    )
}


# ============================================================
# CHECK DATASETS
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING FEATURE DATASETS")
print("=" * 70)

for experiment_name, (
    train_data,
    test_data
) in FEATURE_DATA.items():

    print("\n")
    print(experiment_name)

    print(
        f"Train shape: {train_data.shape}"
    )

    print(
        f"Test shape : {test_data.shape}"
    )


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE SUMMARY FUNCTION
# ============================================================

def create_summary(results_df):

    summary_df = (

        results_df

        .groupby(
            [
                "Configuration",
                "Feature_Set"
            ]
        )

        .agg({

            "RMSE": [
                "mean",
                "std"
            ],

            "MAE": [
                "mean",
                "std"
            ],

            "R2": [
                "mean",
                "std"
            ]
        })

        .reset_index()
    )


    # --------------------------------------------------------
    # Flatten column names
    # --------------------------------------------------------

    summary_df.columns = [

        "_".join(
            col
        ).strip("_")

        for col in summary_df.columns
    ]


    return summary_df


# ============================================================
# SAVE CONFIGURATION RESULTS
# ============================================================

def save_configuration_results(
    results,
    config_name
):

    # --------------------------------------------------------
    # Create detailed DataFrame
    # --------------------------------------------------------

    config_results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    config_summary_df = create_summary(
        config_results_df
    )


    # --------------------------------------------------------
    # Configuration Excel path
    # --------------------------------------------------------

    config_results_path = (
        RESULTS_DIR
        / f"{config_name}_results.xlsx"
    )


    # --------------------------------------------------------
    # Save Excel
    # --------------------------------------------------------

    with pd.ExcelWriter(
        config_results_path,
        engine="openpyxl"
    ) as writer:

        config_results_df.to_excel(
            writer,
            sheet_name="Detailed_Runs",
            index=False
        )

        config_summary_df.to_excel(
            writer,
            sheet_name="Summary_Metrics",
            index=False
        )


    print("\n")
    print("-" * 70)

    print(
        f"{config_name} COMPLETED"
    )

    print(
        f"Runs completed: "
        f"{len(config_results_df)}"
    )

    print(
        f"Configuration results saved to:\n"
        f"{config_results_path}"
    )

    print("-" * 70)


    return config_results_df


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiments():

    # Store results from ALL configurations
    all_results = []


    # ========================================================
    # LOOP THROUGH TRAINING CONFIGURATIONS
    # ========================================================

    for config_name, config in CONFIGURATIONS.items():

        print("\n")
        print("=" * 70)

        print(
            f"STARTING {config_name}"
        )

        print(
            f"Learning Rate : "
            f"{config['learning_rate']}"
        )

        print(
            f"Batch Size    : "
            f"{config['batch_size']}"
        )

        print(
            f"Epochs        : "
            f"{config['num_epochs']}"
        )

        print(
            f"Seeds         : "
            f"{SEEDS}"
        )

        print(
            f"Expected runs : "
            f"{len(FEATURE_DATA) * len(SEEDS)}"
        )

        print("=" * 70)


        # Store results for CURRENT configuration
        config_results = []


        # ====================================================
        # LOOP THROUGH FEATURE SETS
        # ====================================================

        for experiment_name, (
            train_data,
            test_data
        ) in FEATURE_DATA.items():

            print("\n")
            print("-" * 70)

            print(
                f"Feature Experiment: "
                f"{experiment_name}"
            )

            print(
                f"Train shape: "
                f"{train_data.shape}"
            )

            print(
                f"Test shape: "
                f"{test_data.shape}"
            )

            print("-" * 70)


            # =================================================
            # LOOP THROUGH RANDOM SEEDS
            # =================================================

            for seed in SEEDS:

                print("\n")

                print(
                    f"Running "
                    f"{config_name} | "
                    f"{experiment_name} | "
                    f"Seed {seed}"
                )


                # ------------------------------------------------
                # Set NumPy seed
                # ------------------------------------------------

                np.random.seed(seed)


                # ------------------------------------------------
                # Set PyTorch seed
                # ------------------------------------------------

                torch.manual_seed(seed)


                # ------------------------------------------------
                # Set CUDA seed
                # ------------------------------------------------

                if torch.cuda.is_available():

                    torch.cuda.manual_seed_all(
                        seed
                    )


                # ------------------------------------------------
                # Train and evaluate ANN
                # ------------------------------------------------

                result = train_and_evaluate_ANN(

                    train_data=train_data,

                    test_data=test_data,

                    model_class=ANNModel,

                    learning_rate=config[
                        "learning_rate"
                    ],

                    batch_size=config[
                        "batch_size"
                    ],

                    num_epochs=config[
                        "num_epochs"
                    ],

                    # Do not save model
                    save_model=False
                )


                # =================================================
                # STORE RESULT
                # =================================================

                run_result = {

                    "Configuration":
                        config_name,

                    "Feature_Set":
                        experiment_name,

                    "Seed":
                        seed,

                    "Learning_Rate":
                        config[
                            "learning_rate"
                        ],

                    "Batch_Size":
                        config[
                            "batch_size"
                        ],

                    "Epochs":
                        config[
                            "num_epochs"
                        ],

                    "RMSE":
                        result["rmse"],

                    "MAE":
                        result["mae"],

                    "R2":
                        result["r2"],

                    "Best_Val_Loss":
                        result[
                            "best_val_loss"
                        ],

                    "Epochs_Trained":
                        result[
                            "epochs_trained"
                        ]
                }


                # Add to current configuration
                config_results.append(
                    run_result
                )


                # Add to all configurations
                all_results.append(
                    run_result
                )


                print(
                    f"Completed | "
                    f"RMSE: {result['rmse']:.6f} | "
                    f"MAE: {result['mae']:.6f} | "
                    f"R2: {result['r2']:.6f}"
                )


        # ====================================================
        # SAVE CURRENT CONFIGURATION
        # ====================================================

        save_configuration_results(
            results=config_results,
            config_name=config_name
        )


    # ========================================================
    # ALL CONFIGURATIONS COMPLETED
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "ALL CONFIGURATIONS COMPLETED"
    )

    print("=" * 70)


    # ========================================================
    # CREATE FINAL DETAILED RESULTS
    # ========================================================

    final_results_df = pd.DataFrame(
        all_results
    )


    # ========================================================
    # CREATE FINAL SUMMARY
    # ========================================================

    final_summary_df = create_summary(
        final_results_df
    )


    # ========================================================
    # FINAL EXCEL PATH
    # ========================================================

    final_results_path = (
        RESULTS_DIR
        / "ann_final_experiment_results.xlsx"
    )


    # ========================================================
    # SAVE FINAL EXCEL
    # ========================================================

    with pd.ExcelWriter(
        final_results_path,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # All detailed runs
        # ----------------------------------------------------

        final_results_df.to_excel(
            writer,
            sheet_name="Detailed_Runs",
            index=False
        )


        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        final_summary_df.to_excel(
            writer,
            sheet_name="Summary_Metrics",
            index=False
        )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "FINAL RESULTS CREATED"
    )

    print(
        f"Total runs: "
        f"{len(final_results_df)}"
    )

    print(
        f"Expected runs: "
        f"{len(CONFIGURATIONS) * len(FEATURE_DATA) * len(SEEDS)}"
    )

    print(
        f"Final results saved to:\n"
        f"{final_results_path}"
    )

    print("=" * 70)


    return (
        final_results_df,
        final_summary_df
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_experiments()










