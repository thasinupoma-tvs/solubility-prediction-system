import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# Paths Setup
# ============================================================
PROJECT_ROOT = PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_aleatoric_analysis():
    print("=" * 70)
    print("      ALEATORIC UNCERTAINTY ANALYSIS (BIGSOLDB 2.0 DATASET)       ")
    print("=" * 70)

    # Point directly to data/full_dataset/BigSolDBv2.0.csv
    data_path = DATA_DIR / "full_dataset" / "BigSolDBv2.0.csv"

    if not data_path.exists():
        # Fallback check if saved directly in data/
        data_path = DATA_DIR / "BigSolDBv2.0.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Could not locate 'BigSolDBv2.0.csv' in: {DATA_DIR / 'full_dataset'}\n"
            )

    print(f"[OK] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Standardize column names
    solute_col = "SMILES_Solute" if "SMILES_Solute" in df.columns else "solute"
    solvent_col = "SMILES_Solvent" if "SMILES_Solvent" in df.columns else "solvent"
    temp_col = "Temperature_K" if "Temperature_K" in df.columns else "temperature"
    target_col = "LogS(mol/L)" if "LogS(mol/L)" in df.columns else ("LogS" if "LogS" in df.columns else "Actual_LogS")

    # Group by identical experimental conditions (Solute + Solvent + Temperature)
    group_cols = [solute_col, solvent_col, temp_col]
    
    experimental_groups = (
        df.groupby(group_cols)[target_col]
        .agg(['count', 'mean', 'std', 'min', 'max'])
        .reset_index()
    )

    # Filter for duplicate system measurements
    conflicting_systems = experimental_groups[experimental_groups['count'] > 1].copy()
    conflicting_systems['Max_Min_Spread'] = conflicting_systems['max'] - conflicting_systems['min']
    
    mean_system_std = conflicting_systems['std'].mean()

    # Calculate exact Intrinsic RMSE Floor across all duplicate points
    df_duplicates = df.merge(
        conflicting_systems[group_cols + ['mean']],
        on=group_cols,
        how='inner'
    )
    squared_deviations = (df_duplicates[target_col] - df_duplicates['mean']) ** 2
    rmse_floor = np.sqrt(squared_deviations.mean())

    # Print summary metrics
    print(f"\nTotal Dataset Samples Analyzed        : {len(df):,}")
    print(f"Total Duplicate Groups Found          : {len(conflicting_systems):,}")
    print(f"Average Within-Group Std Dev          : {mean_system_std:.2f} LogS")
    print(f"Theoretical Intrinsic RMSE Floor      : {rmse_floor:.2f} LogS")
    print("-" * 70)

    # ============================================================
    # Plot  (Histogram of Within-System Variation)
    # ============================================================
    plt.figure(figsize=(9, 5))
    sns.histplot(conflicting_systems['std'].dropna(), kde=True, color='teal', bins=35, edgecolor='black', alpha=0.7)
    plt.axvline(mean_system_std, color='crimson', linestyle='--', linewidth=2,
                label=f'Mean System SD = {mean_system_std:.3f}')
    plt.title('BigSolDB 2.0 Intrinsic Data Heterogeneity Profile', fontsize=12, fontweight='bold')
    plt.xlabel('Standard Deviation of Measured LogS Within Identical Systems', fontsize=10)
    plt.ylabel('Count of Unique Solution Coordinates', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()

    out_plot_file = RESULTS_DIR / "aleatoric_heterogeneity_profile.png"
    plt.savefig(out_plot_file, dpi=300)
    plt.close()

    # ============================================================
    # Save CSV Outputs
    # ============================================================
    summary_df = pd.DataFrame([{
        "Dataset": "BigSolDB 2.0",
        "Total_Samples": len(df),
        "Duplicate_Groups_Found": len(conflicting_systems),
        "Mean_Within_Group_Std_LogS": round(mean_system_std, 2),
        "Theoretical_Intrinsic_RMSE_Floor": round(rmse_floor, 2),
        "Max_Min_Spread_Mean": round(conflicting_systems['Max_Min_Spread'].mean(), 2),
        "Max_Min_Spread_Max": round(conflicting_systems['Max_Min_Spread'].max(), 2)
    }])
    
    out_summary_file = RESULTS_DIR / "aleatoric_uncertainty_summary.csv"
    out_duplicates_file = RESULTS_DIR / "aleatoric_duplicate_groups.csv"

    summary_df.to_csv(out_summary_file, index=False)
    conflicting_systems.to_csv(out_duplicates_file, index=False)

    print(f"[OK] Summary metrics report saved to: {out_summary_file}")
    print(f"[OK] Heterogeneity plot saved to   : {out_plot_file}\n")
    return summary_df


if __name__ == "__main__":
    run_aleatoric_analysis()