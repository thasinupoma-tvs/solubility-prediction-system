import pandas as pd
from pathlib import Path

from vanthoff import calculate_vanthoff_parameters


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train_combined_features.csv"


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Loading BigSolDB training data")
print("=" * 60)

df = pd.read_csv(TRAIN_FILE)

print(f"Number of rows: {len(df)}")


# ============================================================
# Check required columns
# ============================================================

required_columns = [
    "SMILES_Solute",
    "SMILES_Solvent",
    "Temperature_K",
    "LogS(mol/L)"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# Create solute-solvent groups
# ============================================================

grouped = df.groupby(
    [
        "SMILES_Solute",
        "SMILES_Solvent"
    ]
)


# ============================================================
# Calculate van't Hoff parameters
# ============================================================

results = []

print("\nCalculating van't Hoff parameters...")

for (solute, solvent), group in grouped:

    result = calculate_vanthoff_parameters(group)

    if result is None:
        continue

    result["SMILES_Solute"] = solute
    result["SMILES_Solvent"] = solvent

    results.append(result)


# ============================================================
# Create result dataframe
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 60)
print("Van't Hoff Analysis")
print("=" * 60)

print(
    f"Groups with >= 3 temperatures: "
    f"{len(results_df)}"
)

if len(results_df) > 0:

    print(
        f"\nAverage R²: "
        f"{results_df['r2'].mean():.4f}"
    )

    print(
        f"Median R²: "
        f"{results_df['r2'].median():.4f}"
    )

    print(
        f"Average ΔHsol: "
        f"{results_df['delta_H_kJ_mol'].mean():.2f} kJ/mol"
    )

    print(
        f"Median ΔHsol: "
        f"{results_df['delta_H_kJ_mol'].median():.2f} kJ/mol"
    )

    print("\nR² distribution:")

    print(
        results_df["r2"].describe()
    )


# ============================================================
# Save results
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "vanthoff_analysis.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)