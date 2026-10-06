import pandas as pd
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

VANTHOFF_FILE = (
    RESULTS_DIR /
    "vanthoff_analysis_classified.csv"
)

LOOKUP_FILE = (
    RESULTS_DIR /
    "physics_lookup.csv"
)


# ============================================================
# Load van't Hoff results
# ============================================================

print("=" * 60)
print("Creating Physics Lookup")
print("=" * 60)

vanthoff_df = pd.read_csv(
    VANTHOFF_FILE
)

print(
    f"Total van't Hoff groups: "
    f"{len(vanthoff_df)}"
)


# ============================================================
# Keep only reliable physics groups
# ============================================================

R2_THRESHOLD = 0.90

reliable_physics = vanthoff_df[
    vanthoff_df["r2"] >= R2_THRESHOLD
].copy()


print(
    f"Reliable physics groups: "
    f"{len(reliable_physics)}"
)


# ============================================================
# Keep required columns
# ============================================================

physics_lookup_df = reliable_physics[
    [
        "SMILES_Solute",
        "SMILES_Solvent",
        "delta_H_J_mol",
        "r2"
    ]
].copy()


# ============================================================
# Save physics lookup table
# ============================================================

physics_lookup_df.to_csv(
    LOOKUP_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print(
    f"Lookup entries: "
    f"{len(physics_lookup_df)}"
)

print(
    f"\nPhysics lookup saved to:\n"
    f"{LOOKUP_FILE}"
)

print("\n" + "=" * 60)
print("Completed")
print("=" * 60)