import pandas as pd
from pathlib import Path


# ============================================================
#   Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = (
    DATA_DIR /
    "train_combined_features.csv"
)

INPUT_FILE = (
    RESULTS_DIR /
    "vanthoff_analysis.csv"
)


# ============================================================
#  Load data
# ============================================================

print("=" * 60)
print("Checking van't Hoff Group Quality")
print("=" * 60)

# Load van't Hoff analysis results
df = pd.read_csv(INPUT_FILE)

# Load original training data
data = pd.read_csv(TRAIN_FILE)

print(
    f"Total groups analyzed: {len(df)}"
)

print(
    f"Training rows: {len(data)}"
)


# ============================================================
#  R² threshold
# ============================================================

R2_THRESHOLD = 0.90


# ============================================================
#  Classify groups based on R²
# ============================================================

df["physics_reliable"] = (
    df["r2"] >= R2_THRESHOLD
)


# ============================================================
#  Count reliable and unreliable groups
# ============================================================

reliable = (
    df["physics_reliable"].sum()
)

unreliable = (
    (~df["physics_reliable"]).sum()
)


print("\n" + "=" * 60)
print("Physics Quality")
print("=" * 60)

print(
    f"R² threshold: {R2_THRESHOLD}"
)

print(
    f"Reliable groups  : {reliable}"
)

print(
    f"Unreliable groups: {unreliable}"
)

print(
    f"Reliable percentage: "
    f"{100 * reliable / len(df):.2f}%"
)


# ============================================================
#  Sensitivity to different R² thresholds
# ============================================================

print("\n" + "=" * 60)
print("Sensitivity to R² Threshold")
print("=" * 60)

for threshold in [
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995
]:

    count = (
        df["r2"] >= threshold
    ).sum()

    percentage = (
        100 * count / len(df)
    )

    print(
        f"R² >= {threshold:<5} : "
        f"{count:>5} groups "
        f"({percentage:.2f}%)"
    )


# ============================================================
#  ΔH statistics for reliable groups
# ============================================================

reliable_df = df[
    df["physics_reliable"]
].copy()


print("\n" + "=" * 60)
print("ΔHsol Statistics for Reliable Groups")
print("=" * 60)

print(
    reliable_df[
        "delta_H_kJ_mol"
    ].describe()
)


# ============================================================
# Temperature information
# ============================================================

print("\n" + "=" * 60)
print("Temperature Information")
print("=" * 60)


# Group original training data
grouped = data.groupby(
    [
        "SMILES_Solute",
        "SMILES_Solvent"
    ]
)


temperature_info = []


print(
    "\nCalculating temperature information..."
)


for (solute, solvent), group in grouped:

    # Minimum temperature
    temperature_min = (
        group["Temperature_K"]
        .min()
    )

    # Maximum temperature
    temperature_max = (
        group["Temperature_K"]
        .max()
    )

    # Temperature range
    temperature_range = (
        temperature_max
        - temperature_min
    )

    temperature_info.append(
        {
            "SMILES_Solute": solute,
            "SMILES_Solvent": solvent,
            "temperature_min": temperature_min,
            "temperature_max": temperature_max,
            "temperature_range": temperature_range
        }
    )


# Convert list to DataFrame
temperature_df = pd.DataFrame(
    temperature_info
)


# ============================================================
#  Merge temperature information
# ============================================================

df = df.merge(
    temperature_df,
    on=[
        "SMILES_Solute",
        "SMILES_Solvent"
    ],
    how="left"
)


# ============================================================
#  Number of temperatures per group
# ============================================================

print("\n" + "-" * 60)
print("Number of Temperatures per Group")
print("-" * 60)

# n_temperatures already exists
# in vanthoff_analysis.csv

print(
    df["n_temperatures"].describe()
)


# ============================================================
# Temperature range statistics
# ============================================================

print("\n" + "-" * 60)
print("Temperature Range (K)")
print("-" * 60)

print(
    df["temperature_range"].describe()
)


# ============================================================
#  Temperature statistics for reliable groups
# ============================================================

reliable_df = df[
    df["physics_reliable"]
].copy()


print("\n" + "=" * 60)
print("Temperature Information for Reliable Groups")
print("=" * 60)


print("\nNumber of temperatures:")

print(
    reliable_df[
        "n_temperatures"
    ].describe()
)


print("\nTemperature range (K):")

print(
    reliable_df[
        "temperature_range"
    ].describe()
)


# ============================================================
#  R² quality categories
# ============================================================

print("\n" + "=" * 60)
print("Detailed van't Hoff Quality Analysis")
print("=" * 60)

print("\n" + "-" * 60)
print("R² Quality Categories")
print("-" * 60)


categories = {

    "R² < 0.80":
        df["r2"] < 0.80,

    "0.80 ≤ R² < 0.90":
        (
            (df["r2"] >= 0.80) &
            (df["r2"] < 0.90)
        ),

    "0.90 ≤ R² < 0.95":
        (
            (df["r2"] >= 0.90) &
            (df["r2"] < 0.95)
        ),

    "0.95 ≤ R² < 0.99":
        (
            (df["r2"] >= 0.95) &
            (df["r2"] < 0.99)
        ),

    "R² ≥ 0.99":
        df["r2"] >= 0.99
}


for category, condition in categories.items():

    count = condition.sum()

    percentage = (
        100 * count / len(df)
    )

    print(
        f"{category:<20}: "
        f"{count:>5} groups "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 14. R² vs number of temperatures
# ============================================================

print("\n" + "-" * 60)
print("R² by Number of Temperature Points")
print("-" * 60)


r2_by_temperature_count = (
    df.groupby(
        "n_temperatures"
    )["r2"]
    .agg(
        count="count",
        mean="mean",
        median="median"
    )
    .reset_index()
)


print(
    r2_by_temperature_count
    .to_string(index=False)
)


# ============================================================
#  R² vs temperature range
# ============================================================

print("\n" + "-" * 60)
print("R² by Temperature Range")
print("-" * 60)


df["temperature_range_category"] = pd.cut(

    df["temperature_range"],

    bins=[
        0,
        10,
        20,
        30,
        40,
        50,
        float("inf")
    ],

    labels=[
        "<10 K",
        "10–20 K",
        "20–30 K",
        "30–40 K",
        "40–50 K",
        ">50 K"
    ]
)


range_analysis = (
    df.groupby(
        "temperature_range_category",
        observed=False
    )["r2"]
    .agg(
        count="count",
        mean="mean",
        median="median"
    )
    .reset_index()
)


print(
    range_analysis
    .to_string(index=False)
)


# ============================================================
#  Investigate ΔHsol outliers
# ============================================================

print("\n" + "-" * 60)
print("Potential ΔHsol Outliers")
print("-" * 60)


high_dh = df[
    df["delta_H_kJ_mol"] > 100
]


low_dh = df[
    df["delta_H_kJ_mol"] < -50
]


print(
    f"ΔHsol > 100 kJ/mol : "
    f"{len(high_dh)} groups"
)

print(
    f"ΔHsol < -50 kJ/mol : "
    f"{len(low_dh)} groups"
)


# ============================================================
#  Show ΔHsol outlier details
# ============================================================

print("\n" + "-" * 60)
print("ΔHsol Outlier Details")
print("-" * 60)


outliers = df[
    (
        df["delta_H_kJ_mol"] > 100
    )
    |
    (
        df["delta_H_kJ_mol"] < -50
    )
].copy()


if len(outliers) > 0:

    columns_to_show = [

        "SMILES_Solute",

        "SMILES_Solvent",

        "n_temperatures",

        "temperature_range",

        "r2",

        "delta_H_kJ_mol"
    ]


    print(
        outliers[
            columns_to_show
        ]
        .sort_values(
            "delta_H_kJ_mol"
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No ΔHsol outliers found."
    )


# ============================================================
#  Investigate poor van't Hoff groups
# ============================================================

print("\n" + "-" * 60)
print("Poor van't Hoff Groups")
print("-" * 60)


poor_groups = df[
    df["r2"] < 0.90
].copy()


print(
    f"Number of groups with "
    f"R² < 0.90: "
    f"{len(poor_groups)}"
)


if len(poor_groups) > 0:

    columns_to_show = [

        "SMILES_Solute",

        "SMILES_Solvent",

        "n_temperatures",

        "temperature_range",

        "r2",

        "delta_H_kJ_mol"
    ]


    print(
        "\nFirst 20 groups:"
    )


    print(
        poor_groups[
            columns_to_show
        ]
        .sort_values(
            "r2"
        )
        .head(20)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No poor van't Hoff groups found."
    )


# ============================================================
# Save final classified analysis
# ============================================================

OUTPUT_FILE = (
    RESULTS_DIR /
    "vanthoff_analysis_classified.csv"
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
#  Completed
# ============================================================

print("\n" + "=" * 60)
print("Completed")
print("=" * 60)


print(
    f"Saved classified results to:\n"
    f"{OUTPUT_FILE}"
)