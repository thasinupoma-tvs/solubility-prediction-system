from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Constants
# ============================================================

R = 8.314462618  # J/(mol K)

LN10 = np.log(10.0)


# ============================================================
# Prepare PINN data
# ============================================================

def prepare_pinn_data(
    train_data,
    test_data,
    physics_lookup,
    batch_size=64,
    save_artifacts=False
):

    df_train = train_data.copy()
    df_test = test_data.copy()
    df_physics = physics_lookup.copy()


    # ========================================================
    # Check required columns
    # ========================================================

    required_train_columns = [
        "SMILES_Solute",
        "SMILES_Solvent",
        "Temperature_K",
        "LogS(mol/L)"
    ]

    for column in required_train_columns:

        if column not in df_train.columns:

            raise ValueError(
                f"Training data is missing column: {column}"
            )


    required_physics_columns = [
        "SMILES_Solute",
        "SMILES_Solvent",
        "delta_H_J_mol"
    ]

    for column in required_physics_columns:

        if column not in df_physics.columns:

            raise ValueError(
                f"Physics lookup is missing column: {column}"
            )


    # ========================================================
    # Drop columns with too many missing values
    # ========================================================

    missing_ratio = df_train.isnull().mean()

    valid_cols = missing_ratio[
        missing_ratio < 0.4
    ].index


    # Keep only columns existing in both datasets

    df_train = df_train[
        valid_cols
    ]

    df_test = df_test[
        [
            c for c in valid_cols
            if c in df_test.columns
        ]
    ]


    # ========================================================
    # Fill missing values
    # ========================================================

    train_medians = df_train.median(
        numeric_only=True
    )


    df_train = df_train.fillna(
        train_medians
    )

    df_test = df_test.fillna(
        train_medians
    )


    # ========================================================
    # Merge physics information
    # ========================================================

    df_train = df_train.merge(

        df_physics[
            [
                "SMILES_Solute",
                "SMILES_Solvent",
                "delta_H_J_mol"
            ]
        ],

        on=[
            "SMILES_Solute",
            "SMILES_Solvent"
        ],

        how="left"
    )


    # ========================================================
    # Create physics mask
    # ========================================================

    physics_mask = (
        df_train["delta_H_J_mol"]
        .notna()
    )


    print(
        f"\nPhysics information available for "
        f"{physics_mask.sum()} / {len(df_train)} "
        f"training samples "
        f"({100 * physics_mask.mean():.2f}%)"
    )


    # ========================================================
    # Columns to remove from model input
    # ========================================================

    drop_cols = [

        "SMILES_Solute",
        "SMILES_Solvent",

        "Solvent",
        "Compound_Name",
        "CAS",
        "PubChem_CID",
        "FDA_Approved",
        "Source",

        "delta_H_J_mol"
    ]


    targets_to_drop = [

        "Solubility(mole_fraction)",
        "Solubility(mol/L)",
        "LogS(mol/L)"
    ]


    # ========================================================
    # Create input features
    # ========================================================

    X_train = df_train.drop(

        columns=drop_cols + targets_to_drop,

        errors="ignore"
    )


    X_test = df_test.drop(

        columns=drop_cols + targets_to_drop,

        errors="ignore"
    )


    # ========================================================
    # Keep numeric features only
    # ========================================================

    X_train = X_train.select_dtypes(
        include=["number"]
    )

    X_test = X_test.select_dtypes(
        include=["number"]
    )


    # ========================================================
    # Make sure test columns match training columns
    # ========================================================

    X_test = X_test[
        X_train.columns
    ]


    # ========================================================
    # Target
    # ========================================================

    Y_train = df_train[
        ["LogS(mol/L)"]
    ]

    Y_test = df_test[
        ["LogS(mol/L)"]
    ]


    # ========================================================
    # Original temperature

    # We keep this separately because the PINN physics
    # equation needs the real temperature in Kelvin.
    # ========================================================
    

    T_train = df_train[
        "Temperature_K"
    ].values


    # ========================================================
    # Physics ΔH
    # ========================================================

    delta_H_train = df_train[
        "delta_H_J_mol"
    ].fillna(0.0).values


    # Physics mask

    physics_mask = physics_mask.values.astype(
        np.float32
    )


    # ========================================================
    # Feature names
    # ========================================================

    feature_columns = X_train.columns.tolist()


    # ========================================================
    # Scale input features
    # ========================================================

    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # Save preprocessing artifacts
    # ========================================================

    if save_artifacts:

        joblib.dump(

            train_medians,

            MODEL_DIR /
            "pinn_train_medians.pkl"
        )


        joblib.dump(

            feature_columns,

            MODEL_DIR /
            "pinn_feature_columns.pkl"
        )


        joblib.dump(

            scaler,

            MODEL_DIR /
            "pinn_scaler.pkl"
        )


    # ========================================================
    # Convert to PyTorch tensors
    # ========================================================

    X_train_tensor = torch.tensor(

        X_train_scaled,

        dtype=torch.float32
    )


    X_test_tensor = torch.tensor(

        X_test_scaled,

        dtype=torch.float32
    )


    Y_train_tensor = torch.tensor(

        Y_train.values,

        dtype=torch.float32
    )


    Y_test_tensor = torch.tensor(

        Y_test.values,

        dtype=torch.float32
    )


    T_train_tensor = torch.tensor(

        T_train.reshape(-1, 1),

        dtype=torch.float32
    )


    delta_H_tensor = torch.tensor(

        delta_H_train.reshape(-1, 1),

        dtype=torch.float32
    )


    physics_mask_tensor = torch.tensor(

        physics_mask.reshape(-1, 1),

        dtype=torch.float32
    )


    # ========================================================
    # Create training dataset
    # ========================================================

    train_dataset = TensorDataset(

        X_train_tensor,

        Y_train_tensor,

        T_train_tensor,

        delta_H_tensor,

        physics_mask_tensor
    )


    # ========================================================
    # Test dataset
    # ========================================================

    test_dataset = TensorDataset(

        X_test_tensor,

        Y_test_tensor
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False
    )


    return (

        X_train,

        X_test_tensor,

        Y_test_tensor,

        train_loader,

        test_loader
    )


# ============================================================
# Physics loss
# ============================================================

def calculate_physics_loss(
    predicted_logS,
    temperature,
    delta_H,
    physics_mask
):

    """
    Van't Hoff physics loss.

    log10(S) = -DeltaH/(2.303 R T) + C

    Instead of trying to predict C, we enforce the
    temperature dependence through pairwise differences.
    """

    # --------------------------------------------------------
    # Select samples with reliable physics information
    # --------------------------------------------------------

    valid = (
        physics_mask.squeeze(-1) > 0.5
    )


    if valid.sum() < 2:

        return torch.tensor(
            0.0,
            device=predicted_logS.device
        )


    pred = predicted_logS[valid]

    T = temperature[valid]

    H = delta_H[valid]


    # --------------------------------------------------------
    # Calculate physics quantity
    #
    # Δ logS between two samples:
    #
    # logS_i - logS_j
    #
    # = -ΔH/(2.303R)
    #   × (1/T_i - 1/T_j)
    # --------------------------------------------------------

    coefficient = (
        -H /
        (LN10 * R)
    )


    physics_value = (

        coefficient
        * (1.0 / T)
    )


    # --------------------------------------------------------
    # Remove unknown intercept C
    #
    # The intercept disappears when we subtract
    # the average value.
    # --------------------------------------------------------

    pred_centered = (
        pred -
        pred.mean()
    )


    physics_centered = (
        physics_value -
        physics_value.mean()
    )


    # --------------------------------------------------------
    # Physics residual
    # --------------------------------------------------------

    physics_residual = (

        pred_centered -
        physics_centered
    )


    # --------------------------------------------------------
    # Physics loss
    # --------------------------------------------------------

    physics_loss = torch.mean(
        physics_residual ** 2
    )


    return physics_loss


# ============================================================
# Train and evaluate PINN
# ============================================================

def train_and_evaluate_PINN(

    train_data,

    test_data,

    physics_lookup,

    model_class,

    learning_rate=0.0005,

    num_epochs=300,

    batch_size=64,

    lambda_phys=0.01,

    save_model=False,

    model_filename="best_pinn_model.pth"

):


    # ========================================================
    # Prepare data
    # ========================================================

    (

        X_train,

        X_test_tensor,

        Y_test_tensor,

        train_loader,

        test_loader

    ) = prepare_pinn_data(

        train_data,

        test_data,

        physics_lookup,

        batch_size=batch_size,

        save_artifacts=save_model
    )


    # ========================================================
    # Device
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )


    # ========================================================
    # Initialize model
    # ========================================================

    input_size = X_train.shape[1]


    model = model_class(
        input_size
    ).to(device)


    # ========================================================
    # Data loss
    # ========================================================

    criterion = nn.MSELoss()


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.Adam(

        model.parameters(),

        lr=learning_rate
    )


    # ========================================================
    # Learning-rate scheduler
    # ========================================================

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=10
        )
    )


    # ========================================================
    # Early stopping
    # ========================================================

    best_val_loss = float("inf")

    patience = 20

    patience_counter = 0


    # ========================================================
    # Store losses
    # ========================================================

    train_losses = []

    val_losses = []

    data_losses = []

    physics_losses = []


    # ========================================================
    # Best model
    # ========================================================

    best_model_state = None


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(num_epochs):


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()


        total_train_loss = 0.0

        total_data_loss = 0.0

        total_physics_loss = 0.0


        for (

            X_batch,

            Y_batch,

            T_batch,

            H_batch,

            physics_mask_batch

        ) in train_loader:


            X_batch = X_batch.to(device)

            Y_batch = Y_batch.to(device)

            T_batch = T_batch.to(device)

            H_batch = H_batch.to(device)

            physics_mask_batch = (
                physics_mask_batch.to(device)
            )


            # Clear gradients

            optimizer.zero_grad()


            # ------------------------------------------------
            # ANN prediction
            # ------------------------------------------------

            pred = model(
                X_batch
            )


            # ------------------------------------------------
            # Data loss
            # ------------------------------------------------

            data_loss = criterion(

                pred,

                Y_batch
            )


            # ------------------------------------------------
            # Physics loss
            # ------------------------------------------------

            physics_loss = (
                calculate_physics_loss(

                    predicted_logS=pred,

                    temperature=T_batch,

                    delta_H=H_batch,

                    physics_mask=physics_mask_batch
                )
            )


            # ------------------------------------------------
            # Total PINN loss
            # ------------------------------------------------

            total_loss = (

                data_loss
                +
                lambda_phys * physics_loss
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            total_loss.backward()


            # ------------------------------------------------
            # Update model
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # Accumulate losses
            # ------------------------------------------------

            total_train_loss += (
                total_loss.item()
            )

            total_data_loss += (
                data_loss.item()
            )

            total_physics_loss += (
                physics_loss.item()
            )


        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        avg_train_loss = (

            total_train_loss /
            len(train_loader)
        )


        avg_data_loss = (

            total_data_loss /
            len(train_loader)
        )


        avg_physics_loss = (

            total_physics_loss /
            len(train_loader)
        )


        train_losses.append(
            avg_train_loss
        )

        data_losses.append(
            avg_data_loss
        )

        physics_losses.append(
            avg_physics_loss
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()


        total_val_loss = 0.0


        with torch.no_grad():

            for X_val, Y_val in test_loader:

                X_val = X_val.to(device)

                Y_val = Y_val.to(device)


                val_pred = model(
                    X_val
                )


                val_loss = criterion(

                    val_pred,

                    Y_val
                )


                total_val_loss += (
                    val_loss.item()
                )


        avg_val_loss = (

            total_val_loss /
            len(test_loader)
        )


        val_losses.append(
            avg_val_loss
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            avg_val_loss
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if avg_val_loss < best_val_loss:

            best_val_loss = (
                avg_val_loss
            )

            patience_counter = 0


            best_model_state = {

                k: v.cpu().clone()

                for k, v
                in model.state_dict().items()
            }


        else:

            patience_counter += 1


        if patience_counter >= patience:

            print(
                f"Early stopping at "
                f"epoch {epoch + 1}"
            )

            break


        # ----------------------------------------------------
        # Print progress 
        # ----------------------------------------------------

        if (epoch + 1) % 20 == 0:

            print(

                f"Epoch {epoch + 1} | "

                f"Data Loss: "
                f"{avg_data_loss:.5f} | "

                f"Physics Loss: "
                f"{avg_physics_loss:.5f} | "

                f"Total Loss: "
                f"{avg_train_loss:.5f} | "

                f"Val Loss: "
                f"{avg_val_loss:.5f}"
            )


    # ========================================================
    # Load best model
    # ========================================================

    model.load_state_dict(
        best_model_state
    )


    model.to(device)

    model.eval()


    # ========================================================
    # Save model
    # ========================================================

    if save_model:

        model_path = (
            MODEL_DIR /
            model_filename
        )


        torch.save(

            model.state_dict(),

            model_path
        )


        print(
            f"[INFO] Model saved to: "
            f"{model_path}"
        )


    # ========================================================
    # Final evaluation
    # ========================================================

    X_test_tensor = (
        X_test_tensor.to(device)
    )


    with torch.no_grad():

        Y_pred = model(
            X_test_tensor
        )


    # ========================================================
    # Convert to NumPy
    # ========================================================

    Y_pred_np = (
        Y_pred.cpu().numpy()
    )


    Y_test_np = (
        Y_test_tensor.cpu().numpy()
    )


    # ========================================================
    # Metrics
    # ========================================================

    mse = mean_squared_error(

        Y_test_np,

        Y_pred_np
    )


    rmse = np.sqrt(mse)


    mae = mean_absolute_error(

        Y_test_np,

        Y_pred_np
    )


    r2 = r2_score(

        Y_test_np,

        Y_pred_np
    )


    # ========================================================
    # Print performance
    # ========================================================

    print("\nFinal PINN Performance")

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Physics weight λ: "
        f"{lambda_phys}"
    )

    print("-" * 40)


    # ========================================================
    # Return results
    # ========================================================

    return {

        "model": model,

        "train_losses":
            train_losses,

        "val_losses":
            val_losses,

        "data_losses":
            data_losses,

        "physics_losses":
            physics_losses,

        "y_test":
            Y_test_np,

        "y_pred":
            Y_pred_np,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2,

        "best_val_loss":
            best_val_loss,

        "epochs_trained":
            epoch + 1
    }