from pathlib import Path
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader



PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)



def prepare_data(
    train_data,
    test_data,
    batch_size=64,
    save_artifacts=False
):
    df_train = train_data.copy()
    df_test = test_data.copy()

    # Drop columns with too many missing values
    missing_ratio = df_train.isnull().mean()
    valid_cols = missing_ratio[missing_ratio < 0.4].index

    # Keep only valid columns that exist in both sets
    df_train = df_train[valid_cols]
    df_test = df_test[
        [c for c in valid_cols if c in df_test.columns]
    ]

    # Fill NaN values
    # Median is fitted ONLY on training data
    train_medians = df_train.median(numeric_only=True)

    df_train = df_train.fillna(train_medians)
    df_test = df_test.fillna(train_medians)

    # Drop non-numeric metadata columns
    drop_cols = [
        'SMILES_Solute',
        'SMILES_Solvent',
        'Solvent',
        'Compound_Name',
        'CAS',
        'PubChem_CID',
        'FDA_Approved',
        'Source'
    ]

    targets_to_drop = [
        'Solubility(mole_fraction)',
        'Solubility(mol/L)',
        'LogS(mol/L)'
    ]

    X_train = df_train.drop(
        columns=drop_cols + targets_to_drop,
        errors='ignore'
    )

    X_test = df_test.drop(
        columns=drop_cols + targets_to_drop,
        errors='ignore'
    )

    # Keep only numeric features
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    # Targets
    Y_train = df_train[['LogS(mol/L)']]
    Y_test = df_test[['LogS(mol/L)']]

    # Feature column names
    feature_columns = X_train.columns.tolist()

    # Scale input data
    # Fit ONLY on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing artifacts only when needed. 
    if save_artifacts:

        joblib.dump(
            train_medians,
            MODEL_DIR / "train_medians.pkl"
        )

        joblib.dump(
            feature_columns,
            MODEL_DIR / "feature_columns.pkl"
        )

        joblib.dump(
            scaler,
            MODEL_DIR / "scaler.pkl"
        )

    # Convert to PyTorch tensors
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

    # Create datasets
    train_dataset = TensorDataset(
        X_train_tensor,
        Y_train_tensor
    )

    val_dataset = TensorDataset(
        X_test_tensor,
        Y_test_tensor
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        X_train,
        X_test_tensor,
        Y_test_tensor,
        train_loader,
        val_loader
    )
    
    



def train_and_evaluate_ANN(
    train_data,
    test_data,
    model_class,
    learning_rate=0.0001,
    num_epochs=300,
    batch_size=64,
    dropout=None,
    save_model=False,
    model_filename="best_ann_model.pth"
):

    # Prepare data
    X_train, X_test_tensor, Y_test_tensor, train_loader, val_loader = prepare_data(
        train_data,
        test_data,
        batch_size=batch_size,
        save_artifacts=save_model
    )

    # Initialize Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Initialize Model
    input_size = X_train.shape[1]

    model = model_class(input_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Learning-rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    # Early stopping
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    # Store losses
    train_losses = []
    val_losses = []

    # Keep best model weights
    best_model_state = None

    # Training Loop
    for epoch in range(num_epochs):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        total_train_loss = 0

        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)

            loss = criterion(pred, Y_batch)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = (
            total_train_loss / len(train_loader)
        )

        train_losses.append(avg_train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for X_val, Y_val in val_loader:

                X_val = X_val.to(device)
                Y_val = Y_val.to(device)

                val_pred = model(X_val)

                val_loss = criterion(
                    val_pred,
                    Y_val
                )

                total_val_loss += val_loss.item()

        avg_val_loss = (
            total_val_loss / len(val_loader)
        )

        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # -------------------------
        # Early stopping
        # -------------------------
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            patience_counter = 0

            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch + 1}"
            )

            break

        # Print progress
        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {avg_train_loss:.5f} | "
                f"Val Loss: {avg_val_loss:.5f}"
            )

    # -------------------------
    # Load best model
    # -------------------------
    model.load_state_dict(best_model_state)

    model.to(device)
    model.eval()

    # -------------------------
    # Save model only if requested
    # -------------------------
    if save_model:

        model_path = MODEL_DIR / model_filename

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"[INFO] Model saved to: {model_path}"
        )

    # -------------------------
    # Final Evaluation
    # -------------------------
    X_test_tensor = X_test_tensor.to(device)

    with torch.no_grad():

        Y_pred = model(X_test_tensor)

    # Convert to NumPy
    Y_pred_np = Y_pred.cpu().numpy()
    Y_test_np = Y_test_tensor.cpu().numpy()

    # -------------------------
    # Performance Metrics
    # -------------------------
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

    # Print results
    print("\nFinal Performance")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print("-" * 40)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "y_test": Y_test_np,
        "y_pred": Y_pred_np,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1
    }