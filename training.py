import torch
import pandas as pd
import torch.optim as optim


def train_model(
    model,
    learning_rate,
    train_loader,
    val_loader,
    loss_function,
    num_epochs,
    device,
    save_path,
    patience=5
):
    """
    Train and validate a 3D U-Net model with early stopping.
    """

    train_losses = []
    val_losses = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(mean_train_loss)

        model.eval()
        epoch_val_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                epoch_val_losses.append(loss.item())

        mean_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(mean_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Validation Loss: {mean_val_loss:.4f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping activated.")
            break

    history = pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses
    })

    history.to_csv(f"{save_path}/training_history.csv", index=False)

    return history
