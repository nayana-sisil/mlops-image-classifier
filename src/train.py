import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from dataset import get_dataloaders
from model import create_model


DATA_PATH = "data/processed"
IMAGE_SIZE = 128
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
EPOCHS = 2


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        data_dir=DATA_PATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )

    model = create_model(num_classes=len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    mlflow.set_experiment("Plant Disease Classification")

    with mlflow.start_run():
        # Log configuration
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("model", "ResNet18")

        best_accuracy = 0.0

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            print("=" * 50)
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
            print(f"Accuracy   : {val_accuracy:.4f}")

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                torch.save(
                    model.state_dict(),
                    "models/best_model.pth",
                )

        mlflow.log_metric("best_accuracy", best_accuracy)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            serialization_format="pickle",
        )

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
