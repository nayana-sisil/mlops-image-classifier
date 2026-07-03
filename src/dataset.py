from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(
        root=Path(data_dir),
        transform=transform,
    )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, dataset.classes
