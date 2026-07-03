from src.dataset import get_dataloaders

train_loader, val_loader, classes = get_dataloaders("data/raw")

print(classes)
print(len(train_loader.dataset))
print(len(val_loader.dataset))
