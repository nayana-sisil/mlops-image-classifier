from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


def create_model(num_classes: int):
    # Load pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Replace the final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
