"""
Model utilities for quantization experiments.

This module contains model definitions, data loading, and training utilities
that are shared across different quantization scripts.

Author: Yin Cao
Date: August 8, 2025
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Suppress PyTorch deprecation warnings for NLLLoss2d
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*", category=FutureWarning)


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            import json
            config = json.load(f)
        else:
            # Default to YAML for backward compatibility
            config = yaml.safe_load(f)

    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_name = model_config['name']
    num_classes = model_config['num_classes']

    if model_name == "SimpleCNN":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == "SimpleResNet":
        model = SimpleResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def load_model(model_path: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Load model with optional pretrained weights."""
    model = create_model(config, device)

    model_path = Path(model_path)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model path {model_path} not found, using random weights")

    return model


def load_data(config: Dict[str, Any], data_path: str) -> Tuple[DataLoader, DataLoader]:
    """Load dataset based on configuration."""
    data_config = config['data']
    dataset_name = data_config['dataset']
    batch_size = data_config['batch_size']
    num_workers = data_config.get('num_workers', 2)

    if dataset_name == "CIFAR10":
        return load_cifar10(data_path, batch_size, num_workers)
    elif dataset_name == "YourDataset":
        return load_your_dataset(data_path, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_your_dataset(data_path: str, batch_size: int,
                      num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Load your custom dataset.

    Replace this function with your own dataset loading logic.
    """
    # Example: Custom dataset loading
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    # train_dataset = YourCustomDataset(data_path, train=True, transform=transform)
    # test_dataset = YourCustomDataset(data_path, train=False, transform=transform)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                          shuffle=False, num_workers=num_workers)
    #
    # return train_loader, test_loader

    raise NotImplementedError("Please implement your custom dataset loading logic here")


def load_cifar10(data_path: str, batch_size: int,
                 num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset."""
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Test transforms without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Training data
    train_dataset = datasets.CIFAR10(data_path, train=True, download=True,
                                     transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    # Test data
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True,
                                    transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def create_criterion(config: Dict[str, Any]) -> nn.Module:
    """Create loss function based on configuration."""
    criterion_name = config['training']['criterion']

    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion_name == "MSELoss":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    training_config = config['training']
    optimizer_config = training_config['optimizer']

    lr = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0)
    optimizer_type = optimizer_config['type']

    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        momentum = optimizer_config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer,
                     config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on configuration."""
    training_config = config['training']
    scheduler_config = training_config.get('scheduler')

    if not scheduler_config:
        return None

    scheduler_type = scheduler_config['type']

    if scheduler_type == "StepLR":
        step_size = scheduler_config.get('step_size', 3)
        gamma = scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "CosineAnnealingLR":
        T_max = scheduler_config.get('T_max', 10)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "ReduceLROnPlateau":
        patience = scheduler_config.get('patience', 3)
        factor = scheduler_config.get('factor', 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   max_batches: Optional[int] = None) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy
