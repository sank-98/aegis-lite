"""Train a clean (unmodified) CNN on CIFAR-10."""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn import SimpleCNN

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # CIFAR-10 channel-wise mean and std (computed over the full training set)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=os.path.join(RESULTS_DIR, "data"), train=True,
                                              download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=os.path.join(RESULTS_DIR, "data"), train=False,
                                             download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    save_path = os.path.join(RESULTS_DIR, "clean_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nClean model saved to: {save_path}")
    print(f"Final Test Accuracy: {evaluate(model, test_loader):.2f}%")


if __name__ == "__main__":
    main()
