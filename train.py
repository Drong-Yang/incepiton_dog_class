"""Training script for dog breed classification."""
import torch
import torch.nn as nn
from model import create_model

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (model(imgs).argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            total_loss += loss.item()
            correct += (model(imgs).argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'lr_step': 30,
    'lr_gamma': 0.1,
}
