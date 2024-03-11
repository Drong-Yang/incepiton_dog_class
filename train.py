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
