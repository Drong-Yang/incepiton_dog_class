"""Training utilities: logging, visualization, metrics."""
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.savefig('training_curves.png')
