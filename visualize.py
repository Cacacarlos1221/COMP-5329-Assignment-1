import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Batch Updates')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_accuracy_comparison(accuracies, labels):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    plt.bar(x, accuracies)
    plt.xticks(x, labels)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy_comparison.png')
    plt.close()

def plot_comparison_curves(losses_dict, title, ylabel='Loss', filename='comparison_curves.png'):
    plt.figure(figsize=(12, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_activation_comparison(relu_losses, gelu_losses):
    losses_dict = {
        'ReLU': relu_losses,
        'GELU': gelu_losses
    }
    plot_comparison_curves(losses_dict, 'ReLU vs GELU Activation Functions', filename='activation_comparison.png')

def plot_batchnorm_comparison(with_bn_losses, without_bn_losses):
    losses_dict = {
        'With BatchNorm': with_bn_losses,
        'Without BatchNorm': without_bn_losses
    }
    plot_comparison_curves(losses_dict, 'Effect of Batch Normalization', filename='batchnorm_comparison.png')

def plot_dropout_comparison(dropout_losses_dict):
    plot_comparison_curves(dropout_losses_dict, 'Effect of Different Dropout Rates', filename='dropout_comparison.png')

def plot_optimizer_comparison(sgd_losses, adam_losses):
    losses_dict = {
        'SGD with Momentum': sgd_losses,
        'Adam': adam_losses
    }
    plot_comparison_curves(losses_dict, 'SGD vs Adam Optimizer', filename='optimizer_comparison.png')