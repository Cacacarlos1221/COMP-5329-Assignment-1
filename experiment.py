import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork, Layer, BatchNormalization, Dropout
from visualize import plot_training_loss, plot_confusion_matrix, plot_accuracy_comparison

# Load and preprocess data
def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    # Load data from npy files
    X_train = np.load(train_data_path)
    y_train = np.load(train_label_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_label_path)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    def to_one_hot(y, num_classes=10):
        return np.eye(num_classes)[y.ravel()]
    
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    
    return X_train, y_train, X_test, y_test

def create_model(config, input_dim, output_dim=10):
    model = NeuralNetwork()
    
    # Add layers based on configuration
    hidden_dims = config['hidden_dims']
    dropout_rate = config.get('dropout_rate', 0.0)
    use_batch_norm = config.get('use_batch_norm', False)
    
    # Input layer
    model.add_layer(Layer(input_dim, hidden_dims[0]))
    if use_batch_norm:
        model.add_batch_norm(BatchNormalization(hidden_dims[0]))
    if dropout_rate > 0:
        model.add_dropout(Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(len(hidden_dims)-1):
        model.add_layer(Layer(hidden_dims[i], hidden_dims[i+1]))
        if use_batch_norm:
            model.add_batch_norm(BatchNormalization(hidden_dims[i+1]))
        if dropout_rate > 0:
            model.add_dropout(Dropout(dropout_rate))
    
    # Output layer
    model.add_layer(Layer(hidden_dims[-1], output_dim))
    
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, config):
    start_time = time.time()
    
    # Train model
    losses = model.train(
        X_train, y_train,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 0.01),
        momentum=config.get('momentum', 0.0),
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    training_time = time.time() - start_time
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
    test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'losses': losses,
        'test_predictions': test_predictions
    }

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        'Assignment1-Dataset/train_data.npy',
        'Assignment1-Dataset/train_label.npy',
        'Assignment1-Dataset/test_data.npy',
        'Assignment1-Dataset/test_label.npy'
    )
    
    input_dim = X_train.shape[1]
    
    # Define different model configurations for comparison
    configurations = [
        {
            'name': 'Baseline (Single Hidden Layer)',
            'hidden_dims': [128],
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'Multiple Hidden Layers',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'With Dropout',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.2,
            'use_batch_norm': False,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'With Batch Normalization',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'With Momentum',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'momentum': 0.9,
            'weight_decay': 0.0,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'With Weight Decay',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'momentum': 0.0,
            'weight_decay': 0.0001,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        {
            'name': 'Full Model (All Features)',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.2,
            'use_batch_norm': True,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.01
        }
    ]
    
    results = []
    train_accuracies = []
    test_accuracies = []
    model_names = []
    
    # Train and evaluate each configuration
    for config in configurations:
        print(f"\nTraining model: {config['name']}")
        model = create_model(config, input_dim)
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test, config)
        
        print(f"Training Accuracy: {result['train_accuracy']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        
        results.append(result)
        train_accuracies.append(result['train_accuracy'])
        test_accuracies.append(result['test_accuracy'])
        model_names.append(config['name'])
        
        # Save loss plot for the best model (Full Model)
        if config['name'] == 'Full Model (All Features)':
            plot_training_loss(result['losses'])
            plot_confusion_matrix(np.argmax(y_test, axis=1), result['test_predictions'])
    
    # Plot accuracy comparison
    plt_data = []
    for i in range(len(model_names)):
        plt_data.append(train_accuracies[i])
        plt_data.append(test_accuracies[i])
    
    plt_labels = []
    for name in model_names:
        plt_labels.append(f"{name} (Train)")
        plt_labels.append(f"{name} (Test)")
    
    plot_accuracy_comparison(plt_data, plt_labels)
    
    # Print summary table
    print("\nModel Comparison Summary:")
    print("-" * 100)
    print(f"{'Model Configuration':<30} | {'Train Accuracy':<15} | {'Test Accuracy':<15} | {'Training Time (s)':<20}")
    print("-" * 100)
    
    for i, config in enumerate(configurations):
        print(f"{config['name']:<30} | {train_accuracies[i]:<15.4f} | {test_accuracies[i]:<15.4f} | {results[i]['training_time']:<20.2f}")

if __name__ == '__main__':
    main()