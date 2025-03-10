import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork, Layer, BatchNormalization, Dropout

# Load and preprocess data
def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    # Load data from npy files
    X_train = np.load(train_data_path)
    y_train = np.load(train_label_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_label_path)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    def to_one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]
    
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    
    return X_train, y_train, X_test, y_test

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        'Assignment1-Dataset/train_data.npy',
        'Assignment1-Dataset/train_label.npy',
        'Assignment1-Dataset/test_data.npy',
        'Assignment1-Dataset/test_label.npy'
    )
    
    # Create model
    model = NeuralNetwork()
    
    # Add layers with batch normalization and dropout
    input_dim = X_train.shape[1]
    hidden_dims = [256, 128, 64]  # Multiple hidden layers
    output_dim = 10  # 10 classes
    
    # Input layer
    model.add_layer(Layer(input_dim, hidden_dims[0]))
    model.add_batch_norm(BatchNormalization(hidden_dims[0]))
    model.add_dropout(Dropout(0.2))
    
    # Hidden layers
    for i in range(len(hidden_dims)-1):
        model.add_layer(Layer(hidden_dims[i], hidden_dims[i+1]))
        model.add_batch_norm(BatchNormalization(hidden_dims[i+1]))
        model.add_dropout(Dropout(0.2))
    
    # Output layer
    model.add_layer(Layer(hidden_dims[-1], output_dim))
    
    # Train model
    losses = model.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
    test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
    
    print(f'Training Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()