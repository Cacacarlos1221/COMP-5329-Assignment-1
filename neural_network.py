import numpy as np

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        return y_pred - y_true

class BatchNormalization:
    def __init__(self, input_dim, epsilon=1e-8):
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.epsilon = epsilon
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.rate, size=x.shape) / (1-self.rate)
            return x * self.mask
        return x

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        self.activation = getattr(Activation, activation)
        
        # For momentum
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
        
    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        return self.activation(self.z)
    
    def backward(self, delta, learning_rate=0.01, momentum=0.9, weight_decay=0.0001):
        if self.activation.__name__ == 'relu':
            delta = delta * Activation.relu_derivative(self.z)
        
        d_weights = np.dot(self.input.T, delta) / delta.shape[0]
        d_bias = np.mean(delta, axis=0)
        
        # Add weight decay
        d_weights += weight_decay * self.weights
        
        # Update with momentum
        self.v_w = momentum * self.v_w - learning_rate * d_weights
        self.v_b = momentum * self.v_b - learning_rate * d_bias
        
        self.weights += self.v_w
        self.bias += self.v_b
        
        return np.dot(delta, self.weights.T)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.batch_norms = []
        self.dropouts = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def add_batch_norm(self, batch_norm):
        self.batch_norms.append(batch_norm)
        
    def add_dropout(self, dropout):
        self.dropouts.append(dropout)
        
    def forward(self, x, training=True):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i].forward(x, training)
            if i < len(self.dropouts):
                x = self.dropouts[i].forward(x, training)
        return Activation.softmax(x)
    
    def backward(self, x, y, learning_rate=0.01, momentum=0.9, weight_decay=0.0001):
        m = x.shape[0]
        activations = [x]
        
        # Forward pass
        output = self.forward(x)
        
        # Backward pass
        delta = Loss.cross_entropy_derivative(output, y)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate, momentum, weight_decay)
            
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01, momentum=0.9, weight_decay=0.0001):
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward and backward pass
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, learning_rate, momentum, weight_decay)
                
                # Calculate loss
                loss = Loss.cross_entropy(output, batch_y)
                losses.append(loss)
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {np.mean(losses[-n_samples//batch_size:])}')
        
        return losses
    
    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)