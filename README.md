# Deep Learning Assignment 1: Multi-class Classification with Neural Networks

## Project Description
This project implements a deep neural network for multi-class classification using Python and NumPy. The network architecture incorporates several modern deep learning techniques to achieve optimal performance on the classification task.

### Key Features
- Multi-layer neural network with configurable hidden layers
- ReLU activation function for non-linearity
- Softmax activation in the output layer
- Cross-entropy loss function
- Batch Normalization for training stability
- Dropout layers for regularization
- Momentum-based SGD optimizer
- L2 weight decay for preventing overfitting

## Network Architecture
- Input Layer: Matches input feature dimension
- Hidden Layers: [256, 128, 64] neurons with ReLU activation
- Batch Normalization after each hidden layer
- Dropout (rate=0.2) for regularization
- Output Layer: 10 neurons with Softmax activation

## Implementation Details
### Data Preprocessing
- Feature standardization using StandardScaler
- One-hot encoding for target labels
- Mini-batch training with shuffled data

### Training Process
- Epochs: 100
- Batch Size: 32
- Learning Rate: 0.01
- Momentum: 0.9
- Weight Decay: 0.0001

### Model Components
1. **Activation Functions**
   - ReLU for hidden layers
   - Softmax for output layer

2. **Optimization Techniques**
   - Momentum-based gradient descent
   - L2 regularization (weight decay)
   - Batch normalization
   - Dropout regularization

## Usage Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place your training data in 'train.csv'
   - Place your test data in 'test.csv'
   - Ensure the data follows the required format

3. Run the training script:
   ```
   python train.py
   ```

## Project Structure
- `neural_network.py`: Core implementation of the neural network
- `train.py`: Training script and data preprocessing
- `requirements.txt`: Required Python packages

## Performance Metrics
The model's performance is evaluated using:
- Training accuracy
- Test accuracy
- Cross-entropy loss

Training progress is monitored every 10 epochs, displaying the current loss value.