import json
import math
import random

class ImprovedNeuralNetwork:
    def __init__(self, input_size=7, hidden_sizes=[15, 10], output_size=1, learning_rate=0.01, dropout_rate=0.1):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_sizes = hidden_sizes
        
        # Initialize weights for multiple hidden layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                 for _ in range(layer_sizes[i])]
            b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return max(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return 1 if x > 0 else 0
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function"""
        return x if x > 0 else alpha * x
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return 1 if x > 0 else alpha
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def forward(self, inputs, training=False):
        """Forward pass through the network"""
        self.layer_outputs = [inputs]
        current_input = inputs
        
        for i in range(len(self.weights)):
            # Calculate weighted sum
            layer_input = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = self.biases[i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * self.weights[i][k][j]
                layer_input.append(weighted_sum)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                layer_output = layer_input  # Linear output for regression
            else:  # Hidden layers
                layer_output = [self.leaky_relu(x) for x in layer_input]
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    for k in range(len(layer_output)):
                        if random.random() < self.dropout_rate:
                            layer_output[k] = 0
                        else:
                            layer_output[k] /= (1 - self.dropout_rate)  # Scale remaining neurons
            
            self.layer_outputs.append(layer_output)
            current_input = layer_output
        
        return current_input[0]  # Return single output
    
    def backward(self, inputs, target):
        """Backward pass with support for multiple hidden layers"""
        # Calculate output error
        output = self.layer_outputs[-1][0]
        output_error = target - output
        
        # Collect all layer errors (from output to input)
        all_errors = []
        
        # Output layer error
        all_errors.append([output_error])
        
        # Calculate errors for each hidden layer (backwards)
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            current_layer_size = len(self.layer_outputs[layer_idx + 1])
            next_layer_errors = all_errors[0]
            current_layer_errors = []
            
            for neuron_idx in range(current_layer_size):
                error = 0
                for next_neuron_idx in range(len(next_layer_errors)):
                    error += next_layer_errors[next_neuron_idx] * self.weights[layer_idx + 1][neuron_idx][next_neuron_idx]
                
                # Apply derivative of activation function
                neuron_output = self.layer_outputs[layer_idx + 1][neuron_idx]
                error *= self.leaky_relu_derivative(neuron_output)
                current_layer_errors.append(error)
            
            all_errors.insert(0, current_layer_errors)
        
        # Update weights and biases using collected errors
        for layer_idx in range(len(self.weights)):
            layer_errors = all_errors[layer_idx]
            layer_inputs = self.layer_outputs[layer_idx]
            
            # Update weights
            for input_idx in range(len(layer_inputs)):
                for neuron_idx in range(len(layer_errors)):
                    self.weights[layer_idx][input_idx][neuron_idx] += self.learning_rate * layer_errors[neuron_idx] * layer_inputs[input_idx]
            
            # Update biases
            for neuron_idx in range(len(layer_errors)):
                self.biases[layer_idx][neuron_idx] += self.learning_rate * layer_errors[neuron_idx]
    
    def train(self, training_data, validation_data, epochs=3000):
        """Train with validation monitoring"""
        best_val_error = float('inf')
        patience = 200
        no_improve_count = 0
        
        for epoch in range(epochs):
            # Training
            total_error = 0
            random.shuffle(training_data)
            
            for inputs, target in training_data:
                prediction = self.forward(inputs, training=True)
                self.backward(inputs, target)
                error = (target - prediction) ** 2
                total_error += error
            
            avg_train_error = total_error / len(training_data)
            
            # Validation
            val_error = 0
            for inputs, target in validation_data:
                prediction = self.forward(inputs, training=False)
                val_error += (target - prediction) ** 2
            avg_val_error = val_error / len(validation_data)
            
            # Early stopping
            if avg_val_error < best_val_error:
                best_val_error = avg_val_error
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Train Error: {avg_train_error:.4f}, Val Error: {avg_val_error:.4f}")
    
    def get_weights(self):
        """Return weights and biases for hardcoding"""
        return {
            'weights': self.weights,
            'biases': self.biases,
            'hidden_sizes': self.hidden_sizes
        }

def engineer_features(days, miles, receipts):
    """Create additional features from raw inputs"""
    # Original features
    features = [days, miles, receipts]
    
    # Derived features
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    miles_per_receipt = miles / receipts if receipts > 0 else 0
    
    # Polynomial features
    features.extend([
        miles_per_day,
        receipts_per_day,
        miles_per_receipt,
        days * days  # Non-linear day effect
    ])
    
    return features

def load_improved_training_data():
    """Load and preprocess training data with feature engineering"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Extract all features first
    all_features = []
    all_outputs = []
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        
        features = engineer_features(days, miles, receipts)
        all_features.append(features)
        all_outputs.append(output)
    
    # Calculate normalization parameters for all features
    feature_mins = [min(feature_vals) for feature_vals in zip(*all_features)]
    feature_maxs = [max(feature_vals) for feature_vals in zip(*all_features)]
    output_min, output_max = min(all_outputs), max(all_outputs)
    
    # Normalize data
    training_data = []
    for i, features in enumerate(all_features):
        normalized_features = []
        for j, feature in enumerate(features):
            if feature_maxs[j] != feature_mins[j]:
                norm_feature = (feature - feature_mins[j]) / (feature_maxs[j] - feature_mins[j])
            else:
                norm_feature = 0
            normalized_features.append(norm_feature)
        
        norm_output = (all_outputs[i] - output_min) / (output_max - output_min)
        training_data.append((normalized_features, norm_output))
    
    # Split into train/validation
    random.shuffle(training_data)
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    normalization_params = {
        'feature_mins': feature_mins,
        'feature_maxs': feature_maxs,
        'output_min': output_min,
        'output_max': output_max
    }
    
    return train_data, val_data, normalization_params

def main():
    print("Loading and engineering training data...")
    train_data, val_data, norm_params = load_improved_training_data()
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Try different architectures
    architectures = [
        {'hidden_sizes': [15, 10], 'learning_rate': 0.05},
        {'hidden_sizes': [20], 'learning_rate': 0.03},
        {'hidden_sizes': [25, 15, 10], 'learning_rate': 0.02},
    ]
    
    best_model = None
    best_error = float('inf')
    
    for i, arch in enumerate(architectures):
        print(f"\n=== Testing Architecture {i+1}: {arch} ===")
        nn = ImprovedNeuralNetwork(
            input_size=7,  # 7 engineered features
            hidden_sizes=arch['hidden_sizes'],
            learning_rate=arch['learning_rate']
        )
        
        nn.train(train_data, val_data, epochs=2000)
        
        # Test on validation set
        val_error = 0
        for inputs, target in val_data:
            prediction = nn.forward(inputs, training=False)
            val_error += (target - prediction) ** 2
        avg_val_error = val_error / len(val_data)
        
        print(f"Final validation error: {avg_val_error:.4f}")
        
        if avg_val_error < best_error:
            best_error = avg_val_error
            best_model = nn
    
    print(f"\n=== Best Model (Validation Error: {best_error:.4f}) ===")
    weights = best_model.get_weights()
    
    print("\n# Copy these into your improved inference script:")
    print("NORMALIZATION_PARAMS = {")
    for key, value in norm_params.items():
        print(f"    '{key}': {value},")
    print("}")
    
    print(f"\nHIDDEN_SIZES = {weights['hidden_sizes']}")
    print("WEIGHTS = [")
    for layer_weights in weights['weights']:
        print("    [")
        for neuron_weights in layer_weights:
            print(f"        {neuron_weights},")
        print("    ],")
    print("]")
    
    print("BIASES = [")
    for layer_biases in weights['biases']:
        print(f"    {layer_biases},")
    print("]")

if __name__ == "__main__":
    main() 