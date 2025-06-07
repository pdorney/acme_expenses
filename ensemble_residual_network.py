import json
import math
import random

class ResidualNeuralNetwork:
    def __init__(self, input_size=7, hidden_sizes=[15, 10, 8], output_size=1, learning_rate=0.005, dropout_rate=0.15):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        
        # Initialize weights for multiple hidden layers with residual connections
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.residual_weights = []  # For skip connections
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                 for _ in range(layer_sizes[i])]
            b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Residual connection weights (from input to each hidden layer)
            if i < len(layer_sizes) - 2:  # Not for output layer
                res_limit = math.sqrt(6 / (input_size + layer_sizes[i + 1]))
                res_w = [[random.uniform(-res_limit, res_limit) for _ in range(layer_sizes[i + 1])] 
                         for _ in range(input_size)]
                self.residual_weights.append(res_w)
    
    def leaky_relu(self, x, alpha=0.02):
        return x if x > 0 else alpha * x
    
    def leaky_relu_derivative(self, x, alpha=0.02):
        return 1 if x > 0 else alpha
    
    def swish(self, x):
        """Swish activation function (x * sigmoid(x))"""
        x = max(-500, min(500, x))
        return x / (1 + math.exp(-x))
    
    def swish_derivative(self, x):
        """Derivative of Swish"""
        x = max(-500, min(500, x))
        sigmoid_x = 1 / (1 + math.exp(-x))
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    
    def forward(self, inputs, training=False):
        """Forward pass with residual connections"""
        self.layer_outputs = [inputs]
        self.layer_inputs = []  # Store pre-activation values
        current_input = inputs
        original_input = inputs
        
        for i in range(len(self.weights)):
            # Calculate weighted sum
            layer_input = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = self.biases[i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * self.weights[i][k][j]
                
                # Add residual connection (skip connection from input)
                if i < len(self.residual_weights):
                    for k in range(len(original_input)):
                        weighted_sum += original_input[k] * self.residual_weights[i][k][j]
                
                layer_input.append(weighted_sum)
            
            self.layer_inputs.append(layer_input)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer - linear
                layer_output = layer_input
            else:  # Hidden layers - use Swish activation
                layer_output = [self.swish(x) for x in layer_input]
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    for k in range(len(layer_output)):
                        if random.random() < self.dropout_rate:
                            layer_output[k] = 0
                        else:
                            layer_output[k] /= (1 - self.dropout_rate)
            
            self.layer_outputs.append(layer_output)
            current_input = layer_output
        
        return current_input[0]
    
    def backward(self, inputs, target):
        """Backward pass with residual connections"""
        output = self.layer_outputs[-1][0]
        output_error = target - output
        
        # Collect all layer errors
        all_errors = [[output_error]]
        
        # Calculate errors for hidden layers (backwards)
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            current_layer_size = len(self.layer_outputs[layer_idx + 1])
            next_layer_errors = all_errors[0]
            current_layer_errors = []
            
            for neuron_idx in range(current_layer_size):
                error = 0
                # Error from next layer
                for next_neuron_idx in range(len(next_layer_errors)):
                    error += next_layer_errors[next_neuron_idx] * self.weights[layer_idx + 1][neuron_idx][next_neuron_idx]
                
                # Apply derivative of activation function
                pre_activation = self.layer_inputs[layer_idx][neuron_idx]
                error *= self.swish_derivative(pre_activation)
                current_layer_errors.append(error)
            
            all_errors.insert(0, current_layer_errors)
        
        # Update weights and biases
        for layer_idx in range(len(self.weights)):
            layer_errors = all_errors[layer_idx]
            layer_inputs = self.layer_outputs[layer_idx]
            
            # Update main weights
            for input_idx in range(len(layer_inputs)):
                for neuron_idx in range(len(layer_errors)):
                    self.weights[layer_idx][input_idx][neuron_idx] += self.learning_rate * layer_errors[neuron_idx] * layer_inputs[input_idx]
            
            # Update residual weights
            if layer_idx < len(self.residual_weights):
                original_input = self.layer_outputs[0]
                for input_idx in range(len(original_input)):
                    for neuron_idx in range(len(layer_errors)):
                        self.residual_weights[layer_idx][input_idx][neuron_idx] += self.learning_rate * layer_errors[neuron_idx] * original_input[input_idx]
            
            # Update biases
            for neuron_idx in range(len(layer_errors)):
                self.biases[layer_idx][neuron_idx] += self.learning_rate * layer_errors[neuron_idx]
    
    def train(self, training_data, validation_data, epochs=2000):
        """Train with validation monitoring and adaptive learning rate"""
        best_val_error = float('inf')
        patience = 150
        no_improve_count = 0
        initial_lr = self.learning_rate
        
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
            
            # Adaptive learning rate
            if avg_val_error < best_val_error:
                best_val_error = avg_val_error
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                # Reduce learning rate if no improvement
                if no_improve_count % 50 == 0:
                    self.learning_rate *= 0.8
                    print(f"Reduced learning rate to {self.learning_rate:.6f}")
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Train Error: {avg_train_error:.4f}, Val Error: {avg_val_error:.4f}, LR: {self.learning_rate:.6f}")
        
        # Reset learning rate
        self.learning_rate = initial_lr

class EnsembleNeuralNetwork:
    def __init__(self, num_models=5):
        self.models = []
        self.num_models = num_models
        
        # Create diverse architectures
        architectures = [
            {'hidden_sizes': [12, 8], 'learning_rate': 0.005, 'dropout_rate': 0.1},
            {'hidden_sizes': [16, 12, 6], 'learning_rate': 0.007, 'dropout_rate': 0.15},
            {'hidden_sizes': [20, 10], 'learning_rate': 0.004, 'dropout_rate': 0.2},
            {'hidden_sizes': [14, 10, 8, 4], 'learning_rate': 0.006, 'dropout_rate': 0.12},
            {'hidden_sizes': [18, 14, 8], 'learning_rate': 0.008, 'dropout_rate': 0.18}
        ]
        
        for i in range(num_models):
            arch = architectures[i % len(architectures)]
            model = ResidualNeuralNetwork(
                input_size=7,
                hidden_sizes=arch['hidden_sizes'],
                learning_rate=arch['learning_rate'],
                dropout_rate=arch['dropout_rate']
            )
            self.models.append(model)
    
    def train(self, training_data, validation_data, epochs=2000):
        """Train all models in the ensemble"""
        print(f"Training ensemble of {self.num_models} models...")
        
        for i, model in enumerate(self.models):
            print(f"\n--- Training Model {i+1}/{self.num_models} ---")
            
            # Use different random seeds for diversity
            random.seed(42 + i * 100)
            
            # Train with slightly different data order for each model
            shuffled_training = training_data.copy()
            random.shuffle(shuffled_training)
            
            model.train(shuffled_training, validation_data, epochs)
    
    def predict(self, inputs):
        """Make prediction using ensemble averaging"""
        predictions = []
        for model in self.models:
            pred = model.forward(inputs, training=False)
            predictions.append(pred)
        
        # Return average prediction
        return sum(predictions) / len(predictions)
    
    def predict_with_uncertainty(self, inputs):
        """Return prediction with uncertainty estimate"""
        predictions = []
        for model in self.models:
            pred = model.forward(inputs, training=False)
            predictions.append(pred)
        
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
        std_dev = math.sqrt(variance)
        
        return mean_pred, std_dev

def engineer_features(days, miles, receipts):
    """Enhanced feature engineering"""
    features = [days, miles, receipts]
    
    # Derived features
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    miles_per_receipt = miles / receipts if receipts > 0 else 0
    
    # Non-linear features
    features.extend([
        miles_per_day,
        receipts_per_day,
        miles_per_receipt,
        days * days  # Quadratic day effect
    ])
    
    return features

def load_training_data():
    """Load and preprocess training data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    training_data = []
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        features = engineer_features(days, miles, receipts)
        training_data.append((features, expected))
    
    # Split into train and validation
    random.shuffle(training_data)
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    return train_data, val_data

def main():
    print("Loading training data...")
    train_data, val_data = load_training_data()
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create and train ensemble
    ensemble = EnsembleNeuralNetwork(num_models=5)
    ensemble.train(train_data, val_data, epochs=2500)
    
    # Test on a few examples
    print("\n--- Testing Ensemble ---")
    test_cases = [
        (8, 795, 1645.99, 644.69),  # High error case
        (8, 482, 1411.49, 631.81),  # High error case
        (4, 69, 2321.49, 322.00),   # High error case
        (1, 1082, 1809.49, 446.94), # High error case
        (5, 516, 1878.49, 669.85)   # High error case
    ]
    
    total_error = 0
    for days, miles, receipts, expected in test_cases:
        features = engineer_features(days, miles, receipts)
        prediction, uncertainty = ensemble.predict_with_uncertainty(features)
        error = abs(expected - prediction)
        total_error += error
        
        print(f"Days: {days}, Miles: {miles}, Receipts: ${receipts}")
        print(f"  Expected: ${expected:.2f}")
        print(f"  Predicted: ${prediction:.2f} ± ${uncertainty:.2f}")
        print(f"  Error: ${error:.2f}")
        print()
    
    avg_error = total_error / len(test_cases)
    print(f"Average error on high-error cases: ${avg_error:.2f}")
    
    # Save the best model weights for production use
    print("\nSaving ensemble model...")
    ensemble_data = {
        'models': []
    }
    
    for i, model in enumerate(ensemble.models):
        model_data = {
            'weights': model.weights,
            'biases': model.biases,
            'residual_weights': model.residual_weights,
            'hidden_sizes': model.hidden_sizes,
            'input_size': model.input_size
        }
        ensemble_data['models'].append(model_data)
    
    with open('ensemble_model.json', 'w') as f:
        json.dump(ensemble_data, f, indent=2)
    
    print("Ensemble model saved to ensemble_model.json")

if __name__ == "__main__":
    main() 