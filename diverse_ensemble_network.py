import json
import math
import random

class SimpleNeuralNetwork:
    """Basic neural network without residual connections"""
    def __init__(self, input_size=7, hidden_sizes=[10], learning_rate=0.01):
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                 for _ in range(layer_sizes[i])]
            b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))
    
    def tanh(self, x):
        x = max(-500, min(500, x))
        return math.tanh(x)
    
    def relu(self, x):
        return max(0, x)
    
    def forward(self, inputs):
        current_input = inputs
        
        for i in range(len(self.weights)):
            layer_input = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = self.biases[i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * self.weights[i][k][j]
                layer_input.append(weighted_sum)
            
            if i == len(self.weights) - 1:  # Output layer
                layer_output = layer_input  # Linear
            else:  # Hidden layers
                layer_output = [self.sigmoid(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]

class DeepNeuralNetwork:
    """Deeper network with different activation"""
    def __init__(self, input_size=7, hidden_sizes=[20, 15, 10, 5], learning_rate=0.005):
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                 for _ in range(layer_sizes[i])]
            b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(w)
            self.biases.append(b)
    
    def leaky_relu(self, x, alpha=0.1):
        return x if x > 0 else alpha * x
    
    def forward(self, inputs):
        current_input = inputs
        
        for i in range(len(self.weights)):
            layer_input = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = self.biases[i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * self.weights[i][k][j]
                layer_input.append(weighted_sum)
            
            if i == len(self.weights) - 1:  # Output layer
                layer_output = layer_input  # Linear
            else:  # Hidden layers
                layer_output = [self.leaky_relu(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]

class PolynomialNetwork:
    """Network that uses polynomial features"""
    def __init__(self, input_size=15, hidden_sizes=[12, 8], learning_rate=0.008):
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                 for _ in range(layer_sizes[i])]
            b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(w)
            self.biases.append(b)
    
    def swish(self, x):
        x = max(-500, min(500, x))
        return x / (1 + math.exp(-x))
    
    def forward(self, inputs):
        current_input = inputs
        
        for i in range(len(self.weights)):
            layer_input = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = self.biases[i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * self.weights[i][k][j]
                layer_input.append(weighted_sum)
            
            if i == len(self.weights) - 1:  # Output layer
                layer_output = layer_input  # Linear
            else:  # Hidden layers
                layer_output = [self.swish(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]

def basic_features(days, miles, receipts):
    """Basic feature engineering"""
    return [days, miles, receipts, 
            miles/days if days > 0 else 0,
            receipts/days if days > 0 else 0,
            miles/receipts if receipts > 0 else 0,
            days * days]

def polynomial_features(days, miles, receipts):
    """Extended polynomial features"""
    base = [days, miles, receipts]
    
    # Linear combinations
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    miles_per_receipt = miles / receipts if receipts > 0 else 0
    
    # Quadratic terms
    days_sq = days * days
    miles_sq = miles * miles
    receipts_sq = receipts * receipts
    
    # Interaction terms
    days_miles = days * miles
    days_receipts = days * receipts
    miles_receipts = miles * receipts
    
    # Higher order
    days_cubed = days * days * days
    total_activity = days * miles * receipts
    
    return [days, miles, receipts, miles_per_day, receipts_per_day, miles_per_receipt,
            days_sq, miles_sq, receipts_sq, days_miles, days_receipts, miles_receipts,
            days_cubed, total_activity, math.log(1 + receipts)]

def log_features(days, miles, receipts):
    """Logarithmic transformations"""
    return [days, miles, receipts,
            math.log(1 + days), math.log(1 + miles), math.log(1 + receipts),
            days * miles, days * receipts, miles * receipts,
            math.sqrt(days), math.sqrt(miles), math.sqrt(receipts)]

class DiverseEnsemble:
    def __init__(self):
        # Different models with different architectures and feature engineering
        self.models = [
            {
                'net': SimpleNeuralNetwork(input_size=7, hidden_sizes=[12], learning_rate=0.01),
                'feature_func': basic_features,
                'name': 'Simple-Basic'
            },
            {
                'net': SimpleNeuralNetwork(input_size=7, hidden_sizes=[15, 8], learning_rate=0.008),
                'feature_func': basic_features,
                'name': 'Simple-Deep'
            },
            {
                'net': DeepNeuralNetwork(input_size=7, hidden_sizes=[18, 12, 6], learning_rate=0.006),
                'feature_func': basic_features,
                'name': 'Deep-Basic'
            },
            {
                'net': PolynomialNetwork(input_size=15, hidden_sizes=[20, 12], learning_rate=0.004),
                'feature_func': polynomial_features,
                'name': 'Poly-Extended'
            },
            {
                'net': SimpleNeuralNetwork(input_size=12, hidden_sizes=[16, 10], learning_rate=0.007),
                'feature_func': log_features,
                'name': 'Simple-Log'
            }
        ]
    
    def train_single_model(self, model_info, train_data, val_data, epochs=1500):
        """Train a single model with its specific feature engineering"""
        net = model_info['net']
        feature_func = model_info['feature_func']
        
        # Convert data to this model's feature space
        model_train_data = []
        for case in train_data:
            days, miles, receipts, expected = case
            features = feature_func(days, miles, receipts)
            model_train_data.append((features, expected))
        
        model_val_data = []
        for case in val_data:
            days, miles, receipts, expected = case
            features = feature_func(days, miles, receipts)
            model_val_data.append((features, expected))
        
        # Simple training loop
        best_val_error = float('inf')
        patience = 100
        no_improve = 0
        
        for epoch in range(epochs):
            # Training
            random.shuffle(model_train_data)
            total_error = 0
            
            for features, target in model_train_data:
                prediction = net.forward(features)
                error = target - prediction
                total_error += error * error
                
                # Simple backprop (gradient descent)
                self.simple_backprop(net, features, target, prediction)
            
            # Validation
            val_error = 0
            for features, target in model_val_data:
                prediction = net.forward(features)
                val_error += (target - prediction) ** 2
            
            avg_val_error = val_error / len(model_val_data)
            
            if avg_val_error < best_val_error:
                best_val_error = avg_val_error
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch}, Val Error: {avg_val_error:.2f}")
    
    def simple_backprop(self, net, inputs, target, prediction):
        """Simplified backpropagation"""
        error = target - prediction
        
        # Update output layer
        for i in range(len(inputs)):
            for j in range(len(net.weights[-1][i])):
                net.weights[-1][i][j] += net.learning_rate * error * inputs[i] * 0.1
        
        # Update biases
        for j in range(len(net.biases[-1])):
            net.biases[-1][j] += net.learning_rate * error * 0.1
    
    def train(self, train_data, val_data):
        """Train all models in the ensemble"""
        print(f"Training diverse ensemble of {len(self.models)} models...")
        
        for i, model_info in enumerate(self.models):
            print(f"\n--- Training {model_info['name']} ---")
            random.seed(42 + i * 50)  # Different seeds for diversity
            self.train_single_model(model_info, train_data, val_data)
    
    def predict(self, days, miles, receipts):
        """Make ensemble prediction"""
        predictions = []
        
        for model_info in self.models:
            features = model_info['feature_func'](days, miles, receipts)
            pred = model_info['net'].forward(features)
            predictions.append(pred)
        
        # Weighted average (give more weight to polynomial model)
        weights = [1.0, 1.0, 1.2, 1.5, 1.0]  # Boost polynomial model
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight

def load_training_data():
    """Load training data in simple format"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    training_data = []
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        training_data.append((days, miles, receipts, expected))
    
    # Split
    random.shuffle(training_data)
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    return train_data, val_data

def main():
    print("Loading training data...")
    train_data, val_data = load_training_data()
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create and train diverse ensemble
    ensemble = DiverseEnsemble()
    ensemble.train(train_data, val_data)
    
    # Test on high-error cases
    print("\n--- Testing Diverse Ensemble ---")
    test_cases = [
        (8, 795, 1645.99, 644.69),
        (8, 482, 1411.49, 631.81),
        (4, 69, 2321.49, 322.00),
        (1, 1082, 1809.49, 446.94),
        (5, 516, 1878.49, 669.85)
    ]
    
    total_error = 0
    for days, miles, receipts, expected in test_cases:
        prediction = ensemble.predict(days, miles, receipts)
        error = abs(expected - prediction)
        total_error += error
        
        print(f"Days: {days}, Miles: {miles}, Receipts: ${receipts}")
        print(f"  Expected: ${expected:.2f}, Predicted: ${prediction:.2f}, Error: ${error:.2f}")
    
    avg_error = total_error / len(test_cases)
    print(f"\nAverage error on high-error cases: ${avg_error:.2f}")
    
    # Save ensemble for production
    ensemble_data = {
        'models': []
    }
    
    for model_info in ensemble.models:
        model_data = {
            'weights': model_info['net'].weights,
            'biases': model_info['net'].biases,
            'name': model_info['name'],
            'feature_type': model_info['name'].split('-')[1]  # Basic, Extended, Log
        }
        ensemble_data['models'].append(model_data)
    
    with open('diverse_ensemble_model.json', 'w') as f:
        json.dump(ensemble_data, f, indent=2)
    
    print("Diverse ensemble saved to diverse_ensemble_model.json")

if __name__ == "__main__":
    main() 