import json
import math
import random

class ReceiptCappedNetwork:
    """Network with receipt capping features"""
    def __init__(self, input_size=8, hidden_sizes=[12, 8], learning_rate=0.006):
        self.learning_rate = learning_rate
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
                layer_output = layer_input
            else:  # Hidden layers
                layer_output = [self.sigmoid(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]
    
    def backward(self, inputs, target):
        """Simple backpropagation with clipping"""
        prediction = self.forward(inputs)
        error = target - prediction
        
        # Clip error to prevent instability
        error = max(-100, min(100, error))
        
        # Update output layer weights and biases with smaller steps
        if len(self.weights) > 0 and len(self.weights[-1]) > 0:
            for i in range(len(self.weights[-1])):  # Fixed indexing
                for j in range(len(self.weights[-1][i])):
                    if i < len(inputs):  # Safety check
                        gradient = self.learning_rate * error * inputs[i] * 0.1
                        self.weights[-1][i][j] += max(-1, min(1, gradient))
        
        if len(self.biases) > 0:
            for j in range(len(self.biases[-1])):
                gradient = self.learning_rate * error * 0.1
                self.biases[-1][j] += max(-1, min(1, gradient))
    
    def train(self, training_data, epochs=1000):
        best_error = float('inf')
        patience = 100
        no_improve = 0
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_error = 0
            
            for inputs, target in training_data:
                prediction = self.forward(inputs)
                
                # Clip prediction to prevent overflow
                prediction = max(-10000, min(10000, prediction))
                target = max(-10000, min(10000, target))
                
                error = (target - prediction) ** 2
                if error > 1000000:  # Cap error to prevent overflow
                    error = 1000000
                total_error += error
                
                # Apply penalty for over-prediction on high receipt cases
                if len(inputs) > 2 and inputs[2] > 1000 and prediction > target:
                    penalty_error = (prediction - target) * 0.5  # Smaller penalty
                    target_adjusted = target - penalty_error * 0.01  # Smaller adjustment
                    self.backward(inputs, target_adjusted)
                else:
                    self.backward(inputs, target)
            
            avg_error = total_error / len(training_data)
            
            if avg_error < best_error:
                best_error = avg_error
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch}, Error: {avg_error:.2f}")

def receipt_capped_features(days, miles, receipts):
    """Feature engineering with receipt capping"""
    # Legacy system likely caps reasonable receipts
    daily_cap = 200  # Reasonable daily expense cap
    reasonable_receipts = min(receipts, days * daily_cap)
    excess_receipts = max(0, receipts - days * daily_cap)
    
    # Basic features
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    reasonable_per_day = reasonable_receipts / days if days > 0 else 0
    
    return [
        days, miles, reasonable_receipts, excess_receipts,
        miles_per_day, reasonable_per_day, receipts_per_day,
        days * days  # Non-linear day effect
    ]

def high_receipt_features(days, miles, receipts):
    """Specialized features for high receipt cases"""
    # Focus on ratios and caps for high receipt cases
    receipt_ratio = receipts / (days * 150) if days > 0 else 1  # Ratio to reasonable amount
    excess_factor = max(1, receipt_ratio)  # How much over reasonable
    
    miles_per_day = miles / days if days > 0 else 0
    receipt_penalty = min(receipts * 0.3, days * 100)  # Penalty for excess
    
    return [
        days, miles, receipts, receipt_ratio, excess_factor,
        miles_per_day, receipt_penalty
    ]

class HybridReimbursementSystem:
    def __init__(self):
        # Main model for normal cases
        self.main_model = ReceiptCappedNetwork(input_size=8, hidden_sizes=[12, 8], learning_rate=0.005)
        
        # Specialized model for high receipt cases  
        self.high_receipt_model = ReceiptCappedNetwork(input_size=7, hidden_sizes=[10, 6], learning_rate=0.008)
        
        # Thresholds
        self.high_receipt_threshold = 1000
    
    def is_high_receipt_case(self, days, miles, receipts):
        """Determine if this is a high receipt case"""
        return receipts > self.high_receipt_threshold or receipts > days * 200
    
    def train(self, training_data):
        """Train both models on appropriate data subsets"""
        normal_cases = []
        high_receipt_cases = []
        
        # Split training data
        for days, miles, receipts, expected in training_data:
            if self.is_high_receipt_case(days, miles, receipts):
                features = high_receipt_features(days, miles, receipts)
                high_receipt_cases.append((features, expected))
            
            # All cases go to main model with capped features
            features = receipt_capped_features(days, miles, receipts)
            normal_cases.append((features, expected))
        
        print(f"Training main model on {len(normal_cases)} cases...")
        self.main_model.train(normal_cases, epochs=1200)
        
        if high_receipt_cases:
            print(f"Training high-receipt model on {len(high_receipt_cases)} cases...")
            self.high_receipt_model.train(high_receipt_cases, epochs=800)
        else:
            print("No high-receipt cases found for specialized training")
    
    def predict(self, days, miles, receipts):
        """Make prediction using appropriate model"""
        if self.is_high_receipt_case(days, miles, receipts):
            # Use specialized model for high receipt cases
            features = high_receipt_features(days, miles, receipts)
            high_pred = self.high_receipt_model.forward(features)
            
            # Also get main model prediction with capped features
            main_features = receipt_capped_features(days, miles, receipts)
            main_pred = self.main_model.forward(main_features)
            
            # Blend predictions (favor the lower one to avoid over-prediction)
            return min(high_pred, main_pred * 1.1)
        else:
            # Use main model for normal cases
            features = receipt_capped_features(days, miles, receipts)
            return self.main_model.forward(features)

def load_training_data():
    """Load training data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    training_data = []
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        training_data.append((days, miles, receipts, expected))
    
    return training_data

def main():
    print("Loading training data...")
    train_data = load_training_data()
    print(f"Total training samples: {len(train_data)}")
    
    # Analyze high-receipt cases
    high_receipt_count = sum(1 for days, miles, receipts, _ in train_data 
                           if receipts > 1000 or receipts > days * 200)
    print(f"High-receipt cases: {high_receipt_count}")
    
    # Create and train hybrid system
    hybrid_system = HybridReimbursementSystem()
    hybrid_system.train(train_data)
    
    # Test on problematic cases
    print("\n--- Testing Hybrid System ---")
    test_cases = [
        (8, 795, 1645.99, 644.69),   # Expected low, got high
        (8, 482, 1411.49, 631.81),   # Expected low, got high  
        (4, 69, 2321.49, 322.00),    # Very high receipts, expected low
        (1, 1082, 1809.49, 446.94),  # High miles + receipts
        (5, 516, 1878.49, 669.85)    # High receipts
    ]
    
    total_error = 0
    for days, miles, receipts, expected in test_cases:
        prediction = hybrid_system.predict(days, miles, receipts)
        error = abs(expected - prediction)
        total_error += error
        
        case_type = "HIGH-RECEIPT" if hybrid_system.is_high_receipt_case(days, miles, receipts) else "NORMAL"
        print(f"Days: {days}, Miles: {miles}, Receipts: ${receipts} [{case_type}]")
        print(f"  Expected: ${expected:.2f}, Predicted: ${prediction:.2f}, Error: ${error:.2f}")
    
    avg_error = total_error / len(test_cases)
    print(f"\nAverage error on problem cases: ${avg_error:.2f}")
    
    # Save the hybrid system
    model_data = {
        'main_model': {
            'weights': hybrid_system.main_model.weights,
            'biases': hybrid_system.main_model.biases,
        },
        'high_receipt_model': {
            'weights': hybrid_system.high_receipt_model.weights,
            'biases': hybrid_system.high_receipt_model.biases,
        },
        'high_receipt_threshold': hybrid_system.high_receipt_threshold
    }
    
    with open('hybrid_model.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print("Hybrid model saved to hybrid_model.json")

if __name__ == "__main__":
    main() 