#!/usr/bin/env python3
import sys
import json
import math

class EnsembleReimbursementCalculator:
    def __init__(self, model_file='ensemble_model.json'):
        """Load the trained ensemble model"""
        with open(model_file, 'r') as f:
            self.ensemble_data = json.load(f)
        
        self.models = []
        for model_data in self.ensemble_data['models']:
            model = {
                'weights': model_data['weights'],
                'biases': model_data['biases'],
                'residual_weights': model_data['residual_weights'],
                'hidden_sizes': model_data['hidden_sizes'],
                'input_size': model_data['input_size']
            }
            self.models.append(model)
    
    def swish(self, x):
        """Swish activation function"""
        x = max(-500, min(500, x))
        return x / (1 + math.exp(-x))
    
    def engineer_features(self, days, miles, receipts):
        """Feature engineering to match training"""
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
    
    def forward_single_model(self, model, inputs):
        """Forward pass for a single model with residual connections"""
        current_input = inputs
        original_input = inputs
        
        for i in range(len(model['weights'])):
            # Calculate weighted sum
            layer_input = []
            for j in range(len(model['weights'][i][0])):
                weighted_sum = model['biases'][i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * model['weights'][i][k][j]
                
                # Add residual connection (skip connection from input)
                if i < len(model['residual_weights']):
                    for k in range(len(original_input)):
                        weighted_sum += original_input[k] * model['residual_weights'][i][k][j]
                
                layer_input.append(weighted_sum)
            
            # Apply activation function
            if i == len(model['weights']) - 1:  # Output layer - linear
                layer_output = layer_input
            else:  # Hidden layers - use Swish activation
                layer_output = [self.swish(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]
    
    def predict(self, days, miles, receipts):
        """Make ensemble prediction"""
        features = self.engineer_features(days, miles, receipts)
        
        predictions = []
        for model in self.models:
            pred = self.forward_single_model(model, features)
            predictions.append(pred)
        
        # Return average prediction
        return sum(predictions) / len(predictions)
    
    def predict_with_uncertainty(self, days, miles, receipts):
        """Return prediction with uncertainty estimate"""
        features = self.engineer_features(days, miles, receipts)
        
        predictions = []
        for model in self.models:
            pred = self.forward_single_model(model, features)
            predictions.append(pred)
        
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
        std_dev = math.sqrt(variance)
        
        return mean_pred, std_dev

# Global calculator instance
calculator = None

def calculate_reimbursement(days, miles, receipts):
    """Main function called by evaluation system"""
    global calculator
    
    # Initialize calculator if not done yet
    if calculator is None:
        try:
            calculator = EnsembleReimbursementCalculator('ensemble_model.json')
        except FileNotFoundError:
            # Fallback to a simple rule-based system if ensemble model not available
            print("Ensemble model not found, using fallback calculation", file=sys.stderr)
            return max(0, days * 50 + miles * 0.3 + receipts * 0.4)
    
    result = calculator.predict(days, miles, receipts)
    return max(0, result)  # Ensure non-negative

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ensemble_calculate_reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        reimbursement = calculate_reimbursement(days, miles, receipts)
        print(f"{reimbursement:.2f}")
        
    except ValueError:
        print("Error: Invalid number format")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 