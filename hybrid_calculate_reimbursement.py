#!/usr/bin/env python3
import sys
import json
import math

class HybridReimbursementCalculator:
    def __init__(self, model_file='hybrid_model.json'):
        """Load the trained hybrid model"""
        try:
            with open(model_file, 'r') as f:
                self.model_data = json.load(f)
            
            self.main_model = self.model_data['main_model']
            self.high_receipt_model = self.model_data['high_receipt_model']
            self.high_receipt_threshold = self.model_data['high_receipt_threshold']
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))
    
    def forward_network(self, model, inputs):
        """Forward pass for a single network"""
        current_input = inputs
        
        for i in range(len(model['weights'])):
            layer_input = []
            for j in range(len(model['weights'][i][0])):
                weighted_sum = model['biases'][i][j]
                for k in range(len(current_input)):
                    weighted_sum += current_input[k] * model['weights'][i][k][j]
                layer_input.append(weighted_sum)
            
            if i == len(model['weights']) - 1:  # Output layer
                layer_output = layer_input
            else:  # Hidden layers
                layer_output = [self.sigmoid(x) for x in layer_input]
            
            current_input = layer_output
        
        return current_input[0]
    
    def receipt_capped_features(self, days, miles, receipts):
        """Feature engineering with receipt capping"""
        daily_cap = 200
        reasonable_receipts = min(receipts, days * daily_cap)
        excess_receipts = max(0, receipts - days * daily_cap)
        
        miles_per_day = miles / days if days > 0 else 0
        receipts_per_day = receipts / days if days > 0 else 0
        reasonable_per_day = reasonable_receipts / days if days > 0 else 0
        
        return [
            days, miles, reasonable_receipts, excess_receipts,
            miles_per_day, reasonable_per_day, receipts_per_day,
            days * days
        ]
    
    def high_receipt_features(self, days, miles, receipts):
        """Specialized features for high receipt cases"""
        receipt_ratio = receipts / (days * 150) if days > 0 else 1
        excess_factor = max(1, receipt_ratio)
        
        miles_per_day = miles / days if days > 0 else 0
        receipt_penalty = min(receipts * 0.3, days * 100)
        
        return [
            days, miles, receipts, receipt_ratio, excess_factor,
            miles_per_day, receipt_penalty
        ]
    
    def is_high_receipt_case(self, days, miles, receipts):
        """Determine if this is a high receipt case"""
        return receipts > self.high_receipt_threshold or receipts > days * 200
    
    def predict(self, days, miles, receipts):
        """Make prediction using hybrid approach"""
        if not self.model_loaded:
            # Fallback to simple calculation if model not loaded
            return max(0, days * 50 + miles * 0.3 + receipts * 0.4)
        
        if self.is_high_receipt_case(days, miles, receipts):
            # Use specialized model for high receipt cases
            features = self.high_receipt_features(days, miles, receipts)
            high_pred = self.forward_network(self.high_receipt_model, features)
            
            # Also get main model prediction with capped features
            main_features = self.receipt_capped_features(days, miles, receipts)
            main_pred = self.forward_network(self.main_model, main_features)
            
            # Blend predictions (favor the lower one to avoid over-prediction)
            return min(high_pred, main_pred * 1.1)
        else:
            # Use main model for normal cases
            features = self.receipt_capped_features(days, miles, receipts)
            return self.forward_network(self.main_model, features)

# Global calculator instance
calculator = None

def calculate_reimbursement(days, miles, receipts):
    """Main function called by evaluation system"""
    global calculator
    
    # Initialize calculator if not done yet
    if calculator is None:
        calculator = HybridReimbursementCalculator('hybrid_model.json')
    
    result = calculator.predict(days, miles, receipts)
    return max(0, result)  # Ensure non-negative

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 hybrid_calculate_reimbursement.py <days> <miles> <receipts>")
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