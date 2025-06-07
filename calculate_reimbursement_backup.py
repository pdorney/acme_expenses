#!/usr/bin/env python3
import sys
import math

# Normalization parameters (from our successful training)
NORMALIZATION_PARAMS = {
    'days_min': 1,
    'days_max': 14,
    'miles_min': 5,
    'miles_max': 1317.07,
    'receipts_min': 1.42,
    'receipts_max': 2503.46,
    'output_min': 117.24,
    'output_max': 2337.73,
}

WEIGHTS = {
    'w1': [
        [4.577030196574988, 0.0765762872493861, -0.08916371542350271, -1.2446091459518527, -0.5327194017629768, -0.1150872005190735, -0.7122456442721444, 2.1482702334049857, -0.2725030641025548, 0.33253585660690077],
        [-2.1017955955102137, -0.7852870390330338, 0.10758736845235793, -0.3211841600694197, -0.6815921566062082, -0.6447521565952634, -0.9362107811606476, -0.5756903613408794, -0.6081145127922404, -0.2715359374783945],
        [0.6029548628415357, -1.4292990212333103, -3.0381917770111477, -0.598049197030554, -0.8709904265811883, -8.74981415955423, -0.9255543133811259, -0.020489436522677446, 0.028557350295421082, -1.6982576662265834],
    ],
    'b1': [-1.990642451328135, -1.2913564460960676, -0.7051609337304152, -1.0213086978988015, -1.1528490305799372, 2.733034599576669, -1.000537685284205, -0.8203914268378837, -1.768076620165756, -1.4512108657061042],
    'w2': [
        [-0.6508916983665907],
        [0.10507429356093365],
        [1.544971884059025],
        [-0.9789546857926139],
        [-0.4852940973758344],
        [-0.8803766616395182],
        [-0.6119389639229874],
        [1.2102221988407376],
        [-0.21491274033881214],
        [0.4936442984827437],
    ],
    'b2': [0.48279882038730826]
}

def sigmoid(x):
    """Sigmoid activation function with clipping to prevent overflow"""
    x = max(-500, min(500, x))  # Clip to prevent overflow
    return 1 / (1 + math.exp(-x))

def normalize_input(days, miles, receipts):
    norm_days = (days - NORMALIZATION_PARAMS['days_min']) / (NORMALIZATION_PARAMS['days_max'] - NORMALIZATION_PARAMS['days_min'])
    norm_miles = (miles - NORMALIZATION_PARAMS['miles_min']) / (NORMALIZATION_PARAMS['miles_max'] - NORMALIZATION_PARAMS['miles_min'])
    norm_receipts = (receipts - NORMALIZATION_PARAMS['receipts_min']) / (NORMALIZATION_PARAMS['receipts_max'] - NORMALIZATION_PARAMS['receipts_min'])
    
    # Clamp to [0, 1]
    norm_days = max(0, min(1, norm_days))
    norm_miles = max(0, min(1, norm_miles))
    norm_receipts = max(0, min(1, norm_receipts))
    
    return [norm_days, norm_miles, norm_receipts]

def denormalize_output(normalized_output):
    return normalized_output * (NORMALIZATION_PARAMS['output_max'] - NORMALIZATION_PARAMS['output_min']) + NORMALIZATION_PARAMS['output_min']

def predict(days, miles, receipts):
    # Normalize inputs
    inputs = normalize_input(days, miles, receipts)
    
    # First layer (input -> hidden)
    hidden = []
    for j in range(10):
        activation = WEIGHTS['b1'][j]
        for i in range(3):
            activation += inputs[i] * WEIGHTS['w1'][i][j]
        hidden.append(sigmoid(activation))  # Use sigmoid like in training
    
    # Second layer (hidden -> output)
    output = WEIGHTS['b2'][0]
    for j in range(10):
        output += hidden[j] * WEIGHTS['w2'][j][0]
    
    # Denormalize output
    return denormalize_output(output)

def calculate_reimbursement(days, miles, receipts):
    result = predict(days, miles, receipts)
    return max(0, result)  # Ensure non-negative

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
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