#!/usr/bin/env python3
import json
import subprocess

# Load test cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("Testing first 10 cases:")
total_error = 0

for i in range(10):
    case = cases[i]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    # Get prediction from our model
    result = subprocess.run(['python3', 'calculate_reimbursement.py', str(days), str(miles), str(receipts)], 
                          capture_output=True, text=True)
    predicted = float(result.stdout.strip())
    error = abs(predicted - expected)
    total_error += error
    
    print(f'Case {i+1}: {days} days, {miles} miles, ${receipts} receipts')
    print(f'  Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}')
    print()

print(f"Average error on first 10 cases: ${total_error/10:.2f}") 