#!/usr/bin/env python3
from calculate_reimbursement import normalize_input, WEIGHTS, sigmoid, denormalize_output

inputs = normalize_input(3, 93, 1.42)
print(f'Normalized inputs: {inputs}')

hidden = []
for j in range(10):
    activation = WEIGHTS['b1'][j]
    for i in range(3):
        activation += inputs[i] * WEIGHTS['w1'][i][j]
    hidden.append(sigmoid(activation))
    
print(f'Hidden layer: {hidden}')

output = WEIGHTS['b2'][0]
for j in range(10):
    output += hidden[j] * WEIGHTS['w2'][j][0]
    
print(f'Raw output: {output}')
final = denormalize_output(output)
print(f'Final output: {final}')
print(f'Max(0, final): {max(0, final)}') 