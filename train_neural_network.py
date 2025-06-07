import json
import math
import random

class SimpleNeuralNetwork:
    def __init__(self, input_size=3, hidden_size=10, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # Input to hidden weights (3 inputs: days, miles, receipts)
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        # Hidden bias
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        # Hidden to output weights
        self.w2 = [[random.uniform(-0.5, 0.5)] for _ in range(hidden_size)]
        # Output bias
        self.b2 = [random.uniform(-0.5, 0.5)]
    
    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow"""
        x = max(-500, min(500, x))  # Clip to prevent overflow
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, inputs):
        """Forward pass through the network"""
        # Input to hidden layer
        self.hidden_input = []
        for i in range(len(self.w1[0])):  # For each hidden neuron
            weighted_sum = self.b1[i]
            for j in range(len(inputs)):  # For each input
                weighted_sum += inputs[j] * self.w1[j][i]
            self.hidden_input.append(weighted_sum)
        
        # Apply sigmoid activation to hidden layer
        self.hidden_output = [self.sigmoid(x) for x in self.hidden_input]
        
        # Hidden to output layer
        output_input = self.b2[0]
        for i in range(len(self.hidden_output)):
            output_input += self.hidden_output[i] * self.w2[i][0]
        
        # For regression, we don't apply activation to output (linear output)
        self.output = output_input
        return self.output
    
    def backward(self, inputs, target):
        """Backward pass (backpropagation)"""
        # Calculate output layer error
        output_error = target - self.output
        
        # Calculate hidden layer errors
        hidden_errors = []
        for i in range(len(self.hidden_output)):
            error = output_error * self.w2[i][0] * self.sigmoid_derivative(self.hidden_output[i])
            hidden_errors.append(error)
        
        # Update output weights and bias
        for i in range(len(self.w2)):
            self.w2[i][0] += self.learning_rate * output_error * self.hidden_output[i]
        self.b2[0] += self.learning_rate * output_error
        
        # Update hidden weights and biases
        for i in range(len(self.w1)):  # For each input
            for j in range(len(self.w1[i])):  # For each hidden neuron
                self.w1[i][j] += self.learning_rate * hidden_errors[j] * inputs[i]
        
        for i in range(len(self.b1)):
            self.b1[i] += self.learning_rate * hidden_errors[i]
    
    def train(self, training_data, epochs=1000):
        """Train the network on the provided data"""
        for epoch in range(epochs):
            total_error = 0
            random.shuffle(training_data)  # Shuffle data each epoch
            
            for inputs, target in training_data:
                prediction = self.forward(inputs)
                self.backward(inputs, target)
                error = (target - prediction) ** 2
                total_error += error
            
            avg_error = total_error / len(training_data)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Average Error: {avg_error:.2f}")
    
    def get_weights(self):
        """Return weights and biases for hardcoding"""
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

def load_training_data():
    """Load and preprocess training data from public_cases.json"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    training_data = []
    
    # Calculate normalization parameters
    days_values = [case['input']['trip_duration_days'] for case in data]
    miles_values = [case['input']['miles_traveled'] for case in data]
    receipts_values = [case['input']['total_receipts_amount'] for case in data]
    outputs = [case['expected_output'] for case in data]
    
    # Simple normalization (min-max scaling)
    days_min, days_max = min(days_values), max(days_values)
    miles_min, miles_max = min(miles_values), max(miles_values)
    receipts_min, receipts_max = min(receipts_values), max(receipts_values)
    output_min, output_max = min(outputs), max(outputs)
    
    print(f"Data ranges:")
    print(f"  Days: {days_min} to {days_max}")
    print(f"  Miles: {miles_min} to {miles_max}")
    print(f"  Receipts: {receipts_min} to {receipts_max}")
    print(f"  Output: {output_min} to {output_max}")
    
    for case in data:
        # Normalize inputs to [0, 1]
        days_norm = (case['input']['trip_duration_days'] - days_min) / (days_max - days_min)
        miles_norm = (case['input']['miles_traveled'] - miles_min) / (miles_max - miles_min)
        receipts_norm = (case['input']['total_receipts_amount'] - receipts_min) / (receipts_max - receipts_min)
        
        # Normalize output to [0, 1]
        output_norm = (case['expected_output'] - output_min) / (output_max - output_min)
        
        inputs = [days_norm, miles_norm, receipts_norm]
        training_data.append((inputs, output_norm))
    
    normalization_params = {
        'days_min': days_min, 'days_max': days_max,
        'miles_min': miles_min, 'miles_max': miles_max,
        'receipts_min': receipts_min, 'receipts_max': receipts_max,
        'output_min': output_min, 'output_max': output_max
    }
    
    return training_data, normalization_params

def main():
    print("Loading training data...")
    training_data, norm_params = load_training_data()
    print(f"Loaded {len(training_data)} training samples")
    
    print("\nInitializing neural network...")
    nn = SimpleNeuralNetwork(input_size=3, hidden_size=10, output_size=1, learning_rate=0.1)
    
    print("\nTraining neural network...")
    nn.train(training_data, epochs=2000)
    
    print("\nTesting on first few samples:")
    for i in range(5):
        inputs, target = training_data[i]
        prediction = nn.forward(inputs)
        
        # Denormalize for display
        days = inputs[0] * (norm_params['days_max'] - norm_params['days_min']) + norm_params['days_min']
        miles = inputs[1] * (norm_params['miles_max'] - norm_params['miles_min']) + norm_params['miles_min']
        receipts = inputs[2] * (norm_params['receipts_max'] - norm_params['receipts_min']) + norm_params['receipts_min']
        target_denorm = target * (norm_params['output_max'] - norm_params['output_min']) + norm_params['output_min']
        pred_denorm = prediction * (norm_params['output_max'] - norm_params['output_min']) + norm_params['output_min']
        
        print(f"  {days:.0f}d, {miles:.0f}mi, ${receipts:.2f} -> Expected: ${target_denorm:.2f}, Predicted: ${pred_denorm:.2f}")
    
    print("\nFinal weights and normalization parameters:")
    weights = nn.get_weights()
    
    print("\n# Copy these into your inference script:")
    print("NORMALIZATION_PARAMS = {")
    for key, value in norm_params.items():
        print(f"    '{key}': {value},")
    print("}")
    
    print("\nWEIGHTS = {")
    print("    'w1': [")
    for row in weights['w1']:
        print(f"        {row},")
    print("    ],")
    print(f"    'b1': {weights['b1']},")
    print("    'w2': [")
    for row in weights['w2']:
        print(f"        {row},")
    print("    ],")
    print(f"    'b2': {weights['b2']}")
    print("}")

if __name__ == "__main__":
    main() 