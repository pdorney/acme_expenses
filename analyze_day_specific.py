import json
import numpy as np
from sklearn.linear_model import LinearRegression

def load_cases(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def fit_model(cases, days):
    # Filter cases for the given days
    filtered_cases = [case for case in cases if case['input']['trip_duration_days'] == days]
    if not filtered_cases:
        return None, None
    
    # Extract receipts and expected outputs
    receipts = np.array([case['input']['total_receipts_amount'] for case in filtered_cases]).reshape(-1, 1)
    expected = np.array([case['expected_output'] for case in filtered_cases])
    
    # Fit linear model
    model = LinearRegression()
    model.fit(receipts, expected)
    
    return model.coef_[0], model.intercept_

def fit_model_long_trips(cases):
    long_trips = [case for case in cases if 7 <= case['input']['trip_duration_days'] <= 10]
    receipts = [case['input']['total_receipts_amount'] for case in long_trips]
    expected = [case['expected_output'] for case in long_trips]
    model = LinearRegression()
    model.fit(np.array(receipts).reshape(-1, 1), expected)
    return model.coef_[0], model.intercept_

def fit_model_days_4_to_6(cases):
    days_4_to_6 = [case for case in cases if 4 <= case['input']['trip_duration_days'] <= 6]
    receipts = [case['input']['total_receipts_amount'] for case in days_4_to_6]
    expected = [case['expected_output'] for case in days_4_to_6]
    model = LinearRegression()
    model.fit(np.array(receipts).reshape(-1, 1), expected)
    return model.coef_[0], model.intercept_

def main():
    cases = load_cases('public_cases.json')
    
    for day in [1, 2, 3]:
        coef, intercept = fit_model(cases, day)
        if coef is not None:
            print(f"Days {day}: residual_receipts ≈ {coef:.4f} × receipts + {intercept:.4f}")
        else:
            print(f"Days {day}: No data available")
    
    coef, intercept = fit_model_days_4_to_6(cases)
    print(f"Days 4-6: residual_receipts ≈ {coef:.4f} × receipts + {intercept:.4f}")
    
    coef, intercept = fit_model_long_trips(cases)
    print(f"Days 7-10+: residual_receipts ≈ {coef:.4f} × receipts + {intercept:.4f}")

if __name__ == "__main__":
    main() 