import json

def load_cases(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_high_mileage(cases):
    high_mileage = [case for case in cases if case['input']['miles_traveled'] >= 1000]
    
    print(f"\nFound {len(high_mileage)} cases with 1000+ miles")
    
    # Group by trip duration
    by_duration = {}
    for case in high_mileage:
        days = case['input']['trip_duration_days']
        if days not in by_duration:
            by_duration[days] = []
        by_duration[days].append(case)
    
    print("\nAnalysis by trip duration:")
    for days in sorted(by_duration.keys()):
        cases = by_duration[days]
        avg_output = sum(c['expected_output'] for c in cases) / len(cases)
        avg_miles = sum(c['input']['miles_traveled'] for c in cases) / len(cases)
        avg_receipts = sum(c['input']['total_receipts_amount'] for c in cases) / len(cases)
        
        print(f"\n{days} days ({len(cases)} cases):")
        print(f"  Avg reimbursement: ${avg_output:.2f}")
        print(f"  Avg miles: {avg_miles:.1f}")
        print(f"  Avg receipts: ${avg_receipts:.2f}")
        
        # Show a sample case
        sample = cases[0]
        print(f"\n  Sample case:")
        print(f"    Miles: {sample['input']['miles_traveled']}")
        print(f"    Receipts: ${sample['input']['total_receipts_amount']}")
        print(f"    Expected: ${sample['expected_output']}")

if __name__ == "__main__":
    cases = load_cases('public_cases.json')
    analyze_high_mileage(cases) 