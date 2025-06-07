import json

def load_cases(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(case['input']['trip_duration_days'], 
             case['input']['miles_traveled'], 
             case['input']['total_receipts_amount'], 
             case['expected_output']) for case in data]

def analyze_per_diem_patterns(cases):
    print("=== Per Diem Analysis ===")
    
    # Group by days and look for patterns
    by_days = {}
    for days, miles, receipts, expected in cases:
        if days not in by_days:
            by_days[days] = []
        by_days[days].append((miles, receipts, expected))
    
    for days in sorted(by_days.keys())[:6]:  # Look at 1-6 day trips
        cases_for_days = by_days[days]
        print(f"\n{days}-day trips:")
        
        # Find cases with minimal mileage and receipts to isolate per diem
        min_cases = sorted(cases_for_days, key=lambda x: x[0] + x[1])[:3]
        for miles, receipts, expected in min_cases:
            # Estimate mileage component (assume 0.58/mile for first 100, then less)
            if miles <= 100:
                mileage_est = miles * 0.58
            else:
                mileage_est = 100 * 0.58 + (miles - 100) * 0.3  # rough estimate
            
            # Estimate receipt component (assume small multiplier for low receipts)
            receipt_est = receipts * 0.5  # rough guess
            
            # Calculate implied per diem
            implied_per_diem = expected - mileage_est - receipt_est
            
            print(f"  {miles}mi, ${receipts:.2f} → ${expected:.2f}")
            print(f"    Est mileage: ${mileage_est:.2f}, receipts: ${receipt_est:.2f}")
            print(f"    Implied per diem: ${implied_per_diem:.2f} (vs {days * 100} base)")

def analyze_mileage_patterns(cases):
    print("\n=== Mileage Analysis ===")
    
    # Look at cases with minimal receipts to isolate mileage effect
    low_receipt_cases = [case for case in cases if case[2] < 50]  # < $50 receipts
    low_receipt_cases.sort(key=lambda x: x[1])  # Sort by miles
    
    print("Low-receipt cases (to isolate mileage):")
    for days, miles, receipts, expected in low_receipt_cases[:10]:
        # Estimate per diem (assume $100/day minus some offset)
        if days <= 3:
            per_diem_est = days * 100
        elif days <= 6:
            per_diem_est = days * 100 - 75  # rough guess based on interviews
        else:
            per_diem_est = days * 100 - 150  # rough guess
        
        # Calculate implied mileage + receipt value
        remaining = expected - per_diem_est
        per_mile_rate = remaining / miles if miles > 0 else 0
        
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → ${expected:.2f}")
        print(f"    After per diem (${per_diem_est}): ${remaining:.2f}")
        print(f"    Implied rate: ${per_mile_rate:.3f}/mile")

def analyze_receipt_patterns(cases):
    print("\n=== Receipt Analysis ===")
    
    # Look at cases with similar days/miles but different receipts
    similar_cases = []
    for i, (days1, miles1, receipts1, expected1) in enumerate(cases):
        for j, (days2, miles2, receipts2, expected2) in enumerate(cases[i+1:], i+1):
            # Find cases with similar days/miles but different receipts
            if (days1 == days2 and abs(miles1 - miles2) < 20 and 
                abs(receipts1 - receipts2) > 50):
                similar_cases.append(((days1, miles1, receipts1, expected1),
                                    (days2, miles2, receipts2, expected2)))
    
    print("Cases with similar days/miles, different receipts:")
    for (case1, case2) in similar_cases[:5]:
        days1, miles1, receipts1, expected1 = case1
        days2, miles2, receipts2, expected2 = case2
        
        receipt_diff = receipts2 - receipts1
        expected_diff = expected2 - expected1
        receipt_impact = expected_diff / receipt_diff if receipt_diff != 0 else 0
        
        print(f"  Case 1: {days1}d, {miles1}mi, ${receipts1:.2f} → ${expected1:.2f}")
        print(f"  Case 2: {days2}d, {miles2}mi, ${receipts2:.2f} → ${expected2:.2f}")
        print(f"  Receipt impact: ${receipt_impact:.3f} per dollar")

def main():
    cases = load_cases('public_cases.json')
    print(f"Loaded {len(cases)} cases")
    
    analyze_per_diem_patterns(cases)
    analyze_mileage_patterns(cases)
    analyze_receipt_patterns(cases)

if __name__ == "__main__":
    main() 