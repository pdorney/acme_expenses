import json

def load_cases(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(case['input']['trip_duration_days'], 
             case['input']['miles_traveled'], 
             case['input']['total_receipts_amount'], 
             case['expected_output']) for case in data]

def analyze_long_trips(cases):
    print("=== Long Trip Analysis (7+ days) ===")
    
    long_trips = [case for case in cases if case[0] >= 7]
    long_trips.sort(key=lambda x: x[0])  # Sort by days
    
    print(f"Found {len(long_trips)} long trips")
    
    # Group by days
    by_days = {}
    for days, miles, receipts, expected in long_trips:
        if days not in by_days:
            by_days[days] = []
        by_days[days].append((miles, receipts, expected))
    
    for days in sorted(by_days.keys()):
        cases_for_days = by_days[days]
        print(f"\n{days}-day trips ({len(cases_for_days)} cases):")
        
        # Show a few examples
        examples = sorted(cases_for_days, key=lambda x: x[2])[:5]  # Sort by receipts
        for miles, receipts, expected in examples:
            per_day_reimbursement = expected / days
            miles_per_day = miles / days
            receipts_per_day = receipts / days
            
            print(f"  {miles}mi, ${receipts:.0f} → ${expected:.0f}")
            print(f"    Per day: ${per_day_reimbursement:.0f}, {miles_per_day:.0f}mi/day, ${receipts_per_day:.0f}/day")
            
            # Try to reverse engineer components
            # Assume very conservative per diem for long trips
            if days >= 10:
                estimated_per_diem = days * 50  # very conservative
            elif days >= 8:
                estimated_per_diem = days * 60  # conservative
            else:
                estimated_per_diem = days * 70  # somewhat conservative
            
            remaining = expected - estimated_per_diem
            print(f"    After conservative per diem (${estimated_per_diem}): ${remaining:.0f}")
            
            if remaining < 0:
                print(f"    → Negative remaining! Per diem estimate too high")
            else:
                if miles > 0:
                    effective_rate_per_mile = remaining / miles
                    print(f"    → Effective rate: ${effective_rate_per_mile:.3f}/mile (incl. receipts)")

def analyze_receipt_impact_long_trips(cases):
    print("\n=== Receipt Impact on Long Trips ===")
    
    long_trips = [case for case in cases if case[0] >= 7]
    
    # Find pairs with similar days/miles but different receipts
    similar_pairs = []
    for i, (days1, miles1, receipts1, expected1) in enumerate(long_trips):
        for j, (days2, miles2, receipts2, expected2) in enumerate(long_trips[i+1:], i+1):
            if (days1 == days2 and abs(miles1 - miles2) < 50 and 
                abs(receipts1 - receipts2) > 200):
                similar_pairs.append(((days1, miles1, receipts1, expected1),
                                    (days2, miles2, receipts2, expected2)))
    
    print(f"Found {len(similar_pairs)} similar pairs")
    for (case1, case2) in similar_pairs[:5]:
        days1, miles1, receipts1, expected1 = case1
        days2, miles2, receipts2, expected2 = case2
        
        receipt_diff = receipts2 - receipts1
        expected_diff = expected2 - expected1
        
        print(f"\n  {days1}d trips:")
        print(f"    Case 1: {miles1}mi, ${receipts1:.0f} → ${expected1:.0f}")
        print(f"    Case 2: {miles2}mi, ${receipts2:.0f} → ${expected2:.0f}")
        print(f"    Receipt increase: ${receipt_diff:.0f}")
        print(f"    Reimbursement change: ${expected_diff:.0f}")
        
        if receipt_diff > 0:
            receipt_impact = expected_diff / receipt_diff
            print(f"    Impact: ${receipt_impact:.3f} per dollar (negative = penalty)")

def main():
    cases = load_cases('public_cases.json')
    analyze_long_trips(cases)
    analyze_receipt_impact_long_trips(cases)

if __name__ == "__main__":
    main() 