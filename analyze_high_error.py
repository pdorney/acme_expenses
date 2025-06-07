import subprocess
import re
import json

def run_eval():
    result = subprocess.run(['./eval.sh'], capture_output=True, text=True)
    return result.stdout

def extract_high_error_cases(eval_output):
    # Extract high-error cases using regex
    pattern = r'Case (\d+): (\d+) days, (\d+) miles, \$(\d+\.\d+) receipts\s+Expected: \$(\d+\.\d+), Got: \$(\d+\.\d+), Error: \$(\d+\.\d+)'
    matches = re.findall(pattern, eval_output)
    cases = []
    for match in matches:
        case_num, days, miles, receipts, expected, got, error = match
        cases.append({
            'case_num': int(case_num),
            'days': float(days),
            'miles': float(miles),
            'receipts': float(receipts),
            'expected': float(expected),
            'got': float(got),
            'error': float(error)
        })
    # Sort by error (descending) and take top 20
    cases.sort(key=lambda x: x['error'], reverse=True)
    return cases[:20]

def group_cases(cases):
    # Group by trip length
    by_days = {}
    for case in cases:
        days = case['days']
        if days not in by_days:
            by_days[days] = []
        by_days[days].append(case)
    
    # Group by mileage
    by_miles = {}
    for case in cases:
        miles = case['miles']
        if miles not in by_miles:
            by_miles[miles] = []
        by_miles[miles].append(case)
    
    # Group by receipt amounts
    by_receipts = {}
    for case in cases:
        receipts = case['receipts']
        if receipts not in by_receipts:
            by_receipts[receipts] = []
        by_receipts[receipts].append(case)
    
    return by_days, by_miles, by_receipts

def print_summary(cases, by_days, by_miles, by_receipts):
    print("Top 20 High-Error Cases:")
    for case in cases:
        print(f"Case {case['case_num']}: {case['days']} days, {case['miles']} miles, ${case['receipts']} receipts")
        print(f"  Expected: ${case['expected']}, Got: ${case['got']}, Error: ${case['error']}")
    
    print("\nGrouped by Trip Length (days):")
    for days, group in by_days.items():
        print(f"{days} days: {len(group)} cases")
    
    print("\nGrouped by Mileage:")
    for miles, group in by_miles.items():
        print(f"{miles} miles: {len(group)} cases")
    
    print("\nGrouped by Receipt Amounts:")
    for receipts, group in by_receipts.items():
        print(f"${receipts}: {len(group)} cases")

if __name__ == "__main__":
    eval_output = run_eval()
    high_error_cases = extract_high_error_cases(eval_output)
    by_days, by_miles, by_receipts = group_cases(high_error_cases)
    print_summary(high_error_cases, by_days, by_miles, by_receipts) 