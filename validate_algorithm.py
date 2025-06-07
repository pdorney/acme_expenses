import sys
sys.path.append('.')
from calculate_reimbursement import calculate_reimbursement

# Test a few specific cases manually
test_cases = [
    (3, 93, 1.42, 364.51),
    (1, 55, 3.6, 126.06),
    (1, 47, 17.97, 128.91),
    (2, 13, 4.67, 203.52),
    (5, 130, 306.9, 574.1),
    (5, 173, 1337.9, 1443.96),
    (1, 141, 10.15, 195.14)
]

print('Manual validation:')
for days, miles, receipts, expected in test_cases:
    actual = calculate_reimbursement(days, miles, receipts)
    error = abs(actual - expected)
    print(f'  {days}d, {miles}mi, ${receipts}: expected ${expected}, got ${actual}, error ${error:.2f}')

print('\nDetailed breakdown for case 1 (3d, 93mi, $1.42):')
days, miles, receipts = 3, 93, 1.42
from math import ceil, floor

per_diem = 100 * days
per_diem_offset = (ceil(days / 3) - 1) * 75
per_diem_total = per_diem - per_diem_offset
print(f'  Per diem: {per_diem} - {per_diem_offset} = {per_diem_total}')

capped_receipts = min(receipts, 1081)
print(f'  Capped receipts: {capped_receipts}')

mileage_rate = 0.33
mileage_offset = 127.6
mileage_total = max(0, mileage_rate * (miles - mileage_offset))
print(f'  Mileage: max(0, {mileage_rate} * ({miles} - {mileage_offset})) = {mileage_total}')

reimbursement = per_diem_total + mileage_total + capped_receipts
print(f'  Base reimbursement: {per_diem_total} + {mileage_total} + {capped_receipts} = {reimbursement}')

receipts_fraction = receipts - floor(receipts)
roundoff_penalty = 8
if round(receipts_fraction * 100) == 49 or round(receipts_fraction * 100) == 99:
    roundoff_penalty = 208
print(f'  Receipts fraction: {receipts_fraction}, penalty: {roundoff_penalty}')

final = round(reimbursement - roundoff_penalty, 2)
print(f'  Final: {reimbursement} - {roundoff_penalty} = {final}') 