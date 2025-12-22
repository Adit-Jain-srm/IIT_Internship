import pandas as pd
from pathlib import Path
import numpy as np

base_path = Path('temperatures_range')
temp_ranges = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-85']

print("="*80)
print("ANALYZING DIMENSIONS FOR BALANCED TABLE CREATION")
print("="*80)
print("\nEach range has 10 different readings/files")
print("Goal: Create tables with uniform dimensions\n")

all_data = {}

for temp_range in temp_ranges:
    temp_folder = base_path / temp_range
    csv_files = sorted(list(temp_folder.glob('*.csv')))
    
    print(f"\n{temp_range}°C - 10 Files Analysis:")
    print("-" * 80)
    
    file_rows = []
    for i, csv_file in enumerate(csv_files, 1):
        df = pd.read_csv(csv_file)
        rows = len(df)
        file_rows.append(rows)
        print(f"  File {i:2d} ({csv_file.name}): {rows:,} rows")
    
    min_rows = min(file_rows)
    max_rows = max(file_rows)
    avg_rows = np.mean(file_rows)
    
    print(f"\n  Min: {min_rows:,} | Max: {max_rows:,} | Avg: {avg_rows:,.1f}")
    print(f"  Range: {max_rows - min_rows} rows")
    
    all_data[temp_range] = {
        'file_rows': file_rows,
        'min': min_rows,
        'max': max_rows,
        'avg': avg_rows,
        'files': len(csv_files)
    }

print("\n" + "="*80)
print("GLOBAL STATISTICS")
print("="*80)

all_minimums = [all_data[tr]['min'] for tr in temp_ranges]
all_maximums = [all_data[tr]['max'] for tr in temp_ranges]

global_min = min(all_minimums)
global_max = max(all_maximums)

print(f"\nAcross ALL ranges:")
print(f"  Global minimum rows per file: {global_min:,}")
print(f"  Global maximum rows per file: {global_max:,}")
print(f"  Range: {global_max - global_min} rows")

print(f"\n" + "="*80)
print("FILTERING STRATEGIES FOR BALANCED DIMENSIONS")
print("="*80)

print(f"\nSTRATEGY: Uniform Rows Per File (RECOMMENDED)")
print(f"  Filter each file to: {global_min:,} rows")
print(f"  Result per range: {global_min:,} rows × 10 files = {global_min * 10:,} rows")
print(f"  Grand total: {global_min * 10 * 6:,} rows")

strategy_rows = global_min * 10
strategy_total = strategy_rows * 6

print(f"\n  Advantages:")
print(f"    ✓ Each file has identical dimensions: (1, {global_min})")
print(f"    ✓ Each range has identical total rows: ({global_min * 10:,})")
print(f"    ✓ Perfect balance across all 6 ranges")
print(f"    ✓ All 10 readings equally represented")
print(f"    ✓ Best for 3D tensor: (6 ranges, 10 readings, {global_min} rows)")

print(f"\n  Data retention by range:")
for temp_range in temp_ranges:
    total_original = sum(all_data[temp_range]['file_rows'])
    retained = strategy_rows
    retention_pct = (retained / total_original) * 100
    print(f"    {temp_range}°C: {retained:,} / {total_original:,} ({retention_pct:.2f}%)")

print(f"\n" + "="*80)
print(f"FINAL RECOMMENDATION: Use {global_min:,} rows per file")
print(f"  • Final Dataset Shape: (6 ranges, 10 readings, {global_min} rows)")
print(f"  • Perfect dimensions for structured analysis")
print(f"  • Each range equally represented")
print(f"  • Total rows: {strategy_total:,}")
print("="*80)
