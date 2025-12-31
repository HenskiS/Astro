"""
Check data quality for pairs to see if missing dates cause feature differences
"""
import pandas as pd

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

print("="*100)
print("CHECKING DATA QUALITY FOR ALL PAIRS")
print("="*100)
print()

# Load EURUSD as reference (perfect match)
reference_df = pd.read_csv(f'data/EURUSD_1day_with_spreads.csv')
reference_df['date'] = pd.to_datetime(reference_df['date'])
reference_dates = set(reference_df['date'].dt.normalize())

print(f"Reference (EURUSD): {len(reference_dates)} dates")
print()

for pair in pairs:
    df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Normalize dates for comparison
    pair_dates = set(df['date'].dt.normalize())

    # Check for missing dates compared to reference
    missing_from_pair = reference_dates - pair_dates
    extra_in_pair = pair_dates - reference_dates

    print(f"{pair:8s}: {len(pair_dates):5d} dates | Missing: {len(missing_from_pair):4d} | Extra: {len(extra_in_pair):4d}")

    if len(missing_from_pair) > 0:
        # Check if any missing dates are in 2016Q1
        q1_dates = [d for d in missing_from_pair if d.year == 2016 and 1 <= d.month <= 3]
        if q1_dates:
            print(f"         Missing in 2016Q1: {len(q1_dates)} dates")
            if len(q1_dates) <= 5:
                for d in sorted(q1_dates)[:5]:
                    print(f"           {d.date()}")

print()
