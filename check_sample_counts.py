"""
Check if any pairs have sample counts between 100-101
This would confirm if the threshold difference matters
"""
import pandas as pd
from production_simulation import calculate_features, create_targets

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']

# Check first training period (2010-2015 for 2016Q1)
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print(f"Checking sample counts for training period: {train_start.date()} to {train_end.date()}")
print()

for pair in PAIRS:
    df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = calculate_features(df)
    df = create_targets(df)

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    train_df = train_df.dropna()

    high_count = train_df['target_breakout_high'].sum()
    low_count = train_df['target_breakout_low'].sum()

    status = ""
    if high_count == 100 or high_count == 101 or low_count == 100 or low_count == 101:
        status = " ⚠ AFFECTED BY THRESHOLD"
    elif high_count <= 100 or low_count <= 100:
        status = " ❌ WOULD BE SKIPPED"

    print(f"{pair:8s}: high={high_count:4.0f}, low={low_count:4.0f}{status}")
