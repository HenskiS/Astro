"""
Check how many rows exist after Dec 20 for target calculation
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI

pair = 'NZDUSD'
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')
problem_date = pd.Timestamp('2015-12-20')

print("="*100)
print(f"CHECKING ROW COUNT AFTER DEC 20 FOR {pair}")
print("="*100)
print()

# BACKTEST - uses full dataset
df_full = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df_full['date'] = pd.to_datetime(df_full['date'])
df_full = df_full.set_index('date')
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

# Find Dec 20 row
dec20_idx = df_full.index.get_loc(df_full[df_full.index.normalize() == problem_date.normalize()].index[0])
rows_after_dec20_full = len(df_full) - dec20_idx - 1

print(f"BACKTEST (full dataset):")
print(f"  Total rows: {len(df_full)}")
print(f"  Dec 20 row index: {dec20_idx}")
print(f"  Rows after Dec 20: {rows_after_dec20_full}")
print(f"  Can calculate target with shift(-10): {rows_after_dec20_full >= 10}")

# Show what rows would be used for target
if rows_after_dec20_full >= 10:
    target_rows = df_full.iloc[dec20_idx:dec20_idx+11]
    print(f"  Target looks at rows: {target_rows.index[0].date()} to {target_rows.index[-1].date()}")
print()

# PRODUCTION - uses data through train_end + 10 days
api = MockBrokerAPI(data_dir='data')
data_end_date = train_end + pd.Timedelta(days=10)  # Dec 31
prod_data = api.get_history(pair, count=999999, end_date=data_end_date)

# Find Dec 20 row
dec20_idx_prod = prod_data.index.get_loc(prod_data[prod_data.index.normalize() == problem_date.normalize()].index[0])
rows_after_dec20_prod = len(prod_data) - dec20_idx_prod - 1

print(f"PRODUCTION (data through {data_end_date.date()}):")
print(f"  Total rows: {len(prod_data)}")
print(f"  Dec 20 row index: {dec20_idx_prod}")
print(f"  Rows after Dec 20: {rows_after_dec20_prod}")
print(f"  Can calculate target with shift(-10): {rows_after_dec20_prod >= 10}")

# Show what rows would be used for target
if rows_after_dec20_prod >= 10:
    target_rows_prod = prod_data.iloc[dec20_idx_prod:dec20_idx_prod+11]
    print(f"  Target looks at rows: {target_rows_prod.index[0].date()} to {target_rows_prod.index[-1].date()}")
else:
    print(f"  [ISSUE] Not enough rows! Target would be NaN or use partial data")
print()

print("COMPARISON:")
print("-"*100)
if rows_after_dec20_full == rows_after_dec20_prod:
    print(f"  Rows after Dec 20: MATCH ({rows_after_dec20_full})")
else:
    print(f"  Rows after Dec 20: DIFFER")
    print(f"    Backtest: {rows_after_dec20_full}")
    print(f"    Production: {rows_after_dec20_prod}")
    print(f"    Difference: {abs(rows_after_dec20_full - rows_after_dec20_prod)}")
print()

# Check what target calculation would produce
print("ACTUAL TARGET CALCULATION:")
print("-"*100)

# Backtest
df_back = calculate_features(df_full.copy())
df_back = create_targets(df_back)
dec20_back = df_back[df_back.index.normalize() == problem_date.normalize()].iloc[0]
print(f"Backtest Dec 20 target_breakout_high: {dec20_back['target_breakout_high']}")

# Production
prod_data_calc = calculate_features(prod_data.copy())
prod_data_calc = create_targets(prod_data_calc)
dec20_prod = prod_data_calc[prod_data_calc.index.normalize() == problem_date.normalize()].iloc[0]
print(f"Production Dec 20 target_breakout_high: {dec20_prod['target_breakout_high']}")

if dec20_back['target_breakout_high'] != dec20_prod['target_breakout_high']:
    print()
    print(f"[ISSUE] Targets differ! This is why HIGH models are different.")
print()
