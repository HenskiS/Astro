"""
Verify that adding 10 days of data fixes the target calculation
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI

pair = 'USDCAD'
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print("VERIFYING TARGET FIX FOR USDCAD")
print("="*100)
print()

# OLD PRODUCTION APPROACH (data only up to train_end)
print("1. OLD PRODUCTION (data up to train_end only):")
print("-"*100)
api = MockBrokerAPI(data_dir='data')
training_data_old = api.get_history(pair, count=999999, end_date=train_end)
training_data_old = calculate_features(training_data_old)
training_data_old = create_targets(training_data_old)
training_data_old = training_data_old[(training_data_old.index >= train_start) &
                                       (training_data_old.index <= train_end)]
training_data_old = training_data_old.dropna()
print(f"  target_breakout_high sum: {training_data_old['target_breakout_high'].sum()}")
print()

# NEW PRODUCTION APPROACH (data up to train_end + 10 days)
print("2. NEW PRODUCTION (data up to train_end + 10 days):")
print("-"*100)
data_end_date = train_end + pd.Timedelta(days=10)
training_data_new = api.get_history(pair, count=999999, end_date=data_end_date)
training_data_new = calculate_features(training_data_new)
training_data_new = create_targets(training_data_new)
training_data_new = training_data_new[(training_data_new.index >= train_start) &
                                       (training_data_new.index <= train_end)]
training_data_new = training_data_new.dropna()
print(f"  target_breakout_high sum: {training_data_new['target_breakout_high'].sum()}")
print()

# BACKTEST APPROACH (full dataset)
print("3. BACKTEST (full dataset):")
print("-"*100)
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df_with_features = calculate_features(df.copy())
df_with_targets = create_targets(df_with_features)
train_df_backtest = df_with_targets[(df_with_targets.index >= train_start) &
                                     (df_with_targets.index <= train_end)].copy()
train_df_backtest = train_df_backtest.dropna()
print(f"  target_breakout_high sum: {train_df_backtest['target_breakout_high'].sum()}")
print()

# COMPARISON
print("4. COMPARISON:")
print("-"*100)
print(f"  OLD Production: {training_data_old['target_breakout_high'].sum()} HIGH targets")
print(f"  NEW Production: {training_data_new['target_breakout_high'].sum()} HIGH targets")
print(f"  Backtest:       {train_df_backtest['target_breakout_high'].sum()} HIGH targets")
print()

if training_data_new['target_breakout_high'].sum() == train_df_backtest['target_breakout_high'].sum():
    print("[SUCCESS] New production matches backtest!")
else:
    print(f"[ISSUE] Still differs by {abs(training_data_new['target_breakout_high'].sum() - train_df_backtest['target_breakout_high'].sum())}")
print()
