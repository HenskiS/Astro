"""
COMPARE TRAINING DATA
=====================
Check if backtest and production use the same training data
"""
import pandas as pd
from mock_broker_api import MockBrokerAPI
from production_simulation import calculate_features, create_targets

print("="*100)
print("COMPARING TRAINING DATA FOR 2016Q1")
print("="*100)
print()

api = MockBrokerAPI(data_dir='data')
pair = 'EURUSD'

# Production approach
print("PRODUCTION TRAINING (2010-2015):")
print("-"*100)
train_end_date = pd.Timestamp('2015-12-21')
train_start_date = pd.Timestamp('2010-01-01')

# Get data
training_data_prod = api.get_history(pair, count=999999, end_date=train_end_date)
print(f"Raw data: {len(training_data_prod)} rows, {training_data_prod.index[0].date()} to {training_data_prod.index[-1].date()}")

# Calculate features on FULL history
training_data_prod = calculate_features(training_data_prod)
training_data_prod = create_targets(training_data_prod)

# Filter to training window
training_data_prod = training_data_prod[(training_data_prod.index >= train_start_date) &
                                       (training_data_prod.index <= train_end_date)]
training_data_prod = training_data_prod.dropna()

print(f"After features+targets+filter: {len(training_data_prod)} rows")
print(f"Date range: {training_data_prod.index[0].date()} to {training_data_prod.index[-1].date()}")
print(f"Target breakout high: {training_data_prod['target_breakout_high'].sum()} / {len(training_data_prod)} = {training_data_prod['target_breakout_high'].mean():.4f}")
print(f"Target breakout low: {training_data_prod['target_breakout_low'].sum()} / {len(training_data_prod)} = {training_data_prod['target_breakout_low'].mean():.4f}")
print()

# Backtest approach (from generate_predictions_quarterly.py)
print("BACKTEST TRAINING (2010-2015):")
print("-"*100)

# Load ALL data
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print(f"Raw data (ALL): {len(df)} rows")

# Calculate features on ALL data
from production_simulation import calculate_features as calc_feat
df_with_features = calc_feat(df.copy())
df_with_features = create_targets(df_with_features)

# Filter to training window
train_df = df_with_features[(df_with_features.index >= train_start_date) &
                            (df_with_features.index <= train_end_date)].copy()
train_df = train_df.dropna()

print(f"After features+targets+filter: {len(train_df)} rows")
print(f"Date range: {train_df.index[0].date()} to {train_df.index[-1].date()}")
print(f"Target breakout high: {train_df['target_breakout_high'].sum()} / {len(train_df)} = {train_df['target_breakout_high'].mean():.4f}")
print(f"Target breakout low: {train_df['target_breakout_low'].sum()} / {len(train_df)} = {train_df['target_breakout_low'].mean():.4f}")
print()

# Compare
print("="*100)
print("COMPARISON")
print("="*100)
print(f"Production rows: {len(training_data_prod)}")
print(f"Backtest rows: {len(train_df)}")
print(f"Row count match: {len(training_data_prod) == len(train_df)}")
print()

if len(training_data_prod) == len(train_df):
    # Check if targets match
    common_dates = training_data_prod.index.intersection(train_df.index)
    print(f"Common dates: {len(common_dates)}")

    if len(common_dates) > 0:
        prod_high = training_data_prod.loc[common_dates, 'target_breakout_high']
        back_high = train_df.loc[common_dates, 'target_breakout_high']
        high_match = (prod_high == back_high).sum()

        prod_low = training_data_prod.loc[common_dates, 'target_breakout_low']
        back_low = train_df.loc[common_dates, 'target_breakout_low']
        low_match = (prod_low == back_low).sum()

        print(f"High target matches: {high_match}/{len(common_dates)} ({high_match/len(common_dates)*100:.1f}%)")
        print(f"Low target matches: {low_match}/{len(common_dates)} ({low_match/len(common_dates)*100:.1f}%)")

        if high_match < len(common_dates):
            print("\nHigh target mismatches (first 5):")
            mismatches = common_dates[prod_high.values != back_high.values][:5]
            for date in mismatches:
                print(f"  {date.date()}: Production={prod_high.loc[date]}, Backtest={back_high.loc[date]}")
else:
    print("[ERROR] Row counts differ! Training data is NOT the same!")

print()
