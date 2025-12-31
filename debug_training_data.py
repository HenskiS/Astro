"""
DEBUG TRAINING DATA
===================
Compare training data preparation between quarterly generation and production simulation.
"""
import pandas as pd
import numpy as np
import pickle
from mock_broker_api import MockBrokerAPI
from production_simulation import calculate_features, XGB_CONFIG

# Initialize
api = MockBrokerAPI(data_dir='data')
pair = 'EURUSD'
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print("COMPARING TRAINING DATA PREPARATION")
print("="*100)
print()

# Method 1: Production simulation style
print("Method 1: Production Simulation Style")
print("-"*100)
history = api.get_history(pair, count=999999, end_date=train_end)
print(f"Raw data shape: {history.shape}")
print(f"Date range: {history.index.min()} to {history.index.max()}")

features = calculate_features(history)
print(f"Features shape before dropna: {features.shape}")

features = features.dropna()
print(f"Features shape after dropna: {features.shape}")

# Create targets (production style - manual loop)
full_data = api._load_pair_data(pair)
features['breakout_high'] = 0
features['breakout_low'] = 0

for i in range(len(features) - 10):
    current_date = features.index[i]
    future_10d_indices = list(range(i+1, min(i+11, len(features))))

    if len(future_10d_indices) < 10:
        continue

    current_high_20d = features.iloc[i]['high_20d']
    current_low_20d = features.iloc[i]['low_20d']

    # Get future data
    future_dates = [features.index[j] for j in future_10d_indices]
    future_data = full_data.loc[future_dates]

    if (future_data['high'] > current_high_20d).any():
        features.iloc[i, features.columns.get_loc('breakout_high')] = 1

    if (future_data['low'] < current_low_20d).any():
        features.iloc[i, features.columns.get_loc('breakout_low')] = 1

# Remove last 10 days (no targets)
features_train = features.iloc[:-10].copy()

print(f"Training data shape: {features_train.shape}")
print(f"Breakout high: {features_train['breakout_high'].sum()} / {len(features_train)} = {features_train['breakout_high'].mean():.3f}")
print(f"Breakout low: {features_train['breakout_low'].sum()} / {len(features_train)} = {features_train['breakout_low'].mean():.3f}")
print()

# Method 2: Quarterly generation style
print("Method 2: Quarterly Generation Style")
print("-"*100)

# Load ALL data
full_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
full_df['date'] = pd.to_datetime(full_df['date'])
full_df = full_df.set_index('date')
if full_df.index.tz is not None:
    full_df.index = full_df.index.tz_localize(None)

print(f"Raw data shape (ALL): {full_df.shape}")
print(f"Date range (ALL): {full_df.index.min()} to {full_df.index.max()}")

# Calculate features on ALL data
from generate_predictions_quarterly import calculate_technical_features, create_targets
full_df_features = calculate_technical_features(full_df)
print(f"Features shape (ALL) before targets: {full_df_features.shape}")

full_df_features = create_targets(full_df_features)
print(f"Features shape (ALL) after targets: {full_df_features.shape}")

# Filter to training period
train_df = full_df_features[full_df_features.index <= train_end].copy()
print(f"Filtered to train_end: {train_df.shape}")

train_df = train_df.dropna()
print(f"After dropna: {train_df.shape}")

print(f"Breakout high: {train_df['target_breakout_high'].sum()} / {len(train_df)} = {train_df['target_breakout_high'].mean():.3f}")
print(f"Breakout low: {train_df['target_breakout_low'].sum()} / {len(train_df)} = {train_df['target_breakout_low'].mean():.3f}")
print()

# Compare specific samples
print("="*100)
print("SAMPLE COMPARISON (Last 10 training samples)")
print("="*100)

print("\nProduction simulation targets (last 10):")
print(features_train[['high_20d', 'low_20d', 'breakout_high', 'breakout_low']].tail(10))

print("\nQuarterly generation targets (last 10):")
print(train_df[['high_20d', 'low_20d', 'target_breakout_high', 'target_breakout_low']].tail(10))

# Check if they match
common_dates = features_train.index.intersection(train_df.index)
print(f"\nCommon dates: {len(common_dates)}")

if len(common_dates) > 0:
    prod_targets = features_train.loc[common_dates, ['breakout_high', 'breakout_low']]
    quarterly_targets = train_df.loc[common_dates, ['target_breakout_high', 'target_breakout_low']]

    high_matches = (prod_targets['breakout_high'].values == quarterly_targets['target_breakout_high'].values).sum()
    low_matches = (prod_targets['breakout_low'].values == quarterly_targets['target_breakout_low'].values).sum()

    print(f"High target matches: {high_matches} / {len(common_dates)} = {high_matches/len(common_dates)*100:.1f}%")
    print(f"Low target matches: {low_matches} / {len(common_dates)} = {low_matches/len(common_dates)*100:.1f}%")

    # Show some mismatches
    high_mismatches = prod_targets[prod_targets['breakout_high'] != quarterly_targets['target_breakout_high']]
    if len(high_mismatches) > 0:
        print(f"\nSample high target mismatches ({len(high_mismatches)} total):")
        print(high_mismatches.head())
