"""
Debug why NZDUSD 2016-03-23 predictions differ when training data should match
"""
import pandas as pd
import pickle
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI
from xgboost import XGBClassifier

pair = 'NZDUSD'
prediction_date = pd.Timestamp('2016-03-23')
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print(f"DEBUGGING {pair} MISMATCH ON {prediction_date.date()}")
print("="*100)
print()

# Load saved predictions
with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

back_df = backtest_preds['2016Q1'][pair]
prod_df = prod_preds['2016Q1'][pair]

back_row = back_df[back_df.index.normalize() == prediction_date.normalize()].iloc[0]
prod_row = prod_df[prod_df.index.normalize() == prediction_date.normalize()].iloc[0]

print("SAVED PREDICTIONS:")
print("-"*100)
print(f"Backtest  - HIGH: {back_row['breakout_high_prob']:.6f}, LOW: {back_row['breakout_low_prob']:.6f}")
print(f"Production - HIGH: {prod_row['breakout_high_prob']:.6f}, LOW: {prod_row['breakout_low_prob']:.6f}")
print(f"Difference - HIGH: {abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob']):.6f}, LOW: {abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob']):.6f}")
print()

# Check if features used for prediction match
print("FEATURES USED FOR PREDICTION:")
print("-"*100)
print(f"Backtest  - high_20d: {back_row['high_20d']:.6f}, low_20d: {back_row['low_20d']:.6f}, close: {back_row['close']:.6f}")
print(f"Production - high_20d: {prod_row['high_20d']:.6f}, low_20d: {prod_row['low_20d']:.6f}, close: {prod_row['close']:.6f}")
print()

if abs(back_row['high_20d'] - prod_row['high_20d']) > 0.0001:
    print("[ISSUE] Features differ - prediction made with different input data!")
else:
    print("[OK] Features match - issue must be in model training")
print()

# Compare training data
print("TRAINING DATA COMPARISON:")
print("-"*100)

# Backtest training
df_back = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df_back['date'] = pd.to_datetime(df_back['date'])
df_back = df_back.set_index('date')
if df_back.index.tz is not None:
    df_back.index = df_back.index.tz_localize(None)

df_back = calculate_features(df_back.copy())
df_back = create_targets(df_back)
train_df_back = df_back[(df_back.index >= train_start) & (df_back.index <= train_end)].copy()
train_df_back = train_df_back.dropna()

# Production training (with fix #1)
api = MockBrokerAPI(data_dir='data')
data_end_date = train_end + pd.Timedelta(days=10)
train_df_prod = api.get_history(pair, count=999999, end_date=data_end_date)
train_df_prod = calculate_features(train_df_prod)
train_df_prod = create_targets(train_df_prod)
train_df_prod = train_df_prod[(train_df_prod.index >= train_start) & (train_df_prod.index <= train_end)]
train_df_prod = train_df_prod.dropna()

print(f"Backtest training:   {len(train_df_back)} rows, HIGH targets: {train_df_back['target_breakout_high'].sum()}, LOW targets: {train_df_back['target_breakout_low'].sum()}")
print(f"Production training: {len(train_df_prod)} rows, HIGH targets: {train_df_prod['target_breakout_high'].sum()}, LOW targets: {train_df_prod['target_breakout_low'].sum()}")
print()

if len(train_df_back) != len(train_df_prod):
    print(f"[ISSUE] Different number of training rows!")
elif train_df_back['target_breakout_high'].sum() != train_df_prod['target_breakout_high'].sum():
    print(f"[ISSUE] Different HIGH target counts!")
elif train_df_back['target_breakout_low'].sum() != train_df_prod['target_breakout_low'].sum():
    print(f"[ISSUE] Different LOW target counts!")
else:
    print(f"[OK] Training data matches - targets are identical")
print()
