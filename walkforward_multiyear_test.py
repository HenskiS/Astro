"""
MULTI-YEAR WALK-FORWARD VALIDATION
===================================
Tests the strategy across multiple years to see if performance is consistent
or if 2025 was just a lucky year.

Approach:
- Use 10-month training window (matching original)
- Test on multiple 4-month periods across different years
- Compare CAGRs to see variance across market regimes
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')

print("="*100)
print("MULTI-YEAR WALK-FORWARD VALIDATION")
print("="*100)
print()

LOOKBACK_PERIOD = 80
FORWARD_PERIODS = 24
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

# Define test periods across multiple years
# Each period: 10 months train -> 4 months test
TEST_PERIODS = [
    {'name': '2021 Test', 'train_start': '2020-01-01', 'train_end': '2020-11-01', 'test_start': '2020-11-01', 'test_end': '2021-03-01'},
    {'name': '2022 Test', 'train_start': '2021-01-01', 'train_end': '2021-11-01', 'test_start': '2021-11-01', 'test_end': '2022-03-01'},
    {'name': '2023 Test', 'train_start': '2022-01-01', 'train_end': '2022-11-01', 'test_start': '2022-11-01', 'test_end': '2023-03-01'},
    {'name': '2024 Test', 'train_start': '2023-01-01', 'train_end': '2023-11-01', 'test_start': '2023-11-01', 'test_end': '2024-03-01'},
    {'name': '2025 Test', 'train_start': '2024-01-01', 'train_end': '2024-11-01', 'test_start': '2024-11-01', 'test_end': '2025-03-01'},
]

def add_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute_slot'] = df.index.minute // 15
    df['day_of_week'] = df.index.dayofweek
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1
    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)
    return df

def calculate_features(df, lookback=LOOKBACK_PERIOD):
    df = df.copy()
    df = add_time_features(df)
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_100'] = df['close'].ewm(span=100).mean()
    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']
    df['volume_ma'] = df['volume'].rolling(96).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['return_1p'] = df['close'].pct_change(1)
    df['return_4p'] = df['close'].pct_change(4)
    df['return_16p'] = df['close'].pct_change(16)
    df['return_96p'] = df['close'].pct_change(96)
    df['spread_ma'] = df['spread_pct'].rolling(96).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']
    return df

def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    df = df.copy()
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()
    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)
    return df

feature_cols = [
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}p',
    'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
    'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'atr_pct', 'volume_ratio',
    'return_1p', 'return_4p', 'return_16p', 'return_96p',
    'spread_pct', 'spread_ratio',
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

# Load data
print("Loading data...")
all_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    df = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])
    all_data[pair] = df
    print(f"  {pair}: {len(df)} samples from {df.index.min().date()} to {df.index.max().date()}")

print()

# Test each period
print("="*100)
print("TESTING ACROSS MULTIPLE YEARS")
print("="*100)
print()

all_period_predictions = {}

for period in TEST_PERIODS:
    print(f"{period['name']}:")
    print(f"  Train: {period['train_start']} to {period['train_end']}")
    print(f"  Test:  {period['test_start']} to {period['test_end']}")

    period_predictions = {}

    for pair in PAIRS:
        df = all_data[pair]

        # Split data
        train_df = df[(df.index >= period['train_start']) & (df.index < period['train_end'])]
        test_df = df[(df.index >= period['test_start']) & (df.index < period['test_end'])]

        if len(train_df) < 1000 or len(test_df) == 0:
            print(f"    {pair}: Insufficient data (train={len(train_df)}, test={len(test_df)})")
            continue

        # Train
        X_train = train_df[feature_cols]
        y_train_high = train_df['breakout_high']
        y_train_low = train_df['breakout_low']

        model_high = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
        model_high.fit(X_train, y_train_high, verbose=False)

        model_low = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
        model_low.fit(X_train, y_train_low, verbose=False)

        # Predict
        X_test = test_df[feature_cols]
        pred_high = model_high.predict_proba(X_test)[:, 1]
        pred_low = model_low.predict_proba(X_test)[:, 1]

        preds_df = pd.DataFrame({
            'breakout_high_prob': pred_high,
            'breakout_low_prob': pred_low,
            f'high_{LOOKBACK_PERIOD}p': test_df[f'high_{LOOKBACK_PERIOD}p'],
            f'low_{LOOKBACK_PERIOD}p': test_df[f'low_{LOOKBACK_PERIOD}p'],
            'close': test_df['close']
        }, index=test_df.index)

        period_predictions[pair] = preds_df

    all_period_predictions[period['name']] = period_predictions
    print(f"  Generated predictions for {len(period_predictions)} pairs")
    print()

# Save all period predictions
print("Saving multi-year predictions...")
for period_name, preds in all_period_predictions.items():
    safe_name = period_name.replace(' ', '_').lower()
    with open(f'test_predictions_15m_{safe_name}.pkl', 'wb') as f:
        pickle.dump(preds, f)
    print(f"  Saved: test_predictions_15m_{safe_name}.pkl")

print()
print("="*100)
print("DONE!")
print("="*100)
print()
print("Next: Run backtests on each period to compare CAGRs across years")
print()
