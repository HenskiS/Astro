"""
WALK-FORWARD VALIDATION: 15M STRATEGY
======================================
This is the PROPER way to validate a trading strategy.

Approach:
1. Train on initial window (e.g., 1 month)
2. Trade on next window (e.g., 2 weeks)
3. Retrain adding the traded period
4. Trade on next window
5. Repeat...

This simulates real production where models are retrained periodically.
Results show what you'd actually achieve trading live with periodic retraining.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

print("="*100)
print("WALK-FORWARD VALIDATION: 15M BREAKOUT STRATEGY")
print("="*100)
print()

# Parameters
LOOKBACK_PERIOD = 80
FORWARD_PERIODS = 24
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

# Walk-forward parameters
INITIAL_TRAIN_MONTHS = 1  # Train on first month
TRADE_WEEKS = 2  # Trade for 2 weeks, then retrain
RETRAIN_MODE = 'expanding'  # 'expanding' = use all historical data, 'rolling' = fixed window

print(f"Walk-Forward Configuration:")
print(f"  Initial training: {INITIAL_TRAIN_MONTHS} month(s)")
print(f"  Trading period: {TRADE_WEEKS} week(s)")
print(f"  Retrain mode: {RETRAIN_MODE}")
print()


def add_time_features(df):
    """Add time-based features"""
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
    """Calculate technical features"""
    df = df.copy()
    df = add_time_features(df)

    # Breakout levels
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    # EMAs
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_100'] = df['close'].ewm(span=100).mean()
    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Volume
    df['volume_ma'] = df['volume'].rolling(96).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Momentum
    df['return_1p'] = df['close'].pct_change(1)
    df['return_4p'] = df['close'].pct_change(4)
    df['return_16p'] = df['close'].pct_change(16)
    df['return_96p'] = df['close'].pct_change(96)

    # Spread
    df['spread_ma'] = df['spread_pct'].rolling(96).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    """Calculate targets"""
    df = df.copy()
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()
    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)
    return df


# Feature columns
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

    # Calculate features and targets on full dataset (this is fine - no leakage)
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    df = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    all_data[pair] = df
    print(f"  {pair}: {len(df)} samples from {df.index.min()} to {df.index.max()}")

print()

# Determine walk-forward windows
all_dates = sorted(set.union(*[set(df.index) for df in all_data.values()]))
min_date = min(all_dates)
max_date = max(all_dates)

print(f"Full date range: {min_date} to {max_date}")
print()

# Create walk-forward windows
initial_train_end = min_date + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
trade_period_length = pd.DateOffset(weeks=TRADE_WEEKS)

windows = []
current_train_end = initial_train_end
current_test_start = initial_train_end
current_test_end = current_test_start + trade_period_length

while current_test_end <= max_date:
    windows.append({
        'train_start': min_date,
        'train_end': current_train_end,
        'test_start': current_test_start,
        'test_end': current_test_end
    })

    # Move to next window
    current_train_end = current_test_end  # Expanding window
    current_test_start = current_test_end
    current_test_end = current_test_start + trade_period_length

print(f"Created {len(windows)} walk-forward windows")
print()

# Execute walk-forward validation
print("="*100)
print("WALK-FORWARD EXECUTION")
print("="*100)
print()

all_predictions = {}  # Store predictions for later backtesting

for window_idx, window in enumerate(windows):
    print(f"Window {window_idx + 1}/{len(windows)}:")
    print(f"  Train: {window['train_start'].date()} to {window['train_end'].date()}")
    print(f"  Test:  {window['test_start'].date()} to {window['test_end'].date()}")

    # Train models for each pair
    for pair in PAIRS:
        df = all_data[pair]

        # Split data
        train_df = df[(df.index >= window['train_start']) & (df.index < window['train_end'])]
        test_df = df[(df.index >= window['test_start']) & (df.index < window['test_end'])]

        if len(train_df) < 1000:  # Skip if insufficient training data
            continue

        if len(test_df) == 0:  # Skip if no test data
            continue

        # Train models
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

        # Generate predictions on test period
        X_test = test_df[feature_cols]
        pred_high = model_high.predict_proba(X_test)[:, 1]
        pred_low = model_low.predict_proba(X_test)[:, 1]

        # Store predictions
        preds_df = pd.DataFrame({
            'breakout_high_prob': pred_high,
            'breakout_low_prob': pred_low,
            f'high_{LOOKBACK_PERIOD}p': test_df[f'high_{LOOKBACK_PERIOD}p'],
            f'low_{LOOKBACK_PERIOD}p': test_df[f'low_{LOOKBACK_PERIOD}p'],
            'close': test_df['close']
        }, index=test_df.index)

        if pair not in all_predictions:
            all_predictions[pair] = []
        all_predictions[pair].append(preds_df)

    print(f"  Models trained and predictions generated")
    print()

# Combine all predictions
print("Combining predictions...")
combined_predictions = {}
for pair in PAIRS:
    if pair in all_predictions and len(all_predictions[pair]) > 0:
        combined_predictions[pair] = pd.concat(all_predictions[pair])
        print(f"  {pair}: {len(combined_predictions[pair])} predictions")

print()

# Save walk-forward predictions
print("Saving walk-forward predictions...")
with open('test_predictions_15m_walkforward.pkl', 'wb') as f:
    pickle.dump(combined_predictions, f)
print("Saved: test_predictions_15m_walkforward.pkl")
print()

print("="*100)
print("WALK-FORWARD VALIDATION COMPLETE")
print("="*100)
print()
print("Next step:")
print("  Run backtest on walk-forward predictions to see realistic performance")
print("  python backtest_15m_optimized.py  # modify to load 'test_predictions_15m_walkforward.pkl'")
print()
print("This represents what you'd actually achieve in production with periodic retraining.")
print()
