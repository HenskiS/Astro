"""
VALIDATION: STRICT TIME-SERIES TRAIN/TEST SPLIT
================================================
This script implements a rigorous validation framework to ensure
no data contamination between train and test sets.

Key differences from standard approach:
1. Split by DATE first (not by row count)
2. Calculate features SEPARATELY on train and test
3. Add buffer period between train/test to eliminate window overlap
4. Generate predictions only on truly out-of-sample test data

This allows us to verify that backtest results are not inflated by leakage.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

print("="*100)
print("VALIDATION: STRICT TIME-SERIES TRAIN/TEST SPLIT")
print("="*100)
print()

# Parameters
LOOKBACK_PERIOD = 80  # 20 hours of 15m data
FORWARD_PERIODS = 24  # 6 hours ahead
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

# CRITICAL: Define fixed date split (not percentage split)
# This ensures train and test are completely separate time periods
# Based on data range: Oct 2024 - Jan 2026
TRAIN_END_DATE = '2024-11-30'  # Train on data up to this date
BUFFER_DAYS = 3  # Gap between train and test to prevent window overlap
TEST_START_DATE = '2024-12-03'  # Test starts after buffer

print(f"Train period: All data up to {TRAIN_END_DATE}")
print(f"Buffer period: {BUFFER_DAYS} days (prevents rolling window overlap)")
print(f"Test period: {TEST_START_DATE} onwards")
print()


def add_time_features(df):
    """Add intraday-specific features"""
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
    """Calculate technical features - NO FUTURE DATA"""
    df = df.copy()
    df = add_time_features(df)

    # Breakout levels
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']

    # Distance to breakout levels
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    # EMAs
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_100'] = df['close'].ewm(span=100).mean()

    # Price relative to EMAs
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
    """Calculate breakout targets using FUTURE data (labels only)"""
    df = df.copy()

    # Future high/low over next 6 hours
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()

    # Current breakout levels
    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']

    # Did price break out in the next 6 hours?
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)

    return df


# Feature columns
feature_cols = [
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}p',
    'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
    'macd', 'macd_signal', 'macd_hist',
    'rsi_14', 'atr_pct', 'volume_ratio',
    'return_1p', 'return_4p', 'return_16p', 'return_96p',
    'spread_pct', 'spread_ratio',
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

# Load raw data
print("Loading 15m data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
    print(f"  {pair}: {len(df)} candles from {df.index.min()} to {df.index.max()}")
print()

# Split data by DATE (not by percentage)
print("="*100)
print("SPLITTING DATA BY TIME")
print("="*100)
print()

train_data = {}
test_data = {}
buffer_data = {}

for pair in PAIRS:
    df = all_raw_data[pair]

    # Split by date
    train_df = df[df.index < TRAIN_END_DATE].copy()
    test_df = df[df.index >= TEST_START_DATE].copy()
    buffer_df = df[(df.index >= TRAIN_END_DATE) & (df.index < TEST_START_DATE)].copy()

    train_data[pair] = train_df
    test_data[pair] = test_df
    buffer_data[pair] = buffer_df

    print(f"{pair}:")
    print(f"  Train: {len(train_df):>6,} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Buffer: {len(buffer_df):>6,} samples (excluded from both train and test)")
    print(f"  Test: {len(test_df):>6,} samples ({test_df.index.min()} to {test_df.index.max()})")
    print()

# Calculate features SEPARATELY for train and test
print("="*100)
print("CALCULATING FEATURES (SEPARATELY FOR TRAIN AND TEST)")
print("="*100)
print()

train_features = {}
test_features = {}

for pair in PAIRS:
    print(f"{pair}:")

    # Calculate features on TRAIN data only
    train_df = calculate_features(train_data[pair], LOOKBACK_PERIOD)
    train_df = calculate_targets(train_df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    train_df = train_df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])
    train_features[pair] = train_df
    print(f"  Train features calculated: {len(train_df)} valid samples")

    # Calculate features on TEST data only
    # CRITICAL: Test features use only test period data (no train data in rolling windows)
    test_df = calculate_features(test_data[pair], LOOKBACK_PERIOD)
    test_df = calculate_targets(test_df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    test_df = test_df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])
    test_features[pair] = test_df
    print(f"  Test features calculated: {len(test_df)} valid samples")
    print()

# Train models
print("="*100)
print("TRAINING MODELS (ON TRAIN DATA ONLY)")
print("="*100)
print()

trained_models = {}

for pair in PAIRS:
    train_df = train_features[pair]

    X_train = train_df[feature_cols]
    y_train_high = train_df['breakout_high']
    y_train_low = train_df['breakout_low']

    # Train breakout high model
    model_high = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model_high.fit(X_train, y_train_high)

    # Train breakout low model
    model_low = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model_low.fit(X_train, y_train_low)

    trained_models[pair] = {
        'model_high': model_high,
        'model_low': model_low,
        'feature_cols': feature_cols
    }

    print(f"{pair}: Models trained on {len(X_train)} samples")

print()

# Generate predictions on TEST data
print("="*100)
print("GENERATING PREDICTIONS (ON TEST DATA ONLY)")
print("="*100)
print()

test_predictions = {}

for pair in PAIRS:
    test_df = test_features[pair]
    X_test = test_df[feature_cols]

    model_high = trained_models[pair]['model_high']
    model_low = trained_models[pair]['model_low']

    # Predict probabilities
    pred_high = model_high.predict_proba(X_test)[:, 1]
    pred_low = model_low.predict_proba(X_test)[:, 1]

    # Create predictions dataframe
    preds_df = pd.DataFrame({
        'breakout_high_prob': pred_high,
        'breakout_low_prob': pred_low,
        f'high_{LOOKBACK_PERIOD}p': test_df[f'high_{LOOKBACK_PERIOD}p'],
        f'low_{LOOKBACK_PERIOD}p': test_df[f'low_{LOOKBACK_PERIOD}p'],
        'close': test_df['close']
    }, index=test_df.index)

    test_predictions[pair] = preds_df

    print(f"{pair}:")
    print(f"  Predictions: {len(preds_df)}")
    print(f"  Date range: {preds_df.index.min()} to {preds_df.index.max()}")
    print(f"  Avg high prob: {pred_high.mean():.3f}")
    print(f"  Avg low prob: {pred_low.mean():.3f}")

print()

# Save validated predictions
print("Saving validated predictions...")
with open('test_predictions_15m_validated.pkl', 'wb') as f:
    pickle.dump(test_predictions, f)
print("Saved: test_predictions_15m_validated.pkl")
print()

# Save validated models
with open('models_15m_validated.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("Saved: models_15m_validated.pkl")
print()

print("="*100)
print("VALIDATION COMPLETE!")
print("="*100)
print()
print("Next steps:")
print("  1. Run backtest on validated predictions:")
print("     python backtest_15m_optimized.py  # but modify to load 'test_predictions_15m_validated.pkl'")
print("  2. Compare CAGR with original results")
print("  3. If results are similar, original results were valid")
print("  4. If results are significantly worse, there was contamination")
print()
