"""
PRODUCTION MODEL TRAINING
=========================
Train 15-minute breakout models for production deployment.

Usage:
    python train_models.py [--months 10]

This script:
- Loads data from ../../data_15m/ (project root)
- Trains models using last N months (default: 10)
- Saves models to ../models/ (production_trader/models/)
- Uses same feature calculation as production strategy
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Train production models')
parser.add_argument('--months', type=int, default=10,
                    help='Number of months to use for training (default: 10)')
args = parser.parse_args()

TRAINING_MONTHS = args.months
LOOKBACK_PERIOD = 80  # 20 hours of 15m data
FORWARD_PERIODS = 24  # 6 hours ahead
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Paths (relative to production_trader/scripts/)
DATA_DIR = Path(__file__).parent.parent.parent / 'data_15m'  # ../../data_15m
MODELS_DIR = Path(__file__).parent.parent / 'models'  # ../models (production_trader/models/)
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 100)
print("PRODUCTION MODEL TRAINING")
print("=" * 100)
print()
print(f"Training window: Last {TRAINING_MONTHS} months")
print(f"Data directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")
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
    """Calculate technical features for 15m data"""
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
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                    abs(df['low'] - df['close'].shift(1))))
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
    """Calculate breakout targets for next 6 hours"""
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


# Feature columns (must match feature_calculator.py output)
feature_cols = [
    # Breakout features
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}p',
    # EMAs
    'price_vs_ema12', 'price_vs_ema26', 'price_vs_ema50', 'price_vs_ema100',
    # MACD
    'macd', 'macd_signal', 'macd_hist',
    # RSI
    'rsi_14',
    # Volatility
    'atr_pct',
    # Volume
    'volume_ratio',
    # Momentum
    'return_1p', 'return_4p', 'return_16p', 'return_96p',
    # Spread
    'spread_pct', 'spread_ratio',
    # Time features
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

print(f"Using {len(feature_cols)} features")
print()

# Load and prepare data
print("=" * 100)
print("LOADING DATA")
print("=" * 100)
print()

all_data = {}
for pair in PAIRS:
    file_path = DATA_DIR / f'{pair}_15m.csv'

    if not file_path.exists():
        print(f"WARNING: {file_path} not found, skipping {pair}")
        continue

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Filter to last N months
    cutoff_date = df.index.max() - pd.DateOffset(months=TRAINING_MONTHS)
    df = df[df.index >= cutoff_date]

    all_data[pair] = df
    print(f"  {pair}: {len(df):,} candles from {df.index.min()} to {df.index.max()}")

if len(all_data) == 0:
    print("ERROR: No data loaded! Check data_15m/ directory exists with CSV files.")
    sys.exit(1)

print()

# Calculate features and targets
print("=" * 100)
print("CALCULATING FEATURES")
print("=" * 100)
print()

for pair in all_data.keys():
    df = all_data[pair]
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    all_data[pair] = df
    print(f"  {pair}: {len(df.columns)} features calculated")

print()

# Train models
print("=" * 100)
print("TRAINING MODELS")
print("=" * 100)
print()

trained_models = {}

for pair in all_data.keys():
    df = all_data[pair]

    # Drop NaN rows
    df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    # Use all N months for training
    train_data = df_clean

    print(f"{pair}:")
    print(f"  Training samples: {len(train_data):,}")
    print(f"  Date range: {train_data.index.min()} to {train_data.index.max()}")

    # Prepare training data
    X_train = train_data[feature_cols]
    y_train_high = train_data['breakout_high']
    y_train_low = train_data['breakout_low']

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

    # Save individual model files
    high_path = MODELS_DIR / f'xgboost_15m_{pair}_high.pkl'
    with open(high_path, 'wb') as f:
        pickle.dump(model_high, f)

    low_path = MODELS_DIR / f'xgboost_15m_{pair}_low.pkl'
    with open(low_path, 'wb') as f:
        pickle.dump(model_low, f)

    trained_models[pair] = {
        'model_high': model_high,
        'model_low': model_low,
        'feature_cols': feature_cols,
        'train_samples': len(train_data)
    }

    print(f"  Models saved: {high_path.name}, {low_path.name}")
    print()

# Save combined models file (for convenience)
combined_path = Path(__file__).parent.parent / 'models_15m.pkl'
with open(combined_path, 'wb') as f:
    pickle.dump(trained_models, f)

print("=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)
print()
print(f"Trained {len(trained_models)} pairs")
print(f"Models saved to: {MODELS_DIR}")
print(f"Combined file: {combined_path}")
print()
print("Next steps:")
print("  1. Validate: python backtest_15m_optimized.py --predictions test_predictions_15m_continuous.pkl")
print("  2. Deploy: Upload models/ directory to server")
print("  3. Test: python main.py --dry-run")
print()
