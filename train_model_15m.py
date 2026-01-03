"""
TRAIN 15-MINUTE BREAKOUT MODEL
===============================
Ultra high-frequency adaptation of the breakout strategy

Key adaptations from 1h:
- Lookback: 80 periods = 20 hours (vs 80h)
- Forward target: 24 periods = 6 hours (vs 24h)
- Same intraday features (hour, sessions, etc.)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from datetime import datetime, timedelta
import pickle
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING 15-MINUTE BREAKOUT MODEL")
print("="*100)
print()

# Parameters for 15m timeframe
LOOKBACK_PERIOD = 80  # 20 hours of 15m data (same as 1h: 80 periods)
FORWARD_PERIODS = 24  # 6 hours ahead (same as 1h: predict next 24 periods)
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

def add_time_features(df):
    """Add intraday-specific features"""
    df = df.copy()

    # Hour of day (0-23)
    df['hour'] = df.index.hour

    # 15-minute slot within hour (0-3)
    df['minute_slot'] = df.index.minute // 15

    # Day of week (0-6)
    df['day_of_week'] = df.index.dayofweek

    # Trading session indicators
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1

    # Weekend proximity
    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)

    return df


def calculate_features(df, lookback=LOOKBACK_PERIOD):
    """Calculate technical features for 15m data"""
    df = df.copy()

    # Add time features first
    df = add_time_features(df)

    # Breakout levels (using lookback period = 20 hours)
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']

    # Distance to breakout levels
    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    # EMAs (adjusted for 15m data)
    df['ema_12'] = df['close'].ewm(span=12).mean()  # 3 hours
    df['ema_26'] = df['close'].ewm(span=26).mean()  # 6.5 hours
    df['ema_50'] = df['close'].ewm(span=50).mean()  # 12.5 hours
    df['ema_100'] = df['close'].ewm(span=100).mean()  # 25 hours

    # Price relative to EMAs
    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14 periods = 3.5 hours)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Volatility (ATR for 15m data)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Volume features
    df['volume_ma'] = df['volume'].rolling(96).mean()  # 24h average (96 * 15m)
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Recent momentum (shorter periods for 15m)
    df['return_1p'] = df['close'].pct_change(1)   # Last 15m
    df['return_4p'] = df['close'].pct_change(4)   # Last hour
    df['return_16p'] = df['close'].pct_change(16) # Last 4 hours
    df['return_96p'] = df['close'].pct_change(96) # Last 24 hours

    # Spread analysis
    df['spread_ma'] = df['spread_pct'].rolling(96).mean()  # 24h avg
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    """
    Calculate breakout targets for next 6 hours (24 * 15m periods)
    """
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


# Load and prepare data
print("Loading 15m data...")
all_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_data[pair] = df
    print(f"  {pair}: {len(df)} candles from {df.index.min()} to {df.index.max()}")

print()

# Calculate features for each pair
print("Calculating features...")
for pair in PAIRS:
    df = all_data[pair]
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    all_data[pair] = df
    print(f"  {pair}: {len(df.columns)} features")

print()

# Define feature columns
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

# Train models using 70/30 split
print("="*100)
print("TRAINING MODELS")
print("="*100)
print()

trained_models = {}

for pair in PAIRS:
    df = all_data[pair]

    # Drop NaN rows
    df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    # Split train/test (70/30)
    split_idx = int(len(df_clean) * 0.7)
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]

    print(f"{pair}:")
    print(f"  Training: {len(train_data)} samples ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Testing: {len(test_data)} samples ({test_data.index.min()} to {test_data.index.max()})")

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

    # Store models
    trained_models[pair] = {
        'model_high': model_high,
        'model_low': model_low,
        'feature_cols': feature_cols,
        'train_samples': len(train_data),
        'test_samples': len(test_data)
    }

    print(f"  Models trained")
    print()

# Save models
print("Saving models...")

# Save combined models file (for convenience)
with open('models_15m.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("Models saved to: models_15m.pkl")

# Also save individual model files (for production trader)
import os
os.makedirs('models', exist_ok=True)

for pair, model_dict in trained_models.items():
    # Save high breakout model
    high_path = f'models/xgboost_15m_{pair}_high.pkl'
    with open(high_path, 'wb') as f:
        pickle.dump(model_dict['model_high'], f)

    # Save low breakout model
    low_path = f'models/xgboost_15m_{pair}_low.pkl'
    with open(low_path, 'wb') as f:
        pickle.dump(model_dict['model_low'], f)

    print(f"  Saved: {high_path} and {low_path}")

print("Individual model files saved to: models/ directory")
print()

# Generate predictions on test set
print("="*100)
print("GENERATING TEST PREDICTIONS")
print("="*100)
print()

test_predictions = {}

for pair in PAIRS:
    df = all_data[pair]
    df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    split_idx = int(len(df_clean) * 0.7)
    test_data = df_clean.iloc[split_idx:]

    X_test = test_data[feature_cols]

    model_high = trained_models[pair]['model_high']
    model_low = trained_models[pair]['model_low']

    # Predict probabilities
    pred_high = model_high.predict_proba(X_test)[:, 1]
    pred_low = model_low.predict_proba(X_test)[:, 1]

    # Create predictions dataframe
    preds_df = pd.DataFrame({
        'breakout_high_prob': pred_high,
        'breakout_low_prob': pred_low,
        f'high_{LOOKBACK_PERIOD}p': test_data[f'high_{LOOKBACK_PERIOD}p'],
        f'low_{LOOKBACK_PERIOD}p': test_data[f'low_{LOOKBACK_PERIOD}p'],
        'close': test_data['close']
    }, index=test_data.index)

    test_predictions[pair] = preds_df

    print(f"{pair}:")
    print(f"  Predictions: {len(preds_df)}")
    print(f"  Date range: {preds_df.index.min()} to {preds_df.index.max()}")
    print(f"  Avg high prob: {pred_high.mean():.3f}")
    print(f"  Avg low prob: {pred_low.mean():.3f}")
    print()

# Save test predictions
print("Saving test predictions...")
with open('test_predictions_15m.pkl', 'wb') as f:
    pickle.dump(test_predictions, f)
print("Predictions saved to: test_predictions_15m.pkl")
print()

print("="*100)
print("DONE!")
print("="*100)
print()
print("Next steps:")
print("  1. Run backtest: python backtest_15m.py")
print("  2. Compare with 1h strategy performance")
print("  3. Optimize parameters if needed")
