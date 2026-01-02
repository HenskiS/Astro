"""
TRAIN 1-HOUR BREAKOUT MODEL
============================
Adapts the daily breakout strategy for 1-hour timeframe with intraday features
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from datetime import datetime, timedelta
import pickle
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING 1-HOUR BREAKOUT MODEL")
print("="*100)
print()

# Parameters for 1h timeframe
LOOKBACK_PERIOD = 80  # ~3.3 days of hourly data (was 20 days for daily)
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']
DATA_DIR = 'data_1h'

def add_time_features(df):
    """Add intraday-specific features"""
    df = df.copy()

    # Hour of day (0-23)
    df['hour'] = df.index.hour

    # Day of week (0-6)
    df['day_of_week'] = df.index.dayofweek

    # Trading session indicators
    # Asian: 0-8 UTC, European: 7-16 UTC, US: 13-22 UTC
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1

    # Weekend proximity (Friday evening, Sunday evening)
    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)

    return df


def calculate_features(df, lookback=LOOKBACK_PERIOD):
    """Calculate technical features for 1h data"""
    df = df.copy()

    # Add time features first
    df = add_time_features(df)

    # Breakout levels (using lookback period)
    df[f'high_{lookback}h'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}h'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}h'] = df[f'high_{lookback}h'] - df[f'low_{lookback}h']

    # Distance to breakout levels
    df['dist_to_high'] = (df[f'high_{lookback}h'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}h']) / df['close']

    # EMAs (adjusted for hourly data)
    df['ema_12'] = df['close'].ewm(span=12).mean()  # ~0.5 day
    df['ema_26'] = df['close'].ewm(span=26).mean()  # ~1 day
    df['ema_50'] = df['close'].ewm(span=50).mean()  # ~2 days
    df['ema_100'] = df['close'].ewm(span=100).mean()  # ~4 days

    # Price relative to EMAs
    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema26'] = (df['close'] - df['ema_26']) / df['close']
    df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['price_vs_ema100'] = (df['close'] - df['ema_100']) / df['close']

    # MACD (adjusted for hourly)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14 hour period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Volatility (ATR for hourly data)
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
    df['volume_ma'] = df['volume'].rolling(24).mean()  # 24h average
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Recent momentum (shorter periods for intraday)
    df['return_1h'] = df['close'].pct_change(1)
    df['return_4h'] = df['close'].pct_change(4)
    df['return_12h'] = df['close'].pct_change(12)
    df['return_24h'] = df['close'].pct_change(24)

    # Spread analysis
    df['spread_ma'] = df['spread_pct'].rolling(24).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=24):
    """
    Calculate breakout targets for next 24 hours (1 day)
    """
    df = df.copy()

    # Future high/low over next 24 hours
    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()

    # Current breakout levels
    high_level = df[f'high_{lookback}h']
    low_level = df[f'low_{lookback}h']

    # Did price break out in the next 24 hours?
    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)

    return df


# Load and prepare data
print("Loading 1h data...")
all_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1h.csv'
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
    df = calculate_targets(df, LOOKBACK_PERIOD, forward_periods=24)
    all_data[pair] = df
    print(f"  {pair}: {len(df.columns)} features")

print()

# Define feature columns
feature_cols = [
    # Breakout features
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}h',
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
    'return_1h', 'return_4h', 'return_12h', 'return_24h',
    # Spread
    'spread_pct', 'spread_ratio',
    # Time features
    'hour', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

print(f"Using {len(feature_cols)} features:")
for f in feature_cols:
    print(f"  - {f}")
print()

# Train models using monthly rolling window
print("="*100)
print("TRAINING MODELS (MONTHLY ROLLING)")
print("="*100)
print()

trained_models = {}

# Use first 70% of data for training, rest for testing
for pair in PAIRS:
    df = all_data[pair]

    # Drop NaN rows
    df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    # Split train/test (70/30)
    split_idx = int(len(df_clean) * 0.7)
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]

    print(f"{pair}:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Testing samples: {len(test_data)}")

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

    print(f"  Models trained successfully")
    print()

# Save models
print("Saving models...")
with open('models_1h.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("Models saved to: models_1h.pkl")
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
        f'high_{LOOKBACK_PERIOD}h': test_data[f'high_{LOOKBACK_PERIOD}h'],
        f'low_{LOOKBACK_PERIOD}h': test_data[f'low_{LOOKBACK_PERIOD}h'],
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
with open('test_predictions_1h.pkl', 'wb') as f:
    pickle.dump(test_predictions, f)
print("Predictions saved to: test_predictions_1h.pkl")
print()

print("="*100)
print("DONE!")
print("="*100)
print()
print("Next steps:")
print("  1. Run backtest: python backtest_1h.py")
print("  2. Optimize parameters for 1h timeframe")
print("  3. Compare results with daily strategy")
