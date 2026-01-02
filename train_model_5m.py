"""
TRAIN 5-MINUTE BREAKOUT MODEL
==============================
Extreme high-frequency adaptation

Key adaptations:
- Lookback: 240 periods = 20 hours (vs 80 periods for 15m)
- Forward target: 72 periods = 6 hours (vs 24 periods for 15m)
- Same intraday features
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING 5-MINUTE BREAKOUT MODEL")
print("="*100)
print()

# Parameters for 5m timeframe
LOOKBACK_PERIOD = 240  # 20 hours (240 * 5m)
FORWARD_PERIODS = 72   # 6 hours (72 * 5m)
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']  # 8 pairs
DATA_DIR = 'data_5m'

def add_time_features(df):
    """Add intraday features"""
    df = df.copy()

    df['hour'] = df.index.hour
    df['minute_slot'] = df.index.minute // 5
    df['day_of_week'] = df.index.dayofweek

    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = (df['european_session'] + df['us_session']) > 1

    df['friday_close'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
    df['sunday_open'] = ((df['day_of_week'] == 6) & (df['hour'] <= 3)).astype(int)

    return df


def calculate_features(df, lookback=LOOKBACK_PERIOD):
    """Calculate features for 5m data"""
    df = df.copy()

    df = add_time_features(df)

    # Breakout levels (20 hours)
    df[f'high_{lookback}p'] = df['high'].rolling(lookback).max()
    df[f'low_{lookback}p'] = df['low'].rolling(lookback).min()
    df[f'range_{lookback}p'] = df[f'high_{lookback}p'] - df[f'low_{lookback}p']

    df['dist_to_high'] = (df[f'high_{lookback}p'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df[f'low_{lookback}p']) / df['close']

    # EMAs (adjusted for 5m)
    df['ema_12'] = df['close'].ewm(span=12).mean()   # 1 hour
    df['ema_36'] = df['close'].ewm(span=36).mean()   # 3 hours
    df['ema_60'] = df['close'].ewm(span=60).mean()   # 5 hours
    df['ema_120'] = df['close'].ewm(span=120).mean() # 10 hours

    df['price_vs_ema12'] = (df['close'] - df['ema_12']) / df['close']
    df['price_vs_ema36'] = (df['close'] - df['ema_36']) / df['close']
    df['price_vs_ema60'] = (df['close'] - df['ema_60']) / df['close']
    df['price_vs_ema120'] = (df['close'] - df['ema_120']) / df['close']

    # MACD
    df['macd'] = df['ema_12'] - df['ema_36']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14 periods = 1.17 hours)
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
    df['volume_ma'] = df['volume'].rolling(288).mean()  # 24h (288 * 5m)
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Momentum
    df['return_1p'] = df['close'].pct_change(1)    # 5m
    df['return_12p'] = df['close'].pct_change(12)  # 1h
    df['return_48p'] = df['close'].pct_change(48)  # 4h
    df['return_288p'] = df['close'].pct_change(288) # 24h

    # Spread
    df['spread_ma'] = df['spread_pct'].rolling(288).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    """Calculate targets for next 6 hours"""
    df = df.copy()

    df['future_high'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_low'] = df['low'].shift(-1).rolling(forward_periods).min()

    high_level = df[f'high_{lookback}p']
    low_level = df[f'low_{lookback}p']

    df['breakout_high'] = (df['future_high'] > high_level).astype(int)
    df['breakout_low'] = (df['future_low'] < low_level).astype(int)

    return df


# Load data
print("Loading 5m data...")
all_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_5m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_data[pair] = df
    print(f"  {pair}: {len(df)} candles from {df.index.min()} to {df.index.max()}")

print()

# Calculate features
print("Calculating features...")
for pair in PAIRS:
    df = all_data[pair]
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    all_data[pair] = df
    print(f"  {pair}: {len(df.columns)} features")

print()

# Define features
feature_cols = [
    'dist_to_high', 'dist_to_low', f'range_{LOOKBACK_PERIOD}p',
    'price_vs_ema12', 'price_vs_ema36', 'price_vs_ema60', 'price_vs_ema120',
    'macd', 'macd_signal', 'macd_hist',
    'rsi_14',
    'atr_pct',
    'volume_ratio',
    'return_1p', 'return_12p', 'return_48p', 'return_288p',
    'spread_pct', 'spread_ratio',
    'hour', 'minute_slot', 'day_of_week',
    'asian_session', 'european_session', 'us_session', 'session_overlap',
    'friday_close', 'sunday_open'
]

print(f"Using {len(feature_cols)} features")
print()

# Train models
print("="*100)
print("TRAINING MODELS")
print("="*100)
print()

trained_models = {}

for pair in PAIRS:
    df = all_data[pair]

    df_clean = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    # 70/30 split
    split_idx = int(len(df_clean) * 0.7)
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]

    print(f"{pair}:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Testing: {len(test_data)} samples")

    X_train = train_data[feature_cols]
    y_train_high = train_data['breakout_high']
    y_train_low = train_data['breakout_low']

    # Train models
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
        'feature_cols': feature_cols,
        'train_samples': len(train_data),
        'test_samples': len(test_data)
    }

    print(f"  Models trained")
    print()

# Save models
print("Saving models...")
with open('models_5m.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("Models saved to: models_5m.pkl")
print()

# Generate predictions
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

    pred_high = model_high.predict_proba(X_test)[:, 1]
    pred_low = model_low.predict_proba(X_test)[:, 1]

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

# Save predictions
print("Saving test predictions...")
with open('test_predictions_5m.pkl', 'wb') as f:
    pickle.dump(test_predictions, f)
print("Predictions saved to: test_predictions_5m.pkl")
print()

print("="*100)
print("DONE!")
print("="*100)
print()
print("Next: python backtest_5m.py")
