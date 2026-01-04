"""
WALK-FORWARD WITH FIXED TRAINING WINDOW
========================================
Uses a ROLLING window of fixed size (matching original 70% = ~10 months)
to see if training window size explains the performance difference.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')

print("="*100)
print("WALK-FORWARD VALIDATION: FIXED TRAINING WINDOW")
print("="*100)
print()

# Parameters
LOOKBACK_PERIOD = 80
FORWARD_PERIODS = 24
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

# Walk-forward parameters - MATCHING ORIGINAL APPROACH
TRAIN_MONTHS = 10  # Match original 70% (~10 months)
TRADE_WEEKS = 2    # Trade for 2 weeks, then retrain

print(f"Walk-Forward Configuration:")
print(f"  Training window: {TRAIN_MONTHS} months (FIXED - rolls forward)")
print(f"  Trading period: {TRADE_WEEKS} weeks")
print(f"  This matches the original training data amount!")
print()


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
    print(f"  {pair}: {len(df)} samples")

print()

# Create walk-forward windows with FIXED training size
all_dates = sorted(set.union(*[set(df.index) for df in all_data.values()]))
min_date = min(all_dates)
max_date = max(all_dates)

train_window_length = pd.DateOffset(months=TRAIN_MONTHS)
trade_period_length = pd.DateOffset(weeks=TRADE_WEEKS)

windows = []
# Start after we have enough training data
current_train_end = min_date + train_window_length
current_test_start = current_train_end
current_test_end = current_test_start + trade_period_length

while current_test_end <= max_date:
    # ROLLING window: always use last TRAIN_MONTHS months
    train_start = current_train_end - train_window_length

    windows.append({
        'train_start': train_start,  # ROLLS forward (not fixed at min_date)
        'train_end': current_train_end,
        'test_start': current_test_start,
        'test_end': current_test_end
    })

    # Move to next window
    current_train_end = current_test_end
    current_test_start = current_test_end
    current_test_end = current_test_start + trade_period_length

print(f"Created {len(windows)} walk-forward windows")
print(f"Each uses {TRAIN_MONTHS} months of training data (FIXED)")
print()

# Execute walk-forward validation
print("="*100)
print("WALK-FORWARD EXECUTION")
print("="*100)
print()

all_predictions = {}

for window_idx, window in enumerate(windows):
    print(f"Window {window_idx + 1}/{len(windows)}:")
    print(f"  Train: {window['train_start'].date()} to {window['train_end'].date()}")
    print(f"  Test:  {window['test_start'].date()} to {window['test_end'].date()}")

    # Calculate training window size
    train_days = (window['train_end'] - window['train_start']).days
    print(f"  Training window: {train_days} days")

    for pair in PAIRS:
        df = all_data[pair]

        train_df = df[(df.index >= window['train_start']) & (df.index < window['train_end'])]
        test_df = df[(df.index >= window['test_start']) & (df.index < window['test_end'])]

        if len(train_df) < 1000 or len(test_df) == 0:
            continue

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

        if pair not in all_predictions:
            all_predictions[pair] = []
        all_predictions[pair].append(preds_df)

    print()

# Combine predictions
print("Combining predictions...")
combined_predictions = {}
for pair in PAIRS:
    if pair in all_predictions and len(all_predictions[pair]) > 0:
        combined_predictions[pair] = pd.concat(all_predictions[pair])
        print(f"  {pair}: {len(combined_predictions[pair])} predictions")

print()

# Save
print("Saving fixed-window predictions...")
with open('test_predictions_15m_walkforward_fixed.pkl', 'wb') as f:
    pickle.dump(combined_predictions, f)
print("Saved: test_predictions_15m_walkforward_fixed.pkl")
print()

print("="*100)
print("DONE!")
print("="*100)
print()
print("This uses the SAME training window size as your original approach,")
print("but retrains every 2 weeks (like production would).")
print()
print("Run backtest to compare with original results.")
print()
