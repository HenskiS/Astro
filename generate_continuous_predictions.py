"""
GENERATE CONTINUOUS HISTORICAL PREDICTIONS
===========================================
Simulates production trading by:
1. Using a 10-month rolling training window
2. Retraining models every month
3. Generating predictions for the next month
4. No lookahead bias - models only see past data

This creates continuous predictions for the entire historical period.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

print("="*100)
print("GENERATING CONTINUOUS HISTORICAL PREDICTIONS")
print("="*100)
print()

# Configuration
LOOKBACK_PERIOD = 80
FORWARD_PERIODS = 24
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
DATA_DIR = 'data_15m'

TRAIN_WINDOW_MONTHS = 10  # Use 10 months of data for training
RETRAIN_FREQUENCY_MONTHS = 1  # Retrain every month

# Start date: Need 10 months of training data first
START_TRAIN_DATE = '2020-01-01'
START_PREDICT_DATE = '2020-11-01'  # Start making predictions after 10 months
END_DATE = '2026-02-01'  # End of our data (extended to full range)


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
    """Calculate technical indicators - ONLY uses past data"""
    df = df.copy()
    df = add_time_features(df)

    # Range features
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

    # Returns
    df['return_1p'] = df['close'].pct_change(1)
    df['return_4p'] = df['close'].pct_change(4)
    df['return_16p'] = df['close'].pct_change(16)
    df['return_96p'] = df['close'].pct_change(96)

    # Spread
    df['spread_ma'] = df['spread_pct'].rolling(96).mean()
    df['spread_ratio'] = df['spread_pct'] / df['spread_ma']

    return df


def calculate_targets(df, lookback=LOOKBACK_PERIOD, forward_periods=FORWARD_PERIODS):
    """Calculate forward-looking targets for training"""
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


print("Loading and preparing data...")
print("-" * 100)

# Load all data
all_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate features and targets for the FULL dataset
    # (We'll only use what we're "allowed" to see at each point in time)
    df = calculate_features(df, LOOKBACK_PERIOD)
    df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)
    df = df.dropna(subset=feature_cols + ['breakout_high', 'breakout_low'])

    all_data[pair] = df
    print(f"  {pair}: {len(df):,} samples from {df.index.min().date()} to {df.index.max().date()}")

print()


print("="*100)
print("GENERATING PREDICTIONS WITH ROLLING WINDOW")
print("="*100)
print()
print(f"Training window: {TRAIN_WINDOW_MONTHS} months")
print(f"Retrain frequency: Every {RETRAIN_FREQUENCY_MONTHS} month(s)")
print(f"Start date: {START_PREDICT_DATE}")
print(f"End date: {END_DATE}")
print()


# Store all predictions
all_predictions = {pair: [] for pair in PAIRS}

# Rolling window prediction
current_date = pd.Timestamp(START_PREDICT_DATE, tz='UTC')
end_date = pd.Timestamp(END_DATE, tz='UTC')
iteration = 0

while current_date < end_date:
    iteration += 1

    # Calculate training window (10 months before current_date)
    train_end = current_date
    train_start = current_date - relativedelta(months=TRAIN_WINDOW_MONTHS)

    # Calculate prediction window (next 1 month)
    predict_start = current_date
    predict_end = current_date + relativedelta(months=RETRAIN_FREQUENCY_MONTHS)

    print(f"Iteration {iteration}:")
    print(f"  Train:   {train_start.date()} to {train_end.date()} ({TRAIN_WINDOW_MONTHS} months)")
    print(f"  Predict: {predict_start.date()} to {predict_end.date()} ({RETRAIN_FREQUENCY_MONTHS} month)")

    # Train models for each pair
    for pair in PAIRS:
        df = all_data[pair]

        # Get training data (ONLY data before predict_start)
        train_df = df[(df.index >= train_start) & (df.index < train_end)]

        # Get prediction data
        predict_df = df[(df.index >= predict_start) & (df.index < predict_end)]

        if len(train_df) < 1000:
            print(f"    {pair}: Insufficient training data ({len(train_df)} samples)")
            continue

        if len(predict_df) == 0:
            print(f"    {pair}: No data to predict")
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

        # Generate predictions
        X_predict = predict_df[feature_cols]
        pred_high = model_high.predict_proba(X_predict)[:, 1]
        pred_low = model_low.predict_proba(X_predict)[:, 1]

        # Store predictions
        preds_df = pd.DataFrame({
            'breakout_high_prob': pred_high,
            'breakout_low_prob': pred_low,
            f'high_{LOOKBACK_PERIOD}p': predict_df[f'high_{LOOKBACK_PERIOD}p'],
            f'low_{LOOKBACK_PERIOD}p': predict_df[f'low_{LOOKBACK_PERIOD}p'],
            'close': predict_df['close']
        }, index=predict_df.index)

        all_predictions[pair].append(preds_df)

        print(f"    {pair}: {len(preds_df):,} predictions")

    print()

    # Move to next period
    current_date = predict_end


print("="*100)
print("COMBINING AND SAVING PREDICTIONS")
print("="*100)
print()

# Combine all predictions for each pair
final_predictions = {}
for pair in PAIRS:
    if all_predictions[pair]:
        combined = pd.concat(all_predictions[pair]).sort_index()
        final_predictions[pair] = combined
        print(f"{pair}: {len(combined):,} total predictions")
        print(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")

# Save
with open('test_predictions_15m_continuous.pkl', 'wb') as f:
    pickle.dump(final_predictions, f)

print()
print("Saved: test_predictions_15m_continuous.pkl")
print()

print("="*100)
print("VERIFICATION")
print("="*100)
print()

# Verify no gaps in predictions
for pair in PAIRS:
    preds = final_predictions[pair]

    # Check for date gaps (should be 15min intervals)
    time_diffs = preds.index.to_series().diff()
    expected_diff = pd.Timedelta(minutes=15)

    # Allow for weekend gaps (> 2 days is normal)
    large_gaps = time_diffs[time_diffs > pd.Timedelta(days=2)]

    if len(large_gaps) > 0:
        print(f"{pair}: Found {len(large_gaps)} gaps > 2 days (likely weekends/holidays)")
    else:
        print(f"{pair}: No unexpected gaps")

print()
print("="*100)
print("DONE!")
print("="*100)
print()
print("Next steps:")
print("  1. Update plot_10pct_equity_curve.py to use 'test_predictions_15m_continuous.pkl'")
print("  2. Run backtest with continuous predictions")
print()
