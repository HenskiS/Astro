"""
GENERATE PREDICTIONS - WEEKLY RETRAINING
=========================================
Train models with weekly retraining (every 7 days).

CRITICAL: 10-day gap between train and test to prevent target leakage.
Since we predict 10-day breakouts, targets at day T use data from T+1 to T+10.
We must ensure no overlap between training targets and test period.
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

print("="*100)
print("GENERATING MODEL PREDICTIONS - WEEKLY RETRAINING")
print("="*100)
print()

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# CRITICAL: 10-day gap between train end and test start
GAP_DAYS = 10

print(f"IMPORTANT: Using {GAP_DAYS}-day gap between train and test")
print("This ensures no leakage from 10-day forward-looking targets")
print()

# Generate weekly periods (2016-2025)
# Train on last 6 years, leave 10-day gap, test for 1 week
TEST_PERIODS = []

start_date = pd.Timestamp('2016-01-01')
end_date = pd.Timestamp('2025-12-31')
current_date = start_date

week_num = 0
while current_date < end_date:
    week_num += 1

    # Test period: 7 days starting from current_date
    test_start = current_date
    test_end = current_date + pd.Timedelta(days=6)

    # Train period: 6 years of data, ending 10 days before test start
    train_end = test_start - pd.Timedelta(days=GAP_DAYS + 1)  # +1 to ensure full 10-day gap
    train_start = train_end - pd.Timedelta(days=365*6)

    TEST_PERIODS.append({
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'test_start': test_start.strftime('%Y-%m-%d'),
        'test_end': test_end.strftime('%Y-%m-%d'),
        'name': f'{test_start.year}W{week_num:02d}',
        'gap_days': (test_start - train_end).days - 1
    })

    # Move to next week
    current_date = test_end + pd.Timedelta(days=1)

print(f"Total weeks to process: {len(TEST_PERIODS)}")
print(f"Date range: {TEST_PERIODS[0]['train_start']} to {TEST_PERIODS[-1]['test_end']}")
print()

# Verify all periods have proper gap
print("Verifying 10-day gaps...")
gap_violations = 0
for period in TEST_PERIODS:
    if period['gap_days'] < GAP_DAYS:
        print(f"  WARNING: {period['name']} has only {period['gap_days']}-day gap")
        gap_violations += 1

if gap_violations == 0:
    print(f"  VERIFIED: All {len(TEST_PERIODS)} periods have proper {GAP_DAYS}-day gap")
else:
    print(f"  ERROR: {gap_violations} periods have insufficient gap")
    exit(1)

print()

XGB_CONFIG = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

def calculate_technical_features(df):
    """Calculate technical features"""
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df

def create_targets(df):
    """Create breakout targets - these look 10 days forward"""
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    future_low_10d = df['low'].rolling(10).min().shift(-10)

    current_high_20d = df['high'].rolling(20).max()
    current_low_20d = df['low'].rolling(20).min()

    df['target_breakout_high'] = (future_high_10d > current_high_20d).astype(int)
    df['target_breakout_low'] = (future_low_10d < current_low_20d).astype(int)

    return df

def train_models_for_period(pair_data, train_start, train_end):
    """Train models for a specific pair and time period"""
    df = pair_data.copy()
    df = calculate_technical_features(df)
    df = create_targets(df)

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    train_df = train_df.dropna()

    if len(train_df) < 1000:
        return None

    feature_cols = [col for col in train_df.columns if col not in
                   ['target_breakout_high', 'target_breakout_low',
                    'open', 'high', 'low', 'close', 'volume']]

    X_train = train_df[feature_cols]
    models = {}

    y_high = train_df['target_breakout_high']
    if y_high.sum() > 100:
        model_high = XGBClassifier(**XGB_CONFIG)
        model_high.fit(X_train, y_high, verbose=False)
        models['breakout_high'] = {'model': model_high, 'features': feature_cols}

    y_low = train_df['target_breakout_low']
    if y_low.sum() > 100:
        model_low = XGBClassifier(**XGB_CONFIG)
        model_low.fit(X_train, y_low, verbose=False)
        models['breakout_low'] = {'model': model_low, 'features': feature_cols}

    return models if len(models) == 2 else None

# Load raw data
print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = os.path.join(DATA_DIR, f'{pair}_1day_with_spreads.csv')
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print()

# Sample every 4th week for faster processing (monthly-ish)
# Comment out this line to train all weeks
TEST_PERIODS = [p for i, p in enumerate(TEST_PERIODS) if i % 4 == 0]
print(f"Sampling to {len(TEST_PERIODS)} periods (every 4th week for speed)")
print()

# Train models and generate predictions
all_predictions = {}
weeks_processed = 0

for period in TEST_PERIODS:
    weeks_processed += 1
    print(f"[{weeks_processed}/{len(TEST_PERIODS)}] Processing {period['name']}...")

    period_models = {}
    for pair in PAIRS:
        models = train_models_for_period(all_raw_data[pair], period['train_start'], period['train_end'])
        if models:
            period_models[pair] = models

    if len(period_models) == 0:
        print(f"  SKIP - no models trained")
        continue

    period_predictions = {}

    for pair in PAIRS:
        if pair not in period_models:
            continue

        df = all_raw_data[pair].copy()
        df = calculate_technical_features(df)

        test_df = df[(df.index >= period['test_start']) & (df.index <= period['test_end'])].copy()
        test_df = test_df.dropna()

        if len(test_df) == 0:
            continue

        models = period_models[pair]
        feature_cols = models['breakout_high']['features']
        X_test = test_df[feature_cols]

        breakout_high_probs = models['breakout_high']['model'].predict_proba(X_test)[:, 1]
        breakout_low_probs = models['breakout_low']['model'].predict_proba(X_test)[:, 1]

        predictions_df = pd.DataFrame({
            'date': test_df.index,
            'open': test_df['open'],
            'high': test_df['high'],
            'low': test_df['low'],
            'close': test_df['close'],
            'high_20d': test_df['high_20d'],
            'low_20d': test_df['low_20d'],
            'breakout_high_prob': breakout_high_probs,
            'breakout_low_prob': breakout_low_probs
        })

        period_predictions[pair] = predictions_df

    if len(period_predictions) > 0:
        all_predictions[period['name']] = period_predictions
        total_preds = sum(len(df) for df in period_predictions.values())
        print(f"  {len(period_predictions)} pairs, {total_preds} predictions")

print()

# Save predictions
output_file = 'model_predictions_weekly.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(all_predictions, f)

print("="*100)
print(f"Weekly predictions saved to {output_file}")
print()
print(f"Processed {len(all_predictions)} periods with {GAP_DAYS}-day gap")
print()
print("="*100)
