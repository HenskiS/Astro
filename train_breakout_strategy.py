"""
Train models for the two most predictable patterns:
1. 10-day breakout prediction (when to trade breakouts)
2. 5-day range prediction (when to trade ranges)

Then backtest a combined strategy
"""
import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BREAKOUT vs RANGE TRADING STRATEGY")
print("Training models for the two most predictable patterns")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'models'
TEST_SPLIT_DATE = '2024-01-01'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Range features
    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['close']
    df['position_in_range'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

    return df


def create_targets(df):
    """Create the two key prediction targets"""

    # Target 1: Will price break above 20-day high in next 10 days? (UPWARD BREAKOUT)
    future_high_10d = df['high'].rolling(10).max().shift(-10)
    current_high_20d = df['high'].rolling(20).max()
    df['target_breakout_high_10d'] = (future_high_10d > current_high_20d).astype(int)

    # Target 2: Will price break below 20-day low in next 10 days? (DOWNWARD BREAKOUT)
    future_low_10d = df['low'].rolling(10).min().shift(-10)
    current_low_20d = df['low'].rolling(20).min()
    df['target_breakout_low_10d'] = (future_low_10d < current_low_20d).astype(int)

    # Target 3: Will price stay within 20-day range for next 5 days? (RANGE-BOUND)
    future_high_5d = df['high'].rolling(5).max().shift(-5)
    future_low_5d = df['low'].rolling(5).min().shift(-5)
    df['target_stay_in_range_5d'] = (
        (future_high_5d <= current_high_20d) &
        (future_low_5d >= current_low_20d)
    ).astype(int)

    # For backtesting: actual 10-day high/low breakout returns
    df['future_high_10d'] = df['high'].rolling(10).max().shift(-10)
    df['future_low_10d'] = df['low'].rolling(10).min().shift(-10)

    return df


technical_features = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'atr_pct',
    'volatility_10d', 'volatility_20d',
    'bb_position',
    'momentum_10', 'momentum_20',
    'range_20d', 'position_in_range'
]

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
    'eval_metric': 'logloss',
    'early_stopping_rounds': 30
}

# Load and prepare data
print("Loading data...")
all_data = {}

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df = calculate_technical_features(df)
        df = create_targets(df)
        df = df.dropna()

        all_data[pair] = df
        print(f"  {pair}: {len(df)} days")

    except Exception as e:
        print(f"  {pair}: ERROR - {str(e)}")
        continue

print()
print(f"Loaded {len(all_data)} pairs")
print()

# Train models for each pair
print("="*100)
print("TRAINING MODELS")
print("="*100)
print()

trained_models = {}

for pair in PAIRS:
    if pair not in all_data:
        continue

    print(f"\n{pair}")
    print("-" * 80)

    df = all_data[pair].copy()
    train_df = df[df.index < TEST_SPLIT_DATE]
    test_df = df[df.index >= TEST_SPLIT_DATE]

    print(f"  Data: {len(train_df)} train / {len(test_df)} test")

    trained_models[pair] = {}

    # Model 1: Upward Breakout Predictor
    print(f"\n  [1] UPWARD BREAKOUT (10-day)")

    target = 'target_breakout_high_10d'
    available_features = [f for f in technical_features if f in train_df.columns]

    train_subset = train_df[available_features + [target]].dropna()
    test_subset = test_df[available_features + [target]].dropna()

    X_train = train_subset[available_features].values
    y_train = train_subset[target].values
    X_test = test_subset[available_features].values
    y_test = test_subset[target].values

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    baseline = 1 - y_test.mean()  # Predict no breakout
    improvement = acc - baseline

    print(f"      Accuracy: {acc*100:.2f}% (baseline: {baseline*100:.2f}%, +{improvement*100:.2f}%)")
    print(f"      Precision: {prec:.3f}, Recall: {rec:.3f}")

    trained_models[pair]['breakout_high'] = {
        'model': model,
        'features': available_features,
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    }

    # Model 2: Downward Breakout Predictor
    print(f"\n  [2] DOWNWARD BREAKOUT (10-day)")

    target = 'target_breakout_low_10d'
    train_subset = train_df[available_features + [target]].dropna()
    test_subset = test_df[available_features + [target]].dropna()

    X_train = train_subset[available_features].values
    y_train = train_subset[target].values
    X_test = test_subset[available_features].values
    y_test = test_subset[target].values

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    baseline = 1 - y_test.mean()
    improvement = acc - baseline

    print(f"      Accuracy: {acc*100:.2f}% (baseline: {baseline*100:.2f}%, +{improvement*100:.2f}%)")
    print(f"      Precision: {prec:.3f}, Recall: {rec:.3f}")

    trained_models[pair]['breakout_low'] = {
        'model': model,
        'features': available_features,
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    }

    # Model 3: Range-Bound Predictor
    print(f"\n  [3] RANGE-BOUND (5-day)")

    target = 'target_stay_in_range_5d'
    train_subset = train_df[available_features + [target]].dropna()
    test_subset = test_df[available_features + [target]].dropna()

    X_train = train_subset[available_features].values
    y_train = train_subset[target].values
    X_test = test_subset[available_features].values
    y_test = test_subset[target].values

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    baseline = 1 - y_test.mean()
    improvement = acc - baseline

    print(f"      Accuracy: {acc*100:.2f}% (baseline: {baseline*100:.2f}%, +{improvement*100:.2f}%)")
    print(f"      Precision: {prec:.3f}, Recall: {rec:.3f}")

    trained_models[pair]['range_bound'] = {
        'model': model,
        'features': available_features,
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    }

    # Save models
    model_file = os.path.join(OUTPUT_DIR, f'{pair}_breakout_strategy.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(trained_models[pair], f)
    print(f"\n  Saved: {model_file}")

print()
print("="*100)
print("TRAINING COMPLETE")
print("="*100)
print()

print("Summary:")
print()
print(f"{'Pair':<10} {'Upward BO':>12} {'Downward BO':>12} {'Range':>12}")
print("-" * 80)

for pair in PAIRS:
    if pair in trained_models:
        up = trained_models[pair]['breakout_high']['accuracy']
        down = trained_models[pair]['breakout_low']['accuracy']
        range_acc = trained_models[pair]['range_bound']['accuracy']
        print(f"{pair:<10} {up*100:>11.2f}% {down*100:>11.2f}% {range_acc*100:>11.2f}%")

print()
print("="*100)
print("TRADING STRATEGY")
print("="*100)
print()

print("How to use these models:")
print()
print("1. UPWARD BREAKOUT MODEL (Precision is key)")
print("   - When model predicts breakout with high confidence:")
print("   - Enter LONG position at market open")
print("   - Set stop-loss at 20-day low (the support)")
print("   - Target: New high within 10 days")
print("   - Risk/Reward: Typically 1:2 or better")
print()

print("2. DOWNWARD BREAKOUT MODEL (Precision is key)")
print("   - When model predicts breakout with high confidence:")
print("   - Enter SHORT position at market open")
print("   - Set stop-loss at 20-day high (the resistance)")
print("   - Target: New low within 10 days")
print("   - Risk/Reward: Typically 1:2 or better")
print()

print("3. RANGE-BOUND MODEL (For mean reversion)")
print("   - When model predicts range-bound with high confidence:")
print("   - Wait for price near range extremes")
print("   - SELL at 20-day high (fade resistance)")
print("   - BUY at 20-day low (fade support)")
print("   - Target: Return to middle of range")
print("   - Stop: If breaks range (switch to breakout mode)")
print()

print("4. COMBINED STRATEGY")
print("   - Check ALL three models each day")
print("   - If BOTH breakout models agree on direction: STRONG TREND SIGNAL")
print("   - If range model says range-bound: TRADE MEAN REVERSION")
print("   - If models disagree: STAY OUT (unclear regime)")
print()

print("5. POSITION SIZING")
print("   - Higher precision = larger position")
print("   - Use model probability as confidence score")
print("   - Example: 70% confidence = 0.5% risk, 90% confidence = 1% risk")
print()

print("6. PORTFOLIO APPROACH")
print("   - Run models on all 8 pairs daily")
print("   - Take top 2-3 highest-confidence setups")
print("   - Diversify across different pairs")
print("   - Max 20% of capital per pair")
print()

print("="*100)
print("Next step: Backtest this strategy on test set")
print("="*100)
