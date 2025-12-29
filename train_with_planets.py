"""
Train XGBoost models WITH planetary features and compare against baseline
Tests if geocentric planetary positions correlate with forex movements
"""
import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING FOREX MODELS WITH PLANETARY FEATURES")
print("Comparing: Technical Only vs Technical + Planetary")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'models'
TEST_SPLIT_DATE = '2024-01-01'
PREDICTION_HORIZON = 3  # 3-day forward returns

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  Model: XGBoost (per-pair)")
print(f"  Prediction horizon: {PREDICTION_HORIZON} days")
print(f"  Train/Test split: {TEST_SPLIT_DATE}")
print()


def calculate_features(df):
    """Calculate technical features (same as original script)"""
    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # EMAs
    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    return df


# Load all pairs with planetary features
print("Loading pairs with planetary features...")
all_data = {}

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f'{pair}_1day_with_planets.csv')

    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate technical features
        df = calculate_features(df)
        df = df.dropna()

        # Calculate target
        df[f'target_return_{PREDICTION_HORIZON}d'] = df['close'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
        df['target'] = (df[f'target_return_{PREDICTION_HORIZON}d'] > 0).astype(int)

        all_data[pair] = df
        print(f"  {pair}: {len(df)} days")

    except Exception as e:
        print(f"  {pair}: ERROR - {str(e)}")
        continue

print()
print(f"Loaded {len(all_data)} pairs")
print()

# Add cross-pair features
print("Calculating cross-pair features...")

for date in all_data['EURUSD'].index:
    returns_on_date = []
    for pair, df in all_data.items():
        if date in df.index:
            returns_on_date.append(df.loc[date, 'return_1d'])

    avg_return = np.mean(returns_on_date) if len(returns_on_date) > 0 else 0

    for pair, df in all_data.items():
        if date in df.index:
            df.loc[date, 'market_sentiment'] = avg_return

for pair, df in all_data.items():
    df['relative_strength_vs_market'] = df['return_1d'] - df['market_sentiment']

print("Cross-pair features calculated")
print()

# Define feature sets
technical_features = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'atr_pct',
    'volatility_10d', 'volatility_20d',
    'bb_position',
    'momentum_10', 'momentum_20',
    'market_sentiment',
    'relative_strength_vs_market'
]

# Identify planetary features (all features not in technical_features or basic columns)
sample_df = all_data[list(all_data.keys())[0]]
basic_columns = ['open', 'high', 'low', 'close', 'volume', 'target',
                 f'target_return_{PREDICTION_HORIZON}d']
planetary_features = [col for col in sample_df.columns
                      if col not in technical_features
                      and col not in basic_columns
                      and not col.startswith('ema_')
                      and not col.startswith('bb_')]

print(f"Feature sets defined:")
print(f"  Technical features: {len(technical_features)}")
print(f"  Planetary features: {len(planetary_features)}")
print()

print("Sample planetary features:")
for feat in planetary_features[:10]:
    print(f"  - {feat}")
if len(planetary_features) > 10:
    print(f"  ... and {len(planetary_features) - 10} more")
print()

# XGBoost configuration
XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 20
}

# Train models for each pair - TWO VERSIONS
print("="*100)
print("TRAINING MODELS (BOTH VERSIONS)")
print("="*100)
print()

baseline_models = {}
planetary_models = {}
comparison_results = []

for pair in PAIRS:
    if pair not in all_data:
        continue

    print(f"\n{'='*100}")
    print(f"{pair}")
    print('='*100)

    df = all_data[pair].copy()

    # Remove rows with missing target
    df = df.dropna(subset=['target'])

    # Split train/test
    train_df = df[df.index < TEST_SPLIT_DATE]
    test_df = df[df.index >= TEST_SPLIT_DATE]

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    if len(train_df) < 100 or len(test_df) < 10:
        print(f"  SKIPPED: Not enough data")
        continue

    # Calculate scale_pos_weight
    y_train_temp = train_df['target'].values
    neg_count = (y_train_temp == 0).sum()
    pos_count = (y_train_temp == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # ===========================
    # MODEL 1: BASELINE (Technical Only)
    # ===========================
    print(f"\n  [1] BASELINE MODEL (Technical features only)")
    print(f"  {'-'*50}")

    # Check which features are available
    available_technical = [f for f in technical_features if f in train_df.columns]

    X_train_baseline = train_df[available_technical].dropna()
    y_train_baseline = train_df.loc[X_train_baseline.index, 'target'].values

    X_test_baseline = test_df[available_technical].dropna()
    y_test_baseline = test_df.loc[X_test_baseline.index, 'target'].values

    print(f"      Features: {len(available_technical)}")
    print(f"      Train samples: {len(X_train_baseline)}")
    print(f"      Test samples: {len(X_test_baseline)}")

    model_baseline = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model_baseline.fit(
        X_train_baseline, y_train_baseline,
        eval_set=[(X_test_baseline, y_test_baseline)],
        verbose=False
    )

    y_test_pred_baseline = model_baseline.predict(X_test_baseline)
    baseline_acc = accuracy_score(y_test_baseline, y_test_pred_baseline)

    print(f"      Test accuracy: {baseline_acc*100:.2f}%")

    baseline_models[pair] = {
        'model': model_baseline,
        'features': available_technical,
        'test_acc': baseline_acc
    }

    # ===========================
    # MODEL 2: WITH PLANETARY FEATURES
    # ===========================
    print(f"\n  [2] PLANETARY MODEL (Technical + Planetary)")
    print(f"  {'-'*50}")

    # Check which planetary features are available
    available_planetary = [f for f in planetary_features if f in train_df.columns]
    all_features = available_technical + available_planetary

    X_train_planets = train_df[all_features].dropna()
    y_train_planets = train_df.loc[X_train_planets.index, 'target'].values

    X_test_planets = test_df[all_features].dropna()
    y_test_planets = test_df.loc[X_test_planets.index, 'target'].values

    print(f"      Features: {len(all_features)} ({len(available_technical)} tech + {len(available_planetary)} planetary)")
    print(f"      Train samples: {len(X_train_planets)}")
    print(f"      Test samples: {len(X_test_planets)}")

    model_planets = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model_planets.fit(
        X_train_planets, y_train_planets,
        eval_set=[(X_test_planets, y_test_planets)],
        verbose=False
    )

    y_test_pred_planets = model_planets.predict(X_test_planets)
    planets_acc = accuracy_score(y_test_planets, y_test_pred_planets)

    print(f"      Test accuracy: {planets_acc*100:.2f}%")

    # Feature importance for planetary features
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model_planets.feature_importances_
    }).sort_values('importance', ascending=False)

    top_planetary = feature_importance[
        feature_importance['feature'].isin(available_planetary)
    ].head(5)

    if len(top_planetary) > 0:
        print(f"\n      Top 5 planetary features:")
        for idx, row in top_planetary.iterrows():
            print(f"        {row['feature']:40s}: {row['importance']:.4f}")

    planetary_models[pair] = {
        'model': model_planets,
        'features': all_features,
        'test_acc': planets_acc
    }

    # ===========================
    # COMPARISON
    # ===========================
    improvement = planets_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  COMPARISON:")
    print(f"  {'='*50}")
    print(f"    Baseline (Technical):     {baseline_acc*100:>6.2f}%")
    print(f"    With Planets:             {planets_acc*100:>6.2f}%")
    print(f"    Improvement:              {improvement*100:>+6.2f}% ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print(f"    Status: ✓ PLANETS HELP!")
    elif improvement < 0:
        print(f"    Status: ✗ Planets hurt performance")
    else:
        print(f"    Status: = No difference")

    comparison_results.append({
        'pair': pair,
        'baseline_acc': baseline_acc,
        'planets_acc': planets_acc,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'test_samples': len(X_test_baseline)
    })

# ===========================
# OVERALL SUMMARY
# ===========================
print("\n\n" + "="*100)
print("FINAL COMPARISON SUMMARY")
print("="*100)
print()

results_df = pd.DataFrame(comparison_results)

print(f"{'Pair':<10} {'Baseline':>12} {'W/ Planets':>12} {'Improvement':>15} {'Test Samples':>15}")
print("-" * 100)

for _, row in results_df.iterrows():
    symbol = '✓' if row['improvement'] > 0 else ('✗' if row['improvement'] < 0 else '=')
    print(f"{row['pair']:<10} {row['baseline_acc']*100:>11.2f}% {row['planets_acc']*100:>11.2f}% "
          f"{row['improvement']*100:>+10.2f}% {symbol:>3s} {row['test_samples']:>15,}")

print()
avg_baseline = results_df['baseline_acc'].mean()
avg_planets = results_df['planets_acc'].mean()
avg_improvement = avg_planets - avg_baseline

print(f"AVERAGE PERFORMANCE:")
print(f"  Baseline (Technical only):  {avg_baseline*100:.2f}%")
print(f"  With Planetary features:    {avg_planets*100:.2f}%")
print(f"  Average improvement:        {avg_improvement*100:+.2f}%")
print()

pairs_improved = (results_df['improvement'] > 0).sum()
pairs_worse = (results_df['improvement'] < 0).sum()
pairs_same = (results_df['improvement'] == 0).sum()

print(f"PLANETARY FEATURE IMPACT:")
print(f"  Pairs improved: {pairs_improved}/{len(results_df)}")
print(f"  Pairs worse: {pairs_worse}/{len(results_df)}")
print(f"  Pairs same: {pairs_same}/{len(results_df)}")
print()

if avg_improvement > 0:
    print(f"CONCLUSION: Planetary features provide a {avg_improvement*100:.2f}% average boost! 🌟")
elif avg_improvement < 0:
    print(f"CONCLUSION: Planetary features hurt performance by {abs(avg_improvement)*100:.2f}%")
else:
    print(f"CONCLUSION: Planetary features show no clear impact")

print()

# Save models
baseline_file = os.path.join(OUTPUT_DIR, 'forex_baseline_models.pkl')
planets_file = os.path.join(OUTPUT_DIR, 'forex_with_planets_models.pkl')

with open(baseline_file, 'wb') as f:
    pickle.dump({
        'models': baseline_models,
        'features': 'technical_only',
        'prediction_horizon': PREDICTION_HORIZON
    }, f)

with open(planets_file, 'wb') as f:
    pickle.dump({
        'models': planetary_models,
        'features': 'technical_and_planetary',
        'prediction_horizon': PREDICTION_HORIZON
    }, f)

print(f"Models saved:")
print(f"  Baseline: {baseline_file}")
print(f"  Planetary: {planets_file}")
print()

print("="*100)
print("TRAINING COMPLETE")
print("="*100)
