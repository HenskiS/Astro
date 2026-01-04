"""
Train XGBoost models with REAL astronomical planetary features
Uses actual planetary positions from NASA JPL ephemerides
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
print("TRAINING FOREX MODELS WITH REAL ASTRONOMICAL FEATURES")
print("Using NASA JPL Ephemerides (Actual Planetary Positions)")
print("Comparing: Technical Only vs Technical + Real Astronomy")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'models'
TEST_SPLIT_DATE = '2024-01-01'
PREDICTION_HORIZON = 3

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  Model: XGBoost (per-pair)")
print(f"  Prediction horizon: {PREDICTION_HORIZON} days")
print(f"  Train/Test split: {TEST_SPLIT_DATE}")
print(f"  Astronomical data: REAL (NASA JPL)")
print()


def calculate_features(df):
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

    return df


# Load all pairs with REAL planetary features
print("Loading pairs with REAL astronomical features...")
all_data = {}

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f'{pair}_1day_with_real_planets.csv')

    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df = calculate_features(df)
        df = df.dropna()

        df[f'target_return_{PREDICTION_HORIZON}d'] = df['close'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
        df['target'] = (df[f'target_return_{PREDICTION_HORIZON}d'] > 0).astype(int)

        all_data[pair] = df
        print(f"  ✓ {pair}: {len(df)} days")

    except Exception as e:
        print(f"  ✗ {pair}: ERROR - {str(e)}")
        continue

print()
print(f"Loaded {len(all_data)} pairs with real astronomical data")
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

# Identify astronomical features
sample_df = all_data[list(all_data.keys())[0]]
basic_columns = ['open', 'high', 'low', 'close', 'volume', 'target',
                 f'target_return_{PREDICTION_HORIZON}d']
astronomical_features = [col for col in sample_df.columns
                         if col not in technical_features
                         and col not in basic_columns
                         and not col.startswith('ema_')
                         and not col.startswith('bb_')]

print(f"Feature sets:")
print(f"  • Technical indicators: {len(technical_features)}")
print(f"  • Real astronomical features: {len(astronomical_features)}")
print()

print("Sample astronomical features:")
for feat in astronomical_features[:15]:
    print(f"  • {feat}")
if len(astronomical_features) > 15:
    print(f"  ... and {len(astronomical_features) - 15} more")
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

# Train models
print("="*100)
print("TRAINING MODELS")
print("="*100)
print()

baseline_models = {}
astronomy_models = {}
comparison_results = []

for pair in PAIRS:
    if pair not in all_data:
        continue

    print(f"\n{'='*100}")
    print(f"{pair}")
    print('='*100)

    df = all_data[pair].copy()
    df = df.dropna(subset=['target'])

    train_df = df[df.index < TEST_SPLIT_DATE]
    test_df = df[df.index >= TEST_SPLIT_DATE]

    print(f"  Data: {len(train_df)} train / {len(test_df)} test samples")

    if len(train_df) < 100 or len(test_df) < 10:
        print(f"  SKIPPED: Insufficient data")
        continue

    y_train_temp = train_df['target'].values
    neg_count = (y_train_temp == 0).sum()
    pos_count = (y_train_temp == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # ===== BASELINE MODEL (Technical Only) =====
    print(f"\n  [1] BASELINE MODEL (Technical Only)")
    print(f"  {'-'*50}")

    available_technical = [f for f in technical_features if f in train_df.columns]
    X_train_baseline = train_df[available_technical].dropna()
    y_train_baseline = train_df.loc[X_train_baseline.index, 'target'].values
    X_test_baseline = test_df[available_technical].dropna()
    y_test_baseline = test_df.loc[X_test_baseline.index, 'target'].values

    print(f"      Features: {len(available_technical)}")
    print(f"      Samples: {len(X_train_baseline)} train / {len(X_test_baseline)} test")

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

    # ===== ASTRONOMY MODEL (Technical + Real Astronomy) =====
    print(f"\n  [2] ASTRONOMY MODEL (Technical + Real Planets)")
    print(f"  {'-'*50}")

    available_astronomy = [f for f in astronomical_features if f in train_df.columns]
    all_features = available_technical + available_astronomy

    X_train_astro = train_df[all_features].dropna()
    y_train_astro = train_df.loc[X_train_astro.index, 'target'].values
    X_test_astro = test_df[all_features].dropna()
    y_test_astro = test_df.loc[X_test_astro.index, 'target'].values

    print(f"      Features: {len(all_features)} ({len(available_technical)} tech + {len(available_astronomy)} astro)")
    print(f"      Samples: {len(X_train_astro)} train / {len(X_test_astro)} test")

    model_astro = XGBClassifier(**XGB_CONFIG, scale_pos_weight=scale_pos_weight)
    model_astro.fit(
        X_train_astro, y_train_astro,
        eval_set=[(X_test_astro, y_test_astro)],
        verbose=False
    )

    y_test_pred_astro = model_astro.predict(X_test_astro)
    astro_acc = accuracy_score(y_test_astro, y_test_pred_astro)
    print(f"      Test accuracy: {astro_acc*100:.2f}%")

    # Top astronomical features
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model_astro.feature_importances_
    }).sort_values('importance', ascending=False)

    top_astro = feature_importance[
        feature_importance['feature'].isin(available_astronomy)
    ].head(5)

    if len(top_astro) > 0:
        print(f"\n      Top 5 astronomical features:")
        for idx, row in top_astro.iterrows():
            print(f"        {row['feature']:45s}: {row['importance']:.4f}")

    astronomy_models[pair] = {
        'model': model_astro,
        'features': all_features,
        'test_acc': astro_acc
    }

    # ===== COMPARISON =====
    improvement = astro_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  COMPARISON:")
    print(f"  {'='*50}")
    print(f"    Baseline (Technical):     {baseline_acc*100:>6.2f}%")
    print(f"    With Real Astronomy:      {astro_acc*100:>6.2f}%")
    print(f"    Improvement:              {improvement*100:>+6.2f}% ({improvement_pct:+.1f}%)")

    if improvement > 0.005:  # >0.5% improvement
        print(f"    Status: ✓✓ REAL ASTRONOMY HELPS!")
    elif improvement > 0:
        print(f"    Status: ✓ Slight improvement")
    elif improvement < -0.005:
        print(f"    Status: ✗ Real astronomy hurts")
    else:
        print(f"    Status: = No meaningful difference")

    comparison_results.append({
        'pair': pair,
        'baseline_acc': baseline_acc,
        'astro_acc': astro_acc,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'test_samples': len(X_test_baseline)
    })

# ===== FINAL SUMMARY =====
print("\n\n" + "="*100)
print("FINAL RESULTS: REAL ASTRONOMICAL DATA vs BASELINE")
print("="*100)
print()

results_df = pd.DataFrame(comparison_results)

print(f"{'Pair':<10} {'Baseline':>12} {'W/ Astro':>12} {'Improvement':>15} {'Status':>8} {'Samples':>10}")
print("-" * 100)

for _, row in results_df.iterrows():
    if row['improvement'] > 0.005:
        symbol = '✓✓'
    elif row['improvement'] > 0:
        symbol = '✓'
    elif row['improvement'] < -0.005:
        symbol = '✗'
    else:
        symbol = '='

    print(f"{row['pair']:<10} {row['baseline_acc']*100:>11.2f}% {row['astro_acc']*100:>11.2f}% "
          f"{row['improvement']*100:>+10.2f}% {symbol:>8s} {row['test_samples']:>10,}")

print()
avg_baseline = results_df['baseline_acc'].mean()
avg_astro = results_df['astro_acc'].mean()
avg_improvement = avg_astro - avg_baseline

print(f"AVERAGE PERFORMANCE:")
print(f"  Baseline (Technical only):       {avg_baseline*100:.2f}%")
print(f"  With Real Astronomical data:     {avg_astro*100:.2f}%")
print(f"  Average improvement:             {avg_improvement*100:+.2f}%")
print()

pairs_improved = (results_df['improvement'] > 0).sum()
pairs_significantly_improved = (results_df['improvement'] > 0.005).sum()
pairs_worse = (results_df['improvement'] < -0.005).sum()

print(f"IMPACT ANALYSIS:")
print(f"  Pairs with improvement: {pairs_improved}/{len(results_df)}")
print(f"  Pairs with >0.5% improvement: {pairs_significantly_improved}/{len(results_df)}")
print(f"  Pairs with <-0.5% (worse): {pairs_worse}/{len(results_df)}")
print()

print("="*100)
print("INTERPRETATION:")
print("="*100)

if avg_improvement > 0.01:
    print(f"✓✓ SIGNIFICANT FINDING: Real astronomical positions improve predictions by {avg_improvement*100:.2f}%!")
    print(f"   This suggests potential correlation between planetary positions and forex movements.")
elif avg_improvement > 0.005:
    print(f"✓ MODERATE FINDING: Real astronomy provides {avg_improvement*100:.2f}% improvement.")
    print(f"  Could indicate weak correlation or could be statistical noise.")
elif avg_improvement > 0:
    print(f"≈ MARGINAL: Real astronomy shows {avg_improvement*100:.2f}% improvement (likely noise).")
elif avg_improvement < 0:
    print(f"✗ NO CORRELATION: Real astronomical data does not help ({avg_improvement*100:.2f}%).")
    print(f"  This suggests planetary positions do not correlate with forex movements.")
else:
    print(f"= NEUTRAL: No measurable difference.")

print()

# Save models
baseline_file = os.path.join(OUTPUT_DIR, 'forex_baseline_models.pkl')
astro_file = os.path.join(OUTPUT_DIR, 'forex_with_real_astronomy_models.pkl')

with open(baseline_file, 'wb') as f:
    pickle.dump({
        'models': baseline_models,
        'features': 'technical_only',
        'prediction_horizon': PREDICTION_HORIZON
    }, f)

with open(astro_file, 'wb') as f:
    pickle.dump({
        'models': astronomy_models,
        'features': 'technical_and_real_astronomy',
        'prediction_horizon': PREDICTION_HORIZON,
        'data_source': 'NASA_JPL_DE421'
    }, f)

print(f"Models saved:")
print(f"  • Baseline: {baseline_file}")
print(f"  • With Real Astronomy: {astro_file}")
print()

print("="*100)
print("EXPERIMENT COMPLETE")
print("="*100)
