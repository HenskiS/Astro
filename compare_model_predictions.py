"""
Compare MODEL PREDICTIONS: What probabilities do we get?
=========================================================
"""
import pandas as pd
import numpy as np
import pickle
from production_simulation import ProductionSimulation, calculate_features, XGB_CONFIG
from mock_broker_api import MockBrokerAPI
from xgboost import XGBClassifier

print("="*100)
print("MODEL PREDICTION COMPARISON")
print("="*100)
print()

# Test parameters
TEST_DATE = pd.Timestamp('2022-01-03 14:00:00', tz='UTC')
PAIR = 'EURUSD'
TRAIN_END = pd.Timestamp('2021-12-21')  # Training data cutoff for 2022Q1

print(f"Test date: {TEST_DATE}")
print(f"Pair: {PAIR}")
print(f"Training end: {TRAIN_END}")
print()

# ============================================================================
# Train a fresh model (what simulator does)
# ============================================================================
print("1. Training fresh model (simulator approach):")
print("-" * 60)

api = MockBrokerAPI(data_dir='data')

# Get training data
train_history = api.get_history(PAIR, count=999999, end_date=TRAIN_END)
print(f"Training data: {len(train_history)} days")

# Calculate features and targets
from production_simulation import create_targets
train_data = calculate_features(train_history)
train_data = create_targets(train_data)
train_data = train_data.dropna()

print(f"Training samples after cleaning: {len(train_data)}")

# Train models
feature_cols = [col for col in train_data.columns if col not in
               ['target_breakout_high', 'target_breakout_low',
                'open', 'high', 'low', 'close', 'volume']]

X_train = train_data[feature_cols]
y_high = train_data['target_breakout_high']
y_low = train_data['target_breakout_low']

print(f"Features: {len(feature_cols)}")
print(f"Target high positives: {y_high.sum()}")
print(f"Target low positives: {y_low.sum()}")

model_high = XGBClassifier(**XGB_CONFIG)
model_high.fit(X_train, y_high, verbose=False)

model_low = XGBClassifier(**XGB_CONFIG)
model_low.fit(X_train, y_low, verbose=False)

print("Models trained ✓")
print()

# ============================================================================
# Generate prediction for test date
# ============================================================================
print("2. Generating prediction for test date:")
print("-" * 60)

# Get test data
test_history = api.get_history(PAIR, count=999999, end_date=TEST_DATE)
test_features = calculate_features(test_history)
test_features = test_features.dropna()

print(f"Test data available: {len(test_features)} days")
print(f"Last date: {test_features.index[-1]}")

# Predict
X_test = test_features[feature_cols].iloc[-1:].fillna(0)
high_prob = model_high.predict_proba(X_test)[0, 1]
low_prob = model_low.predict_proba(X_test)[0, 1]

print(f"SIMULATOR PREDICTIONS:")
print(f"  High prob: {high_prob:.4f}")
print(f"  Low prob:  {low_prob:.4f}")
print(f"  Max prob:  {max(high_prob, low_prob):.4f}")
print(f"  Above 0.70 threshold: {max(high_prob, low_prob) > 0.70}")
print()

# ============================================================================
# Compare with backtest predictions
# ============================================================================
print("3. Backtest predictions (from pkl):")
print("-" * 60)

with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_preds = pickle.load(f)

backtest_preds = all_preds['2022Q1'][PAIR]
row = backtest_preds.loc[TEST_DATE]

print(f"BACKTEST PREDICTIONS:")
print(f"  High prob: {row['breakout_high_prob']:.4f}")
print(f"  Low prob:  {row['breakout_low_prob']:.4f}")
print(f"  Max prob:  {max(row['breakout_high_prob'], row['breakout_low_prob']):.4f}")
print(f"  Above 0.70 threshold: {max(row['breakout_high_prob'], row['breakout_low_prob']) > 0.70}")
print()

# ============================================================================
# Comparison
# ============================================================================
print("4. Comparison:")
print("-" * 60)

high_diff = abs(high_prob - row['breakout_high_prob'])
low_diff = abs(low_prob - row['breakout_low_prob'])

print(f"High prob difference: {high_diff:.6f} {'✓' if high_diff < 0.01 else '✗'}")
print(f"Low prob difference:  {low_diff:.6f} {'✓' if low_diff < 0.01 else '✗'}")
print()

if high_diff < 0.01 and low_diff < 0.01:
    print("✓ Predictions MATCH - Simulator and backtest generate same probabilities")
else:
    print("✗ Predictions DIFFER - This is the source of divergence!")
    print()
    print("Possible reasons:")
    print("  - Feature calculation differs")
    print("  - Training data differs")
    print("  - Model randomness (but random_state=42 should prevent this)")
    print("  - fillna(0) in simulator vs dropna in backtest preprocessing")

print()
print("="*100)