"""
DIAGNOSE MODEL ACCURACY
=======================
Proper analysis of model performance with correct metrics.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = 'data_15m'

print("="*80)
print("MODEL ACCURACY DIAGNOSIS")
print("="*80)
print()

# Load predictions for 2023
with open('test_predictions_15m_2023_test.pkl', 'rb') as f:
    preds_2023 = pickle.load(f)

# Load actual outcomes
from train_model_15m import calculate_features, calculate_targets, LOOKBACK_PERIOD, FORWARD_PERIODS

df = pd.read_csv(f'{DATA_DIR}/EURUSD_15m.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = calculate_features(df, LOOKBACK_PERIOD)
df = calculate_targets(df, LOOKBACK_PERIOD, FORWARD_PERIODS)

# Get predictions
eurusd_preds = preds_2023['EURUSD']
common_dates = eurusd_preds.index.intersection(df.index)

print(f"Analyzing EURUSD predictions for 2023")
print(f"Total samples: {len(common_dates)}")
print()

# Collect predictions and actuals
y_true_high = []
y_pred_high = []
y_true_low = []
y_pred_low = []

for date in common_dates:
    if pd.isna(df.loc[date, 'breakout_high']) or pd.isna(df.loc[date, 'breakout_low']):
        continue

    # High breakout
    y_true_high.append(int(df.loc[date, 'breakout_high']))
    y_pred_high.append(int(eurusd_preds.loc[date, 'breakout_high_prob'] > 0.5))

    # Low breakout
    y_true_low.append(int(df.loc[date, 'breakout_low']))
    y_pred_low.append(int(eurusd_preds.loc[date, 'breakout_low_prob'] > 0.5))

y_true_high = np.array(y_true_high)
y_pred_high = np.array(y_pred_high)
y_true_low = np.array(y_true_low)
y_pred_low = np.array(y_pred_low)

print("="*80)
print("BREAKOUT HIGH PREDICTIONS")
print("="*80)
print()

# Confusion matrix
cm_high = confusion_matrix(y_true_high, y_pred_high)
print("Confusion Matrix:")
print(f"                  Predicted: No    Predicted: Yes")
print(f"Actual: No        {cm_high[0,0]:6d}        {cm_high[0,1]:6d}")
print(f"Actual: Yes       {cm_high[1,0]:6d}        {cm_high[1,1]:6d}")
print()

# Calculate metrics
tn_high, fp_high, fn_high, tp_high = cm_high.ravel()
accuracy_high = (tp_high + tn_high) / len(y_true_high)
precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 0
recall_high = tp_high / (tp_high + fn_high) if (tp_high + fn_high) > 0 else 0

print("Metrics:")
print(f"Accuracy:  {accuracy_high:.1%}")
print(f"Precision: {precision_high:.1%} (when model says YES, how often is it right?)")
print(f"Recall:    {recall_high:.1%} (of all actual breakouts, how many did we catch?)")
print()

# Distribution
print("Prediction distribution:")
print(f"Model predicts YES: {y_pred_high.sum()} / {len(y_pred_high)} ({100*y_pred_high.mean():.1f}%)")
print(f"Actual breakouts:   {y_true_high.sum()} / {len(y_true_high)} ({100*y_true_high.mean():.1f}%)")
print()

# Probability distribution
print("Probability distribution (raw model outputs):")
probs_high = [eurusd_preds.loc[date, 'breakout_high_prob'] for date in common_dates
              if not pd.isna(df.loc[date, 'breakout_high'])]
probs_high = np.array(probs_high)
print(f"Mean: {probs_high.mean():.3f}")
print(f"Std:  {probs_high.std():.3f}")
print(f"Min:  {probs_high.min():.3f}")
print(f"Max:  {probs_high.max():.3f}")
print(f"Percentiles: 25%={np.percentile(probs_high, 25):.3f}, "
      f"50%={np.percentile(probs_high, 50):.3f}, "
      f"75%={np.percentile(probs_high, 75):.3f}")
print()

print("="*80)
print("BREAKOUT LOW PREDICTIONS")
print("="*80)
print()

# Confusion matrix
cm_low = confusion_matrix(y_true_low, y_pred_low)
print("Confusion Matrix:")
print(f"                  Predicted: No    Predicted: Yes")
print(f"Actual: No        {cm_low[0,0]:6d}        {cm_low[0,1]:6d}")
print(f"Actual: Yes       {cm_low[1,0]:6d}        {cm_low[1,1]:6d}")
print()

# Calculate metrics
tn_low, fp_low, fn_low, tp_low = cm_low.ravel()
accuracy_low = (tp_low + tn_low) / len(y_true_low)
precision_low = tp_low / (tp_low + fp_low) if (tp_low + fp_low) > 0 else 0
recall_low = tp_low / (tp_low + fn_low) if (tp_low + fn_low) > 0 else 0

print("Metrics:")
print(f"Accuracy:  {accuracy_low:.1%}")
print(f"Precision: {precision_low:.1%}")
print(f"Recall:    {recall_low:.1%}")
print()

# Distribution
print("Prediction distribution:")
print(f"Model predicts YES: {y_pred_low.sum()} / {len(y_pred_low)} ({100*y_pred_low.mean():.1f}%)")
print(f"Actual breakouts:   {y_true_low.sum()} / {len(y_true_low)} ({100*y_true_low.mean():.1f}%)")
print()

# Probability distribution
print("Probability distribution (raw model outputs):")
probs_low = [eurusd_preds.loc[date, 'breakout_low_prob'] for date in common_dates
             if not pd.isna(df.loc[date, 'breakout_low'])]
probs_low = np.array(probs_low)
print(f"Mean: {probs_low.mean():.3f}")
print(f"Std:  {probs_low.std():.3f}")
print(f"Min:  {probs_low.min():.3f}")
print(f"Max:  {probs_low.max():.3f}")
print(f"Percentiles: 25%={np.percentile(probs_low, 25):.3f}, "
      f"50%={np.percentile(probs_low, 50):.3f}, "
      f"75%={np.percentile(probs_low, 75):.3f}")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

if probs_high.std() < 0.01 and probs_low.std() < 0.01:
    print("[CRITICAL] Model outputs have no variance - it's predicting the same value always!")
    print("This means the model failed to learn anything useful.")
elif accuracy_high < 0.52 and accuracy_low < 0.52:
    print("[CRITICAL] Accuracy is at random (50%) - model is not learning!")
elif y_pred_high.sum() == 0 or y_pred_low.sum() == 0:
    print("[CRITICAL] Model never predicts positive class - check threshold!")
else:
    print("[OK] Model is making varied predictions")
    if accuracy_high > 0.55 or accuracy_low > 0.55:
        print(f"[OK] Model has some predictive power (>55% accuracy)")
    else:
        print("[WARNING] Model accuracy is only slightly better than random")
