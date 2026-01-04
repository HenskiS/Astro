"""
FINAL COMPARISON: ALL VALIDATION APPROACHES
============================================
Compares three validation methodologies to show which is most realistic:

1. ORIGINAL: 70/30 split, train once
2. STRICT TIME SPLIT: Date-based split, train once, long test period
3. WALK-FORWARD: Periodic retraining (MOST REALISTIC)
"""
import pandas as pd
import numpy as np
import pickle

print("="*100)
print("FINAL VALIDATION COMPARISON")
print("="*100)
print()

# Load all prediction sets
print("Loading predictions...")

try:
    with open('test_predictions_15m.pkl', 'rb') as f:
        original_preds = pickle.load(f)
    print("  [OK] Original predictions loaded")
except:
    print("  [SKIP] Original predictions not found")
    original_preds = None

try:
    with open('test_predictions_15m_validated.pkl', 'rb') as f:
        validated_preds = pickle.load(f)
    print("  [OK] Validated predictions loaded")
except:
    print("  [SKIP] Validated predictions not found")
    validated_preds = None

try:
    with open('test_predictions_15m_walkforward.pkl', 'rb') as f:
        walkforward_preds = pickle.load(f)
    print("  [OK] Walk-forward predictions loaded")
except:
    print("  [SKIP] Walk-forward predictions not found")
    walkforward_preds = None

print()

# Analyze each prediction set
def analyze_predictions(preds, name):
    print(f"{name}:")
    print("-" * 80)

    if preds is None:
        print("  No data available")
        print()
        return

    # Get date ranges for all pairs
    all_dates = []
    for pair, df in preds.items():
        all_dates.extend(df.index.tolist())

    if len(all_dates) == 0:
        print("  No predictions found")
        print()
        return

    min_date = min(all_dates)
    max_date = max(all_dates)
    days = (max_date - min_date).days

    total_predictions = sum(len(df) for df in preds.values())

    # Calculate average probabilities
    all_high_probs = []
    all_low_probs = []
    for df in preds.values():
        all_high_probs.extend(df['breakout_high_prob'].tolist())
        all_low_probs.extend(df['breakout_low_prob'].tolist())

    avg_high_prob = np.mean(all_high_probs) if len(all_high_probs) > 0 else 0
    avg_low_prob = np.mean(all_low_probs) if len(all_low_probs) > 0 else 0
    max_prob_avg = np.mean([max(h, l) for h, l in zip(all_high_probs, all_low_probs)])

    # Count high confidence predictions (>0.70)
    high_conf_count = sum(1 for h, l in zip(all_high_probs, all_low_probs) if max(h, l) > 0.70)
    high_conf_pct = high_conf_count / len(all_high_probs) * 100 if len(all_high_probs) > 0 else 0

    print(f"  Date range: {min_date.date()} to {max_date.date()} ({days} days)")
    print(f"  Total predictions: {total_predictions:,}")
    print(f"  Avg breakout_high prob: {avg_high_prob:.3f}")
    print(f"  Avg breakout_low prob: {avg_low_prob:.3f}")
    print(f"  Avg max(high, low) prob: {max_prob_avg:.3f}")
    print(f"  High confidence (>0.70): {high_conf_count:,} ({high_conf_pct:.1f}%)")
    print()

# Analyze each set
print("="*100)
print("PREDICTION SET ANALYSIS")
print("="*100)
print()

analyze_predictions(original_preds, "1. ORIGINAL (70/30 split)")
analyze_predictions(validated_preds, "2. STRICT TIME SPLIT (no retrain)")
analyze_predictions(walkforward_preds, "3. WALK-FORWARD (periodic retrain)")

# Summary and recommendations
print("="*100)
print("METHODOLOGY COMPARISON")
print("="*100)
print()

print("1. ORIGINAL (70/30 split, train once)")
print("   Pros:")
print("     - Simple to implement")
print("     - Fast (train once)")
print("     - Uses most data for training")
print("   Cons:")
print("     - Test period may not be representative of future")
print("     - No retraining = doesn't reflect production reality")
print("     - May overestimate performance if market regime changes")
print()

print("2. STRICT TIME SPLIT (date-based split, no retrain)")
print("   Pros:")
print("     - Clear temporal separation")
print("     - Tests model on longer forward period")
print("   Cons:")
print("     - May underestimate performance (no retraining)")
print("     - Small training set = weak model")
print("     - Doesn't reflect production (would retrain in real trading)")
print()

print("3. WALK-FORWARD (periodic retraining) - RECOMMENDED")
print("   Pros:")
print("     - MOST REALISTIC simulation of production")
print("     - Models adapt to changing market conditions")
print("     - Expanding training window = improving models over time")
print("     - Tests actual retraining frequency you'd use")
print("   Cons:")
print("     - Computationally expensive (train many times)")
print("     - More complex to implement")
print()

print("="*100)
print("RECOMMENDATIONS")
print("="*100)
print()

print("FOR DEVELOPMENT & OPTIMIZATION:")
print("  Use Method #1 (Original) - fastest for parameter tuning")
print()

print("FOR REALISTIC PERFORMANCE ESTIMATION:")
print("  Use Method #3 (Walk-Forward) - best reflects production trading")
print()

print("FOR CONSERVATIVE ESTIMATE:")
print("  Use Method #2 (Strict Time Split) - lower bound on performance")
print()

print("="*100)
print("ADDRESSING YOUR ORIGINAL CONCERN")
print("="*100)
print()

print("Were your original results 'too good to be true'?")
print()
print("ANSWER: Probably not contaminated, but validation could be improved.")
print()
print("Analysis:")
print("  1. NO CLEAR DATA CONTAMINATION:")
print("     - Rolling features look backward (no future info)")
print("     - 70/30 split is temporally ordered")
print("     - Targets correctly use future data (they're the labels)")
print()
print("  2. VALIDATION METHODOLOGY:")
print("     - Original method is acceptable for initial testing")
print("     - Walk-forward is better for production confidence")
print("     - Results may vary with retraining frequency")
print()
print("  3. RECOMMENDATION:")
print("     - Run walk-forward backtest to get realistic CAGR estimate")
print("     - If walk-forward CAGR is similar to original, you're good!")
print("     - If walk-forward CAGR is much lower, adjust expectations")
print()

print("="*100)
print("NEXT STEPS")
print("="*100)
print()
print("1. Backtest walk-forward predictions to get realistic performance")
print("2. Compare walk-forward CAGR with original CAGR")
print("3. Use walk-forward results to set production expectations")
print()
print("To run walk-forward backtest:")
print("  Create a copy of backtest_15m_optimized.py")
print("  Change line 172 to load 'test_predictions_15m_walkforward.pkl'")
print("  Run the backtest")
print()
