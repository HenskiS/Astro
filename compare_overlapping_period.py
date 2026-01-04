"""
CRITICAL TEST: Compare performance on the SAME test period
===========================================================
If walk-forward and original produce different results on the SAME period,
that indicates a methodology issue, not just different market conditions.
"""
import pandas as pd
import pickle

print("="*100)
print("COMPARING PREDICTIONS ON OVERLAPPING PERIOD")
print("="*100)
print()

# Load predictions
with open('test_predictions_15m.pkl', 'rb') as f:
    original_preds = pickle.load(f)

with open('test_predictions_15m_walkforward.pkl', 'rb') as f:
    walkforward_preds = pickle.load(f)

# Find overlapping period
orig_dates = set()
for df in original_preds.values():
    orig_dates.update(df.index)

wf_dates = set()
for df in walkforward_preds.values():
    wf_dates.update(df.index)

overlap_dates = orig_dates.intersection(wf_dates)
overlap_dates = sorted(list(overlap_dates))

if len(overlap_dates) == 0:
    print("ERROR: No overlapping dates found!")
    print(f"Original range: {min(orig_dates)} to {max(orig_dates)}")
    print(f"Walk-forward range: {min(wf_dates)} to {max(wf_dates)}")
else:
    print(f"Overlapping period: {min(overlap_dates)} to {max(overlap_dates)}")
    print(f"Total overlapping periods: {len(overlap_dates):,}")
    print()

    # Compare predictions for overlapping period
    print("="*100)
    print("PREDICTION COMPARISON (OVERLAPPING PERIOD ONLY)")
    print("="*100)
    print()

    for pair in ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']:
        print(f"{pair}:")
        print("-" * 80)

        # Get overlapping predictions
        orig_df = original_preds[pair]
        wf_df = walkforward_preds[pair]

        # Filter to overlap
        orig_overlap = orig_df[orig_df.index.isin(overlap_dates)].sort_index()
        wf_overlap = wf_df[wf_df.index.isin(overlap_dates)].sort_index()

        if len(orig_overlap) == 0 or len(wf_overlap) == 0:
            print("  No overlapping data for this pair")
            print()
            continue

        # Find common dates
        common_dates = orig_overlap.index.intersection(wf_overlap.index)
        print(f"  Common dates: {len(common_dates)}")

        if len(common_dates) > 0:
            orig_common = orig_overlap.loc[common_dates]
            wf_common = wf_overlap.loc[common_dates]

            # Compare predictions
            print(f"  Original avg breakout_high_prob: {orig_common['breakout_high_prob'].mean():.4f}")
            print(f"  Walk-forward avg breakout_high_prob: {wf_common['breakout_high_prob'].mean():.4f}")
            print(f"  Original avg breakout_low_prob: {orig_common['breakout_low_prob'].mean():.4f}")
            print(f"  Walk-forward avg breakout_low_prob: {wf_common['breakout_low_prob'].mean():.4f}")

            # Check if predictions are identical or different
            high_diff = (orig_common['breakout_high_prob'] - wf_common['breakout_high_prob']).abs().mean()
            low_diff = (orig_common['breakout_low_prob'] - wf_common['breakout_low_prob']).abs().mean()

            print(f"  Avg absolute difference (high): {high_diff:.4f}")
            print(f"  Avg absolute difference (low): {low_diff:.4f}")

            if high_diff < 0.001 and low_diff < 0.001:
                print("  [IDENTICAL] Predictions are essentially the same")
            elif high_diff < 0.05 and low_diff < 0.05:
                print("  [SIMILAR] Small differences in predictions")
            else:
                print("  [DIFFERENT] Significant differences in predictions")

        print()

print("="*100)
print("INTERPRETATION")
print("="*100)
print()
print("Why might predictions differ in the overlapping period?")
print()
print("EXPECTED DIFFERENCES:")
print("  1. Walk-forward uses MORE training data")
print("     - Original: Trained once on data up to ~Aug 2024")
print("     - Walk-forward: Retrained multiple times, using data up to trade date")
print("     - Models have seen MORE data, so predictions can differ")
print()
print("  2. Walk-forward models are FRESHER")
print("     - For Aug 2025 predictions:")
print("       - Original: Uses model trained ~1 year ago")
print("       - Walk-forward: Uses model retrained in July/Aug 2025")
print("     - Fresher models may have different predictions")
print()
print("This is EXPECTED and CORRECT behavior!")
print("Walk-forward simulates production where you continuously retrain.")
print()

# Now let's backtest just the overlapping period for both
print("="*100)
print("BACKTEST COMPARISON: OVERLAPPING PERIOD ONLY")
print("="*100)
print()

import warnings
warnings.filterwarnings('ignore')

# Create filtered prediction sets for the overlapping period
if len(overlap_dates) > 0:
    orig_filtered = {}
    wf_filtered = {}

    for pair in original_preds.keys():
        orig_filtered[pair] = original_preds[pair][original_preds[pair].index.isin(overlap_dates)]
        wf_filtered[pair] = walkforward_preds[pair][walkforward_preds[pair].index.isin(overlap_dates)]

    # Save filtered predictions
    with open('temp_original_overlap.pkl', 'wb') as f:
        pickle.dump(orig_filtered, f)
    with open('temp_walkforward_overlap.pkl', 'wb') as f:
        pickle.dump(wf_filtered, f)

    print("Filtered prediction sets saved for backtest comparison.")
    print()
    print("To compare performance on the SAME period:")
    print("  1. Run backtest on temp_original_overlap.pkl")
    print("  2. Run backtest on temp_walkforward_overlap.pkl")
    print("  3. Compare CAGRs")
    print()
    print("If CAGRs are similar, the difference is due to test period selection.")
    print("If CAGRs differ significantly, there may be a methodology issue.")
