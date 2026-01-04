"""
ANALYZE POSITION STACKING
==========================
Check how many positions are open simultaneously with 10% sizing
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']
INITIAL_CAPITAL = 500
POSITION_PCT = 0.10


def analyze_positions(predictions, raw_data):
    """Track position counts and exposure"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    capital = INITIAL_CAPITAL
    positions = []
    position_stats = []

    for date in all_dates:
        # Update existing positions
        positions_to_close = []
        for pos in positions:
            pair = pos['pair']
            if date not in raw_data[pair].index:
                continue

            row = raw_data[pair].loc[date]
            pos['bars_held'] += 1

            if pos['direction'] == 'long':
                if not pos.get('trailing_active') and row['bid_high'] >= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['bid_high']
                    continue

                if pos.get('trailing_active'):
                    if row['bid_high'] > pos['peak_price']:
                        pos['peak_price'] = row['bid_high']
                        new_stop = pos['initial_target'] + 0.75 * (pos['peak_price'] - pos['initial_target'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)

                    if row['bid_low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct))
                        continue

            else:
                if not pos.get('trailing_active') and row['ask_low'] <= pos['initial_target']:
                    pos['trailing_active'] = True
                    pos['trailing_stop'] = pos['initial_target']
                    pos['peak_price'] = row['ask_low']
                    continue

                if pos.get('trailing_active'):
                    if row['ask_low'] < pos['peak_price']:
                        pos['peak_price'] = row['ask_low']
                        new_stop = pos['initial_target'] - 0.75 * (pos['initial_target'] - pos['peak_price'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)

                    if row['ask_high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        positions_to_close.append((pos, profit_pct))
                        continue

            if pos['bars_held'] >= 24:
                if pos['direction'] == 'long':
                    exit_price = row['bid_close']
                    profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    exit_price = row['ask_close']
                    profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                positions_to_close.append((pos, profit_pct))

        # Close positions
        for pos, profit_pct in positions_to_close:
            profit_dollars = profit_pct * pos['size']
            capital += profit_dollars
            positions.remove(pos)

        # Look for new signals
        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if date not in raw_data[pair].index:
                continue

            pred = predictions[pair].loc[date]
            max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

            if max_prob <= 0.80:
                continue

            row = raw_data[pair].loc[date]
            if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                direction = 'long'
                entry_price = row['ask_open']
                initial_target = pred['high_80p']
            else:
                direction = 'short'
                entry_price = row['bid_open']
                initial_target = pred['low_80p']

            position_size = capital * POSITION_PCT

            positions.append({
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'initial_target': initial_target,
                'size': position_size,
                'bars_held': 0,
                'confidence': max_prob,
                'trailing_active': False
            })

        # Track position counts
        if len(positions) > 0:
            total_exposure = sum(p['size'] for p in positions) / capital
            pair_counts = {}
            for pos in positions:
                pair_counts[pos['pair']] = pair_counts.get(pos['pair'], 0) + 1

            max_per_pair = max(pair_counts.values()) if pair_counts else 0

            position_stats.append({
                'date': date,
                'total_positions': len(positions),
                'total_exposure_pct': total_exposure,
                'max_positions_per_pair': max_per_pair,
                'pairs_with_positions': len(pair_counts)
            })

    return pd.DataFrame(position_stats)


# ============================================================================
# MAIN
# ============================================================================
print("="*100)
print("ANALYZING POSITION STACKING")
print("="*100)
print()

print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Test one year
with open('test_predictions_15m_2023_test.pkl', 'rb') as f:
    preds = pickle.load(f)

print("Analyzing 2023 test period...")
print()

stats = analyze_positions(preds, all_raw_data)

print("="*100)
print("POSITION STATISTICS")
print("="*100)
print()

print(f"Average positions open: {stats['total_positions'].mean():.1f}")
print(f"Max positions open: {stats['total_positions'].max():.0f}")
print(f"95th percentile: {stats['total_positions'].quantile(0.95):.0f}")
print()

print(f"Average exposure: {stats['total_exposure_pct'].mean():.1%}")
print(f"Max exposure: {stats['total_exposure_pct'].max():.1%}")
print(f"95th percentile: {stats['total_exposure_pct'].quantile(0.95):.1%}")
print()

print(f"Max positions per single pair: {stats['max_positions_per_pair'].max():.0f}")
print(f"Average max per pair: {stats['max_positions_per_pair'].mean():.1f}")
print()

print("Distribution of total positions open:")
print(stats['total_positions'].value_counts().sort_index().head(20))
print()

print("="*100)
print("RISK ANALYSIS")
print("="*100)
print()

if stats['total_exposure_pct'].max() > 1.0:
    print(f"WARNING: Peak exposure is {stats['total_exposure_pct'].max():.1%} of capital!")
    print(f"This means you're over-leveraged at peak times.")
    print()
    print(f"Times with >100% exposure: {(stats['total_exposure_pct'] > 1.0).sum()} periods")
    print(f"Percentage of time: {(stats['total_exposure_pct'] > 1.0).mean():.1%}")
else:
    print(f"Peak exposure is {stats['total_exposure_pct'].max():.1%} - within limits")

print()
print("Recommendation:")
if stats['total_exposure_pct'].max() > 1.0:
    print("  Add position limits OR use dynamic sizing:")
    print("  - Max 1-2 positions per pair")
    print("  - Max 10 total positions")
    print("  - OR reduce position size to 5% to avoid over-leverage")
else:
    print("  Current position stacking is manageable")
    print("  Consider adding position limits for safety")

print()
