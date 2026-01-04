"""
ANALYZE ACTUAL MOVE SIZE
=========================
When model predicts breakout correctly (94% of time):
- How far does price actually move?
- Are we exiting too early at the tiny initial target?
- What's the opportunity we're missing?
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


def analyze_moves(predictions, raw_data):
    """Analyze actual price moves after high-confidence signals"""
    all_dates = sorted(set([d for p in predictions.values() for d in p.index]))

    move_analysis = []

    for date in all_dates:
        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if date not in raw_data[pair].index:
                continue

            pred = predictions[pair].loc[date]
            max_prob = max(pred['breakout_high_prob'], pred['breakout_low_prob'])

            # Only analyze high confidence signals
            if max_prob <= 0.80:
                continue

            row = raw_data[pair].loc[date]

            if pred['breakout_high_prob'] > pred['breakout_low_prob']:
                direction = 'long'
                entry_price = row['ask_open']
                initial_target = pred['high_80p']
                target_distance = (initial_target - entry_price) / entry_price
            else:
                direction = 'short'
                entry_price = row['bid_open']
                initial_target = pred['low_80p']
                target_distance = (entry_price - initial_target) / entry_price

            # Track price movement over next 24 bars
            all_dates_list = list(all_dates)
            current_idx = all_dates_list.index(date)

            max_favorable_move = 0
            target_hit = False
            bars_to_target = None
            max_favorable_bars = 0
            final_profit = 0

            for i in range(1, min(25, len(all_dates_list) - current_idx)):
                future_date = all_dates_list[current_idx + i]
                if future_date not in raw_data[pair].index:
                    continue

                future_row = raw_data[pair].loc[future_date]

                if direction == 'long':
                    # Check if target hit
                    if not target_hit and future_row['bid_high'] >= initial_target:
                        target_hit = True
                        bars_to_target = i

                    # Track max favorable move
                    favorable_move = (future_row['bid_high'] - entry_price) / entry_price
                    if favorable_move > max_favorable_move:
                        max_favorable_move = favorable_move
                        max_favorable_bars = i

                    # Final P&L at 24 bars
                    if i == 24 or i == len(all_dates_list) - current_idx - 1:
                        final_profit = (future_row['bid_close'] - entry_price) / entry_price
                else:
                    # Check if target hit
                    if not target_hit and future_row['ask_low'] <= initial_target:
                        target_hit = True
                        bars_to_target = i

                    # Track max favorable move
                    favorable_move = (entry_price - future_row['ask_low']) / entry_price
                    if favorable_move > max_favorable_move:
                        max_favorable_move = favorable_move
                        max_favorable_bars = i

                    # Final P&L at 24 bars
                    if i == 24 or i == len(all_dates_list) - current_idx - 1:
                        final_profit = (entry_price - future_row['ask_close']) / entry_price

            move_analysis.append({
                'pair': pair,
                'direction': direction,
                'confidence': max_prob,
                'initial_target_pct': target_distance,
                'target_hit': target_hit,
                'bars_to_target': bars_to_target if target_hit else None,
                'max_favorable_pct': max_favorable_move,
                'max_favorable_bars': max_favorable_bars,
                'final_profit_pct': final_profit,
                'target_to_max_ratio': max_favorable_move / target_distance if target_distance > 0 else 0
            })

    return pd.DataFrame(move_analysis)


# ============================================================================
# MAIN
# ============================================================================
print("="*100)
print("ANALYZING ACTUAL MOVE SIZES")
print("="*100)
print()

print("Loading data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

test_files = [
    ('2021', 'test_predictions_15m_2021_test.pkl'),
    ('2022', 'test_predictions_15m_2022_test.pkl'),
    ('2023', 'test_predictions_15m_2023_test.pkl'),
    ('2024', 'test_predictions_15m_2024_test.pkl'),
    ('2025', 'test_predictions_15m_2025_test.pkl'),
]

all_moves = []

for name, filename in test_files:
    try:
        with open(filename, 'rb') as f:
            preds = pickle.load(f)

        moves = analyze_moves(preds, all_raw_data)
        moves['year'] = name
        all_moves.append(moves)
    except FileNotFoundError:
        print(f"{name}: File not found")

if len(all_moves) > 0:
    all_moves_df = pd.concat(all_moves, ignore_index=True)

    print()
    print("="*100)
    print("RESULTS: What happens after high-confidence signals?")
    print("="*100)
    print()

    # Overall statistics
    print(f"Total signals analyzed: {len(all_moves_df):,}")
    print(f"Target hit rate: {all_moves_df['target_hit'].mean():.1%}")
    print()

    print("Initial target size:")
    print(f"  Average: {all_moves_df['initial_target_pct'].mean():.3%}")
    print(f"  Median:  {all_moves_df['initial_target_pct'].median():.3%}")
    print()

    print("Maximum favorable excursion (MFE):")
    print(f"  Average: {all_moves_df['max_favorable_pct'].mean():.3%}")
    print(f"  Median:  {all_moves_df['max_favorable_pct'].median():.3%}")
    print()

    print("How much bigger is the max move vs initial target?")
    print(f"  Average ratio: {all_moves_df['target_to_max_ratio'].mean():.1f}x")
    print(f"  Median ratio:  {all_moves_df['target_to_max_ratio'].median():.1f}x")
    print()

    # For trades that hit target
    target_hit_trades = all_moves_df[all_moves_df['target_hit'] == True]
    if len(target_hit_trades) > 0:
        print("For trades that HIT the initial target (94% of trades):")
        print(f"  Average MFE: {target_hit_trades['max_favorable_pct'].mean():.3%}")
        print(f"  Median MFE:  {target_hit_trades['max_favorable_pct'].median():.3%}")
        print(f"  How much further beyond target: {target_hit_trades['target_to_max_ratio'].mean():.1f}x")
        print()

        print(f"  Bars to hit target: {target_hit_trades['bars_to_target'].mean():.1f} bars (median: {target_hit_trades['bars_to_target'].median():.0f})")
        print(f"  Bars to max move:   {target_hit_trades['max_favorable_bars'].mean():.1f} bars (median: {target_hit_trades['max_favorable_bars'].median():.0f})")
        print()

    print("="*100)
    print("KEY INSIGHT")
    print("="*100)
    print()

    avg_target = all_moves_df['initial_target_pct'].mean()
    avg_mfe = all_moves_df['max_favorable_pct'].mean()
    ratio = all_moves_df['target_to_max_ratio'].mean()

    print(f"We're exiting at {avg_target:.3%} (initial target)")
    print(f"But price typically moves {avg_mfe:.3%} (max favorable)")
    print(f"We're capturing only 1/{ratio:.1f}th of the actual move!")
    print()
    print("Solution: After target hit, use trailing stop to capture more of the move")
    print()

    # Distribution analysis
    print("="*100)
    print("DISTRIBUTION: Max favorable move sizes")
    print("="*100)
    print()

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Percentile | Max Favorable Move")
    print("-" * 35)
    for p in percentiles:
        val = np.percentile(all_moves_df['max_favorable_pct'], p)
        print(f"   {p:2d}th    | {val:>8.3%}")
    print()

    # Year by year
    print("="*100)
    print("BY YEAR")
    print("="*100)
    print()

    for year in ['2021', '2022', '2023', '2024', '2025']:
        year_data = all_moves_df[all_moves_df['year'] == year]
        if len(year_data) > 0:
            print(f"{year}:")
            print(f"  Signals: {len(year_data):>5,}")
            print(f"  Target hit rate: {year_data['target_hit'].mean():>5.1%}")
            print(f"  Avg target: {year_data['initial_target_pct'].mean():>6.3%} -> Avg MFE: {year_data['max_favorable_pct'].mean():>6.3%} ({year_data['target_to_max_ratio'].mean():.1f}x)")
            print()

print()
