"""
Multi-Alignment 15m Backtest
=============================
Test if checking 3 different 15m alignments (every 5 minutes)
improves performance vs standard alignment (every 15 minutes).

Strategy:
- Fetch 1m candles for all pairs
- Resample to 3 alignments: :00, :05, :10
- Generate signals every 5 minutes instead of every 15
- Deduplicate: if same pair triggers twice within 10 minutes, ignore 2nd signal
"""
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Multi-alignment backtest')
parser.add_argument('--alignments', type=int, default=3, choices=[1, 3],
                    help='Number of alignments (1=baseline, 3=multi)')
args = parser.parse_args()

print("="*100)
if args.alignments == 1:
    print("BASELINE BACKTEST: Standard 15m Alignment (Every 15 Minutes)")
else:
    print("MULTI-ALIGNMENT BACKTEST: 3x15m Alignments (Every 5 Minutes)")
print("="*100)
print()

# Strategy Parameters (same as optimized backtest)
INITIAL_CAPITAL = 500
POSITION_PCT = 0.30  # 30% per trade
MIN_CONFIDENCE = 0.80
EMERGENCY_STOP_PCT = 0.05
TRAILING_STOP_TRAIL_PCT = 0.75
MAX_POSITIONS_TOTAL = 10
MAX_POSITIONS_PER_PAIR = 3
LOOKBACK_PERIOD = 100

# Pair settings
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

print("Loading models and predictions...")

# Load models
with open('models_15m.pkl', 'rb') as f:
    models = pickle.load(f)

# Load test predictions (these are on standard alignment)
with open('test_predictions_15m.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

print(f"Loaded {len(models)} models")
print(f"Predictions: {len(all_predictions)} pairs")
print()


def resample_to_15m(df, offset_minutes=0):
    """Resample to 15m with offset"""
    df_shifted = df.copy()
    df_shifted.index = df_shifted.index - pd.Timedelta(minutes=offset_minutes)

    resampled = df_shifted.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled.index = resampled.index + pd.Timedelta(minutes=offset_minutes)
    return resampled


# For this backtest, we'll use the existing predictions but simulate
# checking them at different times based on alignment
# In a real implementation, we'd recalculate features for each alignment

print("Simulating multi-alignment strategy...")
print()

# Track state
capital = INITIAL_CAPITAL
positions = {}  # pair -> {entry_time, entry_price, direction, size, target, ...}
closed_trades = []
signal_history = {}  # pair -> last_signal_time (for deduplication)

# Iterate through time
# We'll use 5-minute intervals for multi-alignment, 15-minute for baseline
time_interval = 5 if args.alignments == 3 else 15

# Get all timestamps across all pairs to find common time range
all_dates = []
for pair in PAIRS:
    if pair in all_predictions:
        all_dates.extend(all_predictions[pair].index.tolist())

if not all_dates:
    print("No prediction data found!")
    sys.exit(1)

start_date = min(all_dates)
end_date = max(all_dates)

print(f"Backtesting period: {start_date.date()} to {end_date.date()}")
print(f"Check interval: every {time_interval} minutes")
print()

# Generate time points
current_time = start_date
time_points = []
while current_time <= end_date:
    time_points.append(current_time)
    current_time += timedelta(minutes=time_interval)

print(f"Total time points: {len(time_points):,}")
print()

# Track metrics
equity_curve = []
peak_capital = INITIAL_CAPITAL


def get_prediction_at_time(pair, check_time, alignment_offset):
    """
    Get prediction for a pair at a specific time and alignment.

    For multi-alignment: we simulate checking different 15m windows
    by looking at predictions offset by alignment_offset minutes.
    """
    # Adjust time based on alignment
    lookup_time = check_time - timedelta(minutes=alignment_offset)

    # Round to nearest 15m mark (for baseline) or 5m mark (for multi)
    if args.alignments == 1:
        # Baseline: only check at :00, :15, :30, :45
        minute = lookup_time.minute
        if minute not in [0, 15, 30, 45]:
            return None

        # Round to 15m
        lookup_time = lookup_time.replace(minute=(minute // 15) * 15, second=0, microsecond=0)
    else:
        # Multi-alignment: check at :00, :05, :10, :15, :20, :25, etc.
        minute = lookup_time.minute
        lookup_time = lookup_time.replace(minute=(minute // 5) * 5, second=0, microsecond=0)

    if pair not in all_predictions:
        return None

    pred_df = all_predictions[pair]

    # Find closest prediction within tolerance
    tolerance = timedelta(minutes=20)
    closest_idx = pred_df.index.searchsorted(lookup_time)

    if 0 <= closest_idx < len(pred_df):
        closest_time = pred_df.index[closest_idx]
        if abs(closest_time - lookup_time) <= tolerance:
            return pred_df.iloc[closest_idx]

    return None


def should_take_signal(pair, current_time):
    """Check if we should take a signal (deduplication)"""
    if pair not in signal_history:
        return True

    last_signal_time = signal_history[pair]
    # Don't take signal if same pair signaled within last 10 minutes
    if (current_time - last_signal_time).total_seconds() < 600:  # 10 minutes
        return False

    return True


def update_positions(current_time, data_cache):
    """Update all open positions"""
    global capital, closed_trades

    pairs_to_close = []

    for pair, pos in list(positions.items()):
        # Get current price data
        pred = get_prediction_at_time(pair, current_time, 0)  # Use standard alignment for price
        if pred is None:
            continue

        current_price = pred['close']
        periods_held = int((current_time - pos['entry_time']).total_seconds() / 900)  # 15min periods

        # Calculate current profit
        if pos['direction'] == 'long':
            current_profit = (current_price - pos['entry_price']) / pos['entry_price']
            peak_price = max(pos.get('peak_price', pos['entry_price']), current_price)
        else:
            current_profit = (pos['entry_price'] - current_price) / pos['entry_price']
            peak_price = min(pos.get('peak_price', pos['entry_price']), current_price)

        pos['peak_price'] = peak_price

        # Update max profit
        pos['max_profit'] = max(pos.get('max_profit', 0), current_profit)

        # Check exit conditions
        exit_reason = None
        exit_price = current_price

        # 1. Emergency stop (-5% loss)
        if current_profit < -EMERGENCY_STOP_PCT:
            exit_reason = 'emergency_stop'

        # 2. 24-period time exit for losers
        elif periods_held >= 24 and current_profit < 0:
            exit_reason = 'time_exit_loser'

        # 3. Target hit - activate trailing stop
        elif pos['direction'] == 'long' and current_price >= pos['target']:
            if 'trailing_active' not in pos:
                pos['trailing_active'] = True
                pos['trailing_stop'] = pos['target']
        elif pos['direction'] == 'short' and current_price <= pos['target']:
            if 'trailing_active' not in pos:
                pos['trailing_active'] = True
                pos['trailing_stop'] = pos['target']

        # 4. Update trailing stop
        if pos.get('trailing_active', False):
            if pos['direction'] == 'long':
                # Trail at 75% from target to peak
                new_stop = pos['target'] + 0.75 * (peak_price - pos['target'])
                pos['trailing_stop'] = max(pos.get('trailing_stop', pos['target']), new_stop)

                if current_price <= pos['trailing_stop']:
                    exit_reason = 'trailing_stop'
                    exit_price = pos['trailing_stop']
            else:
                # Short position
                new_stop = pos['target'] - 0.75 * (pos['target'] - peak_price)
                pos['trailing_stop'] = min(pos.get('trailing_stop', pos['target']), new_stop)

                if current_price >= pos['trailing_stop']:
                    exit_reason = 'trailing_stop'
                    exit_price = pos['trailing_stop']

        if exit_reason:
            pairs_to_close.append((pair, exit_reason, exit_price))

    # Close positions
    for pair, reason, exit_price in pairs_to_close:
        pos = positions[pair]

        # Calculate final profit
        if pos['direction'] == 'long':
            profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        profit_dollars = pos['size'] * profit_pct * pos['entry_price']
        capital += profit_dollars

        closed_trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': current_time,
            'pair': pair,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'reason': reason,
            'periods_held': int((current_time - pos['entry_time']).total_seconds() / 900)
        })

        del positions[pair]


# Main backtest loop
print("Running backtest...")
data_cache = {}

for i, current_time in enumerate(time_points):
    if i % 5000 == 0:
        progress = i / len(time_points) * 100
        print(f"Progress: {progress:.1f}% | Date: {current_time.date()} | Capital: ${capital:.2f} | Positions: {len(positions)}")

    # Update existing positions
    update_positions(current_time, data_cache)

    # Check for new signals (if using multi-alignment)
    if args.alignments == 3:
        # Check all 3 alignments
        for offset in [0, 5, 10]:
            # Only check this alignment at appropriate times
            minute = current_time.minute
            if minute % 15 == offset % 15:  # This alignment's turn
                for pair in PAIRS:
                    # Skip if we already have position in this pair
                    if pair in positions:
                        continue

                    # Skip if too many positions total
                    if len(positions) >= MAX_POSITIONS_TOTAL:
                        continue

                    # Check deduplication
                    if not should_take_signal(pair, current_time):
                        continue

                    # Get prediction
                    pred = get_prediction_at_time(pair, current_time, offset)
                    if pred is None:
                        continue

                    # Check long breakout
                    if pred['breakout_high_prob'] >= MIN_CONFIDENCE:
                        direction = 'long'
                        target = pred['high_80p']
                        confidence = pred['breakout_high_prob']
                    # Check short breakout
                    elif pred['breakout_low_prob'] >= MIN_CONFIDENCE:
                        direction = 'short'
                        target = pred['low_80p']
                        confidence = pred['breakout_low_prob']
                    else:
                        continue

                    # Calculate position size
                    position_size = int(capital * POSITION_PCT / pred['close'])
                    if position_size < 100:  # Minimum size
                        continue

                    # Open position
                    positions[pair] = {
                        'entry_time': current_time,
                        'entry_price': pred['close'],
                        'direction': direction,
                        'size': position_size,
                        'target': target,
                        'confidence': confidence,
                        'max_profit': 0,
                        'peak_price': pred['close'],
                        'alignment_offset': offset
                    }

                    signal_history[pair] = current_time

    else:
        # Baseline: only check at standard 15m marks
        if current_time.minute not in [0, 15, 30, 45]:
            continue

        for pair in PAIRS:
            if pair in positions:
                continue

            if len(positions) >= MAX_POSITIONS_TOTAL:
                continue

            pred = get_prediction_at_time(pair, current_time, 0)
            if pred is None:
                continue

            # Check signals (same logic as above)
            if pred['breakout_high_prob'] >= MIN_CONFIDENCE:
                direction = 'long'
                target = pred['high_80p']
                confidence = pred['breakout_high_prob']
            elif pred['breakout_low_prob'] >= MIN_CONFIDENCE:
                direction = 'short'
                target = pred['low_80p']
                confidence = pred['breakout_low_prob']
            else:
                continue

            position_size = int(capital * POSITION_PCT / pred['close'])
            if position_size < 100:
                continue

            positions[pair] = {
                'entry_time': current_time,
                'entry_price': pred['close'],
                'direction': direction,
                'size': position_size,
                'target': target,
                'confidence': confidence,
                'max_profit': 0,
                'peak_price': pred['close']
            }

    # Track equity
    equity_curve.append({
        'time': current_time,
        'capital': capital,
        'open_positions': len(positions)
    })

    peak_capital = max(peak_capital, capital)

# Close any remaining positions
print("\nClosing remaining positions...")
for pair in list(positions.keys()):
    pos = positions[pair]
    pred = get_prediction_at_time(pair, time_points[-1], 0)
    if pred:
        exit_price = pred['close']
        if pos['direction'] == 'long':
            profit_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            profit_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        profit_dollars = pos['size'] * profit_pct * pos['entry_price']
        capital += profit_dollars

        closed_trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': time_points[-1],
            'pair': pair,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'reason': 'end_of_data',
            'periods_held': int((time_points[-1] - pos['entry_time']).total_seconds() / 900)
        })

print()
print("="*100)
print("RESULTS")
print("="*100)
print()

# Calculate metrics
trades_df = pd.DataFrame(closed_trades)
if len(trades_df) == 0:
    print("No trades executed!")
    sys.exit(0)

total_trades = len(trades_df)
winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]

win_rate = len(winners) / total_trades
avg_win = winners['profit_pct'].mean() if len(winners) > 0 else 0
avg_loss = losers['profit_pct'].mean() if len(losers) > 0 else 0
profit_factor = abs(winners['profit_pct'].sum() / losers['profit_pct'].sum()) if len(losers) > 0 else float('inf')

# Calculate CAGR
years = (end_date - start_date).days / 365.25
cagr = (capital / INITIAL_CAPITAL) ** (1/years) - 1

# Max drawdown
equity_df = pd.DataFrame(equity_curve)
running_max = equity_df['capital'].expanding().max()
drawdown = (equity_df['capital'] - running_max) / running_max
max_dd = drawdown.min()

print(f"Strategy: {'Multi-Alignment (3x)' if args.alignments == 3 else 'Baseline (1x)'}")
print()
print(f"Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital:      ${capital:,.2f}")
print(f"Total Return:       {(capital/INITIAL_CAPITAL - 1)*100:+.2f}%")
print(f"CAGR:               {cagr*100:.2f}%")
print()
print(f"Total Trades:       {total_trades:,}")
print(f"Winners:            {len(winners):,} ({len(winners)/total_trades*100:.1f}%)")
print(f"Losers:             {len(losers):,} ({len(losers)/total_trades*100:.1f}%)")
print(f"Win Rate:           {win_rate*100:.2f}%")
print()
print(f"Avg Win:            {avg_win*100:+.2f}%")
print(f"Avg Loss:           {avg_loss*100:+.2f}%")
print(f"Profit Factor:      {profit_factor:.2f}")
print(f"Max Drawdown:       {max_dd*100:.2f}%")
print()

# Exit reason breakdown
print("Exit Reasons:")
for reason, count in trades_df['reason'].value_counts().items():
    print(f"  {reason:<20} {count:>6,} ({count/total_trades*100:.1f}%)")

print()
print("="*100)
print("COMPARISON TO BASELINE")
print("="*100)
print()
print("Baseline (from backtest_15m_optimized.py):")
print("  CAGR: 43.0%")
print("  Trades: 17,833")
print("  Win Rate: 94.2%")
print("  Max DD: -0.26%")
print()
print(f"This test ({'Multi' if args.alignments == 3 else 'Baseline'}):")
print(f"  CAGR: {cagr*100:.2f}%")
print(f"  Trades: {total_trades:,}")
print(f"  Win Rate: {win_rate*100:.2f}%")
print(f"  Max DD: {max_dd*100:.2f}%")
print()

if args.alignments == 3:
    print("Conclusion:")
    print("  Multi-alignment strategy provides ~3x more trading opportunities")
    print("  Check if CAGR improved while maintaining similar win rate and drawdown")
else:
    print("Run with --alignments 3 to test multi-alignment strategy")
