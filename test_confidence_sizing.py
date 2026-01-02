"""
TEST CONFIDENCE-BASED POSITION SIZING
======================================
Since accuracy increases with confidence, allocate more capital to high-confidence trades

Verified accuracy by confidence:
- 65-70%: 55.1% accuracy
- 70-80%: 58.1% accuracy
- 80-90%: 68.2% accuracy
- 90-100%: 79.6% accuracy
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING CONFIDENCE-BASED POSITION SIZING")
print("="*100)
print()

# Load data
print("Loading data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Fixed parameters (optimized)
INITIAL_CAPITAL = 500
BASE_RISK_PER_TRADE = 0.007  # 0.7% baseline
MIN_CONFIDENCE = 0.65
MAX_TOTAL_POSITIONS = 90
MAX_POSITIONS_PER_PAIR = 12
EMERGENCY_STOP_DAYS = 15
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.40


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

    def update(self, date, high, low, close):
        self.days_held += 1

        if self.direction == 'long':
            current_profit = (close - self.entry_price) / self.entry_price
            intraday_high_profit = (high - self.entry_price) / self.entry_price
            hit_target = high >= self.breakout_target
        else:
            current_profit = (self.entry_price - close) / self.entry_price
            intraday_high_profit = (self.entry_price - low) / self.entry_price
            hit_target = low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        # Check ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
                self.trailing_stop = min(self.trailing_stop, new_stop)

        # Target
        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None

    def calculate_blended_profit(self, final_profit):
        if len(self.partial_exits) == 0:
            return final_profit
        total = 0
        remaining = 1.0
        for exit_profit, exit_pct in self.partial_exits:
            total += exit_profit * exit_pct
            remaining -= exit_pct
        total += final_profit * remaining
        return total


def calculate_risk_multiplier(confidence, sizing_method):
    """Calculate position size multiplier based on confidence and method"""

    if sizing_method == 'flat':
        return 1.0

    elif sizing_method == 'linear':
        # Linear: 0.5x at 0.65 -> 1.5x at 0.95+
        if confidence < 0.65:
            return 0.5
        elif confidence >= 0.95:
            return 1.5
        else:
            # Linear interpolation
            return 0.5 + (confidence - 0.65) * (1.5 - 0.5) / (0.95 - 0.65)

    elif sizing_method == 'conservative_linear':
        # Conservative: 0.7x at 0.65 -> 1.3x at 0.95+
        if confidence < 0.65:
            return 0.7
        elif confidence >= 0.95:
            return 1.3
        else:
            return 0.7 + (confidence - 0.65) * (1.3 - 0.7) / (0.95 - 0.65)

    elif sizing_method == 'aggressive_linear':
        # Aggressive: 0.3x at 0.65 -> 2.0x at 0.95+
        if confidence < 0.65:
            return 0.3
        elif confidence >= 0.95:
            return 2.0
        else:
            return 0.3 + (confidence - 0.65) * (2.0 - 0.3) / (0.95 - 0.65)

    elif sizing_method == 'tiered':
        # Tiered buckets
        if confidence < 0.70:
            return 0.6  # Low confidence: 60%
        elif confidence < 0.80:
            return 1.0  # Medium confidence: 100%
        elif confidence < 0.90:
            return 1.3  # High confidence: 130%
        else:
            return 1.6  # Very high confidence: 160%

    elif sizing_method == 'exponential':
        # Exponential: Grows faster at high confidence
        # 0.65 -> 0.5x, 0.75 -> 0.9x, 0.85 -> 1.3x, 0.95 -> 2.0x
        normalized = (confidence - 0.65) / (0.95 - 0.65)
        return 0.5 + 1.5 * (normalized ** 2)

    elif sizing_method == 'kelly_inspired':
        # Kelly-inspired: Scale by edge
        # Estimated accuracy by confidence (from our analysis):
        if confidence < 0.70:
            accuracy = 0.551
        elif confidence < 0.80:
            accuracy = 0.581
        elif confidence < 0.90:
            accuracy = 0.682
        else:
            accuracy = 0.796

        edge = (accuracy - 0.5) * 2  # Convert to edge
        multiplier = 0.5 + edge * 2  # Scale conservatively
        return max(0.3, min(2.0, multiplier))  # Cap between 0.3x and 2.0x

    return 1.0


def run_backtest_with_sizing(sizing_method):
    """Run full backtest with specific sizing method"""
    capital = INITIAL_CAPITAL
    carried_positions = []
    total_trades = 0
    trades = []
    equity_curve = [INITIAL_CAPITAL]

    for quarter_name, quarter_preds in sorted(all_predictions.items()):
        # Get trading days
        all_trading_days = set()
        prediction_dates = set()

        for pair_df in quarter_preds.values():
            prediction_dates.update(pair_df.index)

        for pair, pair_df in all_raw_data.items():
            if len(prediction_dates) > 0:
                min_date = min(prediction_dates)
                max_date = max(prediction_dates)
                trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
                all_trading_days.update(trading_days)

        all_trading_days = sorted(list(all_trading_days))

        for date in all_trading_days:
            # Get prices
            prices_dict = {}
            for pair, pair_df in all_raw_data.items():
                if date in pair_df.index:
                    row = pair_df.loc[date]
                    prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

            # Update positions
            positions_to_close = []
            for position in carried_positions:
                if position.pair not in prices_dict:
                    continue
                high = prices_dict[position.pair]['high']
                low = prices_dict[position.pair]['low']
                close = prices_dict[position.pair]['close']
                exit_info = position.update(date, high, low, close)
                if exit_info is not None:
                    positions_to_close.append((position, exit_info))

            # Close positions
            for position, exit_info in positions_to_close:
                exit_reason, exit_price, current_profit = exit_info

                if position.direction == 'long':
                    raw_profit = (exit_price - position.entry_price) / position.entry_price
                else:
                    raw_profit = (position.entry_price - exit_price) / position.entry_price

                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)

                capital += profit_dollars
                carried_positions.remove(position)
                total_trades += 1
                equity_curve.append(capital)

                trades.append({
                    'confidence': position.confidence,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

            # Open new positions
            if date not in prediction_dates:
                continue

            if len(carried_positions) >= MAX_TOTAL_POSITIONS:
                continue

            for pair, pair_df in quarter_preds.items():
                if date not in pair_df.index:
                    continue

                pair_positions = [p for p in carried_positions if p.pair == pair]
                if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                    continue

                row = pair_df.loc[date]

                breakout_high_prob = row['breakout_high_prob']
                breakout_low_prob = row['breakout_low_prob']
                max_prob = max(breakout_high_prob, breakout_low_prob)

                if max_prob <= MIN_CONFIDENCE:
                    continue

                # CONFIDENCE-BASED SIZING
                risk_multiplier = calculate_risk_multiplier(max_prob, sizing_method)
                assumed_risk_pct = 0.02
                risk_amount = capital * BASE_RISK_PER_TRADE * risk_multiplier  # Scale by confidence
                price = row['close']
                position_size = risk_amount / (price * assumed_risk_pct)

                if breakout_high_prob > breakout_low_prob:
                    direction = 'long'
                    breakout_level = row['high_20d']
                    target = breakout_level * 1.005
                else:
                    direction = 'short'
                    breakout_level = row['low_20d']
                    target = breakout_level * 0.995

                position = Position(pair, date, price, direction, position_size, target, max_prob)
                carried_positions.append(position)

    # Calculate max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for equity in equity_curve:
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    return capital, total_trades, trades, max_dd


print("="*100)
print("TESTING SIZING METHODS")
print("="*100)
print()

sizing_methods = [
    ('flat', 'Flat 0.7% (baseline)'),
    ('conservative_linear', 'Conservative linear (0.7x-1.3x)'),
    ('linear', 'Linear (0.5x-1.5x)'),
    ('aggressive_linear', 'Aggressive linear (0.3x-2.0x)'),
    ('tiered', 'Tiered (0.6x/1.0x/1.3x/1.6x)'),
    ('exponential', 'Exponential (0.5x-2.0x)'),
    ('kelly_inspired', 'Kelly-inspired (by edge)'),
]

results = []

for method, label in sizing_methods:
    print(f"Testing: {label}...")
    cap, trades, trade_list, max_dd = run_backtest_with_sizing(method)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL

    results.append({
        'method': method,
        'label': label,
        'final_capital': cap,
        'total_return': total_return,
        'total_trades': trades,
        'max_dd': max_dd,
        'trade_list': trade_list
    })

    print(f"  Result: ${cap:>10,.0f} ({total_return:>7.1%}) | DD: {max_dd:>6.1%} | {trades:>5,} trades")
    print()

print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()

# Sort by final capital
results_sorted = sorted(results, key=lambda x: x['final_capital'], reverse=True)

baseline_cap = results[0]['final_capital']
baseline_dd = results[0]['max_dd']

print("Ranking | Method                          | Final Capital | Return   | DD       | vs Baseline")
print("-" * 100)
for i, r in enumerate(results_sorted, 1):
    improvement = r['final_capital'] - baseline_cap
    dd_change = r['max_dd'] - baseline_dd
    print(f"  {i}     | {r['label']:30s} | ${r['final_capital']:>12,.0f} | {r['total_return']:>7.1%} | {r['max_dd']:>7.1%} | {improvement:>+10,.0f} ({dd_change:>+5.1%} DD)")

print()

# Best method
best = results_sorted[0]
print(f"Best method: {best['label']}")
print(f"  Final capital: ${best['final_capital']:,.0f}")
print(f"  Total return: {best['total_return']:.1%}")
print(f"  Max drawdown: {best['max_dd']:.1%}")
print(f"  Improvement over baseline: ${best['final_capital'] - baseline_cap:+,.0f} ({(best['final_capital'] - baseline_cap) / baseline_cap:+.1%})")
print()

# Analyze best method's trades by confidence
if best['method'] != 'flat':
    print("="*100)
    print(f"TRADE ANALYSIS - {best['label']}")
    print("="*100)
    print()

    trades_df = pd.DataFrame(best['trade_list'])

    # Group by confidence buckets
    trades_df['conf_bucket'] = pd.cut(trades_df['confidence'],
                                       bins=[0.65, 0.70, 0.80, 0.90, 1.0],
                                       labels=['65-70%', '70-80%', '80-90%', '90-100%'])

    print("Performance by confidence level:")
    print("-" * 80)
    print("Confidence | Trades | Win Rate | Avg Profit | Sizing Multiplier")
    print("-" * 80)

    for bucket in ['65-70%', '70-80%', '80-90%', '90-100%']:
        subset = trades_df[trades_df['conf_bucket'] == bucket]
        if len(subset) > 0:
            win_rate = (subset['profit_pct'] > 0).mean()
            avg_profit = subset['profit_pct'].mean()
            avg_conf = subset['confidence'].mean()
            multiplier = calculate_risk_multiplier(avg_conf, best['method'])

            print(f"{bucket:10s} | {len(subset):>6,} | {win_rate:>7.1%} | {avg_profit:>+9.2%} | {multiplier:>16.2f}x")

    print()

print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

if best['final_capital'] > baseline_cap * 1.05:
    print(f"STRONG RECOMMENDATION: Adopt {best['label']}")
    print(f"  Improves returns by {(best['final_capital'] - baseline_cap) / baseline_cap:+.1%}")
    print(f"  Drawdown change: {best['max_dd'] - baseline_dd:+.1%}")
elif best['final_capital'] > baseline_cap:
    print(f"CONSIDER: {best['label']}")
    print(f"  Modest improvement of {(best['final_capital'] - baseline_cap) / baseline_cap:+.1%}")
else:
    print("Keep flat sizing - confidence-based sizing doesn't improve performance")

print()
print("="*100)
