"""
OPTIMIZE 1-HOUR STRATEGY PARAMETERS
====================================
Systematically test different parameter combinations to find optimal balance
between hold time, returns, and drawdown
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("OPTIMIZING 1-HOUR STRATEGY PARAMETERS")
print("="*100)
print()

# Load data once
print("Loading data...")
with open('test_predictions_1h.pkl', 'rb') as f:
    predictions = pickle.load(f)

DATA_DIR = 'data_1h'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1h.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
MIN_CONFIDENCE = 0.65
MAX_TOTAL_POSITIONS = 40
MAX_POSITIONS_PER_PAIR = 10
LADDER_SCALE_PCT = 0.40


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence,
                 emergency_hours, emergency_loss, trailing_trigger, trailing_pct, ladder_levels, target_mult):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.hours_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

        # Configurable parameters
        self.emergency_hours = emergency_hours
        self.emergency_loss = emergency_loss
        self.trailing_trigger = trailing_trigger
        self.trailing_pct = trailing_pct
        self.ladder_levels = ladder_levels
        self.target_mult = target_mult

    def update(self, date, high, low, close):
        self.hours_held += 1

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
        if self.ladder_level < len(self.ladder_levels):
            if intraday_high_profit >= self.ladder_levels[self.ladder_level]:
                self.partial_exits.append((self.ladder_levels[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.hours_held >= self.emergency_hours and current_profit < self.emergency_loss:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > self.trailing_trigger:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (high - self.entry_price) * self.trailing_pct
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - low) * self.trailing_pct
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


def run_backtest(emergency_hours, emergency_loss, trailing_trigger, trailing_pct,
                 ladder_levels, target_mult):
    """Run backtest with specific parameters"""

    capital = INITIAL_CAPITAL
    positions = []
    all_trades = []
    equity_curve = [INITIAL_CAPITAL]

    # Get all trading hours
    all_trading_hours = set()
    for pair in PAIRS:
        all_trading_hours.update(predictions[pair].index)

    min_date = min(all_trading_hours)
    max_date = max(all_trading_hours)

    for pair in PAIRS:
        pair_df = all_raw_data[pair]
        hours_in_range = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
        all_trading_hours.update(hours_in_range)

    all_trading_hours = sorted(list(all_trading_hours))

    for date in all_trading_hours:
        # Get prices
        prices_dict = {}
        for pair in PAIRS:
            if date in all_raw_data[pair].index:
                row = all_raw_data[pair].loc[date]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

        # Update positions
        positions_to_close = []
        for position in positions:
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
            positions.remove(position)

            all_trades.append({
                'hours_held': position.hours_held,
                'profit_pct': profit_pct,
                'exit_reason': exit_reason
            })

            equity_curve.append(capital)

        # Open new positions
        if len(positions) >= MAX_TOTAL_POSITIONS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue

            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            assumed_risk_pct = 0.02
            risk_amount = capital * RISK_PER_TRADE
            price = row['close']
            position_size = risk_amount / (price * assumed_risk_pct)

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_80h']
                target = breakout_level * (1 + target_mult)
            else:
                direction = 'short'
                breakout_level = row['low_80h']
                target = breakout_level * (1 - target_mult)

            position = Position(pair, date, price, direction, position_size, target, max_prob,
                              emergency_hours, emergency_loss, trailing_trigger, trailing_pct,
                              ladder_levels, target_mult)
            positions.append(position)

    # Calculate stats
    if len(all_trades) == 0:
        return None

    trades_df = pd.DataFrame(all_trades)
    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate = len(winners) / len(trades_df)
    avg_hold_winners = winners['hours_held'].mean() if len(winners) > 0 else 0
    avg_hold_losers = losers['hours_held'].mean() if len(losers) > 0 else 0

    return {
        'capital': capital,
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'num_trades': len(trades_df),
        'num_winners': len(winners),
        'num_losers': len(losers),
        'avg_hold_winners': avg_hold_winners,
        'avg_hold_losers': avg_hold_losers
    }


# Baseline (standard strategy)
print("Running baseline (standard strategy)...")
baseline = run_backtest(
    emergency_hours=120,
    emergency_loss=-0.04,
    trailing_trigger=0.0025,
    trailing_pct=0.60,
    ladder_levels=[0.004, 0.008],
    target_mult=0.005
)
print(f"Baseline: ${baseline['capital']:,.0f} | Return: {baseline['return']:.1%} | DD: {baseline['max_dd']:.1%} | "
      f"WR: {baseline['win_rate']:.1%} | Avg Hold Winners: {baseline['avg_hold_winners']:.1f}h")
print()

# TEST 1: Trailing Stop Percentage
print("="*100)
print("TEST 1: TRAILING STOP PERCENTAGE")
print("="*100)
print()

trailing_pcts = [0.60, 0.65, 0.70, 0.75]
print("Testing different trailing stop percentages (keeping trigger at 0.0025):")
for pct in trailing_pcts:
    result = run_backtest(120, -0.04, 0.0025, pct, [0.004, 0.008], 0.005)
    improvement = result['capital'] - baseline['capital']
    hold_reduction = baseline['avg_hold_winners'] - result['avg_hold_winners']
    print(f"  {pct:.0%}: ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | DD: {result['max_dd']:>5.1%} | "
          f"WR: {result['win_rate']:>5.1%} | Hold: {result['avg_hold_winners']:>5.1f}h ({hold_reduction:>+5.1f}h) | "
          f"Improvement: ${improvement:>+6,.0f}")
print()

# TEST 2: Trailing Stop Trigger
print("="*100)
print("TEST 2: TRAILING STOP TRIGGER")
print("="*100)
print()

triggers = [0.0015, 0.002, 0.0025, 0.003]
print("Testing different trailing triggers (keeping pct at 60%):")
for trigger in triggers:
    result = run_backtest(120, -0.04, trigger, 0.60, [0.004, 0.008], 0.005)
    improvement = result['capital'] - baseline['capital']
    hold_reduction = baseline['avg_hold_winners'] - result['avg_hold_winners']
    print(f"  {trigger:.3f}: ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | DD: {result['max_dd']:>5.1%} | "
          f"WR: {result['win_rate']:>5.1%} | Hold: {result['avg_hold_winners']:>5.1f}h ({hold_reduction:>+5.1f}h) | "
          f"Improvement: ${improvement:>+6,.0f}")
print()

# TEST 3: Emergency Stop Hours
print("="*100)
print("TEST 3: EMERGENCY STOP HOURS")
print("="*100)
print()

emergency_hours_list = [60, 72, 84, 96, 120]
print("Testing different emergency stop hours (keeping loss at -4%):")
for hours in emergency_hours_list:
    result = run_backtest(hours, -0.04, 0.0025, 0.60, [0.004, 0.008], 0.005)
    improvement = result['capital'] - baseline['capital']
    hold_reduction_losers = baseline['avg_hold_losers'] - result['avg_hold_losers']
    print(f"  {hours:3d}h: ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | DD: {result['max_dd']:>5.1%} | "
          f"WR: {result['win_rate']:>5.1%} | Loser Hold: {result['avg_hold_losers']:>5.1f}h ({hold_reduction_losers:>+5.1f}h) | "
          f"Improvement: ${improvement:>+6,.0f}")
print()

# TEST 4: Target Distance
print("="*100)
print("TEST 4: TARGET DISTANCE")
print("="*100)
print()

targets = [0.003, 0.004, 0.005, 0.006]
print("Testing different target distances:")
for target in targets:
    result = run_backtest(120, -0.04, 0.0025, 0.60, [0.004, 0.008], target)
    improvement = result['capital'] - baseline['capital']
    print(f"  {target:.3f}: ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | DD: {result['max_dd']:>5.1%} | "
          f"WR: {result['win_rate']:>5.1%} | Hold: {result['avg_hold_winners']:>5.1f}h | "
          f"Improvement: ${improvement:>+6,.0f}")
print()

# TEST 5: Ladder Levels
print("="*100)
print("TEST 5: LADDER LEVELS")
print("="*100)
print()

ladders = [
    [0.003, 0.006],  # Fast
    [0.0035, 0.007],  # Medium-fast
    [0.004, 0.008],  # Standard
    [0.005, 0.010],  # Patient
]
print("Testing different ladder levels:")
for ladder in ladders:
    result = run_backtest(120, -0.04, 0.0025, 0.60, ladder, 0.005)
    improvement = result['capital'] - baseline['capital']
    print(f"  {ladder}: ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | DD: {result['max_dd']:>5.1%} | "
          f"WR: {result['win_rate']:>5.1%} | Hold: {result['avg_hold_winners']:>5.1f}h | "
          f"Improvement: ${improvement:>+6,.0f}")
print()

# TEST 6: Best Combinations
print("="*100)
print("TEST 6: PROMISING COMBINATIONS")
print("="*100)
print()

combos = [
    (120, -0.04, 0.0025, 0.60, [0.004, 0.008], 0.005, "Baseline (standard)"),
    (96, -0.04, 0.0025, 0.65, [0.004, 0.008], 0.005, "Tighter trail + faster emergency"),
    (84, -0.04, 0.0025, 0.67, [0.0035, 0.007], 0.004, "Balanced: faster exits all around"),
    (72, -0.035, 0.002, 0.68, [0.0035, 0.007], 0.004, "Aggressive: quick in/out"),
    (96, -0.04, 0.002, 0.65, [0.004, 0.008], 0.005, "Lower trigger + tighter trail"),
]

print("Testing promising parameter combinations:")
results = []
for hours, loss, trigger, trail_pct, ladder, target, label in combos:
    result = run_backtest(hours, loss, trigger, trail_pct, ladder, target)
    result['label'] = label
    result['params'] = (hours, loss, trigger, trail_pct, ladder, target)
    results.append(result)

    improvement = result['capital'] - baseline['capital']
    hold_reduction = baseline['avg_hold_winners'] - result['avg_hold_winners']

    print(f"\n{label}:")
    print(f"  Capital: ${result['capital']:,.0f} ({result['return']:+.1%}) | Improvement: ${improvement:+,.0f}")
    print(f"  Max DD: {result['max_dd']:.1%} | Win Rate: {result['win_rate']:.1%}")
    print(f"  Trades: {result['num_trades']} ({result['num_winners']}W/{result['num_losers']}L)")
    print(f"  Avg Hold Winners: {result['avg_hold_winners']:.1f}h ({result['avg_hold_winners']/24:.1f}d) | Reduction: {hold_reduction:+.1f}h")
    print(f"  Avg Hold Losers: {result['avg_hold_losers']:.1f}h ({result['avg_hold_losers']/24:.1f}d)")

print()
print("="*100)
print("SUMMARY - RANKED BY CAPITAL")
print("="*100)
print()

results_sorted = sorted(results, key=lambda x: x['capital'], reverse=True)
for i, result in enumerate(results_sorted, 1):
    improvement = result['capital'] - baseline['capital']
    hold_reduction = baseline['avg_hold_winners'] - result['avg_hold_winners']
    print(f"{i}. {result['label']:40s} ${result['capital']:>7,.0f} ({result['return']:>+6.1%}) | "
          f"DD: {result['max_dd']:>5.1%} | WR: {result['win_rate']:>5.1%} | "
          f"Hold: {result['avg_hold_winners']:>5.1f}h ({hold_reduction:>+5.1f}h)")

print()
print("="*100)
print("BEST CONFIGURATION")
print("="*100)
print()

best = results_sorted[0]
hours, loss, trigger, trail_pct, ladder, target = best['params']
print(f"Label: {best['label']}")
print(f"Parameters:")
print(f"  Emergency Stop: {hours} hours @ {loss:.1%}")
print(f"  Trailing: Trigger {trigger:.3f}, Trail {trail_pct:.0%}")
print(f"  Ladder: {ladder}")
print(f"  Target: {target:.3f} ({target:.1%})")
print()
print(f"Results:")
print(f"  Capital: ${best['capital']:,.0f}")
print(f"  Return: {best['return']:+.1%}")
print(f"  Max DD: {best['max_dd']:.1%}")
print(f"  Win Rate: {best['win_rate']:.1%}")
print(f"  Avg Hold Winners: {best['avg_hold_winners']:.1f}h ({best['avg_hold_winners']/24:.1f}d)")
print(f"  Avg Hold Losers: {best['avg_hold_losers']:.1f}h ({best['avg_hold_losers']/24:.1f}d)")
print()

print("="*100)
print("DONE!")
print("="*100)
