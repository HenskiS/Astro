"""
OPTIMIZE PROFIT LADDER LEVELS
==============================
Test different ladder configurations to find optimal profit-taking levels.

We'll test:
- Different profit levels (low, medium, high)
- Different scale-out percentages
- Different numbers of levels (2 vs 3 vs 4)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("PROFIT LADDER OPTIMIZATION")
print("="*100)
print()

# Base parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60


def run_backtest_with_ladder(period_predictions, ladder_config):
    """
    Run backtest with specific ladder configuration.

    ladder_config: dict with 'levels' (list of profit %s) and 'scale_pct' (% to exit at each level)
    """

    class LadderPosition:
        def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence, config):
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
            self.max_adverse_excursion = 0
            self.trailing_stop = None
            self.daily_progression = []
            self.partial_exits = []
            self.ladder_level = 0
            self.ladder_levels = config['levels']
            self.scale_pct = config['scale_pct']

        def update(self, date, high, low, close):
            self.days_held += 1

            if self.direction == 'long':
                current_profit = (close - self.entry_price) / self.entry_price
                intraday_high_profit = (high - self.entry_price) / self.entry_price
                intraday_low_profit = (low - self.entry_price) / self.entry_price
                hit_target = high >= self.breakout_target
            else:
                current_profit = (self.entry_price - close) / self.entry_price
                intraday_high_profit = (self.entry_price - low) / self.entry_price
                intraday_low_profit = (self.entry_price - high) / self.entry_price
                hit_target = low <= self.breakout_target

            self.daily_progression.append({
                'day': self.days_held,
                'profit': current_profit,
                'high_profit': intraday_high_profit,
                'low_profit': intraday_low_profit
            })

            self.max_profit = max(self.max_profit, intraday_high_profit)
            self.max_adverse_excursion = min(self.max_adverse_excursion, intraday_low_profit)

            # PROFIT LADDER
            if self.ladder_level < len(self.ladder_levels):
                if intraday_high_profit >= self.ladder_levels[self.ladder_level]:
                    self.partial_exits.append((self.ladder_levels[self.ladder_level], self.scale_pct))
                    self.size *= (1 - self.scale_pct)
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
                if self.direction == 'long':
                    new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                    self.trailing_stop = max(self.trailing_stop, new_stop)
                    hit_stop = low <= self.trailing_stop
                else:
                    new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
                    self.trailing_stop = min(self.trailing_stop, new_stop)
                    hit_stop = high >= self.trailing_stop

                if hit_stop:
                    return 'trailing_stop', self.trailing_stop, current_profit

            # Target hit
            if hit_target:
                return 'target', self.breakout_target, current_profit

            return None

        def calculate_blended_profit(self, final_profit):
            if len(self.partial_exits) == 0:
                return final_profit

            total_weighted_profit = 0
            remaining_size_pct = 1.0

            for exit_profit, exit_pct in self.partial_exits:
                total_weighted_profit += exit_profit * exit_pct
                remaining_size_pct -= exit_pct

            total_weighted_profit += final_profit * remaining_size_pct
            return total_weighted_profit

    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    equity_curve = []
    equity_dates = []

    for date in all_dates:
        prices_dict = {}
        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz in pair_df.index:
                row = pair_df.loc[date_with_tz]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

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

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, current_profit = exit_info

            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars

            position.exit_date = date
            position.exit_price = exit_price
            position.profit_pct = profit_pct
            position.exit_reason = exit_reason

            closed_positions.append(position)
            positions.remove(position)

        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz not in pair_df.index:
                continue

            row = pair_df.loc[date_with_tz]
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
                breakout_level = row['high_20d']
                target = breakout_level * 1.005
            else:
                direction = 'short'
                breakout_level = row['low_20d']
                target = breakout_level * 0.995

            position = LadderPosition(pair, date, price, direction, position_size, target, max_prob, ladder_config)
            positions.append(position)

        equity_curve.append(capital)
        equity_dates.append(date)

    # Calculate yearly metrics
    equity_df = pd.DataFrame({'date': equity_dates, 'equity': equity_curve})
    equity_df['year'] = pd.to_datetime(equity_df['date']).dt.year

    yearly_returns = []
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) == 0:
            continue

        if year == equity_df['year'].min():
            start_capital = INITIAL_CAPITAL
        else:
            prev_year_data = equity_df[equity_df['year'] < year]
            start_capital = prev_year_data['equity'].iloc[-1] if len(prev_year_data) > 0 else INITIAL_CAPITAL

        end_capital = year_data['equity'].iloc[-1]
        year_return = (end_capital - start_capital) / start_capital
        yearly_returns.append(year_return)

    avg_return = np.mean(yearly_returns)
    profitable_years = sum(1 for r in yearly_returns if r > 0)

    # Calculate drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    dd = (equity_series - running_max) / running_max
    max_dd = dd.min()

    num_trades = len(closed_positions)
    win_rate = sum(1 for p in closed_positions if p.profit_pct > 0) / num_trades if num_trades > 0 else 0

    return {
        'avg_return': avg_return,
        'max_dd': max_dd,
        'profitable_years': profitable_years,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'positions': closed_positions
    }


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Combine all periods
all_period_predictions = {}
for period_name, period_preds in all_predictions.items():
    for pair, pred_df in period_preds.items():
        if pair not in all_period_predictions:
            all_period_predictions[pair] = pred_df
        else:
            all_period_predictions[pair] = pd.concat([all_period_predictions[pair], pred_df])

print("Testing ladder configurations...")
print()

# Define configurations to test
configs = []

# BASELINE (no ladder)
configs.append({
    'name': 'BASELINE (No Ladder)',
    'levels': [],  # No ladder
    'scale_pct': 0
})

# Original (too aggressive)
configs.append({
    'name': 'Original (0.3/0.6/0.9%, 25%)',
    'levels': [0.003, 0.006, 0.009],
    'scale_pct': 0.25
})

# Higher levels - just above typical target (~0.75%)
configs.append({
    'name': 'Above Target (0.5/0.8/1.1%, 25%)',
    'levels': [0.005, 0.008, 0.011],
    'scale_pct': 0.25
})

configs.append({
    'name': 'Above Target (0.6/1.0/1.4%, 25%)',
    'levels': [0.006, 0.010, 0.014],
    'scale_pct': 0.25
})

# Wider spacing
configs.append({
    'name': 'Wide Spacing (0.5/1.0/1.5%, 25%)',
    'levels': [0.005, 0.010, 0.015],
    'scale_pct': 0.25
})

configs.append({
    'name': 'Very Wide (0.8/1.5/2.0%, 25%)',
    'levels': [0.008, 0.015, 0.020],
    'scale_pct': 0.25
})

# 2-level ladders
configs.append({
    'name': '2-Level (0.5/1.0%, 33%)',
    'levels': [0.005, 0.010],
    'scale_pct': 0.33
})

configs.append({
    'name': '2-Level (0.6/1.2%, 33%)',
    'levels': [0.006, 0.012],
    'scale_pct': 0.33
})

configs.append({
    'name': '2-Level (0.8/1.5%, 33%)',
    'levels': [0.008, 0.015],
    'scale_pct': 0.33
})

# Smaller scale-outs (more conservative)
configs.append({
    'name': 'Conservative (0.5/1.0/1.5%, 20%)',
    'levels': [0.005, 0.010, 0.015],
    'scale_pct': 0.20
})

configs.append({
    'name': 'Conservative (0.8/1.5/2.0%, 20%)',
    'levels': [0.008, 0.015, 0.020],
    'scale_pct': 0.20
})

# Single level (simple scale-out)
configs.append({
    'name': 'Single (0.8%, 50%)',
    'levels': [0.008],
    'scale_pct': 0.50
})

configs.append({
    'name': 'Single (1.0%, 50%)',
    'levels': [0.010],
    'scale_pct': 0.50
})

configs.append({
    'name': 'Single (1.2%, 50%)',
    'levels': [0.012],
    'scale_pct': 0.50
})

# Run tests
results = []
baseline_result = None

for i, config in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] Testing: {config['name']}...")
    result = run_backtest_with_ladder(all_period_predictions, config)
    result['name'] = config['name']
    result['config'] = config
    results.append(result)

    if 'BASELINE' in config['name']:
        baseline_result = result

print()

# Display results
print("="*100)
print("LADDER CONFIGURATION COMPARISON")
print("="*100)
print()

print(f"{'Configuration':<40} {'Avg Return':>12} {'Max DD':>10} {'Years+':>8} {'Win%':>6}")
print("-" * 90)

# Sort by avg return
sorted_results = sorted(results, key=lambda x: x['avg_return'], reverse=True)

for result in sorted_results:
    years_profitable = f"{result['profitable_years']}/10"

    # Highlight if better than baseline
    marker = ""
    if baseline_result and result['name'] != baseline_result['name']:
        if result['avg_return'] > baseline_result['avg_return']:
            marker = " *"

    print(f"{result['name']:<40} {result['avg_return']:>11.1%} {result['max_dd']:>9.1%} "
          f"{years_profitable:>8} {result['win_rate']:>5.0%}{marker}")

print()

# Top 3 detailed analysis
print("="*100)
print("TOP 3 CONFIGURATIONS - DETAILED ANALYSIS")
print("="*100)
print()

for i, result in enumerate(sorted_results[:3]):
    print(f"#{i+1}: {result['name']}")
    print(f"  Average return: {result['avg_return']:.1%}")
    print(f"  Max drawdown: {result['max_dd']:.1%}")
    print(f"  Profitable years: {result['profitable_years']}/10")
    print(f"  Win rate: {result['win_rate']:.0%}")

    if baseline_result and result['name'] != baseline_result['name']:
        improvement = result['avg_return'] - baseline_result['avg_return']
        dd_change = result['max_dd'] - baseline_result['max_dd']
        print(f"  vs Baseline: {improvement:+.1%} return, {dd_change:+.1%} DD")

    # Analyze ladder usage
    positions = result['positions']
    ladder_usage = {}
    for p in positions:
        num_exits = len(p.partial_exits)
        ladder_usage[num_exits] = ladder_usage.get(num_exits, 0) + 1

    if len(ladder_usage) > 1:
        print(f"  Ladder usage:")
        for num_exits in sorted(ladder_usage.keys()):
            pct = ladder_usage[num_exits] / len(positions)
            print(f"    {num_exits} exits: {ladder_usage[num_exits]} trades ({pct:.1%})")

    # Winner/loser breakdown
    winners = [p for p in positions if p.profit_pct > 0]
    losers = [p for p in positions if p.profit_pct <= 0]
    if len(winners) > 0:
        print(f"  Winners: {len(winners)} trades, avg {np.mean([p.profit_pct for p in winners]):+.2%}")
    if len(losers) > 0:
        print(f"  Losers:  {len(losers)} trades, avg {np.mean([p.profit_pct for p in losers]):+.2%}")

    print()

print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()

best = sorted_results[0]
print(f"Best configuration: {best['name']}")
print(f"  Return improvement: {best['avg_return'] - baseline_result['avg_return']:+.1%} vs baseline")
print(f"  Config: {best['config']['levels']} @ {best['config']['scale_pct']:.0%} each")
print()

# Find best by return/DD ratio
sorted_by_ratio = sorted(results, key=lambda x: x['avg_return']/abs(x['max_dd']) if x['max_dd'] != 0 else 0, reverse=True)
best_ratio = sorted_by_ratio[0]
print(f"Best return/DD ratio: {best_ratio['name']}")
print(f"  Ratio: {best_ratio['avg_return']/abs(best_ratio['max_dd']):.2f}:1")
print(f"  Config: {best_ratio['config']['levels']} @ {best_ratio['config']['scale_pct']:.0%} each")
print()

print("="*100)
print("OPTIMIZATION COMPLETE")
print("="*100)
