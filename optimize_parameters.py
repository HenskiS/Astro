"""
COMPREHENSIVE PARAMETER OPTIMIZATION
Test different combinations of trading parameters to maximize performance
Uses pre-generated predictions for fast iteration
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
from itertools import product
warnings.filterwarnings('ignore')

print("="*100)
print("COMPREHENSIVE PARAMETER OPTIMIZATION")
print("="*100)
print()

INITIAL_CAPITAL = 100000

# Parameter grid to test (reduced for faster testing)
PARAM_GRID = {
    'min_confidence': [0.70, 0.75],  # Focus on 70-75%
    'risk_per_trade': [0.005, 0.007],  # 0.5%, 0.7%
    'emergency_stop_pct': [-0.02, -0.03, -0.04],  # Test all stop levels
    'emergency_stop_days': [10, 15],  # 10 or 15 days
    'trailing_trigger': [0.005, 0.007],  # 0.5%, 0.7%
    'trailing_pct': [0.50, 0.60],  # 50%, 60%
}

# Calculate total combinations
total_combos = np.prod([len(v) for v in PARAM_GRID.values()])
print(f"Testing {total_combos:,} parameter combinations")
print()


class Position:
    """Trading position with configurable parameters"""

    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target,
                 confidence, emergency_stop_pct, emergency_stop_days, trailing_trigger, trailing_pct):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.trailing_stop = None

        # Configurable parameters
        self.emergency_stop_pct = emergency_stop_pct
        self.emergency_stop_days = emergency_stop_days
        self.trailing_trigger = trailing_trigger
        self.trailing_pct = trailing_pct

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

        # Emergency stop
        if self.days_held >= self.emergency_stop_days and current_profit < self.emergency_stop_pct:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > self.trailing_trigger:
                self.trailing_stop = self.entry_price
        else:
            if self.direction == 'long':
                new_stop = self.entry_price + (high - self.entry_price) * self.trailing_pct
                self.trailing_stop = max(self.trailing_stop, new_stop)
                hit_stop = low <= self.trailing_stop
            else:
                new_stop = self.entry_price - (self.entry_price - low) * self.trailing_pct
                self.trailing_stop = min(self.trailing_stop, new_stop)
                hit_stop = high >= self.trailing_stop

            if hit_stop:
                return 'trailing_stop', self.trailing_stop, current_profit

        # Target hit
        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None, close, current_profit


def run_backtest(period_predictions, params):
    """Run backtest with specific parameter set"""
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    # Get all unique dates
    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        return {'return': 0, 'max_dd': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0}

    equity_curve = []

    for date in all_dates:
        # Get prices
        prices_dict = {}
        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz in pair_df.index:
                row = pair_df.loc[date_with_tz]
                prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

        # Update and close positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            high = prices_dict[position.pair]['high']
            low = prices_dict[position.pair]['low']
            close = prices_dict[position.pair]['close']

            exit_reason, exit_price, current_profit = position.update(date, high, low, close)
            if exit_reason is not None:
                positions_to_close.append((position, exit_price, exit_reason))

        for position, exit_price, reason in positions_to_close:
            if position.direction == 'long':
                profit_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                profit_pct = (position.entry_price - exit_price) / position.entry_price

            profit_dollars = profit_pct * (position.size * position.entry_price)
            capital += profit_dollars

            position.profit_pct = profit_pct
            closed_positions.append(position)
            positions.remove(position)

        # Generate signals
        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz not in pair_df.index:
                continue

            row = pair_df.loc[date_with_tz]
            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= params['min_confidence']:
                continue

            # Calculate position size
            assumed_risk_pct = 0.02
            risk_amount = capital * params['risk_per_trade']
            price = row['close']
            position_size = risk_amount / (price * assumed_risk_pct)

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                target = row['high_20d'] * 1.005
            else:
                direction = 'short'
                target = row['low_20d'] * 0.995

            position = Position(
                pair, date, price, direction, position_size, target, max_prob,
                params['emergency_stop_pct'], params['emergency_stop_days'],
                params['trailing_trigger'], params['trailing_pct']
            )
            positions.append(position)

        equity_curve.append(capital)

    # Calculate metrics
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        dd = (equity_series - running_max) / running_max
        max_dd = dd.min()

        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    else:
        max_dd = 0
        sharpe = 0

    num_trades = len(closed_positions)
    win_rate = sum(1 for p in closed_positions if p.profit_pct > 0) / num_trades if num_trades > 0 else 0

    return {
        'return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': num_trades,
        'win_rate': win_rate
    }


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Generate all parameter combinations
param_names = list(PARAM_GRID.keys())
param_values = [PARAM_GRID[k] for k in param_names]
all_param_combos = list(product(*param_values))

print(f"Testing {len(all_param_combos):,} combinations across {len(all_predictions)} periods")
print()

# Test all combinations
best_configs = []
tested = 0

for param_combo in all_param_combos:
    params = dict(zip(param_names, param_combo))

    # Run on all periods
    period_results = []
    for period_name, period_preds in all_predictions.items():
        result = run_backtest(period_preds, params)
        period_results.append(result)

    # Aggregate metrics
    avg_return = np.mean([r['return'] for r in period_results])
    avg_sharpe = np.mean([r['sharpe'] for r in period_results])
    avg_dd = np.mean([r['max_dd'] for r in period_results])
    avg_trades = np.mean([r['trades'] for r in period_results])
    avg_winrate = np.mean([r['win_rate'] for r in period_results])
    profitable_periods = sum(1 for r in period_results if r['return'] > 0)

    best_configs.append({
        **params,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_dd': avg_dd,
        'avg_trades': avg_trades,
        'avg_winrate': avg_winrate,
        'profitable_periods': profitable_periods,
        'return_dd_ratio': avg_return / abs(avg_dd) if avg_dd != 0 else 0
    })

    tested += 1
    if tested % 100 == 0:
        print(f"  Progress: {tested}/{len(all_param_combos)} ({tested/len(all_param_combos)*100:.1f}%)")

print()
print("="*100)
print("TOP 10 CONFIGURATIONS BY AVERAGE RETURN")
print("="*100)
print()

# Sort by return
best_configs.sort(key=lambda x: x['avg_return'], reverse=True)

print(f"{'Rank':<6} {'Conf':>4} {'Risk':>5} {'EmgStop':>8} {'EmgDays':>8} {'TrlTrig':>8} {'TrlPct':>7} "
      f"{'Return':>8} {'Sharpe':>7} {'DD':>7} {'Trades':>7} {'Win%':>5} {'Prof':>5}")
print("-" * 110)

for i, config in enumerate(best_configs[:10], 1):
    print(f"{i:<6} {config['min_confidence']*100:>3.0f}% {config['risk_per_trade']*100:>4.1f}% "
          f"{config['emergency_stop_pct']*100:>7.0f}% {config['emergency_stop_days']:>8.0f} "
          f"{config['trailing_trigger']*100:>7.1f}% {config['trailing_pct']*100:>6.0f}% "
          f"{config['avg_return']:>7.1%} {config['avg_sharpe']:>7.2f} {config['avg_dd']:>6.1%} "
          f"{config['avg_trades']:>7.0f} {config['avg_winrate']:>4.0%} "
          f"{config['profitable_periods']:>2.0f}/5")

print()
print("="*100)
print("TOP 10 CONFIGURATIONS BY SHARPE RATIO")
print("="*100)
print()

# Sort by Sharpe
best_configs.sort(key=lambda x: x['avg_sharpe'], reverse=True)

print(f"{'Rank':<6} {'Conf':>4} {'Risk':>5} {'EmgStop':>8} {'EmgDays':>8} {'TrlTrig':>8} {'TrlPct':>7} "
      f"{'Return':>8} {'Sharpe':>7} {'DD':>7} {'Trades':>7} {'Win%':>5} {'Prof':>5}")
print("-" * 110)

for i, config in enumerate(best_configs[:10], 1):
    print(f"{i:<6} {config['min_confidence']*100:>3.0f}% {config['risk_per_trade']*100:>4.1f}% "
          f"{config['emergency_stop_pct']*100:>7.0f}% {config['emergency_stop_days']:>8.0f} "
          f"{config['trailing_trigger']*100:>7.1f}% {config['trailing_pct']*100:>6.0f}% "
          f"{config['avg_return']:>7.1%} {config['avg_sharpe']:>7.2f} {config['avg_dd']:>6.1%} "
          f"{config['avg_trades']:>7.0f} {config['avg_winrate']:>4.0%} "
          f"{config['profitable_periods']:>2.0f}/5")

print()
print("="*100)
print("CURRENT BASELINE")
print("="*100)
print()
print("Min Confidence: 70%")
print("Risk per Trade: 0.5%")
print("Emergency Stop: -3% after 15 days")
print("Trailing Trigger: 0.5%")
print("Trailing %: 50%")
print()
print("Performance: +85.2% return, 1.40 Sharpe, -23.7% DD, 5/5 profitable")
print()
print("Compare top configurations to baseline to see if improvements are significant.")
