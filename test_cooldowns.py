"""
TEST COOLDOWN PERIODS
Quick test of different cooldown periods on pre-generated predictions
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("COOLDOWN PERIOD OPTIMIZATION")
print("="*100)
print()

# Load predictions
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.03
EMERGENCY_STOP_DAYS = 15

# Test different cooldown periods
COOLDOWN_PERIODS = [0, 1, 2, 3, 5, 7, 10]

class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
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

        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        if self.trailing_stop is None:
            if self.max_profit > 0.005:
                self.trailing_stop = self.entry_price
        else:
            if self.direction == 'long':
                new_stop = self.entry_price + (high - self.entry_price) * 0.5
                self.trailing_stop = max(self.trailing_stop, new_stop)
                hit_stop = low <= self.trailing_stop
            else:
                new_stop = self.entry_price - (self.entry_price - low) * 0.5
                self.trailing_stop = min(self.trailing_stop, new_stop)
                hit_stop = high >= self.trailing_stop

            if hit_stop:
                return 'trailing_stop', self.trailing_stop, current_profit

        if hit_target:
            return 'target', self.breakout_target, current_profit

        return None, close, current_profit


def run_backtest(period_name, period_preds, cooldown_days):
    """Run backtest with specified cooldown"""
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []
    pair_cooldowns = defaultdict(lambda: None)

    # Get all unique dates
    all_dates = set()
    for pair_df in period_preds.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    equity_curve = []

    for date in all_dates:
        # Update positions
        prices_dict = {}
        for pair, pair_df in period_preds.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz in pair_df.index:
                row = pair_df.loc[date_with_tz]
                prices_dict[pair] = {
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                }

        # Close positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue

            high = prices_dict[position.pair]['high']
            low = prices_dict[position.pair]['low']
            close = prices_dict[position.pair]['close']

            result = position.update(date, high, low, close)
            if result[0] is not None:
                positions_to_close.append((position, result[1], result[0]))

        for position, exit_price, reason in positions_to_close:
            profit_pct = ((exit_price - position.entry_price) / position.entry_price if position.direction == 'long'
                         else (position.entry_price - exit_price) / position.entry_price)
            profit_dollars = profit_pct * (position.size * position.entry_price)
            capital += profit_dollars

            position.exit_date = date
            position.profit_pct = profit_pct
            position.exit_reason = reason

            closed_positions.append(position)
            positions.remove(position)
            pair_cooldowns[position.pair] = date + pd.Timedelta(days=cooldown_days)

        # Generate signals
        for pair, pair_df in period_preds.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz not in pair_df.index:
                continue

            # Check cooldown
            if pair_cooldowns[pair] is not None and date < pair_cooldowns[pair]:
                continue

            row = pair_df.loc[date_with_tz]
            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= MIN_CONFIDENCE:
                continue

            # Open position
            assumed_risk_pct = 0.02
            risk_amount = capital * RISK_PER_TRADE
            price = row['close']
            position_size = risk_amount / (price * assumed_risk_pct)

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_20d']
            else:
                direction = 'short'
                breakout_level = row['low_20d']

            target = breakout_level * (1.005 if direction == 'long' else 0.995)
            position = Position(pair, date, price, direction, position_size, target, max_prob)
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
    avg_hold_days = np.mean([p.days_held for p in closed_positions]) if num_trades > 0 else 0

    return {
        'return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': num_trades,
        'win_rate': win_rate,
        'avg_hold_days': avg_hold_days
    }


# Test all cooldown periods
all_results = []

for cooldown in COOLDOWN_PERIODS:
    print(f"Testing {cooldown}-day cooldown:")
    print("-" * 80)

    period_results = []
    for period_name, period_preds in all_predictions.items():
        result = run_backtest(period_name, period_preds, cooldown)
        period_results.append({
            'period': period_name,
            **result
        })
        print(f"  {period_name}: {result['return']:+.1%} ({result['trades']} trades, "
              f"{result['win_rate']:.0%} win, {result['avg_hold_days']:.1f}d hold, {result['max_dd']:.1%} DD)")

    avg_return = np.mean([r['return'] for r in period_results])
    avg_dd = np.mean([r['max_dd'] for r in period_results])
    avg_sharpe = np.mean([r['sharpe'] for r in period_results])
    avg_trades = np.mean([r['trades'] for r in period_results])
    avg_winrate = np.mean([r['win_rate'] for r in period_results])
    avg_hold = np.mean([r['avg_hold_days'] for r in period_results])
    profitable_periods = sum(1 for r in period_results if r['return'] > 0)

    print(f"  AVERAGE: {avg_return:+.1%} ({avg_trades:.0f} trades, "
          f"{avg_winrate:.0%} win, {avg_hold:.1f}d hold, {avg_dd:.1%} DD, {profitable_periods}/5 profitable)")
    print()

    all_results.append({
        'cooldown': cooldown,
        'avg_return': avg_return,
        'avg_dd': avg_dd,
        'avg_sharpe': avg_sharpe,
        'avg_trades': avg_trades,
        'avg_winrate': avg_winrate,
        'avg_hold': avg_hold,
        'profitable_periods': profitable_periods
    })

# Summary
print("="*100)
print("COOLDOWN COMPARISON")
print("="*100)
print()
print(f"{'Cooldown':>10} {'Avg Return':>12} {'Avg DD':>10} {'Sharpe':>8} {'Trades':>8} {'Hold':>6} {'Win%':>6} {'Profitable':>11}")
print("-" * 80)

for r in all_results:
    print(f"{r['cooldown']:>9}d {r['avg_return']:>11.1%} {r['avg_dd']:>9.1%} {r['avg_sharpe']:>8.2f} "
          f"{r['avg_trades']:>8.0f} {r['avg_hold']:>5.1f}d {r['avg_winrate']:>5.0%} {r['profitable_periods']:>7}/5")

print()
print("Current baseline: 3-day cooldown")

# Find best
best_by_return = max(all_results, key=lambda x: x['avg_return'])
best_consistent = [r for r in all_results if r['profitable_periods'] == 5]
if best_consistent:
    best_consistent = max(best_consistent, key=lambda x: x['avg_return'])
    print(f"Best 100% profitable: {best_consistent['cooldown']}-day cooldown, {best_consistent['avg_return']:.1%} avg return")
else:
    print("No configuration with 100% profitable periods")

print(f"Best by return: {best_by_return['cooldown']}-day cooldown, {best_by_return['avg_return']:.1%} avg return")
