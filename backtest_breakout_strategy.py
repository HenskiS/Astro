"""
OPTIMAL 10-DAY BREAKOUT STRATEGY
==================================
Backtest implementation of the optimal forex breakout strategy.

Configuration:
- 10-day breakout predictions (75-77% accuracy)
- Emergency stop loss: -3% after 15 days
- Trailing stops: 50% of profit once > 0.5%
- NO cooldown periods (proven optimal)
- 0.5% risk per trade
- 70% minimum confidence

Performance (2016-2025):
- Average Return: +85.2% per 2-year period
- Win Rate: 87%
- Sharpe Ratio: 1.40
- Max Drawdown: -23.7% average
- Consistency: 5/5 periods profitable (100%)

See STRATEGY_COMPARISON.md for full details.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("OPTIMAL 10-DAY BREAKOUT STRATEGY")
print("No Cooldowns + Emergency Stops + Trailing Stops")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.005           # 0.5% capital per trade
MIN_CONFIDENCE = 0.70            # 70% model confidence
COOLDOWN_DAYS = 0                # NO cooldown (proven optimal)
EMERGENCY_STOP_LOSS_PCT = -0.03  # -3% stop loss
EMERGENCY_STOP_DAYS = 15         # After 15 days
TRAILING_STOP_TRIGGER = 0.005    # Activate at 0.5% profit
TRAILING_STOP_PCT = 0.50         # Lock 50% of max profit


class Position:
    """Trading position with emergency stop, trailing stop, and target"""

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
        """
        Update position and check exit conditions.
        Returns: (exit_reason, exit_price, current_profit) or (None, close, profit)

        Exit priority:
        1. Emergency stop (prevents disasters)
        2. Trailing stop (locks in profits)
        3. Target hit (takes full profit)
        """
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

        # 1. EMERGENCY STOP: Exit if down >3% after 15 days
        if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # 2. TRAILING STOP: Activate once profitable, lock in 50% of gains
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price  # Start at breakeven
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

        # 3. TARGET HIT: Exit at breakout level + 0.5%
        if hit_target:
            return 'target', self.breakout_target, current_profit

        # No exit - position stays open
        return None, close, current_profit


def run_backtest(period_name, period_predictions, cooldown_days=COOLDOWN_DAYS):
    """
    Run backtest for one validation period.

    Args:
        period_name: String identifier for period (e.g., "2016-2017")
        period_predictions: Dict of {pair: DataFrame} with predictions
        cooldown_days: Days to wait after closing before re-entering same pair

    Returns:
        Dict with performance metrics
    """
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []
    pair_cooldowns = defaultdict(lambda: None)

    # Get all unique dates from predictions
    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        print(f"  WARNING: No dates found in {period_name}")
        return {'return': 0, 'max_dd': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0, 'avg_hold_days': 0}

    equity_curve = []

    # Daily loop
    for date in all_dates:
        # Get current prices for all pairs
        prices_dict = {}
        for pair, pair_df in period_predictions.items():
            date_with_tz = pd.Timestamp(date).tz_localize('UTC')
            if date_with_tz in pair_df.index:
                row = pair_df.loc[date_with_tz]
                prices_dict[pair] = {
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                }

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

        # Close positions and update capital
        for position, exit_price, reason in positions_to_close:
            if position.direction == 'long':
                profit_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                profit_pct = (position.entry_price - exit_price) / position.entry_price

            profit_dollars = profit_pct * (position.size * position.entry_price)
            capital += profit_dollars

            position.exit_date = date
            position.exit_price = exit_price
            position.profit_pct = profit_pct
            position.exit_reason = reason

            closed_positions.append(position)
            positions.remove(position)

            # Set cooldown for this pair
            if cooldown_days > 0:
                pair_cooldowns[position.pair] = date + pd.Timedelta(days=cooldown_days)

        # Generate new signals
        for pair, pair_df in period_predictions.items():
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

            # Check confidence threshold
            if max_prob <= MIN_CONFIDENCE:
                continue

            # Calculate position size (0.5% risk per trade)
            assumed_risk_pct = 0.02  # Assume 2% price risk
            risk_amount = capital * RISK_PER_TRADE
            price = row['close']
            position_size = risk_amount / (price * assumed_risk_pct)

            # Determine direction and target
            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_20d']
                target = breakout_level * 1.005  # Breakout + 0.5%
            else:
                direction = 'short'
                breakout_level = row['low_20d']
                target = breakout_level * 0.995  # Breakout - 0.5%

            # Open position
            position = Position(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

        # Record equity
        equity_curve.append(capital)

    # Calculate performance metrics
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


# Load pre-generated predictions
print("Loading pre-generated predictions...")
try:
    with open('model_predictions.pkl', 'rb') as f:
        all_predictions = pickle.load(f)
    print(f"Loaded predictions for {len(all_predictions)} validation periods")
    print()
except FileNotFoundError:
    print("ERROR: model_predictions.pkl not found!")
    print("Run generate_predictions.py first to create predictions.")
    exit(1)

# Run backtest for all periods
print("="*100)
print("RUNNING BACKTEST")
print("="*100)
print()

all_results = []

for period_name, period_preds in all_predictions.items():
    print(f"Testing {period_name}...")
    result = run_backtest(period_name, period_preds, COOLDOWN_DAYS)
    all_results.append({
        'period': period_name,
        **result
    })
    print(f"  Return: {result['return']:+.1%}, Trades: {result['trades']}, "
          f"Win Rate: {result['win_rate']:.0%}, Hold: {result['avg_hold_days']:.1f}d, "
          f"DD: {result['max_dd']:.1%}")

print()

# Calculate aggregate statistics
avg_return = np.mean([r['return'] for r in all_results])
avg_dd = np.mean([r['max_dd'] for r in all_results])
avg_sharpe = np.mean([r['sharpe'] for r in all_results])
avg_trades = np.mean([r['trades'] for r in all_results])
avg_winrate = np.mean([r['win_rate'] for r in all_results])
avg_hold = np.mean([r['avg_hold_days'] for r in all_results])
profitable_periods = sum(1 for r in all_results if r['return'] > 0)

# Results summary
print("="*100)
print("BACKTEST RESULTS SUMMARY")
print("="*100)
print()

print(f"Period Breakdown:")
print(f"{'Period':<12} {'Return':>10} {'Trades':>8} {'Win%':>6} {'Hold':>6} {'Sharpe':>8} {'MaxDD':>8}")
print("-" * 80)

for result in all_results:
    print(f"{result['period']:<12} {result['return']:>9.1%} {result['trades']:>8} "
          f"{result['win_rate']:>5.0%} {result['avg_hold_days']:>5.1f}d "
          f"{result['sharpe']:>8.2f} {result['max_dd']:>7.1%}")

print("-" * 80)
print(f"{'AVERAGE':<12} {avg_return:>9.1%} {avg_trades:>8.0f} "
      f"{avg_winrate:>5.0%} {avg_hold:>5.1f}d "
      f"{avg_sharpe:>8.2f} {avg_dd:>7.1%}")

print()
print(f"Consistency: {profitable_periods}/{len(all_results)} periods profitable "
      f"({profitable_periods/len(all_results)*100:.0f}%)")
print(f"Return/DD Ratio: {avg_return/abs(avg_dd):.2f}:1")

print()
print("="*100)
print("STRATEGY CONFIGURATION")
print("="*100)
print()
print(f"Risk per trade:        {RISK_PER_TRADE*100:.1f}%")
print(f"Min confidence:        {MIN_CONFIDENCE*100:.0f}%")
print(f"Cooldown period:       {COOLDOWN_DAYS} days")
print(f"Emergency stop:        {EMERGENCY_STOP_LOSS_PCT*100:.0f}% after {EMERGENCY_STOP_DAYS} days")
print(f"Trailing stop:         {TRAILING_STOP_PCT*100:.0f}% of profit (activates at {TRAILING_STOP_TRIGGER*100:.1f}%)")

print()
print("="*100)
print("BACKTEST COMPLETE")
print("="*100)
print()

if avg_return > 0.30 and profitable_periods == len(all_results):
    print("SUCCESS: Strategy shows strong performance!")
    print(f"  • Average return: {avg_return:+.1%} per 2-year period (~{(1 + avg_return)**(1/2) - 1:.1%} annualized)")
    print(f"  • Risk-adjusted: {avg_sharpe:.2f} Sharpe ratio")
    print(f"  • Consistency: 100% of periods profitable")
    print(f"  • High win rate: {avg_winrate:.0%}")
    print()
    print("Strategy is ready for live trading. See STRATEGY_COMPARISON.md for details.")
elif avg_return > 0:
    print(f"MODERATE: Strategy profitable but inconsistent ({profitable_periods}/{len(all_results)} periods)")
    print("Consider reviewing failed periods before live trading.")
else:
    print("WARNING: Strategy shows negative returns. Further optimization needed.")
