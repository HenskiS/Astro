"""
ADDITIONAL ACTIVE MANAGEMENT STRATEGIES
========================================
Test various active management approaches beyond the weak momentum scale-out.

Strategies to test:
1. Time-based scale out (e.g., 50% at day 7 regardless of profit)
2. Reverse momentum exit (profitable trade suddenly reverses direction)
3. Volatility spike exit (trade becomes too volatile)
4. Profit ladder (scale out at multiple profit levels)
5. Breakeven stop after X days (if not positive after N days, exit at breakeven)
6. Combination strategies
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("ADDITIONAL ACTIVE MANAGEMENT STRATEGIES")
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


def run_backtest_with_strategy(period_predictions, strategy_func, strategy_name):
    """
    Run backtest with a custom strategy function.
    strategy_func takes (position, date, high, low, close) and returns exit info or None
    """

    class ManagedPosition:
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
            self.max_adverse_excursion = 0
            self.trailing_stop = None
            self.daily_progression = []
            self.partial_exits = []  # Track multiple partial exits

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

            # CUSTOM STRATEGY CHECK
            strategy_exit = strategy_func(self, date, high, low, close)
            if strategy_exit:
                return strategy_exit

            # Emergency stop
            if self.days_held >= EMERGENCY_STOP_DAYS and current_profit < EMERGENCY_STOP_LOSS_PCT:
                return ('emergency_stop', close, current_profit, 1.0)

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
                    return ('trailing_stop', self.trailing_stop, current_profit, 1.0)

            # Target hit
            if hit_target:
                return ('target', self.breakout_target, current_profit, 1.0)

            return None

        def calculate_blended_profit(self, final_profit):
            """Calculate weighted profit with all partial exits"""
            if len(self.partial_exits) == 0:
                return final_profit

            total_weighted_profit = 0
            remaining_size = 1.0

            for exit_profit, exit_pct in self.partial_exits:
                total_weighted_profit += exit_profit * exit_pct
                remaining_size -= exit_pct

            total_weighted_profit += final_profit * remaining_size
            return total_weighted_profit

    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

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
            reason, exit_price, current_profit, exit_pct = exit_info

            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            # Handle partial vs full exit
            if exit_pct < 1.0:
                # Partial exit
                position.partial_exits.append((raw_profit, exit_pct))
                position.size *= (1 - exit_pct)
                # Don't close position, just record partial exit
                continue
            else:
                # Full exit
                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)
                capital += profit_dollars

                position.exit_date = date
                position.exit_price = exit_price
                position.profit_pct = profit_pct
                position.exit_reason = reason

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

            position = ManagedPosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    # Calculate metrics
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    num_trades = len(closed_positions)
    win_rate = sum(1 for p in closed_positions if p.profit_pct > 0) / num_trades if num_trades > 0 else 0
    avg_profit = np.mean([p.profit_pct for p in closed_positions]) if num_trades > 0 else 0

    return {
        'name': strategy_name,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'positions': closed_positions
    }


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Combine all periods for faster testing
all_period_predictions = {}
for period_name, period_preds in all_predictions.items():
    for pair, pred_df in period_preds.items():
        if pair not in all_period_predictions:
            all_period_predictions[pair] = pred_df
        else:
            all_period_predictions[pair] = pd.concat([all_period_predictions[pair], pred_df])

print("Testing strategies...")
print()

# Define strategies to test
strategies = []

# BASELINE: No extra management (just emergency stop, trailing stop, target)
def baseline_strategy(position, date, high, low, close):
    return None

strategies.append(('BASELINE', baseline_strategy))


# STRATEGY 1: Weak momentum scale-out (our current best)
def weak_momentum_strategy(position, date, high, low, close):
    if len(position.partial_exits) == 0 and position.days_held >= 3:
        intraday_high_profit = position.daily_progression[-1]['high_profit']
        if intraday_high_profit >= 0.003:
            recent_profits = [d['profit'] for d in position.daily_progression[-3:]]
            momentum = np.mean([recent_profits[i] - recent_profits[i-1]
                               for i in range(1, len(recent_profits))])
            if momentum < 0.0015:
                # Scale out 50%
                return ('weak_momentum_scale', close, position.daily_progression[-1]['profit'], 0.5)
    return None

strategies.append(('Weak Momentum Scale-Out', weak_momentum_strategy))


# STRATEGY 2: Time-based scale out (50% at day 7)
def time_based_strategy(position, date, high, low, close):
    if len(position.partial_exits) == 0 and position.days_held == 7:
        current_profit = position.daily_progression[-1]['profit']
        if current_profit > 0:  # Only if profitable
            return ('time_scale_day7', close, current_profit, 0.5)
    return None

strategies.append(('Time Scale-Out (Day 7)', time_based_strategy))


# STRATEGY 3: Reverse momentum (profit then sudden reversal)
def reverse_momentum_strategy(position, date, high, low, close):
    if position.days_held >= 4:
        # Check if we had profit but now reversing
        max_profit_so_far = max(d['high_profit'] for d in position.daily_progression)
        current_profit = position.daily_progression[-1]['profit']

        if max_profit_so_far > 0.005:  # Had at least 0.5% profit
            # Calculate recent momentum (last 2 days)
            recent_profits = [d['profit'] for d in position.daily_progression[-3:]]
            recent_momentum = np.mean([recent_profits[i] - recent_profits[i-1]
                                      for i in range(1, len(recent_profits))])

            # If momentum turned strongly negative
            if recent_momentum < -0.003:  # -0.3%/day
                return ('reverse_momentum', close, current_profit, 1.0)
    return None

strategies.append(('Reverse Momentum Exit', reverse_momentum_strategy))


# STRATEGY 4: Volatility spike (sudden increase in daily swings)
def volatility_spike_strategy(position, date, high, low, close):
    if position.days_held >= 5:
        # Calculate recent volatility vs historical
        recent_ranges = [d['high_profit'] - d['low_profit']
                        for d in position.daily_progression[-3:]]
        early_ranges = [d['high_profit'] - d['low_profit']
                       for d in position.daily_progression[:min(5, len(position.daily_progression))]]

        recent_vol = np.mean(recent_ranges)
        early_vol = np.mean(early_ranges)

        # If volatility doubled and we're profitable
        current_profit = position.daily_progression[-1]['profit']
        if recent_vol > early_vol * 2.0 and current_profit > 0.002:
            return ('volatility_spike', close, current_profit, 1.0)
    return None

strategies.append(('Volatility Spike Exit', volatility_spike_strategy))


# STRATEGY 5: Profit ladder (scale out at multiple levels)
def profit_ladder_strategy(position, date, high, low, close):
    intraday_high = position.daily_progression[-1]['high_profit']

    # Scale out 25% at +0.3%, 25% at +0.6%, 25% at +0.9%
    total_exited = sum(exit_pct for _, exit_pct in position.partial_exits)

    if total_exited < 0.25 and intraday_high >= 0.003:
        return ('ladder_0.3', close, position.daily_progression[-1]['profit'], 0.25)
    elif total_exited < 0.50 and intraday_high >= 0.006:
        return ('ladder_0.6', close, position.daily_progression[-1]['profit'], 0.25)
    elif total_exited < 0.75 and intraday_high >= 0.009:
        return ('ladder_0.9', close, position.daily_progression[-1]['profit'], 0.25)

    return None

strategies.append(('Profit Ladder (0.3/0.6/0.9%)', profit_ladder_strategy))


# STRATEGY 6: Breakeven stop after 10 days
def breakeven_stop_strategy(position, date, high, low, close):
    if position.days_held >= 10:
        current_profit = position.daily_progression[-1]['profit']
        # If still down after 10 days, exit at current level
        if current_profit < -0.005:  # Down more than 0.5%
            return ('breakeven_stop', close, current_profit, 1.0)
    return None

strategies.append(('Breakeven Stop (Day 10)', breakeven_stop_strategy))


# STRATEGY 7: Combination - Weak momentum + Reverse momentum
def combo_strategy(position, date, high, low, close):
    # First check weak momentum scale-out
    result = weak_momentum_strategy(position, date, high, low, close)
    if result:
        return result

    # Then check reverse momentum
    return reverse_momentum_strategy(position, date, high, low, close)

strategies.append(('COMBO: Weak Momentum + Reverse', combo_strategy))


# Run all strategies
results = []

for strategy_name, strategy_func in strategies:
    print(f"Testing: {strategy_name}...")
    result = run_backtest_with_strategy(all_period_predictions, strategy_func, strategy_name)
    results.append(result)

print()

# Display results
print("="*100)
print("STRATEGY COMPARISON")
print("="*100)
print()

print(f"{'Strategy':<35} {'Total Return':>12} {'Trades':>8} {'Win Rate':>10} {'Avg Profit':>12}")
print("-" * 90)

baseline_return = results[0]['total_return']

for result in results:
    improvement = result['total_return'] - baseline_return

    print(f"{result['name']:<35} {result['total_return']:>11.1%} {result['num_trades']:>8} "
          f"{result['win_rate']:>9.0%} {result['avg_profit']:>11.3%}")

    if result['name'] != 'BASELINE':
        print(f"  {'vs baseline:':<35} {improvement:>+11.1%}")

print()

# Detailed analysis of top 3 strategies
print("="*100)
print("DETAILED ANALYSIS - TOP 3 STRATEGIES")
print("="*100)
print()

# Sort by total return
sorted_results = sorted(results[1:], key=lambda x: x['total_return'], reverse=True)[:3]

for result in sorted_results:
    print(f"{result['name']}:")
    print(f"  Total return: {result['total_return']:.1%}")
    print(f"  Improvement vs baseline: {result['total_return'] - baseline_return:+.1%}")

    positions = result['positions']
    winners = [p for p in positions if p.profit_pct > 0]
    losers = [p for p in positions if p.profit_pct <= 0]

    if len(winners) > 0:
        print(f"  Winners: {len(winners)} trades, avg: {np.mean([p.profit_pct for p in winners]):+.2%}")
    if len(losers) > 0:
        print(f"  Losers:  {len(losers)} trades, avg: {np.mean([p.profit_pct for p in losers]):+.2%}")

    # Count partial exits
    partial_exit_count = sum(1 for p in positions if len(p.partial_exits) > 0)
    if partial_exit_count > 0:
        print(f"  Partial exits triggered: {partial_exit_count} trades ({partial_exit_count/len(positions):.1%})")

    print()

print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
