"""
BEHAVIORAL PATTERN ANALYSIS
============================
Instead of just looking at P&L levels, analyze the BEHAVIOR of trades:
- Volatility patterns
- Momentum reversals
- Consistency of direction
- Rally/decline characteristics

The goal: Distinguish between:
1. Winners developing slowly (down for many days then recover)
2. Losers having false rallies (briefly positive then collapse)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("BEHAVIORAL PATTERN ANALYSIS - Finding Alpha in Trade Management")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60


class DetailedPosition:
    """Trading position with full daily tracking"""

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
        self.max_adverse_excursion = 0
        self.trailing_stop = None
        self.daily_progression = []

    def update(self, date, high, low, close):
        """Update position and track progression"""
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
            'date': date,
            'profit': current_profit,
            'high_profit': intraday_high_profit,
            'low_profit': intraday_low_profit
        })

        self.max_profit = max(self.max_profit, intraday_high_profit)
        self.max_adverse_excursion = min(self.max_adverse_excursion, intraday_low_profit)

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

        return None, close, current_profit


def calculate_behavior_metrics(trade, check_day):
    """
    Calculate behavioral metrics for a trade up to check_day.

    Metrics:
    - Volatility: std dev of daily P&L changes
    - Momentum: trend of P&L (up or down)
    - Consistency: how many days moved in right direction
    - Max excursion: biggest swing
    - Time above zero: % of days positive
    """
    if len(trade.daily_progression) < check_day:
        return None

    profits = [d['profit'] for d in trade.daily_progression[:check_day]]

    # Calculate daily changes
    changes = [profits[i] - profits[i-1] for i in range(1, len(profits))]

    metrics = {
        'current_profit': profits[-1],
        'max_profit': max([d['high_profit'] for d in trade.daily_progression[:check_day]]),
        'min_profit': min([d['low_profit'] for d in trade.daily_progression[:check_day]]),
        'volatility': np.std(changes) if len(changes) > 0 else 0,
        'positive_days': sum(1 for p in profits if p > 0) / len(profits),
        'positive_momentum': sum(1 for c in changes if c > 0) / len(changes) if len(changes) > 0 else 0,
        'range': max([d['high_profit'] for d in trade.daily_progression[:check_day]]) -
                 min([d['low_profit'] for d in trade.daily_progression[:check_day]])
    }

    return metrics


def run_detailed_backtest(period_name, period_predictions):
    """Run backtest tracking daily progression"""
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        return []

    for date in all_dates:
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

            position = DetailedPosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    return closed_positions


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

print("Running detailed backtest...")
all_positions = []

for period_name, period_preds in all_predictions.items():
    print(f"  Processing {period_name}...")
    positions = run_detailed_backtest(period_name, period_preds)
    all_positions.extend(positions)

winners = [p for p in all_positions if p.profit_pct > 0]
losers = [p for p in all_positions if p.profit_pct <= 0]

print(f"Collected {len(all_positions)} trades")
print()

# Analyze behavior at different checkpoints
print("="*100)
print("BEHAVIORAL DIVERGENCE - Winners vs Losers at Key Days")
print("="*100)
print()

for check_day in [3, 5, 7, 10]:
    print(f"BEHAVIOR AFTER {check_day} DAYS:")
    print()

    winner_metrics = []
    loser_metrics = []

    for trade in winners:
        metrics = calculate_behavior_metrics(trade, check_day)
        if metrics:
            winner_metrics.append(metrics)

    for trade in losers:
        metrics = calculate_behavior_metrics(trade, check_day)
        if metrics:
            loser_metrics.append(metrics)

    if len(winner_metrics) > 0 and len(loser_metrics) > 0:
        print(f"{'Metric':<25} {'Winners':>12} {'Losers':>12} {'Difference':>12}")
        print("-" * 65)

        metrics_to_compare = ['current_profit', 'max_profit', 'min_profit', 'volatility',
                             'positive_days', 'positive_momentum', 'range']

        for metric in metrics_to_compare:
            winner_avg = np.mean([m[metric] for m in winner_metrics])
            loser_avg = np.mean([m[metric] for m in loser_metrics])
            diff = winner_avg - loser_avg

            if metric in ['current_profit', 'max_profit', 'min_profit', 'volatility', 'range']:
                print(f"{metric:<25} {winner_avg:>11.2%} {loser_avg:>11.2%} {diff:>11.2%}")
            else:
                print(f"{metric:<25} {winner_avg:>12.2f} {loser_avg:>12.2f} {diff:>12.2f}")

        print()

print()

# Find specific patterns in losers that went positive
print("="*100)
print("PATTERN ANALYSIS - Losers That Went Positive vs Winners")
print("="*100)
print()

# Losers that went positive
losers_went_positive = [t for t in losers if max(d['high_profit'] for d in t.daily_progression) > 0]

print(f"Analyzing {len(losers_went_positive)} losers that went positive vs {len(winners)} winners")
print()

# When they went positive, what was different?
print("When a trade first crosses positive, compare behavior:")
print()

# For losers that went positive, analyze their behavior when they crossed zero
loser_at_first_positive = []
for trade in losers_went_positive:
    # Find first day it went positive
    for i, day_data in enumerate(trade.daily_progression):
        if day_data['high_profit'] > 0:
            # Look at behavior around this point
            day_of_positive = day_data['day']
            if day_of_positive >= 3:  # Need some history
                # Get profit at days -2, -1, 0
                prev_days = [trade.daily_progression[max(0, i-j)]['profit']
                           for j in range(3, 0, -1) if i-j >= 0]

                # Momentum before going positive
                momentum_before = np.mean([prev_days[j] - prev_days[j-1]
                                          for j in range(1, len(prev_days))])

                loser_at_first_positive.append({
                    'day_went_positive': day_of_positive,
                    'max_profit_reached': trade.max_profit,
                    'final_loss': trade.profit_pct,
                    'momentum_before': momentum_before,
                    'volatility_before': np.std(prev_days) if len(prev_days) > 1 else 0
                })
            break

# For winners, analyze when they went positive
winner_at_first_positive = []
for trade in winners:
    for i, day_data in enumerate(trade.daily_progression):
        if day_data['high_profit'] > 0:
            day_of_positive = day_data['day']
            if day_of_positive >= 3:
                prev_days = [trade.daily_progression[max(0, i-j)]['profit']
                           for j in range(3, 0, -1) if i-j >= 0]

                momentum_before = np.mean([prev_days[j] - prev_days[j-1]
                                          for j in range(1, len(prev_days))])

                winner_at_first_positive.append({
                    'day_went_positive': day_of_positive,
                    'final_profit': trade.profit_pct,
                    'momentum_before': momentum_before,
                    'volatility_before': np.std(prev_days) if len(prev_days) > 1 else 0
                })
            break

if len(loser_at_first_positive) > 0 and len(winner_at_first_positive) > 0:
    print(f"{'Metric':<30} {'Winners':>15} {'Losers':>15} {'Difference':>15}")
    print("-" * 80)

    winner_day = np.mean([x['day_went_positive'] for x in winner_at_first_positive])
    loser_day = np.mean([x['day_went_positive'] for x in loser_at_first_positive])
    print(f"{'Day went positive':<30} {winner_day:>15.1f} {loser_day:>15.1f} {winner_day-loser_day:>15.1f}")

    winner_momentum = np.mean([x['momentum_before'] for x in winner_at_first_positive])
    loser_momentum = np.mean([x['momentum_before'] for x in loser_at_first_positive])
    print(f"{'Momentum before positive':<30} {winner_momentum:>14.3%} {loser_momentum:>14.3%} {winner_momentum-loser_momentum:>14.3%}")

    winner_vol = np.mean([x['volatility_before'] for x in winner_at_first_positive])
    loser_vol = np.mean([x['volatility_before'] for x in loser_at_first_positive])
    print(f"{'Volatility before positive':<30} {winner_vol:>14.3%} {loser_vol:>14.3%} {winner_vol-loser_vol:>14.3%}")

print()
print()

# Test: Exit losers when they go positive with weak momentum
print("="*100)
print("ACTIVE MANAGEMENT RULES - Testing Specific Strategies")
print("="*100)
print()

print("RULE: Exit when positive but momentum is weak")
print("(Trade went positive recently but momentum before was < 0.2%/day)")
print()

# Test this rule
trades_by_rule = {'exit': [], 'hold': []}

for trade in losers_went_positive:
    # Find when it went positive
    for i, day_data in enumerate(trade.daily_progression):
        if day_data['high_profit'] > 0.001:  # At least 0.1% positive
            if i >= 2:
                prev_days = [trade.daily_progression[i-j]['profit'] for j in range(1, min(4, i+1))]
                momentum = np.mean([prev_days[j] - prev_days[j-1] for j in range(1, len(prev_days))])

                if momentum < 0.002:  # Weak momentum (< 0.2%/day)
                    # Would exit here
                    exit_profit = day_data['high_profit']
                    trades_by_rule['exit'].append({
                        'actual': trade.profit_pct,
                        'exit_at': exit_profit
                    })
                else:
                    trades_by_rule['hold'].append({
                        'actual': trade.profit_pct
                    })
            break

if len(trades_by_rule['exit']) > 0:
    avg_actual = np.mean([x['actual'] for x in trades_by_rule['exit']])
    avg_exit = np.mean([x['exit_at'] for x in trades_by_rule['exit']])
    improvement = avg_exit - avg_actual

    print(f"Would exit {len(trades_by_rule['exit'])} losers early:")
    print(f"  Current avg result: {avg_actual:+.2%}")
    print(f"  If exited with rule: {avg_exit:+.2%}")
    print(f"  Improvement: {improvement:+.2%}")
    print()

print()

# Analyze: What if we had a "profit target" for exits when momentum is weak?
print("="*100)
print("SCALING OUT STRATEGY - Take profits on weak momentum")
print("="*100)
print()

print("Test: When trade hits +0.3% but has weak momentum (< 0.15%/day), exit 50%")
print()

# Calculate how many winners and losers this affects
winners_affected = []
losers_affected = []

for trade in winners:
    for i, day_data in enumerate(trade.daily_progression):
        if day_data['high_profit'] > 0.003:  # Hit +0.3%
            if i >= 2:
                prev_days = [trade.daily_progression[i-j]['profit'] for j in range(1, min(4, i+1))]
                momentum = np.mean([prev_days[j] - prev_days[j-1] for j in range(1, len(prev_days))])

                if momentum < 0.0015:  # Weak momentum
                    # Would scale out 50% here
                    scale_profit = day_data['high_profit']
                    final_profit = trade.profit_pct
                    blended_profit = (scale_profit * 0.5) + (final_profit * 0.5)

                    winners_affected.append({
                        'actual': final_profit,
                        'blended': blended_profit
                    })
                    break

for trade in losers_went_positive:
    for i, day_data in enumerate(trade.daily_progression):
        if day_data['high_profit'] > 0.003:  # Hit +0.3%
            if i >= 2:
                prev_days = [trade.daily_progression[i-j]['profit'] for j in range(1, min(4, i+1))]
                momentum = np.mean([prev_days[j] - prev_days[j-1] for j in range(1, len(prev_days))])

                if momentum < 0.0015:  # Weak momentum
                    scale_profit = day_data['high_profit']
                    final_loss = trade.profit_pct
                    blended_profit = (scale_profit * 0.5) + (final_loss * 0.5)

                    losers_affected.append({
                        'actual': final_loss,
                        'blended': blended_profit
                    })
                    break

total_affected = len(winners_affected) + len(losers_affected)
print(f"Would affect {total_affected} trades:")
print(f"  Winners: {len(winners_affected)}")
print(f"  Losers: {len(losers_affected)}")
print()

if len(winners_affected) > 0:
    winner_actual = np.mean([x['actual'] for x in winners_affected])
    winner_blended = np.mean([x['blended'] for x in winners_affected])
    print(f"Winners affected:")
    print(f"  Current: {winner_actual:+.2%}")
    print(f"  Blended: {winner_blended:+.2%}")
    print(f"  Change: {winner_blended - winner_actual:+.2%}")
    print()

if len(losers_affected) > 0:
    loser_actual = np.mean([x['actual'] for x in losers_affected])
    loser_blended = np.mean([x['blended'] for x in losers_affected])
    print(f"Losers affected:")
    print(f"  Current: {loser_actual:+.2%}")
    print(f"  Blended: {loser_blended:+.2%}")
    print(f"  Change: {loser_blended - loser_actual:+.2%}")
    print()

# Estimate total impact
if len(winners_affected) > 0 and len(losers_affected) > 0:
    winner_impact = (winner_blended - winner_actual) * len(winners_affected)
    loser_impact = (loser_blended - loser_actual) * len(losers_affected)
    total_impact = winner_impact + loser_impact

    print(f"Estimated total impact: {total_impact:+.2%} across all trades")
    print(f"  (Winner impact: {winner_impact:+.2%}, Loser impact: {loser_impact:+.2%})")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
