"""
TRADE PROGRESSION ANALYSIS
==========================
Analyze how trades unfold day-by-day to find patterns for active trade management.

Key Questions:
1. Can we identify losing trades early (e.g., "if down after 5 days, likely a loser")?
2. At what point do winning trades typically turn positive?
3. Are there clear divergence patterns between winners and losers?
4. What's the optimal time to cut losses or take profits?
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("TRADE PROGRESSION ANALYSIS")
print("="*100)
print()

# Strategy Parameters (from backtest)
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60


class DetailedPosition:
    """Trading position that tracks daily progression"""

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

        # Track daily progression
        self.daily_progression = []  # List of (day, profit_pct, high_profit, low_profit)

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

        # Record daily progression
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


def run_detailed_backtest(period_name, period_predictions):
    """Run backtest tracking daily progression"""
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    # Get all dates
    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        return []

    # Daily loop
    for date in all_dates:
        # Get current prices
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

        # Update positions
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

        # Close positions
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

        # Generate new signals
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

            # Calculate position size
            assumed_risk_pct = 0.02
            risk_amount = capital * RISK_PER_TRADE
            price = row['close']
            position_size = risk_amount / (price * assumed_risk_pct)

            # Determine direction
            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_20d']
                target = breakout_level * 1.005
            else:
                direction = 'short'
                breakout_level = row['low_20d']
                target = breakout_level * 0.995

            # Open position
            position = DetailedPosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    return closed_positions


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Run detailed backtest
print("Running detailed backtest...")
all_positions = []

for period_name, period_preds in all_predictions.items():
    print(f"  Processing {period_name}...")
    positions = run_detailed_backtest(period_name, period_preds)
    all_positions.extend(positions)

print(f"Collected {len(all_positions)} trades with daily progression data")
print()

# Separate winners and losers
winners = [p for p in all_positions if p.profit_pct > 0]
losers = [p for p in all_positions if p.profit_pct <= 0]

print("="*100)
print("1. DAILY PROGRESSION ANALYSIS - Average P&L by Day")
print("="*100)
print()

# Calculate average profit by day for winners and losers
max_days = 20  # Analyze first 20 days
winner_by_day = defaultdict(list)
loser_by_day = defaultdict(list)

for trade in winners:
    for day_data in trade.daily_progression:
        if day_data['day'] <= max_days:
            winner_by_day[day_data['day']].append(day_data['profit'])

for trade in losers:
    for day_data in trade.daily_progression:
        if day_data['day'] <= max_days:
            loser_by_day[day_data['day']].append(day_data['profit'])

print(f"{'Day':<6} {'Winners Avg':>12} {'Losers Avg':>12} {'Difference':>12} {'Sample Sizes':>20}")
print("-" * 80)

for day in range(1, max_days + 1):
    if day in winner_by_day and day in loser_by_day:
        winner_avg = np.mean(winner_by_day[day])
        loser_avg = np.mean(loser_by_day[day])
        diff = winner_avg - loser_avg
        winner_count = len(winner_by_day[day])
        loser_count = len(loser_by_day[day])

        print(f"Day {day:<2}  {winner_avg:>11.2%}  {loser_avg:>11.2%}  {diff:>11.2%}  W:{winner_count:5} L:{loser_count:4}")

print()

# Key insight: Find the day where divergence is clearest
print("="*100)
print("2. EARLY WARNING SIGNALS - Can we identify losers early?")
print("="*100)
print()

# For each day 1-10, analyze if P&L predicts final outcome
for check_day in [1, 3, 5, 7, 10]:
    winners_at_day = []
    losers_at_day = []

    for trade in winners:
        if len(trade.daily_progression) >= check_day:
            winners_at_day.append(trade.daily_progression[check_day-1]['profit'])

    for trade in losers:
        if len(trade.daily_progression) >= check_day:
            losers_at_day.append(trade.daily_progression[check_day-1]['profit'])

    if len(winners_at_day) > 0 and len(losers_at_day) > 0:
        winner_avg = np.mean(winners_at_day)
        loser_avg = np.mean(losers_at_day)

        # What % of losers are down at this day?
        losers_down = sum(1 for x in losers_at_day if x < 0)
        losers_down_pct = losers_down / len(losers_at_day)

        # What % of winners are down at this day?
        winners_down = sum(1 for x in winners_at_day if x < 0)
        winners_down_pct = winners_down / len(winners_at_day)

        print(f"After DAY {check_day}:")
        print(f"  Winners avg P&L: {winner_avg:+.2%} ({winners_down_pct:.0%} are down)")
        print(f"  Losers avg P&L:  {loser_avg:+.2%} ({losers_down_pct:.0%} are down)")

        # If a trade is down X% at day N, what's the probability it ends as a loser?
        thresholds = [-0.005, -0.01, -0.015, -0.02]
        print(f"  If trade is down at day {check_day}:")

        for threshold in thresholds:
            winners_below = sum(1 for x in winners_at_day if x < threshold)
            losers_below = sum(1 for x in losers_at_day if x < threshold)
            total_below = winners_below + losers_below

            if total_below > 10:  # Need reasonable sample size
                prob_loser = losers_below / total_below
                print(f"    Below {threshold:+.1%}: {prob_loser:.0%} chance of losing ({total_below} trades)")

        print()

print()

# Analyze by time to profitability for winners
print("="*100)
print("3. TIME TO PROFITABILITY - How quickly do winners turn positive?")
print("="*100)
print()

days_to_positive = []
for trade in winners:
    for day_data in trade.daily_progression:
        if day_data['profit'] > 0:
            days_to_positive.append(day_data['day'])
            break

if len(days_to_positive) > 0:
    print(f"Statistics for {len(winners)} winning trades:")
    print(f"  Median days to positive: {np.median(days_to_positive):.1f}")
    print(f"  Average days to positive: {np.mean(days_to_positive):.1f}")
    print(f"  25th percentile: {np.percentile(days_to_positive, 25):.1f} days")
    print(f"  75th percentile: {np.percentile(days_to_positive, 75):.1f} days")
    print()

    # Distribution
    print("Distribution:")
    bins = [0, 1, 2, 3, 5, 7, 10, 100]
    labels = ['Day 1', 'Day 2', 'Day 3', 'Days 4-5', 'Days 6-7', 'Days 8-10', '10+ days']

    for i in range(len(bins) - 1):
        count = sum(1 for d in days_to_positive if bins[i] < d <= bins[i+1])
        pct = count / len(days_to_positive)
        print(f"  {labels[i]:<12}: {count:4} trades ({pct:5.1%})")

print()

# Analyze losers - did they ever get positive?
print("="*100)
print("4. LOSERS THAT GOT POSITIVE - Were there exit opportunities?")
print("="*100)
print()

losers_went_positive = []
for trade in losers:
    max_profit_seen = max(day_data['high_profit'] for day_data in trade.daily_progression)
    if max_profit_seen > 0:
        losers_went_positive.append({
            'trade': trade,
            'max_profit': max_profit_seen,
            'final_loss': trade.profit_pct,
            'day_of_max': next(d['day'] for d in trade.daily_progression if d['high_profit'] == max_profit_seen)
        })

print(f"Out of {len(losers)} losing trades:")
print(f"  {len(losers_went_positive)} went positive ({len(losers_went_positive)/len(losers):.1%})")
print()

if len(losers_went_positive) > 0:
    avg_max_profit = np.mean([x['max_profit'] for x in losers_went_positive])
    avg_final_loss = np.mean([x['final_loss'] for x in losers_went_positive])
    avg_day_of_max = np.mean([x['day_of_max'] for x in losers_went_positive])

    print(f"  Average max profit before losing: {avg_max_profit:+.2%}")
    print(f"  Average final loss: {avg_final_loss:+.2%}")
    print(f"  Average day of max profit: {avg_day_of_max:.1f}")
    print()

    # How much could we have saved?
    total_loss_dollars = sum([x['final_loss'] for x in losers_went_positive])
    total_if_exited_at_breakeven = 0  # Exit at 0%
    saved = total_loss_dollars - total_if_exited_at_breakeven

    print(f"  If we exited these {len(losers_went_positive)} trades at breakeven instead:")
    print(f"    We would save: {abs(saved):.1%} in total losses")

print()

# Analyze exit timing
print("="*100)
print("5. OPTIMAL EXIT TIMING - When should we cut or take profits?")
print("="*100)
print()

print("Current strategy exits:")
for reason in ['target', 'trailing_stop', 'emergency_stop']:
    trades_by_reason = [p for p in all_positions if p.exit_reason == reason]
    if len(trades_by_reason) > 0:
        avg_profit = np.mean([p.profit_pct for p in trades_by_reason])
        avg_days = np.mean([p.days_held for p in trades_by_reason])
        print(f"  {reason:<20}: {len(trades_by_reason):5} trades | Avg: {avg_profit:+.2%} | Days: {avg_days:.1f}")

print()

# Test alternative exit rules
print("What if we added early exit rules?")
print()

# Rule 1: Exit if down >1% after 5 days
print("RULE 1: Exit if down >1% after 5 days")
trades_affected = []
for trade in all_positions:
    if len(trade.daily_progression) >= 5:
        day5_profit = trade.daily_progression[4]['profit']
        if day5_profit < -0.01:
            trades_affected.append({
                'actual_profit': trade.profit_pct,
                'day5_profit': day5_profit,
                'would_exit_at': day5_profit
            })

if len(trades_affected) > 0:
    avg_actual = np.mean([x['actual_profit'] for x in trades_affected])
    avg_if_exited = np.mean([x['would_exit_at'] for x in trades_affected])
    improvement = avg_if_exited - avg_actual

    print(f"  Would affect: {len(trades_affected)} trades")
    print(f"  Current avg result: {avg_actual:+.2%}")
    print(f"  If exited at day 5: {avg_if_exited:+.2%}")
    print(f"  Improvement: {improvement:+.2%}")

    # How many were winners vs losers?
    affected_winners = sum(1 for x in trades_affected if x['actual_profit'] > 0)
    affected_losers = sum(1 for x in trades_affected if x['actual_profit'] <= 0)
    print(f"  Would exit: {affected_winners} winners, {affected_losers} losers")

print()

# Rule 2: Exit if down >1.5% after 7 days
print("RULE 2: Exit if down >1.5% after 7 days")
trades_affected = []
for trade in all_positions:
    if len(trade.daily_progression) >= 7:
        day7_profit = trade.daily_progression[6]['profit']
        if day7_profit < -0.015:
            trades_affected.append({
                'actual_profit': trade.profit_pct,
                'day7_profit': day7_profit,
                'would_exit_at': day7_profit
            })

if len(trades_affected) > 0:
    avg_actual = np.mean([x['actual_profit'] for x in trades_affected])
    avg_if_exited = np.mean([x['would_exit_at'] for x in trades_affected])
    improvement = avg_if_exited - avg_actual

    print(f"  Would affect: {len(trades_affected)} trades")
    print(f"  Current avg result: {avg_actual:+.2%}")
    print(f"  If exited at day 7: {avg_if_exited:+.2%}")
    print(f"  Improvement: {improvement:+.2%}")

    affected_winners = sum(1 for x in trades_affected if x['actual_profit'] > 0)
    affected_losers = sum(1 for x in trades_affected if x['actual_profit'] <= 0)
    print(f"  Would exit: {affected_winners} winners, {affected_losers} losers")

print()

# Rule 3: Exit if down >2% after 10 days
print("RULE 3: Exit if down >2% after 10 days")
trades_affected = []
for trade in all_positions:
    if len(trade.daily_progression) >= 10:
        day10_profit = trade.daily_progression[9]['profit']
        if day10_profit < -0.02:
            trades_affected.append({
                'actual_profit': trade.profit_pct,
                'day10_profit': day10_profit,
                'would_exit_at': day10_profit
            })

if len(trades_affected) > 0:
    avg_actual = np.mean([x['actual_profit'] for x in trades_affected])
    avg_if_exited = np.mean([x['would_exit_at'] for x in trades_affected])
    improvement = avg_if_exited - avg_actual

    print(f"  Would affect: {len(trades_affected)} trades")
    print(f"  Current avg result: {avg_actual:+.2%}")
    print(f"  If exited at day 10: {avg_if_exited:+.2%}")
    print(f"  Improvement: {improvement:+.2%}")

    affected_winners = sum(1 for x in trades_affected if x['actual_profit'] > 0)
    affected_losers = sum(1 for x in trades_affected if x['actual_profit'] <= 0)
    print(f"  Would exit: {affected_winners} winners, {affected_losers} losers")

print()

print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print()
print("Key insights:")
print("1. Check day-by-day P&L divergence between winners and losers")
print("2. Identify at which day you can predict final outcome with confidence")
print("3. See if early exits would improve overall results")
print("4. Consider if losers that went positive offer exit opportunities")
print()
