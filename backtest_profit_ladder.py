"""
BREAKOUT STRATEGY WITH PROFIT LADDER
====================================
Scale out at multiple profit levels: 25% at +0.3%, +0.6%, +0.9%

Expected to:
- Lock in profits early on winners
- Catch losers during their brief rallies
- Reduce drawdowns by taking chips off the table
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("BREAKOUT STRATEGY WITH PROFIT LADDER")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
COOLDOWN_DAYS = 0
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60

# Profit ladder parameters
LADDER_LEVELS = [0.003, 0.006, 0.009]  # 0.3%, 0.6%, 0.9%
LADDER_SCALE_PCT = 0.25  # Scale out 25% at each level


class LadderPosition:
    """Position with profit ladder exits"""

    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size  # Current size (reduces as we scale out)
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.max_adverse_excursion = 0
        self.trailing_stop = None
        self.daily_progression = []
        self.partial_exits = []  # List of (profit, pct) tuples
        self.ladder_level = 0  # Which ladder level we're at (0-3)

    def update(self, date, high, low, close):
        """
        Update position with profit ladder exits.

        Exit priority:
        1. Profit ladder (scale out at profit levels)
        2. Emergency stop
        3. Trailing stop
        4. Target hit
        """
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

        # Track daily progression
        self.daily_progression.append({
            'day': self.days_held,
            'profit': current_profit,
            'high_profit': intraday_high_profit,
            'low_profit': intraday_low_profit
        })

        self.max_profit = max(self.max_profit, intraday_high_profit)
        self.max_adverse_excursion = min(self.max_adverse_excursion, intraday_low_profit)

        # PROFIT LADDER: Check if we hit next level
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                # Scale out at this level
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                # Continue holding remaining position
                return None

        # Emergency stop (applies to remaining position)
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
        """
        Calculate blended profit with all partial exits.

        Returns weighted average of all exits.
        """
        if len(self.partial_exits) == 0:
            return final_profit

        total_weighted_profit = 0
        remaining_size_pct = 1.0

        # Add up all partial exits
        for exit_profit, exit_pct in self.partial_exits:
            total_weighted_profit += exit_profit * exit_pct
            remaining_size_pct -= exit_pct

        # Add final exit
        total_weighted_profit += final_profit * remaining_size_pct

        return total_weighted_profit


def run_backtest(period_name, period_predictions):
    """Run backtest with profit ladder"""
    capital = INITIAL_CAPITAL
    positions = []
    closed_positions = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pd.to_datetime(pair_df.index).tz_localize(None)
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if len(all_dates) == 0:
        return {'return': 0, 'max_dd': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0,
                'avg_hold_days': 0, 'equity_curve': [], 'equity_dates': [], 'closed_positions': []}

    equity_curve = []
    equity_dates = []

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

            # Calculate blended profit (accounts for ladder exits)
            profit_pct = position.calculate_blended_profit(raw_profit)

            # Calculate dollar profit
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars

            position.exit_date = date
            position.exit_price = exit_price
            position.profit_pct = profit_pct  # This is the blended profit
            position.raw_profit = raw_profit   # This is the actual exit profit
            position.exit_reason = exit_reason

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

            position = LadderPosition(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

        equity_curve.append(capital)
        equity_dates.append(date)

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
        'avg_hold_days': avg_hold_days,
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'closed_positions': closed_positions
    }


# Load predictions
print("Loading predictions...")
with open('model_predictions.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

print("Running backtest with profit ladder...")
print()

# Run backtest
all_equity_curves = []
all_equity_dates = []
all_positions = []

for period_name, period_preds in all_predictions.items():
    print(f"Testing {period_name}...")
    result = run_backtest(period_name, period_preds)
    all_equity_curves.extend(result['equity_curve'])
    all_equity_dates.extend(result['equity_dates'])
    all_positions.extend(result['closed_positions'])
    print(f"  {len(result['closed_positions'])} trades, {result['win_rate']:.0%} win rate")

print()

# Calculate yearly results
equity_df = pd.DataFrame({
    'date': all_equity_dates,
    'equity': all_equity_curves
})
equity_df['year'] = pd.to_datetime(equity_df['date']).dt.year

yearly_results = []
for year in sorted(equity_df['year'].unique()):
    year_data = equity_df[equity_df['year'] == year].copy()
    year_positions = [p for p in all_positions if pd.Timestamp(p.exit_date).year == year]

    if len(year_data) == 0:
        continue

    if year == equity_df['year'].min():
        start_capital = INITIAL_CAPITAL
    else:
        prev_year_data = equity_df[equity_df['year'] < year]
        start_capital = prev_year_data['equity'].iloc[-1] if len(prev_year_data) > 0 else INITIAL_CAPITAL

    end_capital = year_data['equity'].iloc[-1]
    year_return = (end_capital - start_capital) / start_capital

    running_max = year_data['equity'].expanding().max()
    dd = (year_data['equity'] - running_max) / running_max
    max_dd = dd.min()

    returns = year_data['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    num_trades = len(year_positions)
    win_rate = sum(1 for p in year_positions if p.profit_pct > 0) / num_trades if num_trades > 0 else 0
    avg_hold = np.mean([p.days_held for p in year_positions]) if num_trades > 0 else 0

    yearly_results.append({
        'year': year,
        'return': year_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': num_trades,
        'win_rate': win_rate,
        'avg_hold_days': avg_hold
    })

# Calculate aggregates
avg_return = np.mean([r['return'] for r in yearly_results])
avg_dd = np.mean([r['max_dd'] for r in yearly_results])
avg_sharpe = np.mean([r['sharpe'] for r in yearly_results])
avg_trades = np.mean([r['trades'] for r in yearly_results])
avg_winrate = np.mean([r['win_rate'] for r in yearly_results])
avg_hold = np.mean([r['avg_hold_days'] for r in yearly_results])
profitable_periods = sum(1 for r in yearly_results if r['return'] > 0)

# Results
print("="*100)
print("BACKTEST RESULTS - WITH PROFIT LADDER")
print("="*100)
print()

print(f"Yearly Breakdown:")
print(f"{'Year':<8} {'Return':>10} {'Trades':>8} {'Win%':>6} {'Hold':>6} {'Sharpe':>8} {'MaxDD':>8}")
print("-" * 80)

for result in yearly_results:
    print(f"{result['year']:<8} {result['return']:>9.1%} {result['trades']:>8} "
          f"{result['win_rate']:>5.0%} {result['avg_hold_days']:>5.1f}d "
          f"{result['sharpe']:>8.2f} {result['max_dd']:>7.1%}")

print("-" * 80)
print(f"{'AVERAGE':<8} {avg_return:>9.1%} {avg_trades:>8.0f} "
      f"{avg_winrate:>5.0%} {avg_hold:>5.1f}d "
      f"{avg_sharpe:>8.2f} {avg_dd:>7.1%}")

print()
print(f"Consistency: {profitable_periods}/{len(yearly_results)} years profitable "
      f"({profitable_periods/len(yearly_results)*100:.0f}%)")
print(f"Return/DD Ratio: {avg_return/abs(avg_dd):.2f}:1")

print()

# Analyze ladder statistics
print("="*100)
print("PROFIT LADDER STATISTICS")
print("="*100)
print()

# Count how many times each ladder level was hit
ladder_stats = {0: 0, 1: 0, 2: 0, 3: 0}  # Number of exits at each level
for p in all_positions:
    ladder_stats[len(p.partial_exits)] += 1

print(f"Ladder usage breakdown:")
print(f"  No ladder exits (0 levels):    {ladder_stats[0]:5} trades ({ladder_stats[0]/len(all_positions):.1%})")
print(f"  Hit 1st level (0.3%):           {ladder_stats[1]:5} trades ({ladder_stats[1]/len(all_positions):.1%})")
print(f"  Hit 2nd level (0.6%):           {ladder_stats[2]:5} trades ({ladder_stats[2]/len(all_positions):.1%})")
print(f"  Hit all 3 levels (0.9%):        {ladder_stats[3]:5} trades ({ladder_stats[3]/len(all_positions):.1%})")
print()

# Analyze performance by ladder usage
for num_exits in range(4):
    trades = [p for p in all_positions if len(p.partial_exits) == num_exits]
    if len(trades) > 0:
        avg_profit = np.mean([p.profit_pct for p in trades])
        winners = sum(1 for p in trades if p.profit_pct > 0)
        win_rate = winners / len(trades)

        level_name = {0: "No exits", 1: "1 exit", 2: "2 exits", 3: "3 exits"}[num_exits]
        print(f"{level_name} ({len(trades)} trades):")
        print(f"  Win rate: {win_rate:.0%}")
        print(f"  Avg profit: {avg_profit:+.2%}")
        print()

# Compare ladder vs no-ladder on specific trades
winners = [p for p in all_positions if p.profit_pct > 0]
losers = [p for p in all_positions if p.profit_pct <= 0]

print(f"Winners analysis:")
winners_with_ladder = [p for p in winners if len(p.partial_exits) > 0]
winners_no_ladder = [p for p in winners if len(p.partial_exits) == 0]
if len(winners_with_ladder) > 0:
    print(f"  With ladder ({len(winners_with_ladder)} trades): {np.mean([p.profit_pct for p in winners_with_ladder]):+.2%} avg")
if len(winners_no_ladder) > 0:
    print(f"  No ladder ({len(winners_no_ladder)} trades):   {np.mean([p.profit_pct for p in winners_no_ladder]):+.2%} avg")
print()

print(f"Losers analysis:")
losers_with_ladder = [p for p in losers if len(p.partial_exits) > 0]
losers_no_ladder = [p for p in losers if len(p.partial_exits) == 0]
if len(losers_with_ladder) > 0:
    print(f"  With ladder ({len(losers_with_ladder)} trades): {np.mean([p.profit_pct for p in losers_with_ladder]):+.2%} avg")
if len(losers_no_ladder) > 0:
    print(f"  No ladder ({len(losers_no_ladder)} trades):   {np.mean([p.profit_pct for p in losers_no_ladder]):+.2%} avg")
print()

print("="*100)
print("COMPARISON TO BASELINE")
print("="*100)
print()

print("Baseline (no active management):")
print("  Average annual return: 53.8%")
print("  Average max DD: -26.6%")
print("  Consistency: 5/10 years (50%)")
print("  Average win rate: 89%")
print()

print("With profit ladder:")
print(f"  Average annual return: {avg_return:.1%}")
print(f"  Average max DD: {avg_dd:.1%}")
print(f"  Consistency: {profitable_periods}/10 years ({profitable_periods*10}%)")
print(f"  Average win rate: {avg_winrate:.0%}")
print()

improvement = avg_return - 0.538
dd_improvement = avg_dd - (-0.266)
print(f"Improvement:")
print(f"  Return: {improvement:+.1%} per year")
print(f"  Drawdown: {dd_improvement:+.1%} (negative is better)")
print(f"  Return/DD ratio: {avg_return/abs(avg_dd):.2f}:1 vs 2.02:1 baseline")
print()
