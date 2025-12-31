"""
BREAKOUT STRATEGY WITH ACTIVE TRADE MANAGEMENT
===============================================
Enhanced version with momentum-based scaling out.

Key Enhancement:
- When trade reaches +0.3% profit but has weak momentum (<0.15%/day), scale out 50%
- Catches losing trades during weak rallies before they collapse
- Minimally affects winners (they have strong momentum)

Expected Impact:
- Reduce average loss on losers by ~60% (from -2.83% to -1.20%)
- Small positive impact on winners (+0.05%)
- Total portfolio improvement: ~9% per trade on affected trades
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

print("="*100)
print("BREAKOUT STRATEGY WITH ACTIVE TRADE MANAGEMENT")
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

# NEW: Active management parameters
SCALE_OUT_PROFIT_TRIGGER = 0.003  # Scale out when at +0.3%
WEAK_MOMENTUM_THRESHOLD = 0.0015  # Momentum < 0.15%/day is "weak"
SCALE_OUT_PCT = 0.50              # Exit 50% of position
MOMENTUM_LOOKBACK = 3             # Look back 3 days for momentum


class ActivePosition:
    """Position with active management based on momentum"""

    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size  # Current size (can be scaled down)
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.days_held = 0
        self.max_profit = 0
        self.max_adverse_excursion = 0
        self.trailing_stop = None
        self.daily_progression = []
        self.scaled_out = False
        self.scale_out_profit = 0  # Track profit from scaled out portion

    def update(self, date, high, low, close):
        """
        Update position with active management.

        Exit priority:
        1. Scale out (new!) - Take partial profit on weak momentum
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

        # NEW: Check for scale-out opportunity
        if not self.scaled_out and self.days_held >= MOMENTUM_LOOKBACK:
            if intraday_high_profit >= SCALE_OUT_PROFIT_TRIGGER:
                # Calculate momentum over last N days
                recent_profits = [d['profit'] for d in self.daily_progression[-MOMENTUM_LOOKBACK:]]
                momentum = np.mean([recent_profits[i] - recent_profits[i-1]
                                   for i in range(1, len(recent_profits))])

                if momentum < WEAK_MOMENTUM_THRESHOLD:
                    # SCALE OUT! Exit 50% at this profit level
                    self.scaled_out = True
                    self.scale_out_profit = intraday_high_profit
                    self.size *= (1 - SCALE_OUT_PCT)  # Reduce position size
                    # Continue holding remaining 50%
                    # Don't return an exit - this is a partial exit

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

        return None, close, current_profit

    def calculate_blended_profit(self, final_profit):
        """
        Calculate blended profit if position was scaled out.

        Returns:
            Weighted average of scale-out profit and final profit
        """
        if self.scaled_out:
            return (self.scale_out_profit * SCALE_OUT_PCT) + (final_profit * (1 - SCALE_OUT_PCT))
        else:
            return final_profit


def run_backtest(period_name, period_predictions):
    """Run backtest with active management"""
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

            exit_reason, exit_price, current_profit = position.update(date, high, low, close)

            if exit_reason is not None:
                positions_to_close.append((position, exit_price, exit_reason))

        # Close positions
        for position, exit_price, reason in positions_to_close:
            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            # Calculate blended profit (accounts for scale-out)
            profit_pct = position.calculate_blended_profit(raw_profit)

            # Calculate dollar profit (uses original size for scale-out portion, current size for remainder)
            if position.scaled_out:
                # Scaled out portion
                scale_out_dollars = position.scale_out_profit * (position.original_size * SCALE_OUT_PCT * position.entry_price)
                # Remaining portion
                remaining_dollars = raw_profit * (position.size * position.entry_price)
                profit_dollars = scale_out_dollars + remaining_dollars
            else:
                profit_dollars = profit_pct * (position.original_size * position.entry_price)

            capital += profit_dollars

            position.exit_date = date
            position.exit_price = exit_price
            position.profit_pct = profit_pct  # This is the blended profit
            position.raw_profit = raw_profit   # This is the actual exit profit
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

            position = ActivePosition(pair, date, price, direction, position_size, target, max_prob)
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

print("Running backtest with active management...")
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
print("BACKTEST RESULTS - WITH ACTIVE MANAGEMENT")
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

# Analyze scale-out statistics
scaled_out_trades = [p for p in all_positions if p.scaled_out]
print("="*100)
print("ACTIVE MANAGEMENT STATISTICS")
print("="*100)
print()

print(f"Scale-out rule triggered: {len(scaled_out_trades)} / {len(all_positions)} trades ({len(scaled_out_trades)/len(all_positions):.1%})")
print()

if len(scaled_out_trades) > 0:
    scaled_winners = [p for p in scaled_out_trades if p.profit_pct > 0]
    scaled_losers = [p for p in scaled_out_trades if p.profit_pct <= 0]

    print(f"Scaled trades breakdown:")
    print(f"  Winners: {len(scaled_winners)} ({len(scaled_winners)/len(scaled_out_trades):.1%})")
    print(f"  Losers:  {len(scaled_losers)} ({len(scaled_losers)/len(scaled_out_trades):.1%})")
    print()

    if len(scaled_winners) > 0:
        avg_blended = np.mean([p.profit_pct for p in scaled_winners])
        avg_raw = np.mean([p.raw_profit for p in scaled_winners])
        print(f"Winners that were scaled:")
        print(f"  Blended profit (with scale-out): {avg_blended:+.2%}")
        print(f"  Raw profit (if held full):       {avg_raw:+.2%}")
        print(f"  Impact:                          {avg_blended - avg_raw:+.2%}")
        print()

    if len(scaled_losers) > 0:
        avg_blended = np.mean([p.profit_pct for p in scaled_losers])
        avg_raw = np.mean([p.raw_profit for p in scaled_losers])
        print(f"Losers that were scaled:")
        print(f"  Blended profit (with scale-out): {avg_blended:+.2%}")
        print(f"  Raw profit (if held full):       {avg_raw:+.2%}")
        print(f"  Impact:                          {avg_blended - avg_raw:+.2%}")
        print()

print("="*100)
print("COMPARISON TO BASELINE")
print("="*100)
print()

print("Baseline (no active management):")
print("  Average annual return: 53.8%")
print("  Consistency: 5/10 years (50%)")
print("  Average win rate: 89%")
print()

print("With active management:")
print(f"  Average annual return: {avg_return:.1%}")
print(f"  Consistency: {profitable_periods}/10 years ({profitable_periods*10}%)")
print(f"  Average win rate: {avg_winrate:.0%}")
print()

improvement = avg_return - 0.538
print(f"Improvement: {improvement:+.1%} per year")
print()
