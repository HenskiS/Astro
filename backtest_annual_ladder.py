"""
BACKTEST ANNUAL RETRAINING - OPTIMAL LADDER
============================================
Test the optimal ladder strategy with annual model retraining
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BACKTEST: ANNUAL RETRAINING WITH OPTIMAL LADDER")
print("="*100)
print()

# Strategy Parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_LOSS_PCT = -0.04
EMERGENCY_STOP_DAYS = 15
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60

# Ladder parameters
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.33


class Position:
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
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

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

        # Check ladder
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
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


def run_backtest(period_predictions, starting_capital):
    """Run backtest for a specific period"""
    capital = starting_capital
    positions = []
    trades = []

    all_dates = set()
    for pair_df in period_predictions.values():
        dates = pair_df.index
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    for date in all_dates:
        prices_dict = {}
        for pair, pair_df in period_predictions.items():
            if date in pair_df.index:
                row = pair_df.loc[date]
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

            trades.append({
                'pair': position.pair,
                'entry_date': position.entry_date,
                'exit_date': date,
                'direction': position.direction,
                'days_held': position.days_held,
                'profit_pct': profit_pct,
                'exit_reason': exit_reason,
                'ladder_hits': position.ladder_level
            })

        # Open new positions
        for pair, pair_df in period_predictions.items():
            if date not in pair_df.index:
                continue

            row = pair_df.loc[date]

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

            position = Position(pair, date, price, direction, position_size, target, max_prob)
            positions.append(position)

    return capital, trades


# Load annual predictions
print("Loading annual predictions...")
with open('model_predictions_annual.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Run backtest year by year
capital = INITIAL_CAPITAL
yearly_results = []

print()
print("Running backtest by year:")
print("-" * 100)

for year_name, year_preds in sorted(all_predictions.items()):
    starting_cap = capital
    ending_cap, trades = run_backtest(year_preds, capital)

    total_return = (ending_cap - starting_cap) / starting_cap

    winners = [t for t in trades if t['profit_pct'] > 0]
    losers = [t for t in trades if t['profit_pct'] <= 0]

    yearly_results.append({
        'year': year_name,
        'starting': starting_cap,
        'ending': ending_cap,
        'return': total_return,
        'trades': len(trades),
        'winners': len(winners),
        'losers': len(losers)
    })

    capital = ending_cap

    print(f"{year_name}: ${starting_cap:>10,.0f} -> ${ending_cap:>10,.0f} ({total_return:>+7.1%}) | "
          f"{len(trades):>3} trades ({len(winners)}/{len(losers)} W/L)")

# Summary statistics
print()
print("="*100)
print("ANNUAL RETRAINING RESULTS")
print("="*100)
print()

total_trades = sum(r['trades'] for r in yearly_results)
total_winners = sum(r['winners'] for r in yearly_results)
total_losers = sum(r['losers'] for r in yearly_results)
win_rate = total_winners / total_trades if total_trades > 0 else 0

profitable_years = sum(1 for r in yearly_results if r['return'] > 0)
avg_return = np.mean([r['return'] for r in yearly_results])

years = len(yearly_results)
cagr = (capital / INITIAL_CAPITAL) ** (1/years) - 1

print(f"Starting Capital:     ${INITIAL_CAPITAL:,.0f}")
print(f"Ending Capital:       ${capital:,.0f}")
print(f"Total Return:         {(capital/INITIAL_CAPITAL - 1):.1%}")
print(f"CAGR:                 {cagr:.1%}")
print()
print(f"Total Trades:         {total_trades}")
print(f"Win Rate:             {win_rate:.1%} ({total_winners}/{total_losers} W/L)")
print(f"Profitable Years:     {profitable_years}/{years}")
print(f"Average Annual:       {avg_return:+.1%}")
print()

# Year by year breakdown
print("Year-by-Year Returns:")
print("-" * 60)
for r in yearly_results:
    print(f"  {r['year']}: {r['return']:>+7.1%}")
print()

# Drawdown analysis
print("Drawdown Analysis:")
equity_values = [INITIAL_CAPITAL]
for r in yearly_results:
    equity_values.append(r['ending'])

peak = INITIAL_CAPITAL
max_dd = 0
for val in equity_values[1:]:
    peak = max(peak, val)
    dd = (val - peak) / peak
    max_dd = min(max_dd, dd)

print(f"  Max Drawdown: {max_dd:.1%}")
print()

print("="*100)
