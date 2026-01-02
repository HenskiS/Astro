"""
OPTIMIZE 5M STRATEGY PARAMETERS
================================
Test different parameter combinations to find optimal settings for 5m trading.

Key areas to optimize:
1. Emergency stop: MUCH shorter than current 144 periods
2. Trailing stop: Tighter trigger and trail %
3. Confidence threshold: Higher to be more selective
4. Lookback period: Maybe shorter for faster breakouts
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from itertools import product
warnings.filterwarnings('ignore')

print("="*100)
print("5M PARAMETER OPTIMIZATION")
print("="*100)
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
AVOID_HOURS = [20, 21, 22]
MAX_TOTAL_POSITIONS = 60
MAX_POSITIONS_PER_PAIR = 15
DATA_DIR = 'data_5m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']

# Parameters to test
PARAM_GRID = {
    'emergency_periods': [24, 36, 48, 72],  # 2h, 3h, 4h, 6h (much shorter!)
    'trailing_trigger': [0.0003, 0.0005, 0.0007, 0.001],  # 0.03% to 0.1%
    'trailing_pct': [0.70, 0.75, 0.80],  # Tighter trailing
    'confidence': [0.65, 0.70, 0.75],  # Higher thresholds
    'ladder_level': [[0.001, 0.002], [0.0015, 0.003]]  # Smaller ladders
}

print("Parameter space:")
print(f"  Emergency stops: {PARAM_GRID['emergency_periods']} periods")
print(f"  Trailing triggers: {PARAM_GRID['trailing_trigger']}")
print(f"  Trailing %: {PARAM_GRID['trailing_pct']}")
print(f"  Confidence: {PARAM_GRID['confidence']}")
print(f"  Ladders: {PARAM_GRID['ladder_level']}")
print()

total_combos = (len(PARAM_GRID['emergency_periods']) *
                len(PARAM_GRID['trailing_trigger']) *
                len(PARAM_GRID['trailing_pct']) *
                len(PARAM_GRID['confidence']) *
                len(PARAM_GRID['ladder_level']))

print(f"Total combinations to test: {total_combos}")
print()


class Position:
    def __init__(self, pair, entry_date, entry_price, direction, size, breakout_target, confidence,
                 emergency_periods, trailing_trigger, trailing_pct, ladder_levels):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.original_size = size
        self.breakout_target = breakout_target
        self.confidence = confidence
        self.periods_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

        # Parameterized
        self.emergency_periods = emergency_periods
        self.trailing_trigger = trailing_trigger
        self.trailing_pct = trailing_pct
        self.ladder_levels = ladder_levels

    def update(self, date, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close):
        self.periods_held += 1

        if self.direction == 'long':
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

        self.max_profit = max(self.max_profit, intraday_high_profit)

        # Ladder
        if self.ladder_level < len(self.ladder_levels):
            if intraday_high_profit >= self.ladder_levels[self.ladder_level]:
                self.partial_exits.append((self.ladder_levels[self.ladder_level], 0.40))
                self.size *= 0.60
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.periods_held >= self.emergency_periods and current_profit < -0.04:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > self.trailing_trigger:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * self.trailing_pct
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * self.trailing_pct
                self.trailing_stop = min(self.trailing_stop, new_stop)

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


# Load predictions
print("Loading predictions...")
with open('test_predictions_5m.pkl', 'rb') as f:
    predictions = pickle.load(f)

# Load raw data
print("Loading raw 5m data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_5m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Get trading periods
all_trading_periods = set()
for pair_df in predictions.values():
    all_trading_periods.update(pair_df.index)
all_trading_periods = sorted(list(all_trading_periods))

print(f"Test period: {len(all_trading_periods):,} periods")
print()

# Test all parameter combinations
results = []
combo_num = 0

for emergency, trigger, trail_pct, conf, ladders in product(
    PARAM_GRID['emergency_periods'],
    PARAM_GRID['trailing_trigger'],
    PARAM_GRID['trailing_pct'],
    PARAM_GRID['confidence'],
    PARAM_GRID['ladder_level']
):
    combo_num += 1

    # Run backtest with these parameters
    capital = INITIAL_CAPITAL
    positions = []
    trades = []

    for date in all_trading_periods:
        # Get prices
        prices_dict = {}
        for pair in PAIRS:
            if date in all_raw_data[pair].index:
                row = all_raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'],
                    'bid_close': row['bid_close'],
                    'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'],
                    'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue

            prices = prices_dict[position.pair]
            exit_info = position.update(
                date,
                prices['bid_high'],
                prices['bid_low'],
                prices['bid_close'],
                prices['ask_high'],
                prices['ask_low'],
                prices['ask_close']
            )
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
                'profit_pct': profit_pct,
                'periods_held': position.periods_held
            })

        # Open new positions
        if len(positions) >= MAX_TOTAL_POSITIONS:
            continue

        if date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue

            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]

            breakout_high_prob = row['breakout_high_prob']
            breakout_low_prob = row['breakout_low_prob']
            max_prob = max(breakout_high_prob, breakout_low_prob)

            if max_prob <= conf:
                continue

            risk_amount = capital * RISK_PER_TRADE
            mid_price = row['close']
            position_size = risk_amount / (mid_price * 0.02)

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_240p']
                target = breakout_level * 1.005
                if pair not in prices_dict:
                    continue
                entry_price = prices_dict[pair]['ask_close']
            else:
                direction = 'short'
                breakout_level = row['low_240p']
                target = breakout_level * 0.995
                if pair not in prices_dict:
                    continue
                entry_price = prices_dict[pair]['bid_close']

            position = Position(pair, date, entry_price, direction, position_size,
                              target, max_prob, emergency, trigger, trail_pct, ladders)
            positions.append(position)

    # Calculate metrics
    if len(trades) == 0:
        continue

    trades_df = pd.DataFrame(trades)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df)

    avg_winner = winners['profit_pct'].mean() if len(winners) > 0 else 0
    avg_loser = losers['profit_pct'].mean() if len(losers) > 0 else 0
    profit_ratio = abs(avg_winner / avg_loser) if len(losers) > 0 and avg_loser != 0 else 0

    avg_winner_hold = winners['periods_held'].mean() if len(winners) > 0 else 0
    avg_loser_hold = losers['periods_held'].mean() if len(losers) > 0 else 0

    results.append({
        'emergency_periods': emergency,
        'trailing_trigger': trigger,
        'trailing_pct': trail_pct,
        'confidence': conf,
        'ladder_levels': str(ladders),
        'final_capital': capital,
        'total_return': total_return,
        'num_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'profit_ratio': profit_ratio,
        'avg_winner_hold_periods': avg_winner_hold,
        'avg_loser_hold_periods': avg_loser_hold,
        'avg_winner_hold_mins': avg_winner_hold * 5,
        'avg_loser_hold_mins': avg_loser_hold * 5
    })

    if combo_num % 20 == 0:
        print(f"Tested {combo_num}/{total_combos} combinations...")

print()
print("="*100)
print("TOP 10 CONFIGURATIONS BY TOTAL RETURN")
print("="*100)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return', ascending=False)

for idx, row in results_df.head(10).iterrows():
    print(f"Rank #{results_df.index.get_loc(idx) + 1}:")
    print(f"  Emergency: {row['emergency_periods']} periods ({row['emergency_periods']*5/60:.1f}h)")
    print(f"  Trailing: {row['trailing_trigger']:.4f} trigger, {row['trailing_pct']:.0%} trail")
    print(f"  Confidence: {row['confidence']:.2f}")
    print(f"  Ladders: {row['ladder_levels']}")
    print(f"  Return: {row['total_return']:+.1%} (${row['final_capital']:.0f})")
    print(f"  Trades: {row['num_trades']:.0f} | Win Rate: {row['win_rate']:.1%}")
    print(f"  Avg Winner: {row['avg_winner']:+.2%} ({row['avg_winner_hold_mins']:.0f}m / {row['avg_winner_hold_mins']/60:.1f}h)")
    print(f"  Avg Loser: {row['avg_loser']:+.2%} ({row['avg_loser_hold_mins']:.0f}m / {row['avg_loser_hold_mins']/60:.1f}h)")
    print(f"  Profit Ratio: {row['profit_ratio']:.2f}:1")
    print()

print("="*100)
print("TOP 10 BY PROFIT RATIO")
print("="*100)
print()

results_df_ratio = results_df[results_df['profit_ratio'] > 0].sort_values('profit_ratio', ascending=False)

for idx, row in results_df_ratio.head(10).iterrows():
    print(f"Rank #{results_df_ratio.index.get_loc(idx) + 1}:")
    print(f"  Emergency: {row['emergency_periods']} periods ({row['emergency_periods']*5/60:.1f}h)")
    print(f"  Trailing: {row['trailing_trigger']:.4f} trigger, {row['trailing_pct']:.0%} trail")
    print(f"  Confidence: {row['confidence']:.2f}")
    print(f"  Return: {row['total_return']:+.1%}")
    print(f"  Win Rate: {row['win_rate']:.1%} | Profit Ratio: {row['profit_ratio']:.2f}:1")
    print(f"  Hold times: {row['avg_winner_hold_mins']/60:.1f}h winners, {row['avg_loser_hold_mins']/60:.1f}h losers")
    print()

# Save results
print("Saving results...")
results_df.to_csv('optimize_5m_results.csv', index=False)
print("Results saved to: optimize_5m_results.csv")
print()

print("="*100)
