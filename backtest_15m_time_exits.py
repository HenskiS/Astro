"""
TEST TIME-BASED EXITS FOR 15M STRATEGY
=======================================
Compare different time-based exit rules:
1. Max holding period (exit after X hours)
2. End-of-day exits (close all at specific hour)
3. Combination of time + profit targets

This tests if forcing exits prevents overnight/weekend gaps
and improves risk management.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING TIME-BASED EXITS - 15M STRATEGY")
print("="*100)
print()

# Fixed parameters
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004
MIN_CONFIDENCE = 0.70
EMERGENCY_STOP_PERIODS = 24
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.001
TRAILING_STOP_PCT = 0.75
LADDER_LEVELS = [0.002, 0.004]
LADDER_SCALE_PCT = 0.40
MAX_TOTAL_POSITIONS = 120
MAX_POSITIONS_PER_PAIR = 15
AVOID_HOURS = [20, 21, 22]
DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Time-based exit configurations to test
TIME_EXIT_CONFIGS = [
    {
        'name': 'BASELINE (No time exits)',
        'max_hold_periods': None,  # No limit
        'eod_exit_hour': None,     # No EOD exit
        'weekend_exit': False
    },
    {
        'name': 'Max 12h hold',
        'max_hold_periods': 48,    # 12 hours
        'eod_exit_hour': None,
        'weekend_exit': False
    },
    {
        'name': 'Max 24h hold',
        'max_hold_periods': 96,    # 24 hours
        'eod_exit_hour': None,
        'weekend_exit': False
    },
    {
        'name': 'EOD Exit at 16:00 UTC',
        'max_hold_periods': None,
        'eod_exit_hour': 16,       # Close all positions at 4pm UTC
        'weekend_exit': False
    },
    {
        'name': 'EOD Exit at 20:00 UTC',
        'max_hold_periods': None,
        'eod_exit_hour': 20,       # Close all positions at 8pm UTC
        'weekend_exit': False
    },
    {
        'name': 'Weekend Exit (Friday close)',
        'max_hold_periods': None,
        'eod_exit_hour': None,
        'weekend_exit': True       # Close all on Friday
    },
    {
        'name': 'Max 12h + Weekend Exit',
        'max_hold_periods': 48,
        'eod_exit_hour': None,
        'weekend_exit': True
    }
]


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
        self.periods_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

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
        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        # Emergency stop
        if self.periods_held >= EMERGENCY_STOP_PERIODS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (bid_high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - ask_low) * TRAILING_STOP_PCT
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


# Load data
print("Loading predictions and data...")
with open('test_predictions_15m.pkl', 'rb') as f:
    predictions = pickle.load(f)

all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

all_trading_periods = set()
for pair_df in predictions.values():
    all_trading_periods.update(pair_df.index)
all_trading_periods = sorted(list(all_trading_periods))

print(f"Test period: {len(all_trading_periods):,} periods")
print()

# Test each configuration
results = []

for config in TIME_EXIT_CONFIGS:
    print(f"Testing: {config['name']}")

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
                    'bid_high': row['bid_high'], 'bid_low': row['bid_low'], 'bid_close': row['bid_close'],
                    'ask_high': row['ask_high'], 'ask_low': row['ask_low'], 'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # === TIME-BASED EXIT CHECKS ===
        time_based_exits = []

        # Check max holding period
        if config['max_hold_periods'] is not None:
            for position in positions:
                if position.periods_held >= config['max_hold_periods']:
                    time_based_exits.append((position, 'max_hold_time'))

        # Check EOD exit
        if config['eod_exit_hour'] is not None:
            if date.hour == config['eod_exit_hour']:
                # Exit all positions
                for position in positions:
                    time_based_exits.append((position, 'eod_exit'))

        # Check weekend exit (Friday)
        if config['weekend_exit']:
            if date.weekday() == 4 and date.hour >= 20:  # Friday after 8pm
                for position in positions:
                    time_based_exits.append((position, 'weekend_exit'))

        # Process time-based exits
        for position, reason in time_based_exits:
            if position.pair not in prices_dict:
                continue

            prices = prices_dict[position.pair]

            # Close at current price
            if position.direction == 'long':
                exit_price = prices['bid_close']
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                exit_price = prices['ask_close']
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)

            capital += profit_dollars
            if position in positions:  # Check if still in list
                positions.remove(position)

            trades.append({
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'periods_held': position.periods_held,
                'exit_reason': reason
            })

        # Update positions (normal stops/targets)
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue

            prices = prices_dict[position.pair]
            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info is not None:
                positions_to_close.append((position, exit_info))

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
                'profit_dollars': profit_dollars,
                'periods_held': position.periods_held,
                'exit_reason': exit_reason
            })

        # Open new positions
        if len(positions) >= MAX_TOTAL_POSITIONS or date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue

            pair_positions = [p for p in positions if p.pair == pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])

            if max_prob <= MIN_CONFIDENCE:
                continue

            risk_amount = capital * RISK_PER_TRADE
            position_size = risk_amount / (row['close'] * 0.02)

            if row['breakout_high_prob'] > row['breakout_low_prob']:
                direction = 'long'
                target = row['high_80p'] * 1.005
                entry_price = prices_dict[pair]['ask_close'] if pair in prices_dict else row['close']
            else:
                direction = 'short'
                target = row['low_80p'] * 0.995
                entry_price = prices_dict[pair]['bid_close'] if pair in prices_dict else row['close']

            position = Position(pair, date, entry_price, direction, position_size, target, max_prob)
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

    avg_winner_hold = winners['periods_held'].mean() * 0.25 if len(winners) > 0 else 0
    avg_loser_hold = losers['periods_held'].mean() * 0.25 if len(losers) > 0 else 0

    # CAGR
    days = (all_trading_periods[-1] - all_trading_periods[0]).days
    years = days / 365
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    # Count time-based exits
    time_exits = trades_df[trades_df['exit_reason'].isin(['max_hold_time', 'eod_exit', 'weekend_exit'])]
    time_exit_pct = len(time_exits) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    print(f"  Return: {total_return:+.1%} (${capital:.0f}) | CAGR: {cagr:.1%}")
    print(f"  Trades: {len(trades_df)} | Win Rate: {win_rate:.1%}")
    print(f"  Profit Ratio: {profit_ratio:.2f}:1")
    print(f"  Hold: {avg_winner_hold:.1f}h winners, {avg_loser_hold:.1f}h losers")
    print(f"  Time exits: {len(time_exits)} ({time_exit_pct:.1f}%)")
    print()

    results.append({
        'config': config['name'],
        'final_capital': capital,
        'return': total_return,
        'cagr': cagr,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'profit_ratio': profit_ratio,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'avg_winner_hold_hours': avg_winner_hold,
        'avg_loser_hold_hours': avg_loser_hold,
        'time_exit_count': len(time_exits),
        'time_exit_pct': time_exit_pct
    })

print("="*100)
print("COMPARISON")
print("="*100)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('return', ascending=False)

print(f"{'Configuration':<30} {'Return':>10} {'CAGR':>10} {'Win%':>8} {'Ratio':>8} {'Time Exits':>12}")
print("-"*100)

for _, row in results_df.iterrows():
    print(f"{row['config']:<30} {row['return']:>9.1%} {row['cagr']:>9.1%} {row['win_rate']:>7.1%} "
          f"{row['profit_ratio']:>7.2f}:1 {row['time_exit_count']:>6.0f} ({row['time_exit_pct']:>4.1f}%)")

print()
print("="*100)
print("ANALYSIS")
print("="*100)
print()

baseline = results_df[results_df['config'].str.contains('BASELINE')].iloc[0]
print(f"Baseline (no time exits): {baseline['return']:+.1%} return, {baseline['cagr']:.1%} CAGR")
print()

print("Impact of time-based exits:")
for _, row in results_df.iterrows():
    if 'BASELINE' in row['config']:
        continue

    return_diff = row['return'] - baseline['return']
    cagr_diff = row['cagr'] - baseline['cagr']

    symbol = "+" if return_diff > 0 else ""
    print(f"  {row['config']:<30} {symbol}{return_diff:>6.1%} return, {symbol}{cagr_diff:>5.1%} CAGR")

print()
print("="*100)
print()

print("Key Insights:")
print("  - Time exits can reduce overnight/weekend gap risk")
print("  - But they may also cut winning trades short")
print("  - Best approach depends on your risk tolerance")
print()

# Save results
results_df.to_csv('time_exit_test_results.csv', index=False)
print("Results saved to: time_exit_test_results.csv")
print()
print("="*100)
