"""
BACKTEST 5-MINUTE BREAKOUT STRATEGY
====================================
Extreme high-frequency adaptation with:
- Emergency: 144 periods (12 hours)
- Trailing: 0.001 trigger (0.1%), 65% trail
- Ladders: [0.002, 0.004] (0.2%, 0.4%)
- Spread costs: CRITICAL at this timeframe

WARNING: Spread costs (~0.015%) are HUGE relative to 5m moves (~0.03%)
Expected to significantly impact profitability.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BACKTEST: 5-MINUTE BREAKOUT STRATEGY (WITH SPREAD COSTS)")
print("="*100)
print()
print("WARNING: Spread costs are EXTREMELY significant at 5m timeframe!")
print("  - Typical 5m move: 0.02-0.05%")
print("  - Round-trip spread: ~0.015%")
print("  - Spread eats 30-75% of profit!")
print()
print("Using bid/ask prices:")
print("  - Long entry: ASK | Long exit: BID")
print("  - Short entry: BID | Short exit: ASK")
print()

# Strategy Parameters (extreme HF)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004  # 0.4%
MIN_CONFIDENCE = 0.65

# Emergency stop: 144 periods = 12 hours
EMERGENCY_STOP_PERIODS = 144
EMERGENCY_STOP_LOSS_PCT = -0.04

# Trailing stop (very tight for 5m)
TRAILING_STOP_TRIGGER = 0.001  # 0.1%
TRAILING_STOP_PCT = 0.65  # 65%

# Ladder (smaller for 5m)
LADDER_LEVELS = [0.002, 0.004]  # 0.2%, 0.4%
LADDER_SCALE_PCT = 0.40

# Position limits
MAX_TOTAL_POSITIONS = 60
MAX_POSITIONS_PER_PAIR = 15

# Avoid high-spread hours
AVOID_HOURS = [20, 21, 22]

# Data
DATA_DIR = 'data_5m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']


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


# Load predictions
print("Loading predictions...")
with open('test_predictions_5m.pkl', 'rb') as f:
    predictions = pickle.load(f)

print(f"Loaded predictions for {len(predictions)} pairs")
print()

# Load raw data
print("Loading raw 5m data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_5m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
    print(f"  {pair}: {len(df)} candles")

print()

# Get trading periods
all_trading_periods = set()
for pair_df in predictions.values():
    all_trading_periods.update(pair_df.index)

all_trading_periods = sorted(list(all_trading_periods))

min_date = min(all_trading_periods)
max_date = max(all_trading_periods)

print(f"Backtesting from {min_date} to {max_date}")
print(f"Total 5m periods: {len(all_trading_periods):,}")
print()

# Run backtest
capital = INITIAL_CAPITAL
positions = []
trades = []
equity_curve = []

for period_idx, date in enumerate(all_trading_periods):
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
            'pair': position.pair,
            'entry_date': position.entry_date,
            'exit_date': date,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.original_size,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'periods_held': position.periods_held,
            'exit_reason': exit_reason,
            'confidence': position.confidence,
            'capital_after': capital
        })

        equity_curve.append(capital)

    # Open new positions
    if len(positions) >= MAX_TOTAL_POSITIONS:
        continue

    current_hour = date.hour
    if current_hour in AVOID_HOURS:
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

        if max_prob <= MIN_CONFIDENCE:
            continue

        assumed_risk_pct = 0.02
        risk_amount = capital * RISK_PER_TRADE
        mid_price = row['close']
        position_size = risk_amount / (mid_price * assumed_risk_pct)

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

        position = Position(pair, date, entry_price, direction, position_size, target, max_prob)
        positions.append(position)

    # Progress
    if (period_idx + 1) % 2000 == 0:
        pct_complete = (period_idx + 1) / len(all_trading_periods) * 100
        print(f"Progress: {period_idx + 1:>6,}/{len(all_trading_periods):,} periods ({pct_complete:>5.1f}%) | "
              f"Capital: ${capital:>10,.0f} | Positions: {len(positions):>3} | "
              f"Trades: {len(trades):>5,}")

print()

# Results
trades_df = pd.DataFrame(trades)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
days = (max_date - min_date).days
years = days / 365
cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]
win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

# Max DD
equity_values = [INITIAL_CAPITAL] + list(trades_df['capital_after'])
peak = INITIAL_CAPITAL
max_dd = 0
for val in equity_values[1:]:
    peak = max(peak, val)
    dd = (val - peak) / peak
    max_dd = min(max_dd, dd)

print("="*100)
print("RESULTS")
print("="*100)
print()
print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Final Capital: ${capital:,.0f}")
print(f"Total Return: {total_return:+.1%}")
print(f"CAGR: {cagr:.1%}")
print(f"Max Drawdown: {max_dd:.1%}")
print()
print(f"Total Trades: {len(trades_df):,}")
print(f"Winners: {len(winners):,} ({win_rate:.1%})")
print(f"Losers: {len(losers):,}")
print()

if len(winners) > 0:
    avg_winner = winners['profit_pct'].mean()
    avg_winner_periods = winners['periods_held'].mean()
    avg_winner_mins = avg_winner_periods * 5
    print(f"Avg Winner: {avg_winner:+.2%} ({avg_winner_mins:.0f} minutes / {avg_winner_mins/60:.1f} hours)")

if len(losers) > 0:
    avg_loser = losers['profit_pct'].mean()
    avg_loser_periods = losers['periods_held'].mean()
    avg_loser_mins = avg_loser_periods * 5
    print(f"Avg Loser: {avg_loser:+.2%} ({avg_loser_mins:.0f} minutes / {avg_loser_mins/60:.1f} hours)")

if len(winners) > 0 and len(losers) > 0:
    profit_ratio = abs(avg_winner / avg_loser)
    print(f"Profit Ratio: {profit_ratio:.2f}:1")

print()

# Exit reasons
print("Exit Reasons:")
exit_reasons = trades_df['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = count / len(trades_df) * 100
    print(f"  {reason:20s} {count:>5,} ({pct:>5.1f}%)")

print()

# Per-pair
print("="*100)
print("PER-PAIR RESULTS")
print("="*100)
print()

for pair in PAIRS:
    pair_trades = trades_df[trades_df['pair'] == pair]
    if len(pair_trades) == 0:
        continue

    pair_winners = pair_trades[pair_trades['profit_pct'] > 0]
    pair_win_rate = len(pair_winners) / len(pair_trades)
    pair_avg_profit = pair_trades['profit_pct'].mean()

    print(f"  {pair:8s} {len(pair_trades):>6,} trades | Win rate: {pair_win_rate:>5.1%} | "
          f"Avg P/L: {pair_avg_profit:>+7.2%}")

print()

print("="*100)
print("COMPARISON: 15M vs 5M")
print("="*100)
print()
print("15M Strategy (4 pairs, 4.4 months):")
print("  CAGR: 137.1%")
print("  Max DD: -1.6%")
print("  Win Rate: 93.8%")
print("  Avg Hold: 58.7 hours winners, 83.6 hours losers")
print()
print(f"5M Strategy (4 pairs, {years:.1f} years):")
print(f"  CAGR: {cagr:.1%}")
print(f"  Max DD: {max_dd:.1%}")
print(f"  Win Rate: {win_rate:.1%}")
if len(winners) > 0 and len(losers) > 0:
    print(f"  Avg Hold: {avg_winner_mins/60:.1f} hours winners, {avg_loser_mins/60:.1f} hours losers")
print()
print("Key insight: Spread costs likely destroy profitability at 5m")
print("  - 5m moves: ~0.03%")
print("  - Spread cost: ~0.015% round-trip")
print("  - Spread eats 50% of profits!")
print()

# Save
print("Saving results...")
trades_df.to_csv('backtest_5m_results.csv', index=False)
print("Results saved to: backtest_5m_results.csv")
print()

print("="*100)
