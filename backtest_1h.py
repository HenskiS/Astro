"""
BACKTEST 1-HOUR BREAKOUT STRATEGY
==================================
Test the breakout strategy on 1-hour timeframe with adjusted parameters
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("BACKTEST: 1-HOUR BREAKOUT STRATEGY")
print("="*100)
print()

# Strategy Parameters (adjusted for 1h timeframe)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004  # 0.4% (more conservative for higher frequency)
MIN_CONFIDENCE = 0.65

# Emergency stop: 120 hours = 5 days (vs 15 days for daily)
EMERGENCY_STOP_HOURS = 120
EMERGENCY_STOP_LOSS_PCT = -0.04

# Trailing stop (tighter for intraday)
TRAILING_STOP_TRIGGER = 0.0025  # 0.25% (vs 0.5% for daily)
TRAILING_STOP_PCT = 0.60

# Ladder parameters (smaller for intraday)
LADDER_LEVELS = [0.004, 0.008]  # 0.4%, 0.8% (vs 0.8%, 1.5% for daily)
LADDER_SCALE_PCT = 0.40

# Position limits
MAX_TOTAL_POSITIONS = 40  # Increased from 20 for 4 pairs (vs 90 for daily with 8 pairs)
MAX_POSITIONS_PER_PAIR = 10  # vs 12 for daily

# Data
DATA_DIR = 'data_1h'
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
        self.hours_held = 0
        self.max_profit = 0
        self.trailing_stop = None
        self.partial_exits = []
        self.ladder_level = 0

    def update(self, date, high, low, close):
        self.hours_held += 1

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
        if self.hours_held >= EMERGENCY_STOP_HOURS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            return 'emergency_stop', close, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                hit_stop = low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price + (high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                hit_stop = high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                new_stop = self.entry_price - (self.entry_price - low) * TRAILING_STOP_PCT
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
with open('test_predictions_1h.pkl', 'rb') as f:
    predictions = pickle.load(f)

print(f"Loaded predictions for {len(predictions)} pairs")
for pair in PAIRS:
    preds = predictions[pair]
    print(f"  {pair}: {len(preds)} predictions from {preds.index.min()} to {preds.index.max()}")
print()

# Load raw price data
print("Loading raw price data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1h.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df
    print(f"  {pair}: {len(df)} candles")
print()

# Run backtest
print("="*100)
print("RUNNING BACKTEST")
print("="*100)
print()

capital = INITIAL_CAPITAL
positions = []
all_trades = []

# Get all trading hours (union of all prediction dates)
all_trading_hours = set()
for pair in PAIRS:
    all_trading_hours.update(predictions[pair].index)

# Get date range
min_date = min(all_trading_hours)
max_date = max(all_trading_hours)

# Extend to include all hours in raw data within range
for pair in PAIRS:
    pair_df = all_raw_data[pair]
    hours_in_range = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
    all_trading_hours.update(hours_in_range)

all_trading_hours = sorted(list(all_trading_hours))

print(f"Backtesting from {min_date} to {max_date}")
print(f"Total hours to process: {len(all_trading_hours):,}")
print()

# Track equity curve
equity_curve = []

for hour_idx, date in enumerate(all_trading_hours):
    # Get prices for this hour
    prices_dict = {}
    for pair in PAIRS:
        if date in all_raw_data[pair].index:
            row = all_raw_data[pair].loc[date]
            prices_dict[pair] = {
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }

    # Update all positions
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

        all_trades.append({
            'pair': position.pair,
            'entry_date': position.entry_date,
            'entry_price': position.entry_price,
            'exit_date': date,
            'exit_price': exit_price,
            'direction': position.direction,
            'hours_held': position.hours_held,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'exit_reason': exit_reason,
            'ladder_hits': position.ladder_level,
            'capital_after': capital,
            'confidence': position.confidence
        })

        equity_curve.append(capital)

    # Open new positions (only on hours with predictions)
    if len(positions) >= MAX_TOTAL_POSITIONS:
        continue

    for pair in PAIRS:
        # Check if we have predictions for this hour
        if date not in predictions[pair].index:
            continue

        # Check per-pair position limit
        pair_positions = [p for p in positions if p.pair == pair]
        if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
            continue

        row = predictions[pair].loc[date]

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

        # Determine direction and target
        if breakout_high_prob > breakout_low_prob:
            direction = 'long'
            breakout_level = row['high_80h']
            target = breakout_level * 1.005
        else:
            direction = 'short'
            breakout_level = row['low_80h']
            target = breakout_level * 0.995

        position = Position(pair, date, price, direction, position_size, target, max_prob)
        positions.append(position)

    # Progress update every 500 hours
    if (hour_idx + 1) % 500 == 0:
        pct_complete = (hour_idx + 1) / len(all_trading_hours) * 100
        print(f"Progress: {hour_idx + 1:>6,}/{len(all_trading_hours):,} hours ({pct_complete:>5.1f}%) | "
              f"Capital: ${capital:>10,.0f} | Open positions: {len(positions):>3} | "
              f"Total trades: {len(all_trades):>5,}")

print()
print("="*100)
print("BACKTEST RESULTS")
print("="*100)
print()

# Calculate statistics
trades_df = pd.DataFrame(all_trades)

winners = trades_df[trades_df['profit_pct'] > 0]
losers = trades_df[trades_df['profit_pct'] <= 0]

total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

# Calculate max drawdown
peak = INITIAL_CAPITAL
max_dd = 0
for eq in equity_curve:
    peak = max(peak, eq)
    dd = (eq - peak) / peak
    max_dd = min(max_dd, dd)

# Time period
start_date = trades_df['entry_date'].min()
end_date = trades_df['exit_date'].max()
days = (end_date - start_date).days
years = days / 365.25
cagr = (capital / INITIAL_CAPITAL) ** (1/years) - 1 if years > 0 else 0

print(f"Starting Capital:     ${INITIAL_CAPITAL:,.0f}")
print(f"Ending Capital:       ${capital:,.0f}")
print(f"Total Return:         {total_return:.1%}")
print(f"CAGR:                 {cagr:.1%}")
print()
print(f"Max Drawdown:         {max_dd:.1%}")
print()
print(f"Total Trades:         {len(trades_df):,}")
print(f"Win Rate:             {win_rate:.1%} ({len(winners)}/{len(losers)} W/L)")
print()
print(f"Avg Winner:           {winners['profit_pct'].mean():+.2%}")
print(f"Avg Loser:            {losers['profit_pct'].mean():+.2%}")
print(f"Avg Hold (Winners):   {winners['hours_held'].mean():.1f} hours ({winners['hours_held'].mean()/24:.1f} days)")
print(f"Avg Hold (Losers):    {losers['hours_held'].mean():.1f} hours ({losers['hours_held'].mean()/24:.1f} days)")
print()

# Exit reason breakdown
print("Exit Reasons:")
for reason in trades_df['exit_reason'].unique():
    count = (trades_df['exit_reason'] == reason).sum()
    pct = count / len(trades_df)
    avg_profit = trades_df[trades_df['exit_reason'] == reason]['profit_pct'].mean()
    print(f"  {reason:20s} {count:>6,} ({pct:>5.1%}) | Avg P/L: {avg_profit:>+7.2%}")
print()

# Results by pair
print("Results by Pair:")
for pair in PAIRS:
    pair_trades = trades_df[trades_df['pair'] == pair]
    if len(pair_trades) > 0:
        pair_winners = pair_trades[pair_trades['profit_pct'] > 0]
        pair_win_rate = len(pair_winners) / len(pair_trades)
        pair_avg_profit = pair_trades['profit_pct'].mean()
        print(f"  {pair:8s} {len(pair_trades):>6,} trades | Win rate: {pair_win_rate:>5.1%} | "
              f"Avg P/L: {pair_avg_profit:>+7.2%}")
print()

print("="*100)
print("COMPARISON WITH DAILY STRATEGY")
print("="*100)
print()
print("Daily Strategy (8 pairs, ~10 years):")
print("  CAGR: 51.4%")
print("  Max DD: -48.2%")
print("  Win Rate: 72.0%")
print()
print(f"1-Hour Strategy (4 pairs, {years:.1f} years):")
print(f"  CAGR: {cagr:.1%}")
print(f"  Max DD: {max_dd:.1%}")
print(f"  Win Rate: {win_rate:.1%}")
print()

# Save results
print("Saving results...")
trades_df.to_csv('backtest_1h_results.csv', index=False)
print("Results saved to: backtest_1h_results.csv")
print()

print("="*100)
print("DONE!")
print("="*100)
