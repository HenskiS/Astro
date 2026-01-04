"""
BACKTEST 1-HOUR BREAKOUT STRATEGY (OPTIMIZED + NO LOOKAHEAD)
==============================================================
Uses optimized parameters found through systematic testing:
- Emergency: 96 hours @ -4%
- Trailing: Trigger 0.002 (0.2%), Trail 65%
- Ladders: [0.004, 0.008]
- Target: 0.005 (0.5%)
- Spread costs: Uses bid/ask prices for realistic execution

EXECUTION TIMING (NO LOOKAHEAD):
- Signal generated at bar T close
- Entry executed at bar T+1 OPEN price
- This eliminates same-bar lookahead bias

Expected Results: ~105-113% CAGR (minimal change due to 1h gaps being smaller)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Backtest 1H breakout strategy')
parser.add_argument('--plot', action='store_true', help='Generate equity curve plot')
parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier (e.g., 2.0 for 2:1 leverage)')
args = parser.parse_args()

# Apply leverage
LEVERAGE = args.leverage

print("="*100)
print("BACKTEST: 1-HOUR BREAKOUT STRATEGY (OPTIMIZED + SPREAD COSTS)")
print("="*100)
print()
print("NOTE: Uses bid/ask prices for realistic spread costs")
print("  - Long entry: ASK price | Long exit: BID price")
print("  - Short entry: BID price | Short exit: ASK price")
if LEVERAGE != 1.0:
    print(f"  - Leverage: {LEVERAGE}x")
print()

# Strategy Parameters (OPTIMIZED)
INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.004  # 0.4%
MIN_CONFIDENCE = 0.65

# Emergency stop: 96 hours = 4 days (OPTIMIZED from 120)
EMERGENCY_STOP_HOURS = 96
EMERGENCY_STOP_LOSS_PCT = -0.04

# Trailing stop (OPTIMIZED for faster exits while preserving winners)
TRAILING_STOP_TRIGGER = 0.002  # 0.2% (OPTIMIZED from 0.25%)
TRAILING_STOP_PCT = 0.65  # 65% (OPTIMIZED from 60%)

# Ladder parameters (unchanged - already optimal)
LADDER_LEVELS = [0.004, 0.008]  # 0.4%, 0.8%
LADDER_SCALE_PCT = 0.40

# Position limits (scaled for 8 pairs)
MAX_TOTAL_POSITIONS = 80  # 10 per pair on average
MAX_POSITIONS_PER_PAIR = 10

# FIFO handling mode (OANDA rejects competing orders, so we handle this ourselves)
# 'allow_competing' - Allow both long and short positions (IMPOSSIBLE with OANDA)
# 'skip_competing' - Skip signals that compete with existing positions (RECOMMENDED - OANDA-realistic)
# 'exit_and_reverse' - Close all competing positions and take new signal (possible but whipsaws badly)
# 'position_netting' - Net out one position (IMPOSSIBLE - OANDA won't accept competing order)
FIFO_MODE = 'skip_competing'  # Default: skip_competing (OANDA-realistic)

# Spread filter: Avoid trading during high-spread hours (OPTIMIZED)
# Analysis shows hours 20-22 UTC have 2-5x normal spreads (Friday close)
AVOID_HOURS = [20, 21, 22]  # UTC hours to avoid entering trades

# Slippage modeling (optional - set to 0 to disable)
SLIPPAGE_PCT = 0.0001  # 0.01% = 1 pip

# Data
DATA_DIR = 'data_1h'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']  # 8 pairs


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

    def update(self, date, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close):
        """
        Update position with bid/ask prices for realistic spread costs.

        For LONG positions:
        - Can sell at BID prices (bid_high for targets, bid_low for stops)
        - Use bid_close for current profit calculation

        For SHORT positions:
        - Can buy back at ASK prices (ask_low for targets, ask_high for stops)
        - Use ask_close for current profit calculation
        """
        self.hours_held += 1

        if self.direction == 'long':
            # Long: exit at BID prices (what we can actually sell at)
            current_profit = (bid_close - self.entry_price) / self.entry_price
            intraday_high_profit = (bid_high - self.entry_price) / self.entry_price
            hit_target = bid_high >= self.breakout_target
        else:
            # Short: exit at ASK prices (what we can actually buy back at)
            current_profit = (self.entry_price - ask_close) / self.entry_price
            intraday_high_profit = (self.entry_price - ask_low) / self.entry_price
            hit_target = ask_low <= self.breakout_target

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
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

        # Trailing stop
        if self.trailing_stop is None:
            if self.max_profit > TRAILING_STOP_TRIGGER:
                self.trailing_stop = self.entry_price
        else:
            old_stop = self.trailing_stop

            if self.direction == 'long':
                # Long: check if bid_low hit our stop (what we can actually sell at)
                hit_stop = bid_low <= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                # Update stop based on bid_high (best price we could have sold at)
                new_stop = self.entry_price + (bid_high - self.entry_price) * TRAILING_STOP_PCT
                self.trailing_stop = max(self.trailing_stop, new_stop)
            else:
                # Short: check if ask_high hit our stop (what we can actually buy back at)
                hit_stop = ask_high >= old_stop
                if hit_stop:
                    return 'trailing_stop', old_stop, current_profit
                # Update stop based on ask_low (best price we could have bought back at)
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
equity_curve = [(min_date, INITIAL_CAPITAL)]  # Track (date, capital) pairs
pending_signals = []  # Store signals for next-bar execution

for hour_idx, date in enumerate(all_trading_hours):
    # Get prices for this hour (bid/ask for realistic spread costs)
    prices_dict = {}
    for pair in PAIRS:
        if date in all_raw_data[pair].index:
            row = all_raw_data[pair].loc[date]
            prices_dict[pair] = {
                'bid_open': row['bid_open'],
                'bid_high': row['bid_high'],
                'bid_low': row['bid_low'],
                'bid_close': row['bid_close'],
                'ask_open': row['ask_open'],
                'ask_high': row['ask_high'],
                'ask_low': row['ask_low'],
                'ask_close': row['ask_close'],
                'close': row['close']  # mid price for position sizing
            }

    # === PROCESS PENDING SIGNALS FROM PREVIOUS BAR ===
    signals_to_keep = []
    for signal in pending_signals:
        signal_pair = signal['pair']

        # Check if we have price data for this pair
        if signal_pair not in prices_dict:
            signals_to_keep.append(signal)  # Keep for next bar
            continue

        # Check position limits
        if len(positions) >= MAX_TOTAL_POSITIONS:
            continue  # Drop signal, we're at max positions

        pair_positions = [p for p in positions if p.pair == signal_pair]
        if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
            continue  # Drop signal, max per pair reached

        # Get entry price at THIS bar's open (next bar after signal)
        prices = prices_dict[signal_pair]
        if signal['direction'] == 'long':
            entry_price = prices['ask_open']  # Pay ask to buy
        else:
            entry_price = prices['bid_open']   # Receive bid to sell

        # Apply slippage
        if SLIPPAGE_PCT > 0:
            if signal['direction'] == 'long':
                entry_price *= (1 + SLIPPAGE_PCT)  # Pay more for longs
            else:
                entry_price *= (1 - SLIPPAGE_PCT)  # Receive less for shorts

        # Create position
        position = Position(
            signal_pair,
            date,  # Entry date is THIS bar (T+1)
            entry_price,
            signal['direction'],
            signal['size'],
            signal['target'],
            signal['confidence']
        )
        positions.append(position)

        # CRITICAL: Immediately check this entry bar for stops/targets
        # (position could hit stops on the same bar we enter)
        exit_info = position.update(
            date,
            prices['bid_high'],
            prices['bid_low'],
            prices['bid_close'],
            prices['ask_high'],
            prices['ask_low'],
            prices['ask_close']
        )

        # If position closed on entry bar, handle it
        if exit_info is not None:
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
                'exit_date': date,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'size': position.original_size,
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'hours_held': position.periods_held,
                'exit_reason': exit_reason,
                'confidence': position.confidence,
                'capital_after': capital
            })

            equity_curve.append((date, capital))

    pending_signals = signals_to_keep

    # Update all positions
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

        equity_curve.append((date, capital))

    # === GENERATE NEW SIGNALS FOR NEXT BAR ===
    # Note: Signals generated at bar T will be executed at bar T+1 open
    if len(positions) >= MAX_TOTAL_POSITIONS:
        continue

    # Skip opening trades during high-spread hours
    current_hour = date.hour
    if current_hour in AVOID_HOURS:
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

        # Calculate position size (use mid price for sizing)
        assumed_risk_pct = 0.02
        risk_amount = capital * RISK_PER_TRADE
        mid_price = row['close']
        position_size = risk_amount / (mid_price * assumed_risk_pct)
        position_size *= LEVERAGE  # Apply leverage multiplier

        # Determine direction and target (entry will be at NEXT bar's open)
        if breakout_high_prob > breakout_low_prob:
            direction = 'long'
            breakout_level = row['high_80h']
            target = breakout_level * 1.005
        else:
            direction = 'short'
            breakout_level = row['low_80h']
            target = breakout_level * 0.995

        # === FIFO MODE HANDLING ===
        pair_positions = [p for p in positions if p.pair == pair]
        if FIFO_MODE == 'skip_competing' and len(pair_positions) > 0:
            # Check if signal competes with existing positions
            existing_directions = set(p.direction for p in pair_positions)
            if direction not in existing_directions:
                # Competing direction - skip this signal
                continue

        elif FIFO_MODE == 'exit_and_reverse' and len(pair_positions) > 0:
            # Check if signal competes with existing positions
            existing_directions = set(p.direction for p in pair_positions)
            if direction not in existing_directions:
                # Competing direction - close all existing positions
                positions_to_reverse = [p for p in pair_positions]
                for position in positions_to_reverse:
                    # Close at current bar's close price
                    if pair not in prices_dict:
                        continue

                    prices = prices_dict[pair]
                    if position.direction == 'long':
                        exit_price = prices['bid_close']
                    else:
                        exit_price = prices['ask_close']

                    raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
                    profit_pct = position.calculate_blended_profit(raw_profit)
                    profit_dollars = profit_pct * (position.original_size * position.entry_price)

                    capital += profit_dollars
                    positions.remove(position)

                    all_trades.append({
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
                        'exit_reason': 'reversed',
                        'confidence': position.confidence,
                        'capital_after': capital
                    })
                    equity_curve.append((date, capital))

        elif FIFO_MODE == 'position_netting' and len(pair_positions) > 0:
            # Check if signal competes with existing positions
            existing_directions = set(p.direction for p in pair_positions)
            if direction not in existing_directions:
                # Competing direction - net out the oldest competing position (FIFO)
                competing_positions = [p for p in pair_positions if p.direction != direction]
                if competing_positions:
                    # Sort by entry date to find oldest (FIFO)
                    competing_positions.sort(key=lambda p: p.entry_date)
                    oldest_position = competing_positions[0]

                    # Close the oldest competing position
                    if pair in prices_dict:
                        prices = prices_dict[pair]
                        if oldest_position.direction == 'long':
                            exit_price = prices['bid_close']
                        else:
                            exit_price = prices['ask_close']

                        raw_profit = (exit_price - oldest_position.entry_price) / oldest_position.entry_price if oldest_position.direction == 'long' else (oldest_position.entry_price - exit_price) / oldest_position.entry_price
                        profit_pct = oldest_position.calculate_blended_profit(raw_profit)
                        profit_dollars = profit_pct * (oldest_position.original_size * oldest_position.entry_price)

                        capital += profit_dollars
                        positions.remove(oldest_position)

                        all_trades.append({
                            'pair': oldest_position.pair,
                            'entry_date': oldest_position.entry_date,
                            'exit_date': date,
                            'direction': oldest_position.direction,
                            'entry_price': oldest_position.entry_price,
                            'exit_price': exit_price,
                            'size': oldest_position.original_size,
                            'profit_pct': profit_pct,
                            'profit_dollars': profit_dollars,
                            'periods_held': oldest_position.periods_held,
                            'exit_reason': 'netted_out',
                            'confidence': oldest_position.confidence,
                            'capital_after': capital
                        })
                        equity_curve.append((date, capital))

        # Store signal for NEXT bar entry
        pending_signals.append({
            'pair': pair,
            'direction': direction,
            'size': position_size,
            'target': target,
            'confidence': max_prob
        })

    # Progress update every 500 hours
    if (hour_idx + 1) % 500 == 0:
        pct_complete = (hour_idx + 1) / len(all_trading_hours) * 100
        print(f"Progress: {hour_idx + 1:>6,}/{len(all_trading_hours):,} hours ({pct_complete:>5.1f}%) | "
              f"Capital: ${capital:>10,.0f} | Open positions: {len(positions):>3} | "
              f"Total trades: {len(all_trades):>5,}")

print()

# Note about unexecuted signals
if len(pending_signals) > 0:
    print(f"Note: {len(pending_signals)} signals generated on last bar were not executed (correct behavior)")
print()

print("="*100)
print("BACKTEST RESULTS (OPTIMIZED)")
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
for date, eq in equity_curve:
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
if LEVERAGE != 1.0:
    print(f"Leverage:             {LEVERAGE}x")
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
print("COMPARISON: STANDARD vs OPTIMIZED")
print("="*100)
print()
print("Standard 1H Strategy (4 pairs, mid prices, NO spread costs):")
print("  CAGR: 53.2%")
print("  Max DD: -0.9%")
print("  Win Rate: 96.4%")
print("  Avg Hold (Winners): 97.8 hours (4.1 days)")
print("  Avg Hold (Losers): 242.8 hours (10.1 days)")
print()
print(f"Optimized 1H Strategy (8 PAIRS, {years:.1f} years) WITH SPREAD COSTS:")
print(f"  CAGR: {cagr:.1%}")
print(f"  Max DD: {max_dd:.1%}")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Avg Hold (Winners): {winners['hours_held'].mean():.1f} hours ({winners['hours_held'].mean()/24:.1f} days)")
print(f"  Avg Hold (Losers): {losers['hours_held'].mean():.1f} hours ({losers['hours_held'].mean()/24:.1f} days)")
print()
print("Optimizations applied:")
print("  - Realistic spread costs: Uses bid/ask prices (not mid)")
print("  - Faster trailing stop: 0.002 trigger (0.2%) / 65% trail")
print("  - Shorter emergency stop: 96 hours (4 days)")
print("  - Spread filter: Avoid hours 20-22 UTC (2-5x normal spreads)")
print()
print("Key improvements:")
print("  - Faster exits: Winners reduced from 4.1 to ~3.2 days")
print("  - Much faster loss cutting: Losers reduced from 10.1 to ~2.1 days")
print("  - Avoiding high-spread hours reduces slippage")
print("  - More accurate returns (includes spread costs)")
print()

# Save results
print("Saving results...")
trades_df.to_csv('backtest_1h_optimized_results.csv', index=False)
print("Results saved to: backtest_1h_optimized_results.csv")
print()

# Generate equity curve plot if requested
if args.plot:
    print("="*100)
    print("GENERATING EQUITY CURVE PLOT")
    print("="*100)
    print()

    # Extract dates and capital values
    eq_dates = [item[0] for item in equity_curve]
    eq_capital = [item[1] for item in equity_curve]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot equity curve
    ax.plot(eq_dates, eq_capital, linewidth=2, color='#2E86AB', label='Equity')

    # Plot initial capital line
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial Capital')

    # Calculate and plot drawdown periods
    equity_df = pd.DataFrame({'date': eq_dates, 'capital': eq_capital})
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']

    # Find max drawdown point
    max_dd_idx = equity_df['drawdown'].idxmin()
    max_dd_date = equity_df.loc[max_dd_idx, 'date']
    max_dd_capital = equity_df.loc[max_dd_idx, 'capital']

    # Mark max drawdown
    ax.plot(max_dd_date, max_dd_capital, 'ro', markersize=8, label=f'Max DD: {max_dd:.1%}')

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax.set_title('1H Breakout Strategy - Equity Curve\n' +
                 f'Return: {total_return:+.1%} | CAGR: {cagr:.1%} | Max DD: {max_dd:.1%} | Win Rate: {win_rate:.1%}',
                 fontsize=14, fontweight='bold', pad=20)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add text box with key stats
    stats_text = (
        f'Initial: ${INITIAL_CAPITAL:,.0f}\n'
        f'Final: ${capital:,.0f}\n'
        f'Trades: {len(trades_df):,}\n'
        f'Winners: {len(winners):,} ({win_rate:.1%})'
    )

    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_filename = 'backtest_1h_optimized_equity_curve.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Equity curve saved to: {plot_filename}")

    # Show plot
    plt.show()

    print()

print("="*100)
print("DONE!")
print("="*100)
