"""
COMPARISON: ORIGINAL VS VALIDATED PREDICTIONS
==============================================
Backtests both prediction sets to quantify the impact of potential contamination.

If there's a significant difference, it indicates data leakage was inflating results.
If results are similar, the original approach was robust.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPARISON: ORIGINAL VS VALIDATED PREDICTIONS")
print("="*100)
print()

# Strategy Parameters (same as backtest_15m_optimized.py)
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
FIFO_MODE = 'skip_competing'
AVOID_HOURS = [20, 21, 22]
SLIPPAGE_PCT = 0.0001

DATA_DIR = 'data_15m'
PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']


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


def run_backtest(predictions, raw_data, name):
    """Run backtest on given predictions"""
    print(f"\nRunning backtest: {name}")
    print("-" * 80)

    # Get all trading periods
    all_trading_periods = set()
    for pair_df in predictions.values():
        all_trading_periods.update(pair_df.index)
    all_trading_periods = sorted(list(all_trading_periods))

    min_date = min(all_trading_periods)
    max_date = max(all_trading_periods)

    print(f"Period: {min_date} to {max_date}")
    print(f"Total periods: {len(all_trading_periods):,}")

    capital = INITIAL_CAPITAL
    positions = []
    trades = []
    pending_signals = []

    for period_idx, date in enumerate(all_trading_periods):
        # Get prices
        prices_dict = {}
        for pair in PAIRS:
            if date in raw_data[pair].index:
                row = raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_open': row['bid_open'],
                    'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'],
                    'bid_close': row['bid_close'],
                    'ask_open': row['ask_open'],
                    'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'],
                    'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Process pending signals
        signals_to_keep = []
        for signal in pending_signals:
            signal_pair = signal['pair']
            if signal_pair not in prices_dict:
                signals_to_keep.append(signal)
                continue

            if len(positions) >= MAX_TOTAL_POSITIONS:
                continue

            pair_positions = [p for p in positions if p.pair == signal_pair]
            if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                continue

            prices = prices_dict[signal_pair]
            if signal['direction'] == 'long':
                entry_price = prices['ask_open']
            else:
                entry_price = prices['bid_open']

            if SLIPPAGE_PCT > 0:
                if signal['direction'] == 'long':
                    entry_price *= (1 + SLIPPAGE_PCT)
                else:
                    entry_price *= (1 - SLIPPAGE_PCT)

            position = Position(
                signal_pair, date, entry_price, signal['direction'],
                signal['size'], signal['target'], signal['confidence']
            )
            positions.append(position)

            exit_info = position.update(
                date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                prices['ask_high'], prices['ask_low'], prices['ask_close']
            )

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
                trades.append({
                    'pair': position.pair,
                    'entry_date': position.entry_date,
                    'exit_date': date,
                    'direction': position.direction,
                    'profit_pct': profit_pct,
                    'profit_dollars': profit_dollars,
                    'periods_held': position.periods_held,
                    'exit_reason': exit_reason
                })

        pending_signals = signals_to_keep

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            prices = prices_dict[position.pair]
            exit_info = position.update(
                date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                prices['ask_high'], prices['ask_low'], prices['ask_close']
            )
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
                'pair': position.pair,
                'entry_date': position.entry_date,
                'exit_date': date,
                'direction': position.direction,
                'profit_pct': profit_pct,
                'profit_dollars': profit_dollars,
                'periods_held': position.periods_held,
                'exit_reason': exit_reason
            })

        # Generate new signals
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

            if breakout_high_prob > breakout_low_prob:
                direction = 'long'
                breakout_level = row['high_80p']
                target = breakout_level * 1.005
            else:
                direction = 'short'
                breakout_level = row['low_80p']
                target = breakout_level * 0.995

            if FIFO_MODE == 'skip_competing' and len(pair_positions) > 0:
                existing_directions = set(p.direction for p in pair_positions)
                if direction not in existing_directions:
                    continue

            assumed_risk_pct = 0.02
            risk_amount = capital * RISK_PER_TRADE
            mid_price = row['close']
            position_size = risk_amount / (mid_price * assumed_risk_pct)

            pending_signals.append({
                'pair': pair,
                'direction': direction,
                'size': position_size,
                'target': target,
                'confidence': max_prob
            })

    # Calculate results
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        print("NO TRADES EXECUTED!")
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    days = (max_date - min_date).days
    years = days / 365
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    winners = trades_df[trades_df['profit_pct'] > 0]
    losers = trades_df[trades_df['profit_pct'] <= 0]
    win_rate = len(winners) / len(trades_df)

    # Calculate max drawdown
    equity_values = [INITIAL_CAPITAL] + list(trades_df['profit_dollars'].cumsum() + INITIAL_CAPITAL)
    peak = INITIAL_CAPITAL
    max_dd = 0
    for val in equity_values[1:]:
        peak = max(peak, val)
        dd = (val - peak) / peak
        max_dd = min(max_dd, dd)

    results = {
        'name': name,
        'initial_capital': INITIAL_CAPITAL,
        'final_capital': capital,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'total_trades': len(trades_df),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'avg_winner': winners['profit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['profit_pct'].mean() if len(losers) > 0 else 0,
        'years': years
    }

    return results


# Load raw data
print("Loading raw data...")
all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_15m.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

# Load original predictions
print("Loading original predictions...")
with open('test_predictions_15m.pkl', 'rb') as f:
    original_predictions = pickle.load(f)

# Load validated predictions
print("Loading validated predictions...")
with open('test_predictions_15m_validated.pkl', 'rb') as f:
    validated_predictions = pickle.load(f)

print()

# Run backtests
results_original = run_backtest(original_predictions, all_raw_data, "ORIGINAL (potential contamination)")
results_validated = run_backtest(validated_predictions, all_raw_data, "VALIDATED (no contamination)")

# Compare results
print()
print("="*100)
print("COMPARISON RESULTS")
print("="*100)
print()

if results_original is None or results_validated is None:
    print("ERROR: One or both backtests failed to generate trades.")
    print("This may be due to insufficient data or mismatched date ranges.")
else:
    print(f"{'Metric':<25} {'Original':<20} {'Validated':<20} {'Difference':<15}")
    print("-" * 80)

    print(f"{'Final Capital':<25} ${results_original['final_capital']:>10,.0f}      ${results_validated['final_capital']:>10,.0f}")
    print(f"{'Total Return':<25} {results_original['total_return']:>10.1%}      {results_validated['total_return']:>10.1%}      {(results_original['total_return'] - results_validated['total_return']):>+10.1%}")
    print(f"{'CAGR':<25} {results_original['cagr']:>10.1%}      {results_validated['cagr']:>10.1%}      {(results_original['cagr'] - results_validated['cagr']):>+10.1%}")
    print(f"{'Max Drawdown':<25} {results_original['max_drawdown']:>10.1%}      {results_validated['max_drawdown']:>10.1%}      {(results_original['max_drawdown'] - results_validated['max_drawdown']):>+10.1%}")
    print(f"{'Win Rate':<25} {results_original['win_rate']:>10.1%}      {results_validated['win_rate']:>10.1%}      {(results_original['win_rate'] - results_validated['win_rate']):>+10.1%}")
    print(f"{'Total Trades':<25} {results_original['total_trades']:>10,}      {results_validated['total_trades']:>10,}")
    print(f"{'Avg Winner':<25} {results_original['avg_winner']:>10.2%}      {results_validated['avg_winner']:>10.2%}      {(results_original['avg_winner'] - results_validated['avg_winner']):>+10.2%}")
    print(f"{'Avg Loser':<25} {results_original['avg_loser']:>10.2%}      {results_validated['avg_loser']:>10.2%}      {(results_original['avg_loser'] - results_validated['avg_loser']):>+10.2%}")

    print()
    print("="*100)
    print("INTERPRETATION")
    print("="*100)
    print()

    cagr_diff = abs(results_original['cagr'] - results_validated['cagr'])

    if cagr_diff < 0.10:  # Less than 10% difference
        print("[PASS] RESULTS ARE SIMILAR (<10% CAGR difference)")
        print("  The original approach appears robust with minimal contamination impact.")
        print(f"  CAGR difference: {cagr_diff:.1%}")
    elif cagr_diff < 0.30:  # 10-30% difference
        print("[WARNING] MODERATE DIFFERENCE (10-30% CAGR difference)")
        print("  Some contamination may be present, but results are still meaningful.")
        print(f"  CAGR difference: {cagr_diff:.1%}")
        print("  Consider using the validated approach for production.")
    else:  # >30% difference
        print("[CRITICAL] SIGNIFICANT DIFFERENCE (>30% CAGR difference)")
        print("  Strong evidence of data contamination in original approach.")
        print(f"  CAGR difference: {cagr_diff:.1%}")
        print("  RECOMMENDATION: Use validated predictions for all future backtests.")

    print()
