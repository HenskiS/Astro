"""
MULTI-YEAR BACKTEST COMPARISON
===============================
Tests the same strategy across 2021-2025 to see if performance is consistent.

This answers: "Was 116% CAGR just a good season, or does it generalize?"
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

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

        if self.ladder_level < len(LADDER_LEVELS):
            if intraday_high_profit >= LADDER_LEVELS[self.ladder_level]:
                self.partial_exits.append((LADDER_LEVELS[self.ladder_level], LADDER_SCALE_PCT))
                self.size *= (1 - LADDER_SCALE_PCT)
                self.ladder_level += 1
                return None

        if self.periods_held >= EMERGENCY_STOP_PERIODS and current_profit < EMERGENCY_STOP_LOSS_PCT:
            exit_price = bid_close if self.direction == 'long' else ask_close
            return 'emergency_stop', exit_price, current_profit

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


def run_backtest(predictions, raw_data):
    """Run backtest - streamlined version"""
    all_trading_periods = set()
    for pair_df in predictions.values():
        all_trading_periods.update(pair_df.index)
    all_trading_periods = sorted(list(all_trading_periods))

    if len(all_trading_periods) == 0:
        return None

    min_date = min(all_trading_periods)
    max_date = max(all_trading_periods)
    capital = INITIAL_CAPITAL
    positions = []
    trades = []
    pending_signals = []

    for date in all_trading_periods:
        prices_dict = {}
        for pair in PAIRS:
            if date in raw_data[pair].index:
                row = raw_data[pair].loc[date]
                prices_dict[pair] = {
                    'bid_open': row['bid_open'], 'bid_high': row['bid_high'],
                    'bid_low': row['bid_low'], 'bid_close': row['bid_close'],
                    'ask_open': row['ask_open'], 'ask_high': row['ask_high'],
                    'ask_low': row['ask_low'], 'ask_close': row['ask_close'],
                    'close': row['close']
                }

        # Process pending signals
        signals_to_keep = []
        for signal in pending_signals:
            if signal['pair'] not in prices_dict or len(positions) >= MAX_TOTAL_POSITIONS:
                if signal['pair'] not in prices_dict:
                    signals_to_keep.append(signal)
                continue
            if len([p for p in positions if p.pair == signal['pair']]) >= MAX_POSITIONS_PER_PAIR:
                continue

            prices = prices_dict[signal['pair']]
            entry_price = prices['ask_open'] if signal['direction'] == 'long' else prices['bid_open']
            if SLIPPAGE_PCT > 0:
                entry_price *= (1 + SLIPPAGE_PCT) if signal['direction'] == 'long' else (1 - SLIPPAGE_PCT)

            position = Position(signal['pair'], date, entry_price, signal['direction'],
                              signal['size'], signal['target'], signal['confidence'])
            positions.append(position)

            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                exit_reason, exit_price, _ = exit_info
                raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
                profit_pct = position.calculate_blended_profit(raw_profit)
                profit_dollars = profit_pct * (position.original_size * position.entry_price)
                capital += profit_dollars
                positions.remove(position)
                trades.append({'profit_pct': profit_pct})

        pending_signals = signals_to_keep

        # Update positions
        positions_to_close = []
        for position in positions:
            if position.pair not in prices_dict:
                continue
            prices = prices_dict[position.pair]
            exit_info = position.update(date, prices['bid_high'], prices['bid_low'], prices['bid_close'],
                                       prices['ask_high'], prices['ask_low'], prices['ask_close'])
            if exit_info:
                positions_to_close.append((position, exit_info))

        for position, exit_info in positions_to_close:
            exit_reason, exit_price, _ = exit_info
            raw_profit = (exit_price - position.entry_price) / position.entry_price if position.direction == 'long' else (position.entry_price - exit_price) / position.entry_price
            profit_pct = position.calculate_blended_profit(raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)
            capital += profit_dollars
            positions.remove(position)
            trades.append({'profit_pct': profit_pct})

        # Generate signals
        if len(positions) >= MAX_TOTAL_POSITIONS or date.hour in AVOID_HOURS:
            continue

        for pair in PAIRS:
            if date not in predictions[pair].index:
                continue
            if len([p for p in positions if p.pair == pair]) >= MAX_POSITIONS_PER_PAIR:
                continue

            row = predictions[pair].loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
            if max_prob <= MIN_CONFIDENCE:
                continue

            if row['breakout_high_prob'] > row['breakout_low_prob']:
                direction = 'long'
                target = row['high_80p'] * 1.005
            else:
                direction = 'short'
                target = row['low_80p'] * 0.995

            pair_positions = [p for p in positions if p.pair == pair]
            if FIFO_MODE == 'skip_competing' and len(pair_positions) > 0:
                if direction not in set(p.direction for p in pair_positions):
                    continue

            risk_amount = capital * RISK_PER_TRADE
            position_size = risk_amount / (row['close'] * 0.02)

            pending_signals.append({
                'pair': pair, 'direction': direction, 'size': position_size,
                'target': target, 'confidence': max_prob
            })

    # Calculate results
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    days = (max_date - min_date).days
    years = max(days / 365, 0.01)
    cagr = (capital / INITIAL_CAPITAL) ** (1 / years) - 1

    winners = trades_df[trades_df['profit_pct'] > 0]
    win_rate = len(winners) / len(trades_df)

    return {
        'cagr': cagr,
        'win_rate': win_rate,
        'total_trades': len(trades_df),
        'final_capital': capital,
        'days': days
    }


print("="*100)
print("MULTI-YEAR BACKTEST COMPARISON")
print("="*100)
print()

# Load raw data
print("Loading raw data...")
all_raw_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{DATA_DIR}/{pair}_15m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print()

# Test each year
test_files = [
    ('2021 Test', 'test_predictions_15m_2021_test.pkl'),
    ('2022 Test', 'test_predictions_15m_2022_test.pkl'),
    ('2023 Test', 'test_predictions_15m_2023_test.pkl'),
    ('2024 Test', 'test_predictions_15m_2024_test.pkl'),
    ('2025 Test', 'test_predictions_15m_2025_test.pkl'),
]

results = []

print("Running backtests...")
print("="*100)
print()

for name, filename in test_files:
    print(f"{name}:")
    try:
        with open(filename, 'rb') as f:
            preds = pickle.load(f)

        r = run_backtest(preds, all_raw_data)
        if r:
            r['name'] = name
            results.append(r)
            print(f"  CAGR: {r['cagr']:>8.1%} | Win Rate: {r['win_rate']:>5.1%} | Trades: {r['total_trades']:>5,} | Days: {r['days']:>4}")
        else:
            print(f"  No trades executed")
    except FileNotFoundError:
        print(f"  File not found: {filename}")
    print()

# Summary
print("="*100)
print("SUMMARY: CAGR ACROSS YEARS")
print("="*100)
print()

if len(results) > 0:
    cagrs = [r['cagr'] for r in results]
    avg_cagr = np.mean(cagrs)
    std_cagr = np.std(cagrs)
    min_cagr = min(cagrs)
    max_cagr = max(cagrs)

    print(f"{'Year':<15} {'CAGR':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['name']:<15} {r['cagr']:>10.1%}")
    print("-" * 30)
    print(f"{'Average':<15} {avg_cagr:>10.1%}")
    print(f"{'Std Dev':<15} {std_cagr:>10.1%}")
    print(f"{'Min':<15} {min_cagr:>10.1%}")
    print(f"{'Max':<15} {max_cagr:>10.1%}")

    print()
    print("="*100)
    print("INTERPRETATION")
    print("="*100)
    print()

    if std_cagr < 0.30:
        print("[CONSISTENT] CAGR is relatively stable across years")
        print(f"  Average: {avg_cagr:.1%} +/- {std_cagr:.1%}")
        print("  The strategy appears to have consistent edge across market regimes")
    elif std_cagr < 0.60:
        print("[MODERATE VARIANCE] CAGR varies somewhat across years")
        print(f"  Average: {avg_cagr:.1%} +/- {std_cagr:.1%}")
        print("  Performance depends on market conditions but shows overall positive edge")
    else:
        print("[HIGH VARIANCE] CAGR is highly inconsistent across years")
        print(f"  Average: {avg_cagr:.1%} +/- {std_cagr:.1%}")
        print("  Strategy performance is heavily regime-dependent")

    print()
    print(f"Expected production CAGR: {avg_cagr:.1%} (based on {len(results)} test periods)")

print()
