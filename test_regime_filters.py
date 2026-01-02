"""
TEST REGIME-BASED FILTERS
==========================
Identify market regimes that correlate with losses and test filtering strategies.

Key insight from loser analysis:
- 2021-2022 had exceptionally high loss rates (32%, 31.2%)
- Worst drawdown (Sept 2022) had HIGH confidence trades failing
- Emergency stops are 34.6% of losses, mostly high confidence

This suggests market regime is the issue, not trade quality.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("REGIME DETECTION & FILTERING")
print("="*100)
print()

# Load data
print("Loading data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

DATA_DIR = 'data'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

all_raw_data = {}
for pair in PAIRS:
    file_path = f'{DATA_DIR}/{pair}_1day_with_spreads.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    all_raw_data[pair] = df

print("Data loaded")
print()

# Fixed parameters (optimized)
INITIAL_CAPITAL = 500
BASE_RISK_PER_TRADE = 0.007
MIN_CONFIDENCE = 0.65
MAX_TOTAL_POSITIONS = 90
MAX_POSITIONS_PER_PAIR = 12
EMERGENCY_STOP_DAYS = 15
EMERGENCY_STOP_LOSS_PCT = -0.04
TRAILING_STOP_TRIGGER = 0.005
TRAILING_STOP_PCT = 0.60
LADDER_LEVELS = [0.008, 0.015]
LADDER_SCALE_PCT = 0.40


def calculate_regime_indicators(df, window=20):
    """Calculate regime indicators for a price series"""
    # ATR (volatility)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volatility regime: normalized ATR
    df['volatility'] = df['atr_pct'].rolling(window).mean()

    # Trend strength: ADX-like indicator
    df['price_change'] = df['close'].pct_change()
    df['abs_price_change'] = abs(df['price_change'])
    df['trend_strength'] = df['abs_price_change'].rolling(window).mean()

    # Choppiness: ratio of actual range to potential range
    df['close_change'] = abs(df['close'] - df['close'].shift(window))
    df['sum_ranges'] = df['tr'].rolling(window).sum()
    df['choppiness'] = df['close_change'] / df['sum_ranges']

    # Reversal frequency: direction changes
    df['direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['direction_changes'] = (df['direction'] != df['direction'].shift(1)).astype(int)
    df['reversal_freq'] = df['direction_changes'].rolling(window).sum() / window

    return df


# Calculate regime indicators for all pairs
print("Calculating regime indicators...")
for pair in PAIRS:
    all_raw_data[pair] = calculate_regime_indicators(all_raw_data[pair])

print("Regime indicators calculated")
print()


def get_regime_state(date, regime_metric='volatility', threshold_high=None, threshold_low=None):
    """Get regime state on a given date (average across all pairs)"""
    values = []
    for pair in PAIRS:
        if date in all_raw_data[pair].index:
            val = all_raw_data[pair].loc[date, regime_metric]
            if not np.isnan(val):
                values.append(val)

    if len(values) == 0:
        return 'unknown'

    avg_value = np.mean(values)

    if threshold_high is not None and threshold_low is not None:
        if avg_value >= threshold_high:
            return 'high'
        elif avg_value <= threshold_low:
            return 'low'
        else:
            return 'medium'

    return avg_value


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


def run_backtest_with_regime_filter(regime_filter=None, size_reduction=1.0):
    """
    Run backtest with optional regime filtering

    regime_filter: dict with 'metric', 'threshold_high', 'threshold_low'
                   If 'high' regime, apply size_reduction or skip (if size_reduction=0)
    size_reduction: Multiplier for position size in bad regime (0=skip, 0.5=half size, 1.0=full size)
    """
    capital = INITIAL_CAPITAL
    carried_positions = []
    trades = []
    equity_curve = [INITIAL_CAPITAL]
    skipped_trades = 0

    for quarter_name, quarter_preds in sorted(all_predictions.items()):
        # Get trading days
        all_trading_days = set()
        prediction_dates = set()

        for pair_df in quarter_preds.values():
            prediction_dates.update(pair_df.index)

        for pair, pair_df in all_raw_data.items():
            if len(prediction_dates) > 0:
                min_date = min(prediction_dates)
                max_date = max(prediction_dates)
                trading_days = pair_df[(pair_df.index >= min_date) & (pair_df.index <= max_date)].index
                all_trading_days.update(trading_days)

        all_trading_days = sorted(list(all_trading_days))

        for date in all_trading_days:
            # Get prices
            prices_dict = {}
            for pair, pair_df in all_raw_data.items():
                if date in pair_df.index:
                    row = pair_df.loc[date]
                    prices_dict[pair] = {'high': row['high'], 'low': row['low'], 'close': row['close']}

            # Update positions
            positions_to_close = []
            for position in carried_positions:
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
                carried_positions.remove(position)
                equity_curve.append(capital)

                trades.append({
                    'exit_reason': exit_reason,
                    'profit_pct': profit_pct,
                    'confidence': position.confidence
                })

            # Open new positions
            if date not in prediction_dates:
                continue

            if len(carried_positions) >= MAX_TOTAL_POSITIONS:
                continue

            # Check regime state
            if regime_filter:
                regime_state = get_regime_state(
                    date,
                    regime_filter['metric'],
                    regime_filter.get('threshold_high'),
                    regime_filter.get('threshold_low')
                )

                if regime_state == 'high' and size_reduction == 0:
                    skipped_trades += 1
                    continue

            for pair, pair_df in quarter_preds.items():
                if date not in pair_df.index:
                    continue

                pair_positions = [p for p in carried_positions if p.pair == pair]
                if len(pair_positions) >= MAX_POSITIONS_PER_PAIR:
                    continue

                row = pair_df.loc[date]

                breakout_high_prob = row['breakout_high_prob']
                breakout_low_prob = row['breakout_low_prob']
                max_prob = max(breakout_high_prob, breakout_low_prob)

                if max_prob <= MIN_CONFIDENCE:
                    continue

                # Apply regime-based sizing
                size_multiplier = 1.0
                if regime_filter and regime_state == 'high':
                    size_multiplier = size_reduction

                assumed_risk_pct = 0.02
                risk_amount = capital * BASE_RISK_PER_TRADE * size_multiplier
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
                carried_positions.append(position)

    # Calculate max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for equity in equity_curve:
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    return capital, len(trades), trades, max_dd, skipped_trades


# ANALYSIS 1: Establish regime thresholds by analyzing historical data
print("="*100)
print("STEP 1: ANALYZING HISTORICAL REGIME CHARACTERISTICS")
print("="*100)
print()

# Calculate regime metrics for each trading day
regime_data = []
for quarter_name, quarter_preds in sorted(all_predictions.items()):
    prediction_dates = set()
    for pair_df in quarter_preds.values():
        prediction_dates.update(pair_df.index)

    for date in sorted(prediction_dates):
        # Get regime metrics (average across all pairs)
        volatility = get_regime_state(date, 'volatility')
        trend_strength = get_regime_state(date, 'trend_strength')
        choppiness = get_regime_state(date, 'choppiness')
        reversal_freq = get_regime_state(date, 'reversal_freq')

        if volatility != 'unknown':
            regime_data.append({
                'date': date,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'choppiness': choppiness,
                'reversal_freq': reversal_freq
            })

regime_df = pd.DataFrame(regime_data)
regime_df['year'] = regime_df['date'].dt.year

print("Regime metrics distribution:")
print("-" * 60)
for metric in ['volatility', 'trend_strength', 'choppiness', 'reversal_freq']:
    p25 = regime_df[metric].quantile(0.25)
    p50 = regime_df[metric].quantile(0.50)
    p75 = regime_df[metric].quantile(0.75)
    p90 = regime_df[metric].quantile(0.90)
    print(f"{metric:20s} | 25%: {p25:.4f} | 50%: {p50:.4f} | 75%: {p75:.4f} | 90%: {p90:.4f}")

print()

# Check regime characteristics during bad years (2021-2022)
bad_years = regime_df[regime_df['year'].isin([2021, 2022])]
good_years = regime_df[~regime_df['year'].isin([2021, 2022])]

print("Regime comparison - Bad years (2021-2022) vs Good years:")
print("-" * 80)
print(f"{'Metric':20s} | {'Bad Years Avg':15s} | {'Good Years Avg':15s} | {'Difference':10s}")
print("-" * 80)
for metric in ['volatility', 'trend_strength', 'choppiness', 'reversal_freq']:
    bad_avg = bad_years[metric].mean()
    good_avg = good_years[metric].mean()
    diff = bad_avg - good_avg
    diff_pct = (bad_avg / good_avg - 1) * 100
    print(f"{metric:20s} | {bad_avg:>15.4f} | {good_avg:>15.4f} | {diff:>+9.4f} ({diff_pct:>+5.1f}%)")

print()

# TESTING: Test different regime filters
print("="*100)
print("STEP 2: TESTING REGIME FILTERS")
print("="*100)
print()

# Baseline
print("Running baseline (no filter)...")
baseline_cap, baseline_trades, baseline_trade_list, baseline_dd, _ = run_backtest_with_regime_filter()
baseline_return = (baseline_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"Baseline: ${baseline_cap:>10,.0f} ({baseline_return:>7.1%}) | DD: {baseline_dd:>6.1%} | {baseline_trades:>5,} trades")
print()

# Test 1: High volatility filter
print("TEST 1: HIGH VOLATILITY FILTERS")
print("-" * 80)
print()

vol_p75 = regime_df['volatility'].quantile(0.75)
vol_p80 = regime_df['volatility'].quantile(0.80)
vol_p85 = regime_df['volatility'].quantile(0.85)
vol_p90 = regime_df['volatility'].quantile(0.90)

volatility_tests = [
    ({'metric': 'volatility', 'threshold_high': vol_p75, 'threshold_low': 0}, 0.0, "Skip top 25% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p75, 'threshold_low': 0}, 0.5, "Half size top 25% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p80, 'threshold_low': 0}, 0.0, "Skip top 20% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p80, 'threshold_low': 0}, 0.5, "Half size top 20% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p85, 'threshold_low': 0}, 0.0, "Skip top 15% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p90, 'threshold_low': 0}, 0.0, "Skip top 10% volatility"),
    ({'metric': 'volatility', 'threshold_high': vol_p90, 'threshold_low': 0}, 0.5, "Half size top 10% volatility"),
]

print(f"{'Filter':35s} | {'Capital':>12s} | {'Return':>8s} | {'DD':>8s} | {'Trades':>7s} | {'Skipped':>8s} | {'vs Base':>10s}")
print("-" * 100)

vol_results = []
for regime_filter, size_reduction, label in volatility_tests:
    cap, trades, trade_list, max_dd, skipped = run_backtest_with_regime_filter(regime_filter, size_reduction)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    vol_results.append({
        'label': label,
        'capital': cap,
        'return': total_return,
        'dd': max_dd,
        'trades': trades,
        'skipped': skipped,
        'improvement': improvement,
        'dd_improvement': dd_improvement
    })

    print(f"{label:35s} | ${cap:>11,.0f} | {total_return:>7.1%} | {max_dd:>7.1%} | {trades:>6,} | {skipped:>7,} | {improvement:>+9,.0f} ({dd_improvement:>+5.1%})")

print()

# Test 2: High choppiness filter
print("TEST 2: HIGH CHOPPINESS FILTERS (choppy/ranging markets)")
print("-" * 80)
print()

chop_p75 = regime_df['choppiness'].quantile(0.75)
chop_p80 = regime_df['choppiness'].quantile(0.80)
chop_p85 = regime_df['choppiness'].quantile(0.85)
chop_p90 = regime_df['choppiness'].quantile(0.90)

choppiness_tests = [
    ({'metric': 'choppiness', 'threshold_high': chop_p75, 'threshold_low': 0}, 0.0, "Skip top 25% choppiness"),
    ({'metric': 'choppiness', 'threshold_high': chop_p75, 'threshold_low': 0}, 0.5, "Half size top 25% choppiness"),
    ({'metric': 'choppiness', 'threshold_high': chop_p80, 'threshold_low': 0}, 0.0, "Skip top 20% choppiness"),
    ({'metric': 'choppiness', 'threshold_high': chop_p85, 'threshold_low': 0}, 0.0, "Skip top 15% choppiness"),
    ({'metric': 'choppiness', 'threshold_high': chop_p90, 'threshold_low': 0}, 0.0, "Skip top 10% choppiness"),
    ({'metric': 'choppiness', 'threshold_high': chop_p90, 'threshold_low': 0}, 0.5, "Half size top 10% choppiness"),
]

print(f"{'Filter':35s} | {'Capital':>12s} | {'Return':>8s} | {'DD':>8s} | {'Trades':>7s} | {'Skipped':>8s} | {'vs Base':>10s}")
print("-" * 100)

chop_results = []
for regime_filter, size_reduction, label in choppiness_tests:
    cap, trades, trade_list, max_dd, skipped = run_backtest_with_regime_filter(regime_filter, size_reduction)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    chop_results.append({
        'label': label,
        'capital': cap,
        'return': total_return,
        'dd': max_dd,
        'trades': trades,
        'skipped': skipped,
        'improvement': improvement,
        'dd_improvement': dd_improvement
    })

    print(f"{label:35s} | ${cap:>11,.0f} | {total_return:>7.1%} | {max_dd:>7.1%} | {trades:>6,} | {skipped:>7,} | {improvement:>+9,.0f} ({dd_improvement:>+5.1%})")

print()

# Test 3: High reversal frequency filter
print("TEST 3: HIGH REVERSAL FREQUENCY FILTERS (whipsaw markets)")
print("-" * 80)
print()

rev_p75 = regime_df['reversal_freq'].quantile(0.75)
rev_p80 = regime_df['reversal_freq'].quantile(0.80)
rev_p85 = regime_df['reversal_freq'].quantile(0.85)
rev_p90 = regime_df['reversal_freq'].quantile(0.90)

reversal_tests = [
    ({'metric': 'reversal_freq', 'threshold_high': rev_p75, 'threshold_low': 0}, 0.0, "Skip top 25% reversal freq"),
    ({'metric': 'reversal_freq', 'threshold_high': rev_p75, 'threshold_low': 0}, 0.5, "Half size top 25% reversal freq"),
    ({'metric': 'reversal_freq', 'threshold_high': rev_p80, 'threshold_low': 0}, 0.0, "Skip top 20% reversal freq"),
    ({'metric': 'reversal_freq', 'threshold_high': rev_p85, 'threshold_low': 0}, 0.0, "Skip top 15% reversal freq"),
    ({'metric': 'reversal_freq', 'threshold_high': rev_p90, 'threshold_low': 0}, 0.0, "Skip top 10% reversal freq"),
    ({'metric': 'reversal_freq', 'threshold_high': rev_p90, 'threshold_low': 0}, 0.5, "Half size top 10% reversal freq"),
]

print(f"{'Filter':35s} | {'Capital':>12s} | {'Return':>8s} | {'DD':>8s} | {'Trades':>7s} | {'Skipped':>8s} | {'vs Base':>10s}")
print("-" * 100)

rev_results = []
for regime_filter, size_reduction, label in reversal_tests:
    cap, trades, trade_list, max_dd, skipped = run_backtest_with_regime_filter(regime_filter, size_reduction)
    total_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL
    improvement = cap - baseline_cap
    dd_improvement = max_dd - baseline_dd

    rev_results.append({
        'label': label,
        'capital': cap,
        'return': total_return,
        'dd': max_dd,
        'trades': trades,
        'skipped': skipped,
        'improvement': improvement,
        'dd_improvement': dd_improvement
    })

    print(f"{label:35s} | ${cap:>11,.0f} | {total_return:>7.1%} | {max_dd:>7.1%} | {trades:>6,} | {skipped:>7,} | {improvement:>+9,.0f} ({dd_improvement:>+5.1%})")

print()

# SUMMARY
print("="*100)
print("SUMMARY - BEST REGIME FILTERS")
print("="*100)
print()

# Combine all results
all_results = vol_results + chop_results + rev_results

# Sort by DD improvement (we want to reduce DD)
sorted_by_dd = sorted(all_results, key=lambda x: x['dd'])

print("RANKED BY DRAWDOWN (best DD first):")
print("-" * 110)
print(f"{'Rank':>4s} | {'Filter':35s} | {'Capital':>12s} | {'DD':>8s} | {'DD Change':>10s} | {'Return Change':>13s}")
print("-" * 110)

for i, result in enumerate(sorted_by_dd[:10], 1):
    print(f"{i:>4d} | {result['label']:35s} | ${result['capital']:>11,.0f} | {result['dd']:>7.1%} | {result['dd_improvement']:>+9.1%} | {result['improvement']:>+12,.0f}")

print()

# Find best risk/reward trade-off
# We want: improved DD without losing too much return
print("BEST RISK/REWARD TRADE-OFFS (DD improvement vs capital loss):")
print("-" * 110)

# Filter for results that improve DD
dd_improved = [r for r in all_results if r['dd_improvement'] < -0.01]  # At least 1% DD improvement

if len(dd_improved) > 0:
    # Sort by capital (best return among DD improvers)
    sorted_dd_improved = sorted(dd_improved, key=lambda x: x['capital'], reverse=True)

    print(f"{'Rank':>4s} | {'Filter':35s} | {'Capital':>12s} | {'DD':>8s} | {'DD Change':>10s} | {'Return Change':>13s}")
    print("-" * 110)

    for i, result in enumerate(sorted_dd_improved[:10], 1):
        print(f"{i:>4d} | {result['label']:35s} | ${result['capital']:>11,.0f} | {result['dd']:>7.1%} | {result['dd_improvement']:>+9.1%} | {result['improvement']:>+12,.0f}")
else:
    print("NO FILTERS IMPROVED DRAWDOWN BY MORE THAN 1%")

print()

# Best overall (Calmar ratio approximation)
best_calmar = max(all_results, key=lambda x: (x['capital'] - INITIAL_CAPITAL) / abs(x['dd']) if x['dd'] < 0 else 0)
print(f"BEST CALMAR RATIO (return/drawdown): {best_calmar['label']}")
print(f"  Final capital: ${best_calmar['capital']:,.0f}")
print(f"  Max drawdown: {best_calmar['dd']:.1%}")
print(f"  Return: {best_calmar['return']:.1%}")
print(f"  Calmar: {best_calmar['return'] / abs(best_calmar['dd']):.2f}")
print()

print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

if best_calmar['dd_improvement'] < -0.05:  # At least 5% DD improvement
    print(f"STRONG RECOMMENDATION: {best_calmar['label']}")
    print(f"  Reduces DD by {abs(best_calmar['dd_improvement']):.1%}")
    if best_calmar['improvement'] < 0:
        print(f"  Cost: ${abs(best_calmar['improvement']):,.0f} in returns ({best_calmar['improvement'] / baseline_cap:.1%})")
    else:
        print(f"  Bonus: ALSO improves returns by ${best_calmar['improvement']:,.0f} ({best_calmar['improvement'] / baseline_cap:+.1%})")
elif best_calmar['dd_improvement'] < -0.02:  # At least 2% DD improvement
    print(f"CONSIDER: {best_calmar['label']}")
    print(f"  Reduces DD by {abs(best_calmar['dd_improvement']):.1%}")
    if best_calmar['improvement'] < 0:
        print(f"  Cost: ${abs(best_calmar['improvement']):,.0f} in returns")
else:
    print("NO SIGNIFICANT DD IMPROVEMENT FOUND FROM REGIME FILTERS")
    print("Alternative strategies:")
    print("  1. Reduce max concurrent positions (90 -> 60)")
    print("  2. Reduce position size globally (0.7% -> 0.5%)")
    print("  3. Accept current DD as cost of high returns")

print()
print("="*100)
