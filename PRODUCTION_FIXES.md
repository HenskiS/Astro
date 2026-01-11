# Production Code Fixes - 2026-01-11

## Summary
Fixed critical differences between production and simulation code to ensure OANDA production trading matches backtested performance.

---

## ✅ Fix #1: Cross-Pair Position Sizing (CRITICAL)

**File**: [production_trader/strategies/strategy_15m.py:360-405](production_trader/strategies/strategy_15m.py#L360-L405)

**Problem**:
- For EURJPY with $100 capital and EURJPY=160, old code calculated: `units = 100 / 160 = 0.625 → 0 units` (integer cast)
- Even if 1 unit was passed, it would trade €1 (~$1.05), not $100!
- Old code didn't understand that OANDA units are ALWAYS in base currency (EUR for EURJPY)

**Solution**:
Added proper currency conversion logic that:
1. **USD-base pairs** (USDJPY, USDCAD, USDCHF): `units = capital` (1 unit = $1)
2. **USD-quote pairs** (EURUSD, GBPUSD, AUDUSD, NZDUSD): `units = capital / price`
3. **Cross pairs** (EURJPY): `units = capital / EURUSD_rate` (convert USD → EUR first)

**Example**: To risk $100 on EURJPY:
- Fetch EURUSD rate: 1.05
- Calculate: `units = 100 / 1.05 = 95 EUR units`
- OANDA trades 95 EUR against JPY, giving ~$100 exposure

**Impact**:
- EURJPY positions will now use correct capital allocation
- Matches backtest behavior for cross pairs

---

## ✅ Fix #2: P/L Verification Logging

**File**: [production_trader/execution/position_manager.py:448-517](production_trader/execution/position_manager.py#L448-L517)

**Problem**:
- Simulation uses percentage-based P/L: `profit = price_change_pct * position_value_usd`
- OANDA uses: `profit = price_diff * units * conversion_factor`
- No way to verify if these match in production!

**Solution**:
Added P/L verification that:
1. Captures account balance BEFORE closing position
2. Captures account balance AFTER closing position
3. Calculates actual P/L from balance change
4. Compares actual vs expected (simulation formula)
5. Logs WARNING if difference > 1%

**Output Example**:
```
INFO: Position closed: EURJPY LONG | P/L: 1.23% ($12.30)
DEBUG: P/L verified: EURJPY | Actual: $12.35, Expected: $12.30
```

Or if there's a discrepancy:
```
WARNING: P/L DISCREPANCY: EURJPY | Actual: $11.80, Expected: $12.30, Diff: -$0.50 (-4.1%)
```

**Impact**:
- You'll see if OANDA's P/L differs from simulation
- Helps identify currency conversion issues in cross pairs
- Uses actual P/L for capital tracking (not expected)

---

## ✅ Fix #3: Exit Price Documentation

**File**: [production_trader/execution/oanda_broker.py:428-453](production_trader/execution/oanda_broker.py#L428-L453)

**Problem**:
- [position_manager.py](production_trader/execution/position_manager.py#L467) calculates exact stop loss prices and passes them to broker
- Production OANDA **ignores** these and fills at market price
- This was undocumented, causing confusion

**Solution**:
Added comprehensive documentation explaining:
- MockBroker uses exact `exit_price` parameter
- OandaBroker **IGNORES** `exit_price` and fills at market
- Production will have slippage vs simulation (realistic!)
- Market execution includes spread widening, gaps, etc.

**Impact**:
- You now understand why production P/L might differ slightly from backtest
- This is NORMAL and EXPECTED in real trading
- Production minute-by-minute checks will likely IMPROVE performance by catching exits faster

---

## 🔍 Minute-by-Minute Position Checks (Already Working!)

**File**: [production_trader/main.py:262-273](production_trader/main.py#L262-L273)

**Status**: ✅ Already implemented correctly

**Behavior**:
- Production checks positions every minute (updates every 60 seconds)
- Backtest only checks at 15-minute bar closes
- Position manager correctly uses intraday extremes (bid_low, ask_high) for stop checks

**Impact**:
- **BENEFICIAL** - catches stops/targets faster than backtest
- Reduces drawdown by exiting bad trades sooner
- No lookahead bias (still using historical bar data)

---

## 📋 Testing Checklist

Run simulation again to verify fixes work correctly:

```bash
cd production_trader
python run_simulation.py \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --data-dir ../data_15m \
  --predictions "c:\Users\hensk\Documents\GitHub\Astro\test_predictions_15m_2024_test.pkl"
```

### Verify:

1. **EURJPY position sizes are reasonable**
   - Should see: "EURJPY: Cross pair conversion - $200.00 / EURUSD 1.05 = 190 EUR units"
   - NOT: "EURJPY: 0 units" or "EURJPY: 1 units"

2. **P/L calculations match**
   - Check for "P/L verified" messages in logs
   - No warnings about discrepancies (simulation uses MockBroker, should match perfectly)

3. **Trade count matches previous results**
   - Should still get ~450 trades for January 2024
   - Win rate should still be ~65.6%
   - Return should still be ~2.2%

4. **EURJPY trades execute**
   - Previously: EURJPY might have had 0-1 trades (wrong sizing)
   - Now: EURJPY should have ~81 trades (from your doc: 81 trades in January)

---

## 🚨 Known Production vs Simulation Differences

These differences are EXPECTED and REALISTIC:

### 1. Exit Prices
- **Simulation**: Exact stop prices using bar extremes
- **Production**: Market execution with potential slippage

### 2. Spread Handling
- **Simulation**: Uses historical spread from CSV data
- **Production**: Uses real-time spread (may be wider during news/low liquidity)

### 3. Execution Timing
- **Simulation**: Executes at bar open/close
- **Production**: Executes immediately when signal fires (within 5 seconds of 15-minute mark)

### 4. P/L Calculation
- **Simulation**: Percentage-based (matches backtest)
- **Production**: OANDA's internal calculation (includes real conversion rates)

**These are GOOD differences** - production reflects real market conditions!

---

## 🎯 Next Steps

### Before Live Trading:

1. **Run full simulation** (January - March 2024) to verify EURJPY trades correctly
2. **Test on OANDA practice account** with $100:
   ```bash
   cd production_trader
   # Edit config.yaml: set account_type: "practice"
   python main.py --config config.yaml
   ```
3. **Monitor first 10 trades** carefully:
   - Check position sizes are correct
   - Verify P/L calculations make sense
   - Watch for "P/L DISCREPANCY" warnings
4. **If EURJPY shows discrepancies > 5%**, investigate currency conversion

### Production Deployment:

Once practice account tests pass:
1. Change config.yaml: `account_type: "live"`
2. Start with small capital ($100-500)
3. Monitor for first week to ensure stability
4. Scale up gradually if performance matches expectations

---

## Files Modified

1. **[production_trader/strategies/strategy_15m.py](production_trader/strategies/strategy_15m.py#L360-L405)**
   - Fixed cross-pair position sizing (EURJPY)
   - Added detailed logging for each pair type

2. **[production_trader/simulation/mock_broker.py](production_trader/simulation/mock_broker.py#L251-L266)**
   - Updated position_value_usd estimation for cross pairs
   - Added warning if cross pair doesn't provide position_value_usd

3. **[production_trader/execution/position_manager.py](production_trader/execution/position_manager.py#L448-L517)**
   - Added P/L verification (actual vs expected)
   - Uses actual P/L for capital tracking in production

4. **[production_trader/execution/oanda_broker.py](production_trader/execution/oanda_broker.py#L428-L453)**
   - Documented exit_price limitation
   - Clarified production vs simulation differences

---

**Date**: 2026-01-11
**Status**: ✅ Ready for testing
**Critical Fix**: Cross-pair position sizing for EURJPY
