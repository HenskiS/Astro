# Astro - Forex Trading System

Automated forex trading system using XGBoost machine learning models for breakout prediction on 8 major currency pairs.

## 📊 Strategy Performance

### 15-Minute Breakout Strategy
- **CAGR:** 116.1%
- **Win Rate:** 52.8%
- **Total Trades:** 2,937 (2018-2023)
- **Max Drawdown:** -14.8%
- **Sharpe Ratio:** 2.1
- **Timeframe:** 15-minute candles
- **Pairs:** EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/JPY

**Exit Strategy:**
- Immediate stop: -5% (peace of mind)
- Ladder exits: 0.2%, 0.4% (40% scale-out each)
- Trailing stop: Activates at 0.1%, trails at 75%
- Emergency stop: 6 hours + -4% loss
- Target: Breakout level + 0.5%

### 1-Hour Breakout Strategy
- **CAGR:** 105.7%
- **Win Rate:** 53.2%
- **Total Trades:** 1,234 (2018-2023)
- **Timeframe:** 1-hour candles
- **Same pairs and exit strategy as 15m**

## 🏗️ Architecture

### Machine Learning Pipeline
- **Model:** XGBoost binary classifiers (16 models: 8 pairs × 2 directions)
- **Features:** Technical indicators (EMAs, RSI, MACD, ATR, Bollinger Bands), breakout levels, time features
- **Training Window:** 9 months rolling
- **Prediction Confidence:** 70% minimum threshold

### Production Trading System
- **Broker:** OANDA v20 REST API
- **Execution:** Next-bar open entry (no lookahead bias)
- **Risk Management:** 0.4% risk per trade, 15% max drawdown, 5% daily loss limit
- **Position Limits:** 120 total positions, 15 per pair
- **FIFO Handling:** Skips competing positions (OANDA compliance)
- **Monitoring:** Telegram alerts + structured logging
- **Safety:** Kill switch, emergency stops, crash recovery

## 📁 Key Scripts

### Training & Backtesting
- **`train_model_15m.py`** - Train XGBoost models for 15m strategy (saves to `models/`)
- **`train_model_1h.py`** - Train XGBoost models for 1h strategy
- **`backtest_15m_optimized.py`** - Backtest 15m strategy with realistic execution
- **`backtest_1h_optimized.py`** - Backtest 1h strategy with realistic execution
- **`analyze_immediate_stop.py`** - Analyze impact of -5% immediate stop loss

### Production Trading
- **`production_trader/`** - Complete production trading system
  - See [production_trader/GUIDE.md](production_trader/GUIDE.md) for deployment instructions
  - See [production_trader/IMPLEMENTATION_STATUS.md](production_trader/IMPLEMENTATION_STATUS.md) for component details

### Data Processing
- **`fetch_oanda_data.py`** - Download historical OANDA price data
- **`check_data_quality.py`** - Validate data completeness and quality
- **`compare_features.py`** - Compare feature calculations across timeframes

### Analysis & Debugging
- **`production_simulation.py`** - Simulate production trading with historical data
- **`verify_no_lookahead.py`** - Verify backtests have no lookahead bias
- **`check_prediction_gaps.py`** - Check for gaps in model predictions
- **`diagnostic_*.py`** - Various diagnostic scripts for debugging edge cases

## 🚀 Quick Start

### 1. Train Models
```bash
# Train 15-minute strategy models
python train_model_15m.py

# Train 1-hour strategy models (optional)
python train_model_1h.py
```

### 2. Run Backtest
```bash
# Backtest with equity curve plot
python backtest_15m_optimized.py --plot

# Test different FIFO modes
python backtest_15m_optimized.py --fifo-mode skip_competing
```

### 3. Deploy Production Trader
See detailed guide: [production_trader/GUIDE.md](production_trader/GUIDE.md)

```bash
cd production_trader

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
nano .env  # Fill in OANDA practice credentials

# Test connection
python execution/oanda_broker.py

# Run dry-run mode (no real orders)
python main.py --dry-run

# Deploy live on practice account
tmux new -s trader
python main.py
# Ctrl+B then D to detach
```

## 📦 Requirements

### Core Dependencies
- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0

### Production Trading
- v20 >= 3.0.25 (OANDA API)
- python-telegram-bot >= 20.0
- pyyaml >= 6.0
- python-dotenv >= 1.0.0

See [production_trader/requirements.txt](production_trader/requirements.txt) for complete list.

## ⚙️ Configuration

### Backtest Parameters
- **Initial Capital:** $500
- **Risk Per Trade:** 0.4% of capital
- **Max Positions:** 120 total, 15 per pair
- **Commission/Spread:** 1 pip (0.01%)
- **Slippage:** 1 pip (0.01%)

### Production Settings
All settings configured in [production_trader/config.yaml](production_trader/config.yaml):
- Capital and risk management
- Strategy parameters
- OANDA API settings (practice/live)
- Telegram notifications
- Logging and state persistence

## 🔒 Safety Features

### Risk Controls
- **Max Drawdown:** 15% from peak → closes all positions
- **Daily Loss Limit:** 5% → stops new positions for rest of day
- **Position Limits:** Enforced at strategy and risk manager levels
- **Immediate Stop Loss:** -5% anytime (peace of mind)

### Emergency Controls
- **KILL_SWITCH:** Create file to trigger immediate shutdown
- **Emergency Stop:** 6 hours + -4% loss
- **Graceful Shutdown:** Ctrl+C saves state before exit
- **Crash Recovery:** Auto-loads last saved state on restart

### Monitoring
- **Telegram Alerts:** Real-time position events, emergency stops, daily summary
- **Structured Logging:** All actions logged to file with rotation
- **Health Checks:** Periodic API connection and system health validation

## 📈 Backtest Validation

### Lookahead Bias Prevention
- ✅ Predictions made at bar T close
- ✅ Entry at bar T+1 open (next bar)
- ✅ Entry bar checked immediately for stops/targets
- ✅ Exit prices use realistic bid/ask spreads
- ✅ Verified with `verify_no_lookahead.py`

### FIFO Compliance
- OANDA rejects competing positions (long + short on same pair)
- Strategy skips signals that compete with existing positions
- Allows position stacking (multiple positions same direction)
- Tested modes: skip_competing (116.1% CAGR), position_netting (113% CAGR)

### Realistic Execution
- Commission: 1 pip per trade
- Slippage: 1 pip per execution
- Spread: 2-3 pips (bid/ask difference)
- High-spread hours avoided: 20-22 UTC
- Entry at next bar open (gap risk included)

## 📊 Data

### Data Sources
- **Historical Data:** OANDA v20 API
- **Timeframes:** 15-minute and 1-hour candles
- **Date Range:** 2018-2023 (5 years)
- **Storage:** CSV files in `data_15m/` and `data_1h/`

### Data Quality
- No missing bars (validated with `check_data_quality.py`)
- Bid/ask prices included for realistic spread modeling
- High/low prices used for intraday stop detection

## 🛠️ Development

### Project Structure
```
Astro/
├── train_model_15m.py           # Model training (15m)
├── train_model_1h.py            # Model training (1h)
├── backtest_15m_optimized.py    # Backtest engine (15m)
├── backtest_1h_optimized.py     # Backtest engine (1h)
├── models/                      # Trained XGBoost models
├── data_15m/                    # 15-minute historical data
├── data_1h/                     # 1-hour historical data
└── production_trader/           # Live trading system
    ├── main.py                  # Main orchestrator
    ├── config.yaml              # Configuration
    ├── strategies/              # Signal generation
    ├── execution/               # OANDA broker, position manager
    ├── risk/                    # Risk management
    ├── state/                   # State persistence
    ├── monitoring/              # Telegram alerts, logging
    ├── models/                  # Model loader, feature calculator
    └── GUIDE.md                 # Deployment guide
```

### Key Implementation Details

**Entry Logic:**
1. Signal generated at bar T close (after features calculated)
2. Order placed at bar T+1 open (next bar)
3. Entry bar checked immediately for stops/targets
4. Long entries use ASK price, short entries use BID price

**Exit Logic:**
- Checked every minute on production system
- Uses bid prices for long exits, ask prices for short exits
- Ladder exits scale out 40% at each level (0.2%, 0.4%)
- Trailing stop follows price with 75% trail percentage
- Emergency stop prevents extended losing positions

**Position Sizing:**
```python
risk_amount = capital * 0.004  # 0.4% risk
stop_distance = 0.04           # 4% stop (emergency + immediate)
position_size = risk_amount / stop_distance  # ~10% of capital
```

## 📝 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This is an automated trading system that involves substantial risk. Past performance does not guarantee future results. Only trade with capital you can afford to lose. The authors are not responsible for any financial losses incurred from using this software.

## 🤝 Contributing

This is a personal trading project. Feel free to fork and adapt for your own use, but be aware that trading performance can vary significantly based on market conditions, execution environment, and parameter choices.

## 📧 Contact

For questions about the implementation or architecture, open an issue on GitHub.

---

**Status:** ✅ Production ready - All components implemented and tested

**Last Updated:** January 2026
