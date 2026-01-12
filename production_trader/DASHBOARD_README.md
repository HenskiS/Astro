# Trading Dashboard

Real-time monitoring dashboard for your production trading system.

## Features

### 📊 Live Monitor
- **Account Metrics**: Balance, daily P/L, open positions, drawdown
- **Interactive Charts**: Candlestick charts with 15-minute data from OANDA
- **Position Markers**: Visual indicators for open positions
- **Real-time Data**: Auto-refreshes every 15 seconds
- **Open Positions Table**: Live P/L tracking for all open trades

### 📜 Trade History
- Filter trades by pair and date range
- Performance metrics (win rate, total P/L, avg P/L)
- Sortable trade history table

### 📈 Performance (Coming Soon)
- Equity curves
- Pair performance breakdown
- Daily/weekly/monthly statistics

---

## Installation

### 1. Install Dependencies

```bash
pip install streamlit plotly
```

Or add to your requirements.txt:
```
streamlit>=1.28.0
plotly>=5.17.0
```

### 2. Verify Environment Variables

Make sure your `.env` file has OANDA credentials:
```env
# Practice account (recommended for testing)
OANDA_PRACTICE_API_KEY=your_practice_key
OANDA_PRACTICE_ACCOUNT_ID=your_practice_account_id

# Or live account
OANDA_API_KEY=your_live_key
OANDA_ACCOUNT_ID=your_live_account_id
```

---

## Usage

### Start the Dashboard

From the project root directory:

```bash
streamlit run production_trader/dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### Run Dashboard + Trader Simultaneously

**Terminal 1** (Trading System):
```bash
cd production_trader
python main.py
```

**Terminal 2** (Dashboard):
```bash
streamlit run production_trader/dashboard.py
```

---

## Dashboard Pages

### Live Monitor

**Top Metrics:**
- Balance with today's P/L
- Number of open positions
- Total trades executed
- Current drawdown from peak

**Chart View:**
- Select any of your 8 trading pairs
- 15-minute candlestick charts
- Green/red markers for open positions
- Current price, 50-bar change, spread

**Open Positions Table:**
- Pair, direction (LONG/SHORT)
- Entry price vs current price
- Position size and confidence
- Unrealized P/L percentage
- Duration (how long position has been open)

### Trade History

- Filter by pairs
- Filter by date range (last 1-30 days)
- Summary metrics: trades, win rate, total P/L, avg P/L
- Detailed table with entry/exit times and P/L

---

## Data Sources

### Live Data (from OANDA)
- ✅ Real-time candle data (cached for 5 minutes)
- ✅ Current account balance
- ✅ Open positions count
- ✅ Current prices for unrealized P/L

### State Data (from files)
- ✅ `state/trading_state.json` - positions, capital, daily P/L
- ✅ `trades_history.csv` - historical trade log (if exists)

---

## Auto-Refresh

The dashboard auto-refreshes every 15 seconds by default.

**To disable:**
- Uncheck "Auto-refresh (15s)" in the sidebar

**Manual refresh:**
- Click "🔄 Refresh Now" button in sidebar
- Press `R` key in browser

---

## Troubleshooting

### Dashboard won't start

**Error: "OANDA credentials not found"**
- Check your `.env` file has correct variables
- Make sure you're running from the project root

**Error: "Failed to connect to OANDA"**
- Verify API key is valid
- Check internet connection
- Try switching between practice/live account

### Charts not loading

**"Failed to load candles"**
- Check OANDA API connection
- Verify pair name is correct
- Clear cache: Click "🔄 Refresh Now"

### Positions not showing

**Check these:**
1. Is `main.py` running? (Positions come from state file)
2. Does `state/trading_state.json` exist?
3. Are there actually open positions?

---

## Future Enhancements

### Phase 2 (Next)
- [ ] Prediction overlays on charts (high/low probabilities)
- [ ] Trade markers on historical charts
- [ ] Equity curve chart
- [ ] Performance by pair breakdown

### Phase 3
- [ ] Log viewer with filtering
- [ ] Backtest comparison page
- [ ] Real-time alerts/notifications
- [ ] Mobile-responsive design

### Phase 4
- [ ] Multi-timeframe charts (1m, 5m, 15m, 1h)
- [ ] Technical indicators (EMA, RSI, MACD)
- [ ] Order book depth visualization
- [ ] Export reports to PDF

---

## Tips

1. **Run on VPS**: Deploy both trader and dashboard on same server
2. **Port Forwarding**: Access dashboard remotely via SSH tunnel:
   ```bash
   ssh -L 8501:localhost:8501 user@your-server
   ```
3. **Performance**: If dashboard is slow, reduce chart candle count to 100
4. **Multiple Monitors**: Open dashboard on second screen while monitoring logs

---

## Architecture

```
┌─────────────────────────────────┐
│    Streamlit Dashboard          │
│    (localhost:8501)            │
└─────────┬───────────────────────┘
          │
    ┌─────┴─────────────┐
    │                   │
┌───▼────────┐   ┌──────▼────────┐
│   OANDA    │   │  State Files  │
│    API     │   │               │
│  (Live)    │   │ - JSON        │
│            │   │ - CSV         │
└────────────┘   └───────────────┘
                        ▲
                        │
              ┌─────────┴─────────┐
              │  Production       │
              │  Trader           │
              │  (main.py)       │
              └───────────────────┘
```

Both processes run independently and read from the same data sources.

---

**Created**: 2026-01-11
**Status**: ✅ Ready to use
**Version**: 1.0 MVP
