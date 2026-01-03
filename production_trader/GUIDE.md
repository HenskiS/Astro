# Production Trader - Quick Start

## Setup (First Time)

```bash
cd /home/forex/Astro/production_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
nano .env  # Fill in OANDA_PRACTICE_* credentials

# Test connection
python execution/oanda_broker.py

# Train models (if not already trained)
cd ..
python train_model_15m.py
cd production_trader
```

## Testing (Dry Run)

```bash
# Start tmux session
tmux new -s trader

# Run dry-run mode (no real orders)
python main.py --dry-run

# Detach: Ctrl+B then D
# Reattach: tmux attach -t trader
```

## Live Trading

```bash
# Start tmux session
tmux new -s trader

# Run live on practice account
python main.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t trader
# Stop: Ctrl+C (graceful shutdown)
```

## Monitoring

```bash
# View logs
tail -f logs/production_trader.log

# Check status
ps aux | grep main.py

# Emergency close all
python scripts/emergency_close_all.py
```