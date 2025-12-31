"""
PRODUCTION SIMULATION - USING QUARTERLY PREDICTIONS
====================================================
Run production simulation but use the pre-generated quarterly predictions.
This tests if the issue is predictions vs backtest logic.

No training required - just loads predictions and trades.
"""
import pandas as pd
import pickle
from production_simulation import ProductionSimulation, INITIAL_CAPITAL

print("="*100)
print("PRODUCTION SIMULATION WITH QUARTERLY PREDICTIONS")
print("="*100)
print()

# Load quarterly predictions
print("Loading quarterly predictions...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    quarterly_data = pickle.load(f)

# Convert to lookup dictionary: {(date, pair): {high_prob, low_prob, high_20d, low_20d}}
predictions_lookup = {}
for quarter_key, quarter_data in quarterly_data.items():
    for pair, pair_data in quarter_data.items():
        for i in range(len(pair_data['date'])):
            date = pd.Timestamp(pair_data['date'][i])
            if date.tz is not None:
                date = date.tz_localize(None)
            date = date.normalize()

            predictions_lookup[(date, pair)] = {
                'high_prob': pair_data['breakout_high_prob'][i],
                'low_prob': pair_data['breakout_low_prob'][i],
                'high_20d': pair_data['high_20d'][i],
                'low_20d': pair_data['low_20d'][i]
            }

print(f"  Loaded {len(predictions_lookup)} predictions")
print()

# Create a modified simulation that uses pre-generated predictions
class ProductionSimWithQuarterlyPreds(ProductionSimulation):
    def __init__(self, predictions_lookup, **kwargs):
        super().__init__(**kwargs)
        self.predictions_lookup = predictions_lookup

    def train_models(self, train_end_date):
        """Override to skip training - we're using pre-generated predictions"""
        print(f"Skipping training (using pre-generated predictions)")
        # Just create dummy models dict so the code doesn't break
        for pair in self.api.get_available_pairs():
            self.models[pair] = {'dummy': True}
        self.last_training_date = train_end_date

    def _process_day(self, current_date):
        """Override to use pre-generated predictions"""
        pairs = self.api.get_available_pairs()

        # Update positions
        positions_to_close = []
        for position in self.positions:
            # Get current prices for the position's pair
            current_prices = self.api.get_current_price(position.pair, current_date)
            exit_info = position.update(
                current_date,
                current_prices['high'],
                current_prices['low'],
                current_prices['close']
            )
            if exit_info:
                positions_to_close.append((position, exit_info))

        # Close positions
        for position, exit_info in positions_to_close:
            self._close_position(position, exit_info, current_date)
            self.positions.remove(position)

        # Use pre-generated predictions
        from production_simulation import MIN_CONFIDENCE, RISK_PER_TRADE, Position

        for pair in pairs:
            # Look up prediction (normalize date for comparison)
            lookup_date = current_date.normalize() if hasattr(current_date, 'normalize') else current_date
            key = (lookup_date, pair)
            if key not in self.predictions_lookup:
                continue

            pred = self.predictions_lookup[key]
            high_prob = pred['high_prob']
            low_prob = pred['low_prob']
            high_20d = pred['high_20d']
            low_20d = pred['low_20d']

            max_prob = max(high_prob, low_prob)

            if max_prob >= MIN_CONFIDENCE:
                # Determine direction
                if high_prob > low_prob:
                    direction = 'long'
                    target = high_20d * 1.005
                else:
                    direction = 'short'
                    target = low_20d * 0.995

                # NO DIRECTIONAL CHECK - match quarterly backtest behavior

                # Enter at same-day close (match quarterly backtest)
                current_price = self.api.get_current_price(pair, current_date)
                entry_price = current_price['close']

                # Calculate position size
                assumed_risk = 0.02
                risk_amount = self.capital * RISK_PER_TRADE
                position_size = risk_amount / (entry_price * assumed_risk)

                # Create position immediately (same-day entry)
                position = Position(pair, current_date, entry_price, direction, position_size, target, max_prob)
                self.positions.append(position)

# Run simulation with quarterly predictions
sim = ProductionSimWithQuarterlyPreds(
    predictions_lookup=predictions_lookup,
    data_dir='data',
    start_date='2016-01-01',
    end_date='2025-12-31'
)

results = sim.run()

print()
print("="*100)
print("COMPARISON")
print("="*100)
print(f"Production sim with quarterly predictions: ${INITIAL_CAPITAL} -> ${results['final_capital']:,.0f}")
print(f"Original quarterly backtest:                ${INITIAL_CAPITAL} -> $34,800,000")
print()
print("If these match, the issue is the on-the-fly predictions.")
print("If they differ, the issue is the backtest logic itself.")
