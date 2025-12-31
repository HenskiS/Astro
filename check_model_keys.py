"""
Check if model keys match between backtest and production
"""
print("BACKTEST model keys:")
print("  models['breakout_high']")
print("  models['breakout_low']")
print()

print("PRODUCTION model keys:")
print("  self.models[pair]['high']")
print("  self.models[pair]['low']")
print()

print("Note the difference:")
print("  Backtest: 'breakout_high', 'breakout_low'")
print("  Production: 'high', 'low'")
print()

print("Both should be accessing the same models...")
print("HIGH model should predict breakout_high")
print("LOW model should predict breakout_low")
