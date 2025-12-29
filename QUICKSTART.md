# Quick Start: Real Astronomical Forex Analysis

## The Problem You Just Solved

Your initial question was spot-on: **"How do we align the actual planetary positions with the dates of the forex data?"**

The original epicycle model was a *simulation* that didn't know where planets actually were on specific historical dates. It just generated patterns based on orbital periods.

## The Solution: Real Astronomy

We now use **NASA JPL ephemerides** to get ACTUAL planetary positions for your forex dates (2010-2025).

## Run the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Merge real astronomical data with forex
python integrate_real_astronomy.py

# Step 2: Train and compare models
python train_with_real_planets.py
```

## What Happens

### Step 1: Integration (5-10 minutes)
```
INTEGRATING REAL ASTRONOMICAL POSITIONS WITH FOREX DATA
Using NASA JPL Ephemerides (DE421)
```

This will:
- Download NASA ephemeris data (~10-20 MB, cached after first run)
- Calculate actual planetary positions for Feb 2010 - Dec 2025
- Generate features:
  - **Ecliptic longitude**: Where each planet is in the sky (0-360°)
  - **Distance**: Real Earth-planet distance in AU
  - **Aspects**: Angular separations (e.g., Sun 120° from Mars)
  - **Lunar phase**: Actual new/full moon dates
- Output: `{PAIR}_1day_with_real_planets.csv` files

### Step 2: Training & Comparison (2-5 minutes)
```
TRAINING FOREX MODELS WITH REAL ASTRONOMICAL FEATURES
Comparing: Technical Only vs Technical + Real Astronomy
```

This will:
- Train TWO XGBoost models per forex pair:
  1. **Baseline**: Your technical indicators only (RSI, MACD, etc.)
  2. **Astronomy**: Technical + Real planetary positions
- Show which model performs better
- Identify which astronomical features matter most

## Example Output

```
EURUSD:
  [1] BASELINE (Technical):       52.34%
  [2] WITH REAL ASTRONOMY:         54.67%
      Improvement: +2.33% ✓✓ REAL ASTRONOMY HELPS!

      Top astronomical features:
        Moon_angle                : 0.0234
        aspect_Sun_Venus          : 0.0189
        Mars_distance             : 0.0156

AVERAGE IMPROVEMENT: +0.73%
Pairs with improvement: 6/8
```

## Interpreting Results

### Positive Finding (Astronomy Helps)
- **>1% improvement**: Significant! Real correlation possible
- **High feature importance**: Specific planetary configs matter
- **Consistent across pairs**: Pattern is robust

**Possible explanations:**
- Lunar/solar cycles affect human psychology/trading
- Seasonal patterns correlate with planetary positions
- Cultural/astrological beliefs create self-fulfilling prophecies
- Statistical coincidence (needs further validation)

### Negative/Neutral Finding (No Correlation)
- **<0.5% improvement**: Within noise threshold
- **Inconsistent**: Helps some pairs, hurts others
- **Low feature importance**: Model ignores planetary features

**This is the expected result!** Most likely outcome is no correlation.

## Key Differences: Real vs Simulation

### Real Astronomy (`integrate_real_astronomy.py`)
```python
# Actual position on March 15, 2020
Mars_angle: 253.7°  # This is where Mars ACTUALLY was
Mars_distance: 1.89 AU  # Actual distance from Earth
```

### Epicycle Simulation (`integrate_planetary_forex.py`)
```python
# Simulated position (not tied to real dates)
Mars_angle: 123.4°  # Generated from epicycle equations
Mars_distance: 4.6 AU  # From deferent + epicycle radii
```

**For correlation analysis**: You MUST use real astronomy!
**For visualization**: The epicycle model is beautiful to watch!

## What Each File Does

| File | Purpose |
|------|---------|
| `real_astronomy.py` | NASA data interface using skyfield |
| `integrate_real_astronomy.py` | Merge real planetary data with forex |
| `train_with_real_planets.py` | Train models with real astronomical features |
| `geocentric_model.py` | Epicycle simulation (for visualization) |
| `visualize.py` | Animated geocentric visualization |

## Visualization (Optional)

Want to see the cosmos in action?

```bash
python visualize.py
```

This shows the *epicycle model* (not real positions, but beautiful!). Watch the planets orbit Earth with their epicycles creating retrograde motion.

## Next Steps After Training

1. **Check the results**: Did astronomy help or hurt?
2. **Examine top features**: Which planets/aspects correlate?
3. **Consider lunar focus**: Moon has strongest cultural significance
4. **Test significance**: Run with randomized dates as control
5. **Extend analysis**: Try different prediction horizons (1-day, 7-day)

## Files Generated

```
data/
├── real_planetary_positions_forex_range.csv  # All astronomical data
├── EURUSD_1day_with_real_planets.csv         # Forex + real planets
├── GBPUSD_1day_with_real_planets.csv
└── ...

models/
├── forex_baseline_models.pkl                 # Technical only
└── forex_with_real_astronomy_models.pkl      # Tech + planets
```

## Scientific Rigor

To claim real correlation, you'd need:
- ✓ Use actual planetary positions (we now have this!)
- □ Test with out-of-sample data (different time periods)
- □ Compare against randomized controls (shuffle dates)
- □ Correct for multiple testing (many features → some will look good by chance)
- □ Test causal mechanisms (why would this work?)

This analysis is step 1: **exploratory data analysis**. Have fun with it! 🌟

---

**Remember**: The most likely result is NO correlation. But the journey of testing it with real data is the fun part!
