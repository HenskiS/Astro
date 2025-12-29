# Planetary-Forex Integration Usage Guide

This guide walks you through combining **REAL astronomical planetary positions** with forex data to test for correlations.

## Quick Start (Real Astronomy - Recommended)

```bash
# 1. Install dependencies (includes skyfield for NASA data)
pip install -r requirements.txt

# 2. Generate REAL planetary data and merge with forex
python integrate_real_astronomy.py

# 3. Train models and compare performance
python train_with_real_planets.py
```

## Alternative: Historical Epicycle Model

For the stylized geocentric visualization:
```bash
python integrate_planetary_forex.py
python train_with_planets.py
```

## Important: Real Astronomy vs Epicycle Simulation

### Real Astronomy (Recommended for Analysis)
- **Uses**: NASA JPL DE421 ephemerides (actual astronomical data)
- **Accuracy**: Positions match the actual sky on those dates
- **Data source**: Same ephemerides NASA uses for space missions
- **Files**: `integrate_real_astronomy.py`, `train_with_real_planets.py`
- **Why**: For meaningful correlation testing, you need ACTUAL planetary positions

### Epicycle Simulation (For Visualization)
- **Uses**: Stylized geocentric model with epicycles (Ptolemaic system)
- **Accuracy**: Approximate patterns, not astronomically accurate to dates
- **Data source**: Simulated using orbital periods and epicycle parameters
- **Files**: `integrate_planetary_forex.py`, `train_with_planets.py`
- **Why**: Beautiful visualization of historical cosmology, but positions don't match real sky

**For forex correlation analysis, always use the REAL astronomy approach!**

## Detailed Workflow

### Step 1: Visualize the Cosmos (Optional but Fun!)

```bash
python visualize.py
```

This shows you the geocentric planetary motion system you're working with. Watch the planets orbit Earth with their epicycles! Close the window when you're done.

### Step 2: Generate & Integrate Planetary Features

```bash
python integrate_planetary_forex.py
```

**What this does:**
- Scans your forex data in `data/` directory
- Generates planetary positions for the matching date range (2010-2025)
- Creates comprehensive planetary features:
  - **Positions**: x, y coordinates for each celestial body
  - **Angles**: Angular positions (radians) + sin/cos encoding
  - **Distances**: How far each body is from Earth
  - **Aspects**: Angular separations between planet pairs (e.g., Sun-Mars angle)
- Merges planetary data with each forex pair
- Outputs: `{PAIR}_1day_with_planets.csv` files

**Example output:**
```
EURUSD: 3,847 days
Planetary features: 154 features
  - Moon_x, Moon_y, Moon_angle, Moon_distance
  - Mercury_x, Mercury_y, Mercury_angle, ...
  - aspect_Moon_Mercury, aspect_Moon_Venus, ...
  - Moon_sin, Moon_cos, ...
```

### Step 3: Train & Compare Models

```bash
python train_with_planets.py
```

**What this does:**
- Trains TWO models per forex pair:
  1. **Baseline**: Technical indicators only (RSI, MACD, EMA, etc.)
  2. **Planetary**: Technical indicators + Planetary features
- Compares test accuracy between both versions
- Shows which planetary features are most important
- Saves both model sets for later use

**Example output:**
```
EURUSD:
  [1] BASELINE MODEL (Technical features only)
      Test accuracy: 52.34%

  [2] PLANETARY MODEL (Technical + Planetary)
      Test accuracy: 54.67%

      Top 5 planetary features:
        aspect_Moon_Venus                        : 0.0234
        Mars_angle                               : 0.0189
        Sun_distance                             : 0.0156
        aspect_Mercury_Jupiter                   : 0.0143
        Venus_sin                                : 0.0128

  COMPARISON:
    Baseline:          52.34%
    With Planets:      54.67%
    Improvement:       +2.33% ✓ PLANETS HELP!
```

### Step 4: Analyze Results

The training script outputs a comprehensive comparison:

```
FINAL COMPARISON SUMMARY
Pair       Baseline    W/ Planets   Improvement    Test Samples
------------------------------------------------------------------------
EURUSD     52.34%      54.67%          +2.33% ✓            367
GBPUSD     51.89%      52.45%          +0.56% ✓            367
USDJPY     50.12%      49.88%          -0.24% ✗            367
...

AVERAGE PERFORMANCE:
  Baseline:           51.45%
  With Planetary:     52.18%
  Average improvement: +0.73%

PLANETARY FEATURE IMPACT:
  Pairs improved: 6/8
  Pairs worse: 2/8
```

## Understanding the Features

### Technical Features (Baseline)
Standard forex indicators:
- **Returns**: 1d, 3d, 5d, 10d price changes
- **EMAs**: 10, 20, 50 period exponential moving averages
- **MACD**: Moving average convergence/divergence
- **RSI**: Relative strength index
- **ATR**: Average true range (volatility)
- **Bollinger Bands**: Price position within bands
- **Market Sentiment**: Cross-pair correlation

### Planetary Features (Added)
Geocentric cosmology indicators:
- **Angles**: Where each planet is in the sky (0-360°)
  - Encoded as sin/cos for cyclical ML compatibility
- **Distances**: How far planets are from Earth
  - Changes due to epicycle motion
- **Aspects**: Angular relationships between planets
  - Example: `aspect_Sun_Mars` = angular separation
  - Values 0° (conjunction), 90° (square), 180° (opposition)
  - Historically significant in astrology

## Files Generated

```
data/
├── planetary_positions_forex_range.csv      # All planetary data
├── EURUSD_1day_with_planets.csv             # Forex + planets merged
├── GBPUSD_1day_with_planets.csv
└── ...

models/
├── forex_baseline_models.pkl                # Technical only
└── forex_with_planets_models.pkl            # Tech + planets
```

## Interpreting Results

### Good Signs (Planets Help!)
- ✓ Test accuracy improves by >1%
- ✓ Multiple pairs show improvement
- ✓ Specific planetary features rank high in importance
- ✓ Improvements are consistent across different pairs

### Neutral/Negative Signs
- ✗ Improvement is <0.5% (could be noise)
- ✗ Some pairs improve, others get worse (inconsistent)
- ✗ Planetary features have very low importance scores
- ✗ Average improvement is negative

### What to Look For
1. **Consistent patterns**: Do the same planetary features help across pairs?
2. **Specific aspects**: Do certain planet pairs (e.g., Sun-Venus) correlate?
3. **Cyclical timing**: Do angular positions (not just aspects) matter?
4. **USD correlation**: Since most pairs involve USD, does that create a planetary pattern?

## Advanced: Exporting Custom Planetary Data

If you want to analyze planetary positions independently:

```python
from geocentric_model import GeocentricModel
from export_data import PlanetaryDataExporter
from datetime import datetime

model = GeocentricModel()
exporter = PlanetaryDataExporter(model)

# Export hourly data (for intraday forex)
df = exporter.export_to_csv(
    filename='planetary_hourly_2024.csv',
    start_date=datetime(2024, 1, 1),
    days=365,
    interval_hours=1.0
)

# Export specific date range
df = exporter.generate_time_series(
    start_date=datetime(2020, 1, 1),
    days=1000,
    interval_hours=6.0  # Every 6 hours
)
```

## Visualization Options

### View Planetary Motion
```bash
python visualize.py
```

### Save Animation
Edit [visualize.py](visualize.py) and uncomment line ~119:
```python
visualizer.save_animation('geocentric_motion.mp4', total_days=365, fps=30)
```

## Historical Context

The epicycle model you're testing against forex was used from Ptolemy (~150 CE) through Aquinas (13th century). While the heliocentric model replaced it scientifically, the question remains: do these geometric patterns correlate with human market behavior?

**Possible mechanisms (if correlations exist):**
1. **Direct influence**: Planetary positions affect human psychology/biology
2. **Cultural**: Traders aware of astrology make correlated decisions
3. **Coincidence**: Pure statistical noise in limited dataset
4. **Hidden variable**: Both correlate with something else (seasons, lunar cycles, etc.)

## Next Steps

1. ✓ Run the integration and training
2. Check if planetary features improve accuracy
3. Analyze which specific planetary features matter most
4. Consider testing:
   - Different prediction horizons (1-day, 7-day)
   - Different forex pairs (crypto, commodities)
   - Specific planetary events (conjunctions, oppositions)
   - Lunar phases separately (Moon has strong cultural significance)

Good luck with your cosmic forex exploration! 🌟📈
