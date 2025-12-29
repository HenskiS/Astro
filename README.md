# Geocentric Planetary Motion Visualization

A Python-based visualization of the geocentric (Earth-centered) model of planetary motion using epicycles, inspired by Ptolemaic and Aquinas cosmology. Includes data export capabilities for correlating planetary positions with forex data or other time series.

## Overview

This project simulates the historical geocentric model where:
- Earth sits at the center of the cosmos
- Planets move on **epicycles** (small circles) whose centers orbit Earth on larger circles called **deferents**
- The seven classical celestial bodies are included: Moon, Mercury, Venus, Sun, Mars, Jupiter, and Saturn

## Features

- **Animated 2D Visualization**: Real-time animation showing planetary motions with orbital trails
- **Epicycle Model**: Historically accurate epicycle-based motion calculations
- **Data Export**: Export planetary position data as time series for analysis
- **Planetary Aspects**: Calculate angular relationships between celestial bodies
- **ML-Ready**: Includes cyclical encoding (sin/cos) for machine learning applications

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. View the Animated Visualization

Run the visualization to see the planets moving around Earth:

```bash
python visualize.py
```

This will display an animated plot showing:
- Planetary positions and orbital trails
- Current day counter
- Color-coded celestial bodies

**Controls:**
- Close the window to exit
- The animation loops continuously

### 2. Export Planetary Position Data

Generate time series data for analysis:

```bash
python export_data.py
```

This will create two CSV files:
- `planetary_positions_daily.csv` - Daily positions for 2 years (long-term patterns)
- `planetary_positions_hourly.csv` - Hourly positions for 3 months (forex correlation)

**Generated Features:**
- `{planet}_x`, `{planet}_y`: Cartesian coordinates
- `{planet}_angle`: Angular position (radians)
- `{planet}_distance`: Distance from Earth
- `aspect_{planet1}_{planet2}`: Angular separation between planets (degrees)
- `{planet}_sin`, `{planet}_cos`: Cyclical encoding for ML

### 3. Custom Data Export

You can customize the export in your own scripts:

```python
from datetime import datetime
from geocentric_model import GeocentricModel
from export_data import PlanetaryDataExporter

# Create model and exporter
model = GeocentricModel()
exporter = PlanetaryDataExporter(model)

# Export with custom parameters
df = exporter.export_to_csv(
    filename='my_data.csv',
    start_date=datetime(2024, 1, 1),
    days=365,
    interval_hours=6.0,  # Every 6 hours
    include_aspects=True,
    include_cyclical=True
)
```

### 4. Correlating with Forex Data

For your forex ML model correlation:

1. Export hourly planetary data aligned with your forex timestamps:
   ```python
   df_planets = exporter.generate_time_series(
       start_date=your_forex_start_date,
       days=your_analysis_period,
       interval_hours=1.0  # Match forex data frequency
   )
   ```

2. Merge with forex data:
   ```python
   import pandas as pd

   df_forex = pd.read_csv('your_forex_data.csv')
   df_forex['timestamp'] = pd.to_datetime(df_forex['timestamp'])

   # Merge on timestamp
   df_combined = pd.merge(df_planets, df_forex, on='timestamp', how='inner')
   ```

3. Use planetary features in your ML model:
   - Angular positions (`_angle`, `_sin`, `_cos`)
   - Distances from Earth (`_distance`)
   - Planetary aspects (angular separations)

## Project Structure

```
Astro/
├── geocentric_model.py      # Core epicycle calculations
├── visualize.py              # Animated visualization
├── export_data.py            # Data export for analysis
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## The Geocentric Model

### Celestial Bodies (in order from Earth):

1. **Moon** - Closest body, rapid motion
2. **Mercury** - Quick epicycle motion
3. **Venus** - Morning/evening star
4. **Sun** - No epicycle (simple circular orbit)
5. **Mars** - Retrograde motion via epicycle
6. **Jupiter** - Slow, majestic motion
7. **Saturn** - Outermost sphere

### Parameters

Each planet has:
- **Deferent radius**: Distance to epicycle center
- **Epicycle radius**: Size of the small circle
- **Deferent period**: Orbital period around Earth
- **Epicycle period**: Rotation period on epicycle

These create the complex looping patterns visible in the animation.

## Historical Context

This model represents the geocentric cosmology that dominated Western thought from Ptolemy (~150 CE) through the medieval period, including Thomas Aquinas' cosmological writings (13th century). While superseded by the heliocentric model, the epicycle system was remarkably successful at predicting planetary positions.

## Next Steps

- Run the visualization to see the planetary dance
- Export data and explore the time series
- Correlate with your forex ML model to look for interesting patterns
- Experiment with different date ranges and time intervals

## Notes

The orbital parameters are stylized for visual clarity while maintaining the character of epicycle motion. For rigorous historical accuracy, consult Ptolemy's *Almagest*.

Enjoy exploring the cosmos as Aquinas saw it!
