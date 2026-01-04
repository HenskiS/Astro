"""
Real astronomical planetary positions using NASA JPL ephemerides
Provides geocentric (Earth-centered) coordinates for actual planetary positions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from skyfield.api import load, utc
from skyfield import almanac
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class RealAstronomyModel:
    """
    Calculates actual planetary positions using NASA JPL ephemerides.
    Provides geocentric (Earth-centered) view of the solar system.
    """

    def __init__(self):
        """Initialize with JPL ephemeris data"""
        print("Loading JPL ephemeris data (this may take a moment on first run)...")

        # Load ephemeris (positions of celestial bodies)
        # This downloads data from NASA JPL if not cached (~10-20 MB)
        self.ts = load.timescale()
        self.eph = load('de421.bsp')  # JPL DE421 ephemeris (1900-2050)

        # Define celestial bodies
        self.bodies = {
            'Sun': self.eph['sun'],
            'Moon': self.eph['moon'],
            'Mercury': self.eph['mercury'],
            'Venus': self.eph['venus'],
            'Mars': self.eph['mars'],
            'Jupiter': self.eph['jupiter barycenter'],  # Jupiter system center
            'Saturn': self.eph['saturn barycenter'],    # Saturn system center
        }

        # Earth for geocentric calculations
        self.earth = self.eph['earth']

        print("Ephemeris loaded successfully!")

    def get_position_for_date(
        self,
        date: datetime,
        body_name: str,
        normalize_scale: float = 1.0
    ) -> Tuple[float, float]:
        """
        Get geocentric position of a celestial body for a specific date.

        Args:
            date: The date/time to calculate position for
            body_name: Name of the celestial body
            normalize_scale: Scaling factor for visualization (1.0 = AU)

        Returns:
            (x, y) position in AU (Astronomical Units) from Earth
        """
        if body_name not in self.bodies:
            raise ValueError(f"Unknown body: {body_name}")

        # Ensure datetime has timezone (UTC)
        if date.tzinfo is None:
            date = date.replace(tzinfo=utc)

        # Convert datetime to Skyfield time
        t = self.ts.from_datetime(date)

        # Calculate position relative to Earth (geocentric)
        body = self.bodies[body_name]
        astrometric = self.earth.at(t).observe(body)

        # Get position in AU (x, y, z coordinates)
        position = astrometric.position.au

        # For 2D geocentric view, we'll use ecliptic plane projection
        # This gives us the "classic" view of planetary positions
        x = position[0] * normalize_scale
        y = position[1] * normalize_scale
        # z = position[2]  # Ignore for 2D projection

        return float(x), float(y)

    def get_all_positions(
        self,
        date: datetime,
        normalize_scale: float = 1.0
    ) -> Dict[str, Tuple[float, float]]:
        """Get positions of all celestial bodies for a date"""
        return {
            name: self.get_position_for_date(date, name, normalize_scale)
            for name in self.bodies.keys()
        }

    def get_ecliptic_longitude(self, date: datetime, body_name: str) -> float:
        """
        Get ecliptic longitude (celestial longitude) of a body.
        This is the traditional astrological position (0-360 degrees).

        Args:
            date: The date to calculate for
            body_name: Name of celestial body

        Returns:
            Ecliptic longitude in degrees (0-360)
        """
        # Ensure datetime has timezone (UTC)
        if date.tzinfo is None:
            date = date.replace(tzinfo=utc)

        t = self.ts.from_datetime(date)
        body = self.bodies[body_name]

        astrometric = self.earth.at(t).observe(body)
        lat, lon, distance = astrometric.ecliptic_latlon()

        return float(lon.degrees % 360)

    def get_distance(self, date: datetime, body_name: str) -> float:
        """
        Get distance from Earth to celestial body in AU.

        Args:
            date: The date to calculate for
            body_name: Name of celestial body

        Returns:
            Distance in Astronomical Units (AU)
        """
        # Ensure datetime has timezone (UTC)
        if date.tzinfo is None:
            date = date.replace(tzinfo=utc)

        t = self.ts.from_datetime(date)
        body = self.bodies[body_name]

        astrometric = self.earth.at(t).observe(body)
        _, _, distance = astrometric.ecliptic_latlon()

        return float(distance.au)

    def get_lunar_phase(self, date: datetime) -> float:
        """
        Get lunar phase (0 = new moon, 0.5 = full moon, 1.0 = new moon).

        Args:
            date: The date to calculate for

        Returns:
            Lunar phase as fraction (0-1)
        """
        # Ensure datetime has timezone (UTC)
        if date.tzinfo is None:
            date = date.replace(tzinfo=utc)

        t = self.ts.from_datetime(date)

        # Calculate phase using Sun-Earth-Moon angle
        sun = self.bodies['Sun']
        moon = self.bodies['Moon']

        e = self.earth.at(t)
        s = e.observe(sun).apparent()
        m = e.observe(moon).apparent()

        # Calculate elongation (angular separation)
        elongation = s.separation_from(m).degrees

        # Convert to phase (0-1)
        # 0° = new moon (0), 180° = full moon (0.5), 360° = new moon (1)
        phase = elongation / 360.0

        return float(phase)


class RealAstronomyDataExporter:
    """Exports real astronomical data for time series analysis"""

    def __init__(self, model: RealAstronomyModel):
        self.model = model

    def generate_time_series(
        self,
        start_date: datetime,
        days: int,
        interval_hours: float = 24.0
    ) -> pd.DataFrame:
        """
        Generate time series of actual planetary positions.

        Args:
            start_date: Starting date
            days: Number of days
            interval_hours: Hours between data points

        Returns:
            DataFrame with real astronomical positions
        """
        interval_days = interval_hours / 24.0
        num_points = int(days / interval_days)

        print(f"Generating {num_points} astronomical data points...")

        data = {'timestamp': []}

        # Initialize columns
        for body_name in self.model.bodies.keys():
            data[f'{body_name}_x'] = []
            data[f'{body_name}_y'] = []
            data[f'{body_name}_angle'] = []  # Ecliptic longitude
            data[f'{body_name}_distance'] = []

        # Add lunar phase
        data['lunar_phase'] = []

        # Generate data points
        for i in range(num_points):
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_points} ({i/num_points*100:.1f}%)")

            day = i * interval_days
            timestamp = start_date + timedelta(days=day)

            # Ensure timestamp has UTC timezone for skyfield
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=utc)

            data['timestamp'].append(timestamp)

            # Get positions for all bodies
            positions = self.model.get_all_positions(timestamp)

            for body_name, (x, y) in positions.items():
                data[f'{body_name}_x'].append(x)
                data[f'{body_name}_y'].append(y)

                # Get ecliptic longitude (astrological position)
                longitude = self.model.get_ecliptic_longitude(timestamp, body_name)
                data[f'{body_name}_angle'].append(longitude)

                # Get distance
                distance = self.model.get_distance(timestamp, body_name)
                data[f'{body_name}_distance'].append(distance)

            # Lunar phase
            lunar_phase = self.model.get_lunar_phase(timestamp)
            data['lunar_phase'].append(lunar_phase)

        print(f"  Complete: {num_points} data points generated")

        df = pd.DataFrame(data)
        return df

    def add_planetary_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate astrological aspects (angular separations).

        Args:
            df: DataFrame with planetary positions

        Returns:
            DataFrame with aspect columns added
        """
        print("Calculating planetary aspects...")

        planets = list(self.model.bodies.keys())

        # Calculate angular separations
        for i, planet1 in enumerate(planets):
            for planet2 in planets[i+1:]:
                angle1 = df[f'{planet1}_angle']
                angle2 = df[f'{planet2}_angle']

                # Angular separation (0-180 degrees)
                separation = np.abs(angle2 - angle1)
                separation = np.minimum(separation, 360 - separation)

                df[f'aspect_{planet1}_{planet2}'] = separation

        print(f"  Calculated {len(planets)*(len(planets)-1)//2} aspect pairs")

        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sin/cos encoding of angles for ML.

        Args:
            df: DataFrame with angles

        Returns:
            DataFrame with cyclical features
        """
        print("Adding cyclical encoding...")

        for body_name in self.model.bodies.keys():
            # Convert degrees to radians for sin/cos
            angle_rad = np.radians(df[f'{body_name}_angle'])
            df[f'{body_name}_sin'] = np.sin(angle_rad)
            df[f'{body_name}_cos'] = np.cos(angle_rad)

        # Add lunar phase cyclical encoding
        df['lunar_phase_sin'] = np.sin(2 * np.pi * df['lunar_phase'])
        df['lunar_phase_cos'] = np.cos(2 * np.pi * df['lunar_phase'])

        return df

    def export_to_csv(
        self,
        filename: str,
        start_date: datetime,
        days: int,
        interval_hours: float = 24.0,
        include_aspects: bool = True,
        include_cyclical: bool = True
    ) -> pd.DataFrame:
        """Export real astronomical data to CSV"""
        df = self.generate_time_series(start_date, days, interval_hours)

        if include_aspects:
            df = self.add_planetary_aspects(df)

        if include_cyclical:
            df = self.add_cyclical_features(df)

        print(f"\nSaving to {filename}...")
        df.to_csv(filename, index=False)
        print(f"Saved! Total features: {len(df.columns)}")

        return df


def main():
    """Example usage"""
    print("="*80)
    print("REAL ASTRONOMICAL PLANETARY POSITIONS")
    print("Using NASA JPL Ephemerides")
    print("="*80)
    print()

    # Initialize
    model = RealAstronomyModel()
    exporter = RealAstronomyDataExporter(model)

    # Export data
    start_date = datetime(2024, 1, 1)

    df = exporter.export_to_csv(
        filename='real_planetary_positions.csv',
        start_date=start_date,
        days=365,
        interval_hours=24.0,
        include_aspects=True,
        include_cyclical=True
    )

    print()
    print("Sample data:")
    print(df.head())
    print()
    print("Date range:", df['timestamp'].min(), "to", df['timestamp'].max())


if __name__ == '__main__':
    main()
