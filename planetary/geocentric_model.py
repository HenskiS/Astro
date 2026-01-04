"""
Geocentric Planetary Motion Model with Epicycles
Based on Ptolemaic/Aquinas cosmology
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PlanetaryBody:
    """Represents a celestial body in the geocentric model with epicycle."""
    name: str
    deferent_radius: float  # Distance from Earth to epicycle center
    epicycle_radius: float  # Radius of the epicycle
    deferent_period: float  # Days for one complete deferent orbit
    epicycle_period: float  # Days for one complete epicycle orbit
    deferent_angle_offset: float = 0.0  # Initial angle on deferent
    epicycle_angle_offset: float = 0.0  # Initial angle on epicycle
    color: str = 'white'


class GeocentricModel:
    """
    Simulates the geocentric (Earth-centered) model of the solar system
    using epicycles as described in medieval cosmology.
    """

    def __init__(self):
        # Classical seven celestial bodies (Ptolemaic system)
        # Distances and periods are approximate/stylized for visual effect
        self.bodies: Dict[str, PlanetaryBody] = {
            'Moon': PlanetaryBody(
                name='Moon',
                deferent_radius=0.8,
                epicycle_radius=0.15,
                deferent_period=27.3,
                epicycle_period=27.3,
                color='lightgray'
            ),
            'Mercury': PlanetaryBody(
                name='Mercury',
                deferent_radius=1.5,
                epicycle_radius=0.3,
                deferent_period=88.0,
                epicycle_period=22.0,
                deferent_angle_offset=np.pi / 4,
                color='darkgray'
            ),
            'Venus': PlanetaryBody(
                name='Venus',
                deferent_radius=2.2,
                epicycle_radius=0.4,
                deferent_period=224.7,
                epicycle_period=40.0,
                deferent_angle_offset=np.pi / 2,
                color='gold'
            ),
            'Sun': PlanetaryBody(
                name='Sun',
                deferent_radius=3.0,
                epicycle_radius=0.0,  # Sun doesn't need epicycle in this model
                deferent_period=365.25,
                epicycle_period=365.25,
                deferent_angle_offset=0,
                color='yellow'
            ),
            'Mars': PlanetaryBody(
                name='Mars',
                deferent_radius=4.0,
                epicycle_radius=0.6,
                deferent_period=687.0,
                epicycle_period=70.0,
                deferent_angle_offset=np.pi,
                color='red'
            ),
            'Jupiter': PlanetaryBody(
                name='Jupiter',
                deferent_radius=5.5,
                epicycle_radius=0.5,
                deferent_period=4333.0,
                epicycle_period=200.0,
                deferent_angle_offset=3 * np.pi / 2,
                color='orange'
            ),
            'Saturn': PlanetaryBody(
                name='Saturn',
                deferent_radius=7.0,
                epicycle_radius=0.4,
                deferent_period=10759.0,
                epicycle_period=300.0,
                deferent_angle_offset=np.pi / 6,
                color='wheat'
            ),
        }

    def get_position(self, body_name: str, day: float) -> Tuple[float, float]:
        """
        Calculate the position of a celestial body at a given day.

        Args:
            body_name: Name of the celestial body
            day: Time in days since epoch

        Returns:
            (x, y) position relative to Earth at origin
        """
        body = self.bodies[body_name]

        # Calculate angle on deferent (main circle)
        deferent_angle = (2 * np.pi * day / body.deferent_period) + body.deferent_angle_offset

        # Position of epicycle center
        deferent_x = body.deferent_radius * np.cos(deferent_angle)
        deferent_y = body.deferent_radius * np.sin(deferent_angle)

        # Calculate angle on epicycle
        epicycle_angle = (2 * np.pi * day / body.epicycle_period) + body.epicycle_angle_offset

        # Position on epicycle relative to epicycle center
        epicycle_x = body.epicycle_radius * np.cos(epicycle_angle)
        epicycle_y = body.epicycle_radius * np.sin(epicycle_angle)

        # Final position is deferent center + epicycle offset
        x = deferent_x + epicycle_x
        y = deferent_y + epicycle_y

        return x, y

    def get_all_positions(self, day: float) -> Dict[str, Tuple[float, float]]:
        """Get positions of all celestial bodies at a given day."""
        return {name: self.get_position(name, day) for name in self.bodies.keys()}

    def get_celestial_sphere_radius(self) -> float:
        """Get the radius of the outermost sphere (for visualization bounds)."""
        max_radius = max(
            body.deferent_radius + body.epicycle_radius
            for body in self.bodies.values()
        )
        return max_radius * 1.2  # Add 20% padding
