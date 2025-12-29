"""
Animated visualization of the geocentric model
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from geocentric_model import GeocentricModel


class GeocentricVisualizer:
    """Creates an animated 2D visualization of the geocentric model."""

    def __init__(self, model: GeocentricModel, days_per_frame: float = 1.0):
        """
        Initialize the visualizer.

        Args:
            model: The geocentric model to visualize
            days_per_frame: How many days advance per animation frame
        """
        self.model = model
        self.days_per_frame = days_per_frame
        self.current_day = 0.0

        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 12), facecolor='black')
        self.ax.set_facecolor('black')

        # Set up the coordinate system
        sphere_radius = self.model.get_celestial_sphere_radius()
        self.ax.set_xlim(-sphere_radius, sphere_radius)
        self.ax.set_ylim(-sphere_radius, sphere_radius)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2, color='white')

        # Draw Earth at the center
        earth = Circle((0, 0), 0.2, color='blue', label='Earth', zorder=10)
        self.ax.add_patch(earth)

        # Draw celestial sphere
        celestial_sphere = Circle((0, 0), sphere_radius, fill=False,
                                   edgecolor='darkblue', linestyle='--',
                                   alpha=0.3, linewidth=1)
        self.ax.add_patch(celestial_sphere)

        # Initialize planet markers and trails
        self.planet_markers = {}
        self.planet_trails = {}
        self.trail_length = 200  # Number of points in trail

        for body_name, body in self.model.bodies.items():
            # Create marker for planet
            marker, = self.ax.plot([], [], 'o', color=body.color,
                                   markersize=8, label=body_name, zorder=5)
            self.planet_markers[body_name] = marker

            # Create trail line
            trail, = self.ax.plot([], [], '-', color=body.color,
                                 alpha=0.3, linewidth=1, zorder=1)
            self.planet_trails[body_name] = {
                'line': trail,
                'x_data': [],
                'y_data': []
            }

        # Add legend
        self.ax.legend(loc='upper right', facecolor='black',
                      edgecolor='white', labelcolor='white')

        # Add title and day counter
        self.title = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                 color='white', fontsize=14,
                                 verticalalignment='top')

        # Style improvements
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.tick_params(colors='white')

    def init_animation(self):
        """Initialize animation (required by FuncAnimation)."""
        for marker in self.planet_markers.values():
            marker.set_data([], [])
        for trail_data in self.planet_trails.values():
            trail_data['line'].set_data([], [])
        self.title.set_text('')
        return list(self.planet_markers.values()) + \
               [trail['line'] for trail in self.planet_trails.values()] + \
               [self.title]

    def update_frame(self, frame):
        """Update function for animation."""
        self.current_day += self.days_per_frame

        # Update planet positions and trails
        positions = self.model.get_all_positions(self.current_day)

        for body_name, (x, y) in positions.items():
            # Update marker position
            self.planet_markers[body_name].set_data([x], [y])

            # Update trail
            trail = self.planet_trails[body_name]
            trail['x_data'].append(x)
            trail['y_data'].append(y)

            # Keep trail length limited
            if len(trail['x_data']) > self.trail_length:
                trail['x_data'].pop(0)
                trail['y_data'].pop(0)

            trail['line'].set_data(trail['x_data'], trail['y_data'])

        # Update title with current day
        self.title.set_text(f'Day: {self.current_day:.1f}\n' +
                           'Geocentric Model (Ptolemaic System)')

        return list(self.planet_markers.values()) + \
               [trail['line'] for trail in self.planet_trails.values()] + \
               [self.title]

    def animate(self, total_days: int = 365, interval: int = 20):
        """
        Create and display the animation.

        Args:
            total_days: Total number of days to simulate
            interval: Milliseconds between frames
        """
        frames = int(total_days / self.days_per_frame)

        anim = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=frames,
            interval=interval,
            blit=True,
            repeat=True
        )

        plt.tight_layout()
        return anim

    def save_animation(self, filename: str, total_days: int = 365,
                       fps: int = 30, dpi: int = 100):
        """
        Save animation to file.

        Args:
            filename: Output filename (e.g., 'planets.mp4' or 'planets.gif')
            total_days: Total number of days to simulate
            fps: Frames per second
            dpi: Dots per inch resolution
        """
        anim = self.animate(total_days, interval=1000/fps)

        print(f"Saving animation to {filename}...")
        anim.save(filename, fps=fps, dpi=dpi)
        print("Animation saved!")


def main():
    """Main function to run the visualization."""
    # Create the model
    model = GeocentricModel()

    # Create visualizer (1 day per frame)
    visualizer = GeocentricVisualizer(model, days_per_frame=2.0)

    # Animate for one Earth year
    print("Starting animation... Close the window to exit.")
    print("The animation will loop continuously.")
    anim = visualizer.animate(total_days=730, interval=20)  # 2 years

    # Optional: Save animation (uncomment to save)
    # visualizer.save_animation('geocentric_motion.mp4', total_days=365, fps=30)

    plt.show()


if __name__ == '__main__':
    main()
