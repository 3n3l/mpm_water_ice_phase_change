from _common.constants import Ice, Snow, PurpleSnow, MagentaSnow
from _common.configurations import Circle, Configuration

snow_presets = [
    Configuration(
        name="Spherefall, Snow",
        information="Snow",
        gravity=-9.81,
        dt=1e-4,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                temperature=-100.0,
                velocity=(0, -2),
                center=(0.5, 0.4),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        dt=1e-4,
        name="Spherefall, Snow vs. Ice",
        information="Water, Ice",
        ambient_temperature=0.0,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.25, 0.4),
                velocity=(0, -2),
                temperature=-100.0,
                radius=0.08,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.75, 0.4),
                velocity=(0, -2),
                temperature=-100.0,
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Snowball Hits Wall",
        information="Snow",
        dt=1e-4,
        gravity=-9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(4, 0),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Snowball Hits Snowball",
        information="Snow",
        dt=1e-4,
        gravity=-9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(3, 0),
                radius=0.08,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.9, 0.56),
                velocity=(-6, 0),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Snowball Hits Snowball, Colored",
        information="Snow",
        dt=1e-4,
        gravity=-9.81,
        geometries=[
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(3, 0),
                radius=0.08,
            ),
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.9, 0.56),
                velocity=(-6, 0),
                radius=0.08,
            ),
        ],
    ),
]
