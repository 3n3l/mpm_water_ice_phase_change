from _common.configurations import Circle, Rectangle, Configuration
from _common.constants import Water

configuration_list = [
    Configuration(
        name="Waterjet Hits Pool",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Rectangle(
                    lower_left=(0.47, 0.9),
                    material=Water,  # pyright: ignore
                    velocity=(0, -2),
                    size=(0.06, 0.06),
                    frame_threshold=i,
                )
                for i in range(1, 200)
            ],
        ],
    ),
    Configuration(
        name="Dam Break",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Centered Dam Break",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.35, 0.0),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Waterjet",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.47, 0.9),
                velocity=(0, -2),
                size=(0.06, 0.06),
                frame_threshold=i,
            )
            for i in range(1, 200)
        ],
    ),
    Configuration(
        name="Spherefall",
        geometries=[
            Circle(
                material=Water,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(0, -1),
                radius=0.1,
            ),
        ],
    ),
    Configuration(
        name="Stationary Pool",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(1.0, 0.25),
                velocity=(0, 0),
            ),
        ],
    ),
]

# Sort alphabetically:
configuration_list.sort(key=lambda c: str.lower(c.name), reverse=False)
