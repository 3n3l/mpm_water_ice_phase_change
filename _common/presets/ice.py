from _common.configurations import Circle, Configuration, Rectangle
from _common.constants import Ice

ice_presets = [
    Configuration(
        name="Spherefall, Ice",
        information="Ice",
        gravity = -9.81,
        dt = 1e-4,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.5, 0.4),
                velocity=(0, -2),
                temperature=-100.0,
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Dropping Cube, Ice",
        dt=1e-4,
        gravity=-9.81,
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.425, 0.425),
                temperature=-10.0,
            )
        ],
    ),
]
