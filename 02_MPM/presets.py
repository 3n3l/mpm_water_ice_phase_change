from _common.configurations import Circle, Configuration, Rectangle
from _common.constants import Ice, Snow, PurpleSnow, MagentaSnow

# Width of the bounding box, TODO: transform points to coordinates in bounding box
configuration_list = [
    Configuration(
        name="Snowball hits wall [1]",
        E=2.8e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=3.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
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
        name="Snowball hits ground [1]",
        E=2.8e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(0, -3),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Ice Cube Fall",
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
    Configuration(
        name="Ice Cube vs. Snow Cube",
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.225, 0.425),
                temperature=-10.0,
            ),
            Rectangle(
                material=Snow,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.625, 0.425),
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball [1]",
        E=2.8e5,  # Young's modulus (1.4e5)
        nu=0.25,  # Poisson's ratio (0.2)
        zeta=8,  # Hardening coefficient (10)
        theta_c=3.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.07, 0.595),
                velocity=(3, 0),
                radius=0.04,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.91, 0.615),
                velocity=(-3, 0),
                radius=0.06,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored) [1]",
        E=2.8e5,  # Young's modulus (1.4e5)
        nu=0.25,  # Poisson's ratio (0.2)
        zeta=8,  # Hardening coefficient (10)
        theta_c=3.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.07, 0.595),
                velocity=(3, 0),
                radius=0.04,
            ),
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.91, 0.615),
                velocity=(-3, 0),
                radius=0.06,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball [2]",
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=5,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=4.0e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(4, 0),
                radius=0.07,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.9, 0.53),
                velocity=(-8, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored) [2]",
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=5,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=4.0e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(4, 0),
                radius=0.07,
            ),
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.9, 0.53),
                velocity=(-8, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (high velocity) [3]",
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=5,  # Hardening coefficient (10)
        theta_c=4.5e-2,  # Critical compression (2.5e-2)
        theta_s=4.0e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(7, 0),
                radius=0.07,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.9, 0.51),
                velocity=(-7, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored, high velocity) [3]",
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=5,  # Hardening coefficient (10)
        theta_c=4.5e-2,  # Critical compression (2.5e-2)
        theta_s=4.0e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(7, 0),
                radius=0.07,
            ),
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.9, 0.51),
                velocity=(-7, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits giant snowball",
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=5,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=4.0e-3,  # Critical stretch (7.5e-3)
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(10, 0),
                radius=0.05,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.79, 0.51),
                velocity=(-1, 0),
                radius=0.15,
            ),
        ],
    ),
]

# Sort by length in descending order:
# TODO: move sorting an stuff to BaseSimuluation or something
# configuration_list.sort(key=lambda c: len(c.name), reverse=True)
