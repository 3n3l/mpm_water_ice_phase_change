from _common.configurations import Circle, Rectangle, Configuration
from _common.constants import Ice, Water

configuration_list = [
    Configuration(
        name="Snowball Smashes Into Wall",
        information="Ice",
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(4, 0),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Snowball Smashes Into Snowball",
        information="Ice",
        geometries=[
            Circle(
                center=(0.15, 0.55),
                velocity=(5, 0),
                material=Ice,  # pyright: ignore
                radius=0.07,
            ),
            Circle(
                center=(0.85, 0.5),
                velocity=(-10, 0),
                material=Ice,  # pyright: ignore
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowballs Smash Into Wall",
        information="Ice",
        geometries=[
            *[
                Circle(
                    velocity=(5, 0),
                    center=(0.25, 0.2 + (i * 0.005)),
                    radius=0.05,
                    temperature=20.0,
                    frame_threshold=i,
                    material=Ice,  # pyright: ignore
                )
                for i in range(10, 110, 25)
            ],
        ],
        ambient_temperature=-20.0,
    ),
    Configuration(
        name="Floating, Melting Ice Cube",
        information="Ice -> Water",
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.425, 0.425),
                temperature=-10.0,
            )
        ],
        ambient_temperature=100.0,
        gravity=0.0,
    ),
    Configuration(
        name="Floating, Melting Ice Ball",
        information="Ice -> Water",
        geometries=[
            Circle(
                velocity=(0, 0),
                center=(0.5, 0.5),
                radius=0.1,
                temperature=-10.0,
                material=Ice,  # pyright: ignore
            )
        ],
        ambient_temperature=100.0,
        gravity=0.0,
    ),
    Configuration(
        name="Melting Ice Cube",
        information="Ice -> Water",
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.2, 0.21),  # a bit higher to fit the grid better
                velocity=(0, 0),
                lower_left=(0.4, 0.0),
                temperature=-10.0,
            ),
        ],
        ambient_temperature=100.0,
    ),
    Configuration(
        name="Waterjet & Lake",
        information="Water",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Rectangle(
                    material=Water,  # pyright: ignore
                    size=(0.04, 0.04),
                    velocity=(0, -3),
                    lower_left=(0.48, 0.48),
                    frame_threshold=i,
                    temperature=20.0,
                )
                for i in range(1, 300)
            ],
        ],
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Dam Break",
        information="Water",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(0.3, 0.3),
                velocity=(0, 0),
            ),
        ],
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Dam Break, Centered",
        information="Water",
        geometries=[
            Rectangle(
                lower_left=(0.35, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(0.3, 0.3),
                velocity=(0, 0),
            ),
        ],
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Lake",
        information="Water",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(1.0, 0.15),
                velocity=(0, 0),
            ),
        ],
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Ice Cubes Dropped Into Lake",
        information="Water, Ice -> Water",
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                size=(1.0, 0.1),
                velocity=(0, 0),
                lower_left=(0, 0),
                frame_threshold=0,
                temperature=50.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.25, 0.35),
                frame_threshold=10,
                temperature=-30.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.45, 0.15),
                frame_threshold=20,
                temperature=-30.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.65, 0.25),
                frame_threshold=30,
                temperature=-30.0,
            ),
        ],
        ambient_temperature=0.0,
    ),
    Configuration(
        name="Freezing Lake",
        information="Water -> Ice",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(1.0, 0.15),
                velocity=(0, 0),
            ),
        ],
        ambient_temperature=-500.0,
    ),
    Configuration(
        name="Waterjet & Ice Cubes",
        information="Water, Ice -> Water",
        ambient_temperature=5.0,
        geometries=[
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    center=(0.04, 0.96),
                    temperature=80.0,
                    velocity=(4, -3),
                    radius=0.025,
                    frame_threshold=i,
                )
                for i in range(1, 300)
            ],
            Rectangle(  # BL
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.6, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # BM
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.69, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # BR
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.78, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # ML
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.645, 0.09),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # MR
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.735, 0.09),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # TM
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.69, 0.18),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        name="Waterjet & Snowballs",
        information="Water, Ice -> Water",
        geometries=[
            *[
                Rectangle(
                    lower_left=(0.47, 0.9),
                    material=Water,  # pyright: ignore
                    temperature=20.0,
                    velocity=(0, -3),
                    size=(0.06, 0.06),
                    frame_threshold=i,
                )
                for i in range(300)
            ],
            *[
                Circle(
                    velocity=(5, 0),
                    center=(0.25, 0.4 + (i * 0.05)),
                    radius=0.05,
                    temperature=20.0,
                    frame_threshold=(i + 1) * 50,
                    material=Ice,  # pyright: ignore
                )
                for i in range(3)
            ],
        ],
    ),
    Configuration(
        name="Waterjet",
        information="Water",
        geometries=[
            Rectangle(
                lower_left=(0.47, 0.9),
                material=Water,  # pyright: ignore
                temperature=20.0,
                velocity=(0, -3),
                size=(0.06, 0.06),
                frame_threshold=i,
            )
            for i in range(1, 300)
        ],
    ),
    Configuration(
        name="Spherefall",
        information="Water",
        geometries=[
            Circle(
                center=(0.5, 0.5),
                material=Water,  # pyright: ignore
                velocity=(0, -3),
                radius=0.1,
            )
        ],
        ambient_temperature=50.0,
    ),
    Configuration(
        name="Spherefall",
        information="Ice",
        geometries=[
            Circle(
                center=(0.5, 0.4),
                velocity=(0, -4),
                material=Ice,  # pyright: ignore
                radius=0.1,
            ),
        ],
        ambient_temperature=-50.0,
    ),
    Configuration(
        name="Spherefall",
        information="Water, Ice",
        geometries=[
            Circle(
                center=(0.25, 0.25),
                velocity=(0, -6),
                temperature=-100.0,
                material=Ice,  # pyright: ignore
                radius=0.1,
            ),
            Circle(
                center=(0.75, 0.25),
                velocity=(0, -6),
                temperature=100.0,
                material=Water,  # pyright: ignore
                radius=0.1,
            ),
        ],
        ambient_temperature=0.0,
    ),
]

# Sort alphabetically:
configuration_list.sort(key=lambda c: str.lower(c.name), reverse=False)

# Add indices:
max_length = len(max(configuration_list, key=lambda c: len(c.name)).name)
for i, c in enumerate(configuration_list):
    c.name = f"{f' ({i})':5s} {configuration_list[i].name:{max_length}s} [{configuration_list[i].information}]"
