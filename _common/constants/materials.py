from _common.constants.colors import ColorRGB
from dataclasses import dataclass


@dataclass
class Material:
    """Defines parameters that represent a material."""

    Conductivity: float = 0.0
    LatentHeat: float = 0.0
    Capacity: float = 0.0
    Density: float = 0.0
    Lambda: float = 0.0
    Color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    Phase: int = 0
    Mu: float = 0.0

    # TODO: proper default variables?!
    E = 1.0
    nu = 1.0
    Zeta: int = 10
    Theta_s: float = 0.0
    Theta_c: float = 0.0


@dataclass
class Water(Material):
    Conductivity = 0.55
    LatentHeat = 334.4
    Capacity = 4.186  # j/dC
    Density = 997.0
    Lambda = 1e32
    Color = ColorRGB.Water
    Phase = 43
    Mu = 0


# TODO: the materials should also define theta_s, theta_c, zeta
#       as this differentiates the ice from snow, but then
#       a multi-material solver would also need to save these
#       per particle


@dataclass
class Ice(Material):
    E = 5.8e5
    nu = 0.1
    Conductivity = 2.33
    LatentHeat = 0.0
    Capacity = 2.093  # j/dC
    Density = 400.0
    Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    Color = ColorRGB.Ice
    Phase = 86
    Mu = E / (2 * (1 + nu))

    Zeta: int = 12
    Theta_s: float = 7.5e-3
    Theta_c: float = 2.5e-2


@dataclass
class Snow(Material):
    E = 1.4e5
    nu = 0.3
    Conductivity = 2.33
    LatentHeat = 0.0
    Capacity = 2.093  # j/dC
    Density = 400.0
    Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    Color = ColorRGB.Ice
    Phase = 129
    Mu = E / (2 * (1 + nu))

    Zeta: int = 8
    Theta_s: float = 6.5e-3
    Theta_c: float = 2.5e-2


@dataclass
class PurpleSnow(Snow):
    Color = ColorRGB.Purple


@dataclass
class MagentaSnow(Snow):
    Color = ColorRGB.Magenta
