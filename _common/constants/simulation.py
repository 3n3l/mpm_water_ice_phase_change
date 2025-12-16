from dataclasses import dataclass


@dataclass
class Classification:
    Empty = 22
    Colliding = 33
    Interior = 44
    Insulated = 55



@dataclass
class State:
    Active = 0
    Hidden = 1


@dataclass
class Simulation:
    """Defines parameters for the simulation."""

    MininumTemperature = -273.15
    MaximumTemperature = 100.0
