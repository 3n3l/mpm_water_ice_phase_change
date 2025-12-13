from _common.samplers import BasePoissonDiskSampler
from _common.constants import State

from apic import APIC

import taichi as ti


@ti.data_oriented
class PoissonDiskSampler(BasePoissonDiskSampler):
    def __init__(self, apic_solver: APIC, r: float = 0.0025, k: int = 300) -> None:
        super().__init__(apic_solver.position_p, apic_solver.state_p, apic_solver.max_particles, r, k)

        # The head points to the last found position, this is the updated number of particles,
        # the solver needs this to keep track of all particles in the simulation.
        apic_solver.n_particles = self._head

        # Particle properties, used to seed new particles:
        self.position_p = apic_solver.position_p
        self.velocity_p = apic_solver.velocity_p
        self.state_p = apic_solver.state_p
        self.color_p = apic_solver.color_p
        self.cx_p = apic_solver.cx_p
        self.cy_p = apic_solver.cy_p

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position
        self.color_p[index] = geometry.color

        # Set properties to default values:
        self.state_p[index] = State.Active
        self.cx_p[index] = 0
        self.cy_p[index] = 0
