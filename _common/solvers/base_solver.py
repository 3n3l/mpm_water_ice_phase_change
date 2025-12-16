from _common.constants import Classification, State
from _common.configurations import Configuration
from abc import ABC, abstractmethod

import taichi as ti


@ti.data_oriented
class BaseSolver(ABC):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        self.max_particles = max_particles
        self.inv_dx = float(n_grid)
        self.n_grid = n_grid
        self.dx = 1 / n_grid
        self.vol_0_p = vol_0

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 5
        self.w_grid = self.n_grid + self.boundary_width + self.boundary_width
        self.w_offset = (-self.boundary_width, -self.boundary_width)
        self.negative_boundary = -self.boundary_width
        self.positive_boundary = self.n_grid + self.boundary_width

        # Variables accessed by kernels must be stored in fields:
        self.ambient_temperature = ti.field(dtype=ti.f32, shape=())
        self.boundary_temperature = ti.field(dtype=ti.f32, shape=())
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.gravity = ti.field(dtype=ti.f32, shape=())
        self.dt = ti.field(dtype=ti.f32, shape=())

        # Properties on cell centers:
        self.classification_c = ti.field(dtype=ti.i32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.temperature_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.velocity_c = ti.Vector.field(2, dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.mass_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Properties on particles:
        self.temperature_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.state_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.phase_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.f32, shape=max_particles)

        # Now we can initialize the colliding boundary (or bounding box) around the domain:
        self.initialize_boundary()

    @ti.func
    def is_valid(self, i: int, j: int) -> bool:
        in_horizontal_bounds = i >= self.negative_boundary and i < self.positive_boundary
        in_vertical_bounds = j >= self.negative_boundary and j < self.positive_boundary
        return in_horizontal_bounds and in_vertical_bounds

    @ti.func
    def is_colliding(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Colliding

    @ti.func
    def is_insulated(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Insulated

    @ti.func
    def is_interior(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Interior

    @ti.func
    def is_empty(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Empty

    @ti.kernel
    def initialize_boundary(self):
        for i, j in self.classification_c:
            is_colliding = not (0 <= i < self.n_grid)
            is_colliding |= not (0 <= j < self.n_grid)
            if is_colliding:
                self.classification_c[i, j] = Classification.Colliding
            else:
                self.classification_c[i, j] = Classification.Empty

    def reset(self, configuration: Configuration):
        self.boundary_temperature[None] = configuration.boundary_temperature
        self.ambient_temperature[None] = configuration.ambient_temperature
        self.gravity[None] = configuration.gravity
        self.dt[None] = configuration.dt
        self.state_p.fill(State.Hidden)
        self.position_p.fill([42, 42])
        self.n_particles[None] = 0

    @abstractmethod
    def substep(self):
        pass

    @abstractmethod
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        pass
