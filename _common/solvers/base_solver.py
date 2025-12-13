from _common.constants import Classification
from abc import ABC, abstractmethod

import taichi as ti


@ti.data_oriented
class BaseSolver(ABC):
    def __init__(self, max_particles: int, n_grid: int, dt: float):
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        # self.max_particles = max_particles
        self.n_grid = n_grid
        # self.n_cells = self.n_grid * self.n_grid
        # self.dx = 1 / self.n_grid
        # self.inv_dx = float(self.n_grid)
        # self.vol_0_p = (self.dx * 0.5) ** 2
        # self.n_dimensions = 2
        # self.dt = dt

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.w_grid = self.n_grid + self.boundary_width + self.boundary_width
        self.w_offset = (-self.boundary_width, -self.boundary_width)
        self.negative_boundary = -self.boundary_width
        self.positive_boundary = self.n_grid + self.boundary_width

        # Properties on cell centers:
        self.classification_c = ti.field(dtype=ti.i32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Properties on particles:
        self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.state_p = ti.field(dtype=ti.f32, shape=max_particles)

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

    @abstractmethod
    def substep(self):
        pass
