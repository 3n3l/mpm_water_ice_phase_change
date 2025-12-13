from _common.solvers import BaseSolver
from _common.constants import State
from abc import ABC

import taichi as ti



@ti.data_oriented
class BasePoissonDiskSampler(ABC):
    def __init__(
        self,
        solver: BaseSolver,
        r: float = 0.002,
        k: int = 30,
    ) -> None:
        # Some of the solver's constants wills be used:
        self.solver = solver
        self.max_particles = solver.max_particles
        self.position_p = solver.position_p
        self.state_p = solver.state_p

        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(2)  # Cell size is bounded by this
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 5
        self.w_grid = self.n_grid + self.boundary_width + self.boundary_width
        self.w_offset = (-self.boundary_width, -self.boundary_width)
        # self.negative_boundary = -self.boundary_width
        # self.positive_boundary = self.n_grid + self.boundary_width

        # Initialize an n-dimension background grid to store samples:
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # We can't use a resizable list, so we point to the head and tail:
        # self._head = ti.field(int, shape=())
        self._head = self.solver.n_particles
        self._tail = ti.field(int, shape=())

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y = self.point_to_index(base_point)
        _min = (ti.max(0, x - 2), ti.min(self.n_grid, x + 3))  # pyright: ignore
        _max = (ti.max(0, y - 2), ti.min(self.n_grid, y + 3))  # pyright: ignore
        distance_min = ti.sqrt(2)  # Maximum possible distance

        # Search in a 3x3 grid neighborhood around the position
        # TODO: compute lower left as base like in mpm
        # TODO: check all distances against < self.r, return immediately
        for i, j in ti.ndrange(_min, _max):
            if (index := self.background_grid[i, j]) != -1:
                # We found a point and can compute the distance:
                found_point = self.position_p[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance

        return distance_min < self.r

    @ti.func
    def _in_bounds(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        in_bounds = 0.0 < point[0] < 1.0 and 0.0 < point[1] < 1.0  # in simulation bounds
        in_bounds &= geometry.in_bounds(point[0], point[1])  # in geometry bounds
        return in_bounds

    @ti.func
    def point_to_index(self, point: ti.template()) -> ti.Vector:  # pyright: ignore
        return ti.cast(point * self.n_grid, ti.i32)  # pyright: ignore

    @ti.func
    def point_fits(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        point_has_been_found = not self._has_collision(point)  # no collision
        point_has_been_found &= self._in_bounds(point, geometry)  # in bounds
        return point_has_been_found

    @ti.func
    def can_sample_more_points(self) -> bool:
        return (self._head[None] < self._tail[None]) and (self._head[None] < self.max_particles)

    @ti.func
    def initialize_grid(self, n_particles: ti.i32, positions: ti.template()):  # pyright: ignore
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            self.background_grid[i, j] = -1

        for p in ti.ndrange(n_particles):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            index = self.point_to_index(self.position_p[p])
            self.background_grid[index] = p

    @ti.func
    def initialize_pointers(self, n_particles: ti.i32):  # pyright: ignore
        self._tail[None] = n_particles + 1
        self._head[None] = n_particles

    @ti.func
    def generate_point_around(self, prev_position: ti.template()) -> ti.Vector:  # pyright: ignore
        theta = ti.random() * 2 * ti.math.pi
        offset = ti.Vector([ti.cos(theta), ti.sin(theta)])
        offset *= (1 + ti.random()) * self.r
        return prev_position + offset

    @ti.func
    def generate_initial_point(self, geometry: ti.template()) -> ti.Vector:  # pyright: ignore
        initial_point = geometry.random_seed()

        n_samples = 0  # otherwise this might not halt
        while not self.point_fits(initial_point, geometry) and n_samples < self.k:
            initial_point = geometry.random_seed()
            n_samples += 1

        index = self.point_to_index(initial_point)
        self.background_grid[index] = self._head[None]

        return initial_point

    @ti.kernel
    def add_geometry(self, geometry: ti.template()):  # pyright: ignore
        # Initialize background grid to the current positions:
        self.initialize_grid(self._head[None], self.position_p)

        # Update pointers, for a fresh sample this will be (0, 1), in the running simulation
        # this will reset this to where we left of, allowing to add more particles:
        self.initialize_pointers(self._head[None])

        # Find a good initial point for this sample run:
        initial_point = self.generate_initial_point(geometry)
        self.solver.add_particle(self._tail[None], initial_point, geometry)
        self._head[None] += 1
        self._tail[None] += 1

        while self.can_sample_more_points():
            prev_position = self.position_p[self._head[None]]
            self._head[None] += 1  # Increment on each iteration

            for _ in range(self.k):
                next_position = self.generate_point_around(prev_position)
                next_index = self.point_to_index(next_position)
                if self.point_fits(next_position, geometry):
                    self.background_grid[next_index] = self._tail[None]
                    self.solver.add_particle(self._tail[None], next_position, geometry)
                    self._tail[None] += 1  # Increment when point is found
