from _common.solvers import StaggeredSolver, PressureSolver
from _common.constants import State, Classification

import taichi as ti


@ti.data_oriented
class APIC(StaggeredSolver):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_grid, vol_0)

        # Properties on particles:
        self.cx_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.cy_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)

        # Offsets for weight computations:
        self.base_offset_x = ti.Vector([0.5, 1.0])
        self.base_offset_y = ti.Vector([1.0, 0.5])
        self.dist_offset_x = ti.Vector([0.0, 0.5])
        self.dist_offset_y = ti.Vector([0.5, 0.0])

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)

    @ti.kernel
    def reset_grids(self):
        for i, j in self.velocity_x:
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.velocity_y:
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_x), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_y), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - self.dist_offset_x
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - self.dist_offset_y

            # Quadratic kernels:
            w_x = self.compute_quadratic_kernel(dist_x)
            w_y = self.compute_quadratic_kernel(dist_y)

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                velocity_x, velocity_y = self.velocity_p[p][0], self.velocity_p[p][1]
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                offset = ti.Vector([i, j])
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx

                # Rasterize to cell faces:
                self.velocity_x[base_x + offset] += weight_x * (velocity_x + (self.cx_p[p] @ dpos_x))
                self.velocity_y[base_y + offset] += weight_y * (velocity_y + (self.cy_p[p] @ dpos_y))
                self.mass_x[base_x + offset] += weight_x
                self.mass_y[base_y + offset] += weight_y

    @ti.kernel
    def classify_cells(self):
        for i, j in self.classification_c:
            # Reset all the cells that don't belong to the colliding boundary:
            if not self.is_colliding(i, j):
                self.classification_c[i, j] = Classification.Empty

        for p in self.velocity_p:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Find the nearest cell and set it to interior:
            i, j = ti.floor(self.position_p[p] * self.inv_dx, dtype=ti.i32)  # pyright: ignore
            if not self.is_colliding(i, j):  # pyright: ignore
                self.classification_c[i, j] = Classification.Interior

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_x:
            if (mass := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass
                collision_right = i >= self.n_grid and self.velocity_x[i, j] > 0
                collision_left = i <= 0 and self.velocity_x[i, j] < 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            if (mass := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass
                self.velocity_y[i, j] += self.gravity[None] * self.dt[None]
                collision_top = j >= self.n_grid and self.velocity_y[i, j] > 0
                collision_bottom = j <= 0 and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0

    @ti.kernel
    def compute_volumes(self):
        # FIXME: this control volume doesn't help with the density correction
        control_volume = 0.5 * self.dx * self.dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.is_interior(i, j):
                self.volume_x[i + 1, j] += control_volume
                self.volume_x[i, j] += control_volume
                self.volume_y[i, j + 1] += control_volume
                self.volume_y[i, j] += control_volume

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_x), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_y), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - self.dist_offset_x
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - self.dist_offset_y

            # Quadratic kernels:
            w_x = self.compute_quadratic_kernel(dist_x)
            w_y = self.compute_quadratic_kernel(dist_y)

            next_velocity = ti.Vector.zero(ti.f32, 2)
            b_x = ti.Vector.zero(ti.f32, 2)
            b_y = ti.Vector.zero(ti.f32, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                x_weight = w_x[i][0] * w_x[j][1]
                y_weight = w_y[i][0] * w_y[j][1]
                offset = ti.Vector([i, j])
                dpos_x = ti.cast(offset, ti.f32) - dist_x
                dpos_y = ti.cast(offset, ti.f32) - dist_y
                x_velocity = x_weight * self.velocity_x[base_x + offset]
                y_velocity = y_weight * self.velocity_y[base_y + offset]
                next_velocity += [x_velocity, y_velocity]
                b_x += x_velocity * dpos_x
                b_y += y_velocity * dpos_y

            # We compute c_x, c_y from b_x, b_y as in https://doi.org/10.1016/j.jcp.2020.109311,
            # this avoids computing the weight gradients and results in less dissipation.
            # C = B @ (D^(-1)), NOTE: one inv_dx is cancelled with one dx in dpos.
            self.cx_p[p] = b_x * 4 * self.inv_dx
            self.cy_p[p] = b_y * 4 * self.inv_dx
            self.velocity_p[p] = next_velocity
            self.position_p[p] += self.dt[None] * next_velocity

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.color_p[index] = geometry.color
        self.phase_p[index] = geometry.phase
        self.position_p[index] = position

        # Set properties to default values:
        self.state_p[index] = State.Active
        self.cx_p[index] = 0
        self.cy_p[index] = 0

    def substep(self) -> None:
        self.reset_grids()
        self.particle_to_grid()
        self.classify_cells()
        self.momentum_to_velocity()
        self.compute_volumes()
        self.pressure_solver.solve()
        self.grid_to_particle()
