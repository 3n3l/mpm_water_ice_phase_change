from _common.constants import Classification, State, Water
from _common.solvers import PressureSolver, StaggeredSolver
from typing import override

import taichi as ti


@ti.data_oriented
class APIC(StaggeredSolver):
    def __init__(self, coupled_solver):
        super().__init__(coupled_solver.max_particles, coupled_solver.n_grid, coupled_solver.vol_0_p)

        # TODO: maybe make all of these fields optional in the base classes and then just pass them from here?
        #       this might not only be more aesthetic, but also more efficient

        # Properties on particles.
        self.position_p = coupled_solver.position_p
        self.velocity_p = coupled_solver.velocity_p
        self.phase_p = coupled_solver.phase_p
        self.state_p = coupled_solver.state_p
        self.mass_p = coupled_solver.mass_p
        self.cx_p = coupled_solver.cx_p
        self.cy_p = coupled_solver.cy_p

        self.n_particles = coupled_solver.n_particles
        self.gravity = coupled_solver.gravity
        self.dt = coupled_solver.dt

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)

        # Set the initial boundary:
        self.initialize_boundary()

    @ti.kernel
    def reset_grids(self):
        for i, j in self.mass_x:
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.mass_y:
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

        for i, j in self.mass_c:
            self.mass_c[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        # NOTE: particles are sorted: [water | ice | uninitialized]
        # for p in ti.ndrange(self.water_divider[None]):
        for p in ti.ndrange(self.n_particles[None]):
            # TODO: this check should not be needed anymore after sorting:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue
            if self.phase_p[p] != Water.Phase:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.0])

            # Quadratic kernels:
            w_x = self.compute_quadratic_kernel(dist_x)
            w_y = self.compute_quadratic_kernel(dist_y)
            w_c = self.compute_quadratic_kernel(dist_c)

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                velocity_x, velocity_y = self.velocity_p[p][0], self.velocity_p[p][1]
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                weight_c = w_c[i][0] * w_c[j][1]
                offset = ti.Vector([i, j])
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx

                # Rasterize to cell centers:
                self.mass_c[base_c + offset] += weight_c * self.mass_p[p]

                # Rasterize to cell faces:
                self.velocity_x[base_x + offset] += weight_x * (velocity_x + (self.cx_p[p] @ dpos_x))
                self.velocity_y[base_y + offset] += weight_y * (velocity_y + (self.cy_p[p] @ dpos_y))
                self.mass_x[base_x + offset] += weight_x
                self.mass_y[base_y + offset] += weight_y

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.mass_x:
            if (mass_x := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass_x
                # Everything outside the visible grid belongs to the simulation boundary,
                # we enforce a free-slip boundary condition by allowing separation.
                if (i >= self.n_grid and self.velocity_x[i, j] > 0) or (i <= 0 and self.velocity_x[i, j] < 0):
                    self.velocity_x[i, j] = 0

        for i, j in self.mass_y:
            if (mass_y := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass_y
                self.velocity_y[i, j] += self.gravity[None] * self.dt[None]
                # Everything outside the visible grid belongs to the simulation boundary,
                # we enforce a free-slip boundary condition by allowing separation.
                if (j >= self.n_grid and self.velocity_y[i, j] > 0) or (j <= 0 and self.velocity_y[i, j] < 0):
                    self.velocity_y[i, j] = 0

    @ti.kernel
    def classify_cells(self):
        # TODO: replace this with phase field entirely?
        for i, j in self.classification_c:
            # Reset all the cells that don't belong to the colliding boundary:
            if not self.is_colliding(i, j):
                self.classification_c[i, j] = Classification.Empty

        for p in self.velocity_p:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue
            if self.phase_p[p] != Water.Phase:
                continue

            # Find the nearest cell and set it to interior:
            i, j = ti.floor(self.position_p[p] * self.inv_dx, dtype=ti.i32)  # pyright: ignore
            if not self.is_colliding(i, j):  # pyright: ignore
                self.classification_c[i, j] = Classification.Interior

    @ti.kernel
    def compute_volumes(self):
        # FIXME: this seems to be wrong, the paper has a sum over CDFs
        control_volume = 0.5 * self.dx * self.dx
        for i, j in self.classification_c:
            if self.classification_c[i, j] == Classification.Interior:
                self.volume_x[i + 1, j] += control_volume
                self.volume_y[i, j + 1] += control_volume
                self.volume_x[i, j] += control_volume
                self.volume_y[i, j] += control_volume

    @override
    def substep(self):
        self.reset_grids()
        self.particle_to_grid()
        self.momentum_to_velocity()
        self.classify_cells()
        self.compute_volumes()
        self.pressure_solver.solve()
