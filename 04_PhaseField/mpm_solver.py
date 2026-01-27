from _common.constants import Classification, State, Water, Ice
from _common.solvers import StaggeredSolver
from typing import override

import taichi as ti


@ti.data_oriented
class MPM(StaggeredSolver):
    def __init__(self, coupled_solver):
        super().__init__(coupled_solver.max_particles, coupled_solver.n_grid, coupled_solver.vol_0_p)

        # Properties on MAC-cells.
        self.JE_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.JP_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Properties on particles.
        self.position_p = coupled_solver.position_p
        self.velocity_p = coupled_solver.velocity_p
        self.phase_p = coupled_solver.phase_p
        self.state_p = coupled_solver.state_p
        self.mass_p = coupled_solver.mass_p
        self.cx_p = coupled_solver.cx_p
        self.cy_p = coupled_solver.cy_p

        self.theta_c_p = coupled_solver.theta_c_p
        self.theta_s_p = coupled_solver.theta_s_p
        self.lambda_p = coupled_solver.lambda_p
        self.zeta_p = coupled_solver.zeta_p
        self.mu_p = coupled_solver.mu_p
        self.FE_p = coupled_solver.FE_p
        self.JE_p = coupled_solver.JE_p
        self.JP_p = coupled_solver.JP_p
        self.C_p = coupled_solver.C_p

        self.n_particles = coupled_solver.n_particles
        self.gravity = coupled_solver.gravity
        self.dt = coupled_solver.dt

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
            self.JE_c[i, j] = 0
            self.JP_c[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        # NOTE: particles are sorted: [water | ice | uninitialized]
        # TODO: water_divider + 1?
        # for p in ti.ndrange(self.water_divider[None], self.ice_divider[None]):
        for p in ti.ndrange(self.n_particles[None]):
            # TODO: this check should not be needed anymore after sorting:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue
            if self.phase_p[p] != Ice.Phase:
                continue

            # Update deformation gradient:
            self.FE_p[p] += (self.dt[None] * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Clamp singular values to simulate plasticity and elasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            self.JE_p[p] = 1.0
            for d in ti.static(range(2)):
                singular_value = ti.f32(sigma[d, d])
                # Clamp singular values to [1 - theta_c, 1 + theta_s]
                # TODO: use this when everything else works
                # clamped = tm.clamp(singular_value, 1 - self.theta_c_p[p], 1 + self.theta_s_p[p])
                clamped = max(singular_value, 1 - self.theta_c_p[p])
                clamped = min(clamped, 1 + self.theta_s_p[p])
                self.JP_p[p] *= singular_value / clamped
                self.JE_p[p] *= clamped
                sigma[d, d] = clamped

            # Reconstruct elastic deformation gradient after plasticity
            self.FE_p[p] = U @ sigma @ V.transpose()

            # Apply ice hardening by adjusting Lame parameters:
            # TODO: this no longer needs fields, as this is now ice only
            hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta_p[p] * (1.0 - self.JP_p[p]))))
            la, mu = self.lambda_p[p] * hardening, self.mu_p[p] * hardening

            # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            piola_kirchhoff = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
            piola_kirchhoff += ti.Matrix.identity(float, 2) * la * self.JE_p[p] * (self.JE_p[p] - 1)

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 4 * self.inv_dx * self.inv_dx  # quadratic interpolation

            # Cauchy stress times dt and D_inv:
            cauchy_stress = -self.dt[None] * self.vol_0_p * D_inv * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore

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

            for i, j in ti.static(ti.ndrange(3, 3)):
                velocity_x, velocity_y = self.velocity_p[p][0], self.velocity_p[p][1]
                weight_c = w_c[i][0] * w_c[j][1]
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                offset = ti.Vector([i, j])
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx
                mass = self.mass_p[p]

                # Rasterize to cell centers:
                self.temperature_c[base_c + offset] += weight_c * mass * self.temperature_p[p]
                # self.inv_lambda_c[base_c + offset] += weight_c * (mass / la)
                # self.capacity_c[base_c + offset] += weight_c * mass * self.capacity_p[p]
                # self.mass_c[base_c + offset] += weight_c * self.mass_p[p]
                self.JE_c[base_c + offset] += weight_c * mass * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * mass * self.JP_p[p]

                # Rasterize to cell faces:
                self.velocity_x[base_x + offset] += weight_x * (mass * velocity_x + affine_x @ dpos_x)
                self.velocity_y[base_y + offset] += weight_y * (mass * velocity_y + affine_y @ dpos_y)
                self.mass_x[base_x + offset] += weight_x * mass
                self.mass_y[base_y + offset] += weight_y * mass

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.mass_x:
            if (ice_mass_x := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= ice_mass_x
                # Everything outside the visible grid belongs to the simulation boundary,
                # we enforce a free-slip boundary condition by allowing separation.
                if (i >= self.n_grid and self.velocity_x[i, j] > 0) or (i <= 0 and self.velocity_x[i, j] < 0):
                    self.velocity_x[i, j] = 0

        for i, j in self.mass_y:
            if (ice_mass_y := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= ice_mass_y
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
    def couple_materials(self):
        for i, j in self.velocity_x:
            combined_mass = self.mass_x[i, j] + self.mass_x[i, j]
            if combined_mass > 0:
                combined_velocity = self.mass_x[i, j] * self.velocity_x[i, j]
                combined_velocity += self.mass_x[i, j] * self.velocity_x[i, j]
                self.velocity_x[i, j] = combined_velocity / combined_mass
                # if (i >= self.n_grid and self.c_velocity_x[i, j] > 0) or (i <= 0 and self.c_velocity_x[i, j] < 0):
                #     self.c_velocity_x[i, j] = 0
        for i, j in self.velocity_y:
            combined_mass = self.mass_y[i, j] + self.mass_y[i, j]
            if combined_mass > 0:
                combined_velocity = self.mass_y[i, j] * self.velocity_y[i, j]
                combined_velocity += self.mass_y[i, j] * self.velocity_y[i, j]
                self.velocity_y[i, j] = combined_velocity / combined_mass
                # if (j >= self.n_grid and self.c_velocity_y[i, j] > 0) or (j <= 0 and self.c_velocity_y[i, j] < 0):
                #     self.c_velocity_y[i, j] = 0

    @override
    def substep(self):
        self.reset_grids()
        self.particle_to_grid()
        self.momentum_to_velocity()
