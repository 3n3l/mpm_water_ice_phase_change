from _common.constants import Classification, State, Water, Ice, Simulation
from solvers import PressureSolver, HeatSolver
from _common.solvers import BaseSolver
from typing import override

import taichi as ti


@ti.data_oriented
class AugmentedMPM(BaseSolver):
    def __init__(self, max_particles: int, n_grid: int, dt: float):
        super().__init__(max_particles, n_grid, dt)

        # Properties on MAC-faces.
        self.classification_x = ti.field(dtype=ti.i32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.classification_y = ti.field(dtype=ti.i32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.velocity_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.velocity_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.volume_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.volume_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.mass_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.mass_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)

        # Properties on MAC-cells.
        self.inv_lambda_c = ti.field(dtype=ti.f64, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.JE_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.JP_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.J_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Properties on particles.
        self.conductivity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_c_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_s_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.lambda_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.zeta_p = ti.field(dtype=ti.i32, shape=max_particles)
        self.FE_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.JE_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.nu_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mu_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.E_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.J_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.C_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.latent_heat_p = ti.field(dtype=ti.f32, shape=max_particles)  # U_p

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)
        self.heat_solver = HeatSolver(self)

        # Set the initial boundary:
        self.initialize_boundary()

    @ti.func
    @override
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.conductivity_p[index] = geometry.conductivity
        self.theta_c_p[index] = geometry.material.Theta_c
        self.theta_s_p[index] = geometry.material.Theta_s
        self.latent_heat_p[index] = geometry.latent_heat
        self.temperature_p[index] = geometry.temperature
        self.lambda_p[index] = geometry.material.Lambda
        self.zeta_p[index] = geometry.material.Zeta
        self.capacity_p[index] = geometry.capacity
        self.velocity_p[index] = geometry.velocity
        self.mu_p[index] = geometry.material.Mu
        self.nu_p[index] = geometry.material.nu
        self.E_p[index] = geometry.material.E
        self.color_p[index] = geometry.color
        self.phase_p[index] = geometry.phase
        self.position_p[index] = position

        # Set properties to default values:
        self.mass_p[index] = self.vol_0_p * geometry.density
        self.FE_p[index] = ti.Matrix([[1, 0], [0, 1]])
        self.C_p[index] = ti.Matrix.zero(float, 2, 2)
        self.state_p[index] = State.Active
        self.JE_p[index] = 1.0
        self.JP_p[index] = 1.0

    @ti.kernel
    def reset_grids(self):
        for i, j in self.classification_x:
            self.conductivity_x[i, j] = 0
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.classification_y:
            self.conductivity_y[i, j] = 0
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

        for i, j in self.classification_c:
            self.temperature_c[i, j] = 0
            self.inv_lambda_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.mass_c[i, j] = 0
            self.JE_c[i, j] = 0
            self.JP_c[i, j] = 0
            self.J_c[i, j] = 0

    @ti.func
    def R(self, M: ti.types.matrix(2, 2, float)) -> ti.types.matrix(2, 2, float):  # pyright: ignore
        # TODO: this might not be needed, as the timestep is so small for explicit MPM anyway
        result = ti.Matrix.identity(float, 2) + M
        while ti.math.determinant(result) < 0:
            result = (ti.Matrix.identity(float, 2) + (0.5 * result)) ** 2
        return result

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Update deformation gradient:
            self.FE_p[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # TODO: R might not be needed for our small timesteps? then remove R and everything
            # self.FE_p[p] = self.R(self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # TODO: could this just be simplified to: (or would this be unstable?)
            # self.FE_p[p] += (self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Remove the deviatoric component from the deformation gradient:
            if self.phase_p[p] == Water.Phase:
                self.FE_p[p] = ti.sqrt(self.JE_p[p]) * ti.Matrix.identity(ti.f32, 2)

            # Clamp singular values to simulate plasticity and elasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            self.JE_p[p] = 1.0
            for d in ti.static(range(self.n_dimensions)):
                singular_value = ti.f32(sigma[d, d])
                clamped = ti.f32(sigma[d, d])
                if self.phase_p[p] == Ice.Phase:
                    # Clamp singular values to [1 - theta_c, 1 + theta_s]
                    clamped = max(clamped, 1 - self.theta_c_p[p])
                    clamped = min(clamped, 1 + self.theta_s_p[p])
                self.JP_p[p] *= singular_value / clamped
                self.JE_p[p] *= clamped
                sigma[d, d] = clamped

            # Reconstruct elastic deformation gradient after plasticity
            self.FE_p[p] = U @ sigma @ V.transpose()

            # # TODO: if elasticity/plasticity is applied in the fluid phase, we also need this corrections:
            # if self.phase_p[p] == Phase.Water:
            #     self.FE_p[p] *= ti.sqrt(self.JP_p[p])
            #     self.JE_p[p] = ti.math.determinant(self.FE_p[p])
            #     self.JP_p[p] = 1.0

            # Apply ice hardening by adjusting Lame parameters:
            la, mu = self.lambda_p[p], self.mu_p[p]
            if self.phase_p[p] == Ice.Phase:
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta_p[p] * (1.0 - self.JP_p[p]))))
                la, mu = la * hardening, mu * hardening

            # Eliminate dilational component explicitly [Jiang 2014, Eqn. 8], then
            # compute deviatoric Piola-Kirchhoff stress P(F) [Jiang 2016, Eqn. 52]:
            FE_deviatoric = self.FE_p[p] * ti.sqrt(self.JE_p[p])
            U_deviatoric, _, V_deviatoric = ti.svd(FE_deviatoric)
            piola_kirchhoff = FE_deviatoric - (U_deviatoric @ V_deviatoric.transpose())
            piola_kirchhoff = (2 * mu * piola_kirchhoff) @ self.FE_p[p].transpose()  # pyright: ignore

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # Cauchy stress times dt and D_inv:
            cauchy_stress = -self.dt * self.vol_0_p * D_inv * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore
            # TODO: use cx, cy vectors here directly?
            # affine_x = cauchy_stress + self.mass_p[p] * self.c_x[p]
            # affine_y = cauchy_stress + self.mass_p[p] * self.c_y[p]

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([2.0, 2.0])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5])

            # Cubic kernels (JST16 Eqn. 122 with x=fx, x=|fx-1|, x=|fx-2|, x=|fx-3|, where fx is the distance
            # between base node and particle position). Based on https://www.bilibili.com/opus/662560355423092789
            # TODO: this could be shortened to x=fx, fx-1, fx-2, fx+1?!
            w_c = [
                ((-0.166 * dist_c**3) + (dist_c**2) - (2 * dist_c) + 1.33),
                ((0.5 * ti.abs(dist_c - 1.0) ** 3) - ((dist_c - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_c - 2.0) ** 3) - ((dist_c - 2.0) ** 2) + 0.66),
                ((-0.166 * ti.abs(dist_c - 3.0) ** 3) + ((dist_c - 3.0) ** 2) - (2 * ti.abs(dist_c - 3.0)) + 1.33),
            ]
            w_x = [
                ((-0.166 * dist_x**3) + (dist_x**2) - (2 * dist_x) + 1.33),
                ((0.5 * ti.abs(dist_x - 1.0) ** 3) - ((dist_x - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_x - 2.0) ** 3) - ((dist_x - 2.0) ** 2) + 0.66),
                ((-0.166 * ti.abs(dist_x - 3.0) ** 3) + ((dist_x - 3.0) ** 2) - (2 * ti.abs(dist_x - 3.0)) + 1.33),
            ]
            w_y = [
                ((-0.166 * dist_y**3) + (dist_y**2) - (2 * dist_y) + 1.33),
                ((0.5 * ti.abs(dist_y - 1.0) ** 3) - ((dist_y - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_y - 2.0) ** 3) - ((dist_y - 2.0) ** 2) + 0.66),
                ((-0.166 * ti.abs(dist_y - 3.0) ** 3) + ((dist_y - 3.0) ** 2) - (2 * ti.abs(dist_y - 3.0)) + 1.33),
            ]

            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]

                # Rasterize to cell centers:
                self.mass_c[base_c + offset] += weight_c * self.mass_p[p]

                temperature = self.mass_p[p] * self.temperature_p[p]
                self.temperature_c[base_c + offset] += weight_c * temperature

                capacity = self.mass_p[p] * self.capacity_p[p]
                self.capacity_c[base_c + offset] += weight_c * capacity

                inv_lambda = self.mass_p[p] / la
                self.inv_lambda_c[base_c + offset] += weight_c * inv_lambda

                self.JE_c[base_c + offset] += weight_c * self.mass_p[p] * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * self.mass_p[p] * self.JP_p[p]
                # TODO: the paper wants to rasterize JE, J and then set JP = J / JE, but this makes no difference
                # self.J_c[base_c + offset] += weight_c * self.mass_p[p] * self.J_p[p]

                # Rasterize to faces:
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx

                self.mass_x[base_x + offset] += weight_x * self.mass_p[p]
                self.mass_y[base_y + offset] += weight_y * self.mass_p[p]

                velocity_x = self.mass_p[p] * self.velocity_p[p][0] + affine_x @ dpos_x
                velocity_y = self.mass_p[p] * self.velocity_p[p][1] + affine_y @ dpos_y
                self.velocity_x[base_x + offset] += weight_x * velocity_x
                self.velocity_y[base_y + offset] += weight_y * velocity_y

                conductivity = self.mass_p[p] * self.conductivity_p[p]
                self.conductivity_x[base_x + offset] += weight_x * conductivity
                self.conductivity_y[base_y + offset] += weight_y * conductivity

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_x:
            if (mass_x := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass_x
                # TODO: use face/cell classfications here
                collision_right = i >= self.n_grid and self.velocity_x[i, j] > 0
                collision_left = i <= 0 and self.velocity_x[i, j] < 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            if (mass_y := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass_y
                self.velocity_y[i, j] += self.gravity[None] * self.dt
                # TODO: use face/cell classfications here
                collision_top = j >= self.n_grid and self.velocity_y[i, j] > 0
                collision_bottom = j <= 0 and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0

        for i, j in self.mass_c:
            if (mass_c := self.mass_c[i, j]) > 0:
                self.temperature_c[i, j] /= mass_c
                self.inv_lambda_c[i, j] /= mass_c
                self.capacity_c[i, j] /= mass_c
                self.JE_c[i, j] /= mass_c
                self.JP_c[i, j] /= mass_c
                # TODO: the paper wants to rasterize JE, J and then set JP = J / JE, but this makes no difference
                # self.J_c[i, j] *= 1 / self.mass_c[i, j]
                # self.JP_c[i, j] = self.J_c[i, j] / self.JE_c[i, j]

    @ti.kernel
    def classify_cells(self):
        # TODO: is it even needed to classify faces?
        # for i, j in self.classification_x:
        #     # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
        #
        #     # The simulation boundary is always colliding.
        #     x_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
        #     x_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
        #     if x_face_is_colliding:
        #         self.classification_x[i, j] = Classification.Colliding
        #         continue
        #
        #     # For convenience later on: a face is marked interior if it has mass.
        #     if self.mass_x[i, j] > 0:
        #         self.classification_x[i, j] = Classification.Interior
        #         continue
        #
        #     # All remaining faces are empty.
        #     self.classification_x[i, j] = Classification.Empty
        #
        # for i, j in self.classification_y:
        #     # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
        #
        #     # The simulation boundary is always colliding.
        #     y_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
        #     y_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
        #     if y_face_is_colliding:
        #         self.classification_y[i, j] = Classification.Colliding
        #         continue
        #
        #     # For convenience later on: a face is marked interior if it has mass.
        #     if self.mass_y[i, j] > 0:
        #         self.classification_y[i, j] = Classification.Interior
        #         continue
        #
        #     # All remaining faces are empty.
        #     self.classification_y[i, j] = Classification.Empty

        for i, j in self.classification_c:
            # TODO: Colliding cells are either assigned the temperature of the object it collides with
            # or a user-defined spatially-varying value depending on the setup.

            # NOTE: currently this is only set in the beginning, as the colliding boundary is fixed:
            # TODO: decide if this should be done here for better integration of colliding objects
            if self.is_colliding(i, j):
                # The boundary temperature is recorded for boundary (colliding) cells:
                self.temperature_c[i, j] = self.boundary_temperature[None]
                continue

            # A cell is marked as colliding if all of its surrounding faces are colliding.
            # cell_is_colliding = self.classification_x[i, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_x[i + 1, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_y[i, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_y[i, j + 1] == Classification.Colliding
            # if cell_is_colliding:
            #     # self.cell_temperature[i, j] = self.ambient_temperature[None]
            #     self.classification_c[i, j] = Classification.Colliding
            #     continue

            # A cell is interior if the cell and all of its surrounding faces have mass.
            cell_is_interior = self.mass_c[i, j] > 0
            cell_is_interior &= self.mass_x[i, j] > 0 and self.mass_x[i + 1, j] > 0
            cell_is_interior &= self.mass_y[i, j] > 0 and self.mass_y[i, j + 1] > 0

            if cell_is_interior:
                self.classification_c[i, j] = Classification.Interior
                continue

            # All remaining cells are empty.
            self.classification_c[i, j] = Classification.Empty

            # If the free surface is being enforced as a Dirichlet temperature condition,
            # the ambient air temperature is recorded for empty cells.
            self.temperature_c[i, j] = self.ambient_temperature[None]

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

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.0])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5])

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
            w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
            w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

            b_x = ti.Vector.zero(ti.f32, 2)
            b_y = ti.Vector.zero(ti.f32, 2)
            next_velocity = ti.Vector.zero(ti.f32, 2)
            next_temperature = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                next_temperature += weight_c * self.temperature_c[base_c + offset]
                velocity_x = weight_x * self.velocity_x[base_x + offset]
                velocity_y = weight_y * self.velocity_y[base_y + offset]
                next_velocity += [velocity_x, velocity_y]
                x_dpos = ti.cast(offset, ti.f32) - dist_x
                y_dpos = ti.cast(offset, ti.f32) - dist_y
                b_x += velocity_x * x_dpos
                b_y += velocity_y * y_dpos

            c_x = 3 * self.inv_dx * b_x  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            c_y = 3 * self.inv_dx * b_y  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            self.C_p[p] = ti.Matrix([[c_x[0], c_y[0]], [c_x[1], c_y[1]]])  # pyright: ignore
            self.position_p[p] += self.dt * next_velocity
            self.velocity_p[p] = next_velocity

            # DONE: set temperature for empty cells
            # DONE: set temperature for particles, ideally per geometry
            # DONE: set heat capacity per particle depending on phase
            # DONE: set heat conductivity per particle depending on phase
            # DONE: set particle mass per phase
            # DONE: set E and nu for each particle depending on phase
            # DONE: apply latent heat
            # TODO: move this to a ti.func? (or keep this here but assign values in func and use when adding particles)
            # TODO: set theta_c, theta_s per phase? Water probably wants very small values, ice depends on temperature
            # TODO: in theory all of the constitutive parameters must be functions of temperature
            #       in the ice phase to range from solid ice to slushy ice?

            # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            if (self.phase_p[p] == Ice.Phase) and (next_temperature >= 0):
                # Ice reached the melting point, additional temperature change is added to heat buffer.
                difference = next_temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is full the particle changes its phase to water,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] >= Water.LatentHeat:
                    self.conductivity_p[p] = Water.Conductivity
                    self.latent_heat_p[p] = Water.LatentHeat
                    self.temperature_p[p] = 0.0
                    self.capacity_p[p] = Water.Capacity
                    self.lambda_p[p] = Water.Lambda
                    self.color_p[p] = Water.Color
                    self.phase_p[p] = Water.Phase
                    self.mass_p[p] = self.vol_0_p * Water.Density
                    self.mu_p[p] = Water.Mu
                    self.FE_p[p] = ti.Matrix.identity(ti.f32, self.n_dimensions)
                    self.JP_p[p] = 1.0
                    self.JE_p[p] = 1.0

            elif (self.phase_p[p] == Water.Phase) and (next_temperature < 0):
                # Water particle reached the freezing point, additional temperature change is added to heat buffer.
                difference = next_temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is empty the particle changes its phase to ice,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] <= Ice.LatentHeat:
                    self.conductivity_p[p] = Ice.Conductivity
                    self.latent_heat_p[p] = Ice.LatentHeat
                    self.temperature_p[p] = 0.0
                    self.capacity_p[p] = Ice.Capacity
                    self.lambda_p[p] = self.lambda_p[p]
                    self.color_p[p] = Ice.Color
                    self.phase_p[p] = Ice.Phase
                    self.mass_p[p] = self.vol_0_p * Ice.Density
                    self.mu_p[p] = self.mu_p[p]
                    self.FE_p[p] = ti.Matrix.identity(ti.f32, self.n_dimensions)
                    self.JP_p[p] = 1.0
                    self.JE_p[p] = 1.0

            else:
                # Freely change temperature according to heat equation, but clamp for safety reasons.
                # NOTE: this also increases time it takes to melt/solidify.
                clamped = ti.min(Simulation.MaximumTemperature, next_temperature)
                clamped = ti.max(Simulation.MininumTemperature, clamped)
                self.temperature_p[p] = clamped

    @override
    def substep(self):
        # TODO: find good dt and number of iterations
        for _ in range(int(2e-3 // self.dt)):
            self.reset_grids()
            self.particle_to_grid()
            self.momentum_to_velocity()
            self.classify_cells()
            self.compute_volumes()
            self.pressure_solver.solve()
            self.heat_solver.solve()
            self.grid_to_particle()
