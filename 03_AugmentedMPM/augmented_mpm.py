from _common.constants import Classification, State, Water, Ice, Simulation
from _common.solvers import PressureSolver, HeatSolver
from _common.solvers import StaggeredSolver
from typing import override

import taichi.math as tm
import taichi as ti


@ti.data_oriented
class AugmentedMPM(StaggeredSolver):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_grid, vol_0)

        # Properties on MAC-faces.
        self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)

        # Properties on MAC-cells.
        self.inv_lambda_c = ti.field(dtype=ti.f64, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.JE_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.JP_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

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
        self.mu_p = ti.field(dtype=ti.f32, shape=max_particles)
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
    def left_hand_offset(self, i: ti.i32, j: ti.i32) -> ti.f32:  # pyright: ignore
        # lambda_c approaches infinity for incompressible materials, this way we end up with the
        # usual pressure equation for cells where a lot of incompressible material has accumulated.
        return (self.JP_c[i, j] / (self.dt[None] * self.JE_c[i, j])) * self.inv_lambda_c[i, j]

    @ti.func
    @override
    def right_hand_offset(self, i: ti.i32, j: ti.i32) -> ti.f32:  # pyright: ignore
        # JE_c approaches 1 for incompressible materials, this way we end up with the usual
        # pressure equation for cells where a lot of incompressible material has accumulated.
        return (1 - self.JE_c[i, j]) / (self.dt[None] * self.JE_c[i, j])

    @ti.func
    @override
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        self.change_particle_material(index, geometry.material)
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position
        self.state_p[index] = State.Active
        self.C_p[index] = ti.Matrix.zero(ti.f32, 2, 2)

    @ti.func
    def change_particle_material(self, p: ti.i32, material: ti.template()):  # pyright: ignore
        self.conductivity_p[p] = material.Conductivity
        self.latent_heat_p[p] = material.LatentHeat
        self.temperature_p[p] = 0.0
        self.capacity_p[p] = material.Capacity
        self.theta_c_p[p] = material.Theta_c
        self.theta_s_p[p] = material.Theta_s
        self.lambda_p[p] = material.Lambda
        self.color_p[p] = material.Color
        self.phase_p[p] = material.Phase
        self.mass_p[p] = self.vol_0_p * material.Density
        self.zeta_p[p] = material.Zeta
        self.mu_p[p] = material.Mu
        self.FE_p[p] = ti.Matrix.identity(ti.f32, 2)
        self.JP_p[p] = 1.0
        self.JE_p[p] = 1.0

    @ti.kernel
    def reset_grids(self):
        for i, j in self.mass_x:
            self.conductivity_x[i, j] = 0
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.mass_y:
            self.conductivity_y[i, j] = 0
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

        for i, j in self.mass_c:
            self.temperature_c[i, j] = 0
            self.inv_lambda_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.mass_c[i, j] = 0
            self.JE_c[i, j] = 0
            self.JP_c[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Update deformation gradient:
            self.FE_p[p] += (self.dt[None] * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Remove the deviatoric component from the deformation gradient:
            if self.phase_p[p] == Water.Phase:
                self.FE_p[p] = ti.sqrt(self.JE_p[p]) * ti.Matrix.identity(ti.f32, 2)

            # Clamp singular values to simulate plasticity and elasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            self.JE_p[p] = 1.0
            for d in ti.static(range(2)):
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
            cauchy_stress = -self.dt[None] * self.vol_0_p * D_inv * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([2.0, 2.0])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5])

            # Cubic kernels:
            w_c = self.compute_cubic_kernel(dist_c)
            w_x = self.compute_cubic_kernel(dist_x)
            w_y = self.compute_cubic_kernel(dist_y)

            for i, j in ti.static(ti.ndrange(4, 4)):
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
                self.inv_lambda_c[base_c + offset] += weight_c * (mass / la)
                self.capacity_c[base_c + offset] += weight_c * mass * self.capacity_p[p]
                self.mass_c[base_c + offset] += weight_c * self.mass_p[p]
                self.JE_c[base_c + offset] += weight_c * mass * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * mass * self.JP_p[p]

                # Rasterize to cell faces:
                self.conductivity_x[base_x + offset] += weight_x * mass * self.conductivity_p[p]
                self.conductivity_y[base_y + offset] += weight_y * mass * self.conductivity_p[p]
                self.velocity_x[base_x + offset] += weight_x * (mass * velocity_x + affine_x @ dpos_x)
                self.velocity_y[base_y + offset] += weight_y * (mass * velocity_y + affine_y @ dpos_y)
                self.mass_x[base_x + offset] += weight_x * mass
                self.mass_y[base_y + offset] += weight_y * mass

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

        for i, j in self.mass_c:
            if (mass_c := self.mass_c[i, j]) > 0:
                self.temperature_c[i, j] /= mass_c
                self.inv_lambda_c[i, j] /= mass_c
                self.capacity_c[i, j] /= mass_c
                self.JE_c[i, j] /= mass_c
                self.JP_c[i, j] /= mass_c

    @ti.kernel
    def classify_cells(self):
        for i, j in self.classification_c:
            if self.is_colliding(i, j):
                # The boundary temperature is recorded for boundary (colliding) cells:
                self.temperature_c[i, j] = self.boundary_temperature[None]
                continue

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

            # Quadratic kernels:
            w_c = self.compute_quadratic_kernel(dist_c)
            w_x = self.compute_quadratic_kernel(dist_x)
            w_y = self.compute_quadratic_kernel(dist_y)

            temperature = 0.0
            velocity = ti.Vector.zero(ti.f32, 2)
            b_x = ti.Vector.zero(ti.f32, 2)
            b_y = ti.Vector.zero(ti.f32, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                temperature += weight_c * self.temperature_c[base_c + offset]
                velocity_x = weight_x * self.velocity_x[base_x + offset]
                velocity_y = weight_y * self.velocity_y[base_y + offset]
                velocity += [velocity_x, velocity_y]
                x_dpos = ti.cast(offset, ti.f32) - dist_x
                y_dpos = ti.cast(offset, ti.f32) - dist_y
                b_x += velocity_x * x_dpos
                b_y += velocity_y * y_dpos

            c_x = 3 * self.inv_dx * b_x  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            c_y = 3 * self.inv_dx * b_y  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            self.C_p[p] = ti.Matrix([[c_x[0], c_y[0]], [c_x[1], c_y[1]]])  # pyright: ignore
            self.position_p[p] += self.dt[None] * velocity
            self.velocity_p[p] = velocity

            # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            if (self.phase_p[p] == Ice.Phase) and (temperature >= 0):
                # Ice reached the melting point, additional temperature change is added to heat buffer.
                difference = temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is full the particle changes its phase to water,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] >= Water.LatentHeat:
                    self.change_particle_material(p, Water)

            elif (self.phase_p[p] == Water.Phase) and (temperature < 0):
                # Water particle reached the freezing point, additional temperature change is added to heat buffer.
                difference = temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is empty the particle changes its phase to ice,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] <= Ice.LatentHeat:
                    self.change_particle_material(p, Ice)

            else:
                # Freely change temperature according to heat equation, but clamp temperature for realism.
                self.temperature_p[p] = tm.clamp(temperature, Simulation.MinTemperature, Simulation.MaxTemperature)

    @override
    def substep(self):
        self.reset_grids()
        self.particle_to_grid()
        self.momentum_to_velocity()
        self.classify_cells()
        self.compute_volumes()
        self.pressure_solver.solve()
        self.heat_solver.solve()
        self.grid_to_particle()
