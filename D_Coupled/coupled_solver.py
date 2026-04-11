from _common.constants import Classification, State, Water, Ice, Simulation
from _common.solvers import HeatSolver, StaggeredSolver
from apic_solver import APIC
from mpm_solver import MPM
from typing import override

import taichi.math as tm
import taichi as ti


@ti.data_oriented
class CoupledSolver(StaggeredSolver):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_grid, vol_0)

        # Properties on MAC-faces.
        self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)

        # Properties on MAC-cells.
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Properties on particles.
        self.conductivity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_c_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_s_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.lambda_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.zeta_p = ti.field(dtype=ti.i32, shape=max_particles)
        self.cx_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.cy_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.mu_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.FE_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.JE_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.C_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.latent_heat_p = ti.field(dtype=ti.f32, shape=max_particles)  # U_p
        # self.water_divider = ti.field(dtype=ti.f32, shape=())  # divides water and ice particles
        # self.ice_divider = ti.field(dtype=ti.f32, shape=())  # divides water and ice particles

        self.apic = APIC(self)
        self.mpm = MPM(self)
        self.heat_solver = HeatSolver(self)

        # Set the initial boundary:
        self.initialize_boundary()

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
            self.mass_x[i, j] = 0

        for i, j in self.mass_y:
            self.conductivity_y[i, j] = 0
            self.velocity_y[i, j] = 0
            self.mass_y[i, j] = 0

        for i, j in self.mass_c:
            self.temperature_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.mass_c[i, j] = 0

    @ti.kernel
    def couple_materials(self):
        for i, j in self.velocity_x:
            self.mass_x[i, j] = self.apic.mass_x[i, j] + self.mpm.mass_x[i, j]
            if (mass_x := self.mass_x[i, j]) > 0:
                velocity_x = self.apic.mass_x[i, j] * self.apic.velocity_x[i, j]
                velocity_x += self.mpm.mass_x[i, j] * self.mpm.velocity_x[i, j]
                self.velocity_x[i, j] = velocity_x / mass_x

                # TODO: mass not needed because not in m2v?
                coupled_conductivity = self.apic.conductivity_x[i, j] + self.mpm.conductivity_x[i, j]
                self.conductivity_x[i, j] = coupled_conductivity / mass_x

                # if (i >= self.n_grid and self.c_velocity_x[i, j] > 0) or (i <= 0 and self.c_velocity_x[i, j] < 0):
                #     self.c_velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            self.mass_y[i, j] = self.apic.mass_y[i, j] + self.mpm.mass_y[i, j]
            if (coupled_mass_y := self.mass_y[i, j]) > 0:
                velocity_y = self.apic.mass_y[i, j] * self.apic.velocity_y[i, j]
                velocity_y += self.mpm.mass_y[i, j] * self.mpm.velocity_y[i, j]
                self.velocity_y[i, j] = velocity_y / coupled_mass_y

                # TODO: mass not needed because not in m2v?
                coupled_conductivity = self.apic.conductivity_y[i, j] + self.mpm.conductivity_y[i, j]
                self.conductivity_y[i, j] = coupled_conductivity / coupled_mass_y

                # if (j >= self.n_grid and self.c_velocity_y[i, j] > 0) or (j <= 0 and self.c_velocity_y[i, j] < 0):
                #     self.c_velocity_y[i, j] = 0

        for i, j in self.mass_c:
            self.mass_c[i, j] = self.apic.mass_c[i, j] + self.mpm.mass_c[i, j]
            if (mass_c := self.mass_c[i, j]) > 0:
                # TODO: mass not needed because not in m2v?
                temperature = self.apic.temperature_c[i, j] + self.mpm.temperature_c[i, j]
                self.temperature_c[i, j] = temperature / mass_c
                capacity = self.apic.capacity_c[i, j] + self.mpm.capacity_c[i, j]
                self.capacity_c[i, j] = capacity / mass_c

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

            c_x = 4 * self.inv_dx * b_x  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            c_y = 4 * self.inv_dx * b_y  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
            self.cx_p[p], self.cy_p[p] = c_x, c_y
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
        self.apic.substep()
        self.mpm.substep()
        self.couple_materials()
        self.classify_cells()
        self.heat_solver.solve()
        self.grid_to_particle()
        # TODO: there is no G2P in the solvers right now
        # TODO: sort particles here instead of checking in each kernel
