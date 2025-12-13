from _common.configurations import Configuration
from _common.solvers import BaseSolver
from _common.constants import State

from typing import override

import taichi as ti

# TODO: move to constants
GRAVITY = ti.Vector([0, -9.81])


@ti.data_oriented
class MPM(BaseSolver):
    def __init__(self, max_particles: int, n_grid: int, dt: float):
        super().__init__(max_particles, n_grid, dt)

        # MPM Parameters that are configuration independent
        # self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.max_particles = max_particles
        # self.n_grid = n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = dt
        self.vol_0_p = (self.dx * 0.5) ** 2

        # Cell properties:
        self.velocity_c = ti.Vector.field(2, dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)
        self.mass_c = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid), offset=self.w_offset)

        # Particle properties:
        # self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        # self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        # self.color_p = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        # self.state_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.FE_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.JE_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.J_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.C_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # Parameters controlled from the GUI:
        self.lambda_0 = ti.field(dtype=ti.f32, shape=())
        self.theta_c = ti.field(dtype=ti.f32, shape=())
        self.theta_s = ti.field(dtype=ti.f32, shape=())
        self.mu_0 = ti.field(dtype=ti.f32, shape=())
        self.zeta = ti.field(dtype=ti.i32, shape=())
        self.nu = ti.field(dtype=ti.f32, shape=())
        self.E = ti.field(dtype=ti.f32, shape=())

    @ti.func
    @override
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position
        self.color_p[index] = geometry.color

        # Set properties to default values:
        self.mass_p[index] = self.vol_0_p * geometry.density
        self.FE_p[index] = ti.Matrix([[1, 0], [0, 1]])
        self.C_p[index] = ti.Matrix.zero(float, 2, 2)
        self.state_p[index] = State.Active
        self.JE_p[index] = 1.0
        self.JP_p[index] = 1.0

    @override
    def reset(self, configuration: Configuration):
        self.lambda_0[None] = configuration.lambda_0
        self.theta_c[None] = configuration.theta_c
        self.theta_s[None] = configuration.theta_s
        self.zeta[None] = configuration.zeta
        self.mu_0[None] = configuration.mu_0
        self.nu[None] = configuration.nu
        self.E[None] = configuration.E
        self.state_p.fill(State.Hidden)
        self.position_p.fill([42, 42])
        self.n_particles[None] = 0

    @ti.kernel
    def reset_grids(self):
        for i, j in self.mass_c:
            self.velocity_c[i, j] = 0
            self.mass_c[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Deformation gradient update
            self.FE_p[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Apply snow hardening by adjusting Lame parameters
            h = ti.max(0.1, ti.min(50, ti.exp(self.zeta[None] * (1.0 - self.JP_p[p]))))
            mu, la = self.mu_0[None] * h, self.lambda_0[None] * h
            U, sigma, V = ti.svd(self.FE_p[p])

            J = 1.0
            for d in ti.static(range(2)):
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - self.theta_c[None])
                singular_value = min(singular_value, 1 + self.theta_s[None])
                self.JP_p[p] *= sigma[d, d] / singular_value
                sigma[d, d] = singular_value
                J *= singular_value

            # Reconstruct elastic deformation gradient after plasticity
            self.FE_p[p] = U @ sigma @ V.transpose()

            # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            piola_kirchhoff = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
            piola_kirchhoff += ti.Matrix.identity(float, 2) * la * J * (J - 1)

            # Cauchy stress times dt and D_inv
            cauchy_stress = -self.dt * self.vol_0_p * 4 * self.inv_dx * self.inv_dx * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]

            # Lower left corner of the interpolation grid:
            # Based on https://www.bilibili.com/opus/662560355423092789
            base = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist = self.position_p[p] * self.inv_dx - ti.cast(base, ti.f32)

            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - dist) ** 2, 0.75 - (dist - 1) ** 2, 0.5 * (dist - 0.5) ** 2]

            # Rasterize mass and velocity
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset - dist, ti.f32) * self.dx
                weight = w[i][0] * w[j][1]
                v = self.mass_p[p] * self.velocity_p[p] + affine @ dpos  # pyright: ignore
                self.mass_c[base + offset] += weight * self.mass_p[p]
                self.velocity_c[base + offset] += weight * v

    @ti.kernel
    def momentum_to_velocity(self):
        """Forward Euler."""
        for i, j in self.mass_c:
            # Normalize velocity, add gravity:
            if self.mass_c[i, j] > 0:
                self.velocity_c[i, j] /= self.mass_c[i, j]
                self.velocity_c[i, j] += self.dt * GRAVITY

            # Sticky simulation boundary:
            if i < 0 or i > self.n_grid or j < 0 or j > self.n_grid:
                self.velocity_c[i, j] = 0

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            base = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)
            dist = self.position_p[p] * self.inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - dist) ** 2, 0.75 - (dist - 1.0) ** 2, 0.5 * (dist - 0.5) ** 2]

            C = ti.Matrix.zero(float, 2, 2)
            v = ti.Vector.zero(float, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - dist
                g_v = self.velocity_c[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                v += weight * g_v

            self.velocity_p[p], self.C_p[p] = v, C
            self.position_p[p] += self.dt * v

    @override
    def substep(self):
        for _ in range(int(2e-3 // self.dt)):
            self.reset_grids()
            self.particle_to_grid()
            self.momentum_to_velocity()
            self.grid_to_particle()
