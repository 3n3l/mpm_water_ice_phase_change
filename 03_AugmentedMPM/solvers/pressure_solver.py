from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from parsing import should_use_direct_solver

import taichi as ti
import numpy as np

GRAVITY = -9.81


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.boundary_width = mpm_solver.boundary_width
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
        self.mpm_solver = mpm_solver
        self.dt = mpm_solver.dt

        self.classification_c = mpm_solver.classification_c
        self.inv_lambda_c = mpm_solver.inv_lambda_c
        self.JE_c = mpm_solver.JE_c
        self.JP_c = mpm_solver.JP_c

        self.velocity_x = mpm_solver.velocity_x
        self.velocity_y = mpm_solver.velocity_y
        self.volume_x = mpm_solver.volume_x
        self.volume_y = mpm_solver.volume_y
        self.mass_x = mpm_solver.mass_x
        self.mass_y = mpm_solver.mass_y

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        dt_inv_dx_sqrd = self.dt * self.inv_dx * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            diagonal = 0.0  # to keep max_num_triplets as low as possible
            idx = (i * self.n_grid) + j  # raveled index

            # We enforce homogeneous Dirichlet pressure boundary conditions at CELLS that have been marked as empty.
            if not self.mpm_solver.is_interior(i, j):
                A[idx, idx] += 1.0
                b[idx] = 0.0
                continue

            # Build the right-hand side of the linear system:
            # NOTE: JE_c approaches 1 for incompressible materials, this way we end up with the usual
            #       pressure equation for cells where a lot of incompressible material has accumulated.
            b[idx] = (1 - self.JE_c[i, j]) / (self.dt * self.JE_c[i, j])

            # Build the left-hand side of the linear system:
            # NOTE: lambda_c approaches infinity for incompressible materials, this way we end up with the
            #       usual pressure equation for cells where a lot of incompressible material has accumulated.
            diagonal += (self.JP_c[i, j] / (self.dt * self.JE_c[i, j])) * self.inv_lambda_c[i, j]

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that have been marked as colliding.
            if not self.mpm_solver.is_colliding(i + 1, j):  # homogeneous Neumann
                inv_rho = self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                b[idx] += self.inv_dx * self.velocity_x[i + 1, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.mpm_solver.is_empty(i + 1, j):  # homogeneous Dirichlet
                    A[idx, idx + self.n_grid] -= dt_inv_dx_sqrd * inv_rho

            if not self.mpm_solver.is_colliding(i - 1, j):  # homogeneous Neumann
                inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                b[idx] -= self.inv_dx * self.velocity_x[i, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.mpm_solver.is_empty(i - 1, j):  # homogeneous Dirichlet
                    A[idx, idx - self.n_grid] -= dt_inv_dx_sqrd * inv_rho

            if not self.mpm_solver.is_colliding(i, j + 1):  # homogeneous Neumann
                inv_rho = self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                b[idx] += self.inv_dx * self.velocity_y[i, j + 1]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.mpm_solver.is_empty(i, j + 1):  # homogeneous Dirichlet
                    A[idx, idx + 1] -= dt_inv_dx_sqrd * inv_rho

            if not self.mpm_solver.is_colliding(i, j - 1):  # homogeneous Neumann
                inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                b[idx] -= self.inv_dx * self.velocity_y[i, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.mpm_solver.is_empty(i, j - 1):  # homogeneous Dirichlet
                    A[idx, idx - 1] -= dt_inv_dx_sqrd * inv_rho

            A[idx, idx] += diagonal

    @ti.kernel
    def apply_pressure(self, pressure: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            idx = i * self.n_grid + j
            if self.mpm_solver.is_interior(i - 1, j) or self.mpm_solver.is_interior(i, j):
                # NOTE: we add the pressure, instead of subtracting it.
                if not (self.mpm_solver.is_colliding(i - 1, j) or self.mpm_solver.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - self.n_grid]
                    inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                    self.velocity_x[i, j] += inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_x[i, j] = 0
            if self.mpm_solver.is_interior(i, j - 1) or self.mpm_solver.is_interior(i, j):
                if not (self.mpm_solver.is_colliding(i, j - 1) or self.mpm_solver.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - 1]
                    inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                    self.velocity_y[i, j] += inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_y[i, j] = 0

    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system:
        if should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            p = solver.solve(b)
            # FIXME: remove this debugging statements or move to test file
            solver_succeeded, pressure = solver.info(), p.to_numpy()
            assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b, atol=1e-6, max_iter=500)
            p, _ = solver.solve()

        # Correct pressure:
        self.apply_pressure(p)
