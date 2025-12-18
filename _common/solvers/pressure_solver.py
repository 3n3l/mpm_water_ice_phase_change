from _common.solvers.staggered_solver import StaggeredSolver
from taichi.linalg import SparseMatrixBuilder, SparseCG

import taichi as ti


@ti.data_oriented
class PressureSolver:
    def __init__(self, solver: StaggeredSolver) -> None:
        self.w_cells = solver.w_grid * solver.w_grid
        self.solver = solver

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        dt_inv_dx_sqrd = self.solver.dt[None] * self.solver.inv_dx * self.solver.inv_dx
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            diagonal = 0.0  # to keep max_num_triplets as low as possible
            idx = (i * self.solver.w_grid) + j  # raveled index

            # We enforce homogeneous Dirichlet pressure boundary conditions at CELLS that have been marked as empty.
            if not self.solver.is_interior(i, j):
                A[idx, idx] += 1.0
                b[idx] = 0.0
                continue

            # Build the left-hand side of the linear system:
            diagonal += self.solver.left_hand_offset(i, j)

            # Build the right-hand side of the linear system:
            b[idx] = self.solver.right_hand_offset(i, j)

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that have been marked as colliding.
            if not self.solver.is_colliding(i + 1, j):  # homogeneous Neumann
                inv_rho = self.solver.volume_x[i + 1, j] / self.solver.mass_x[i + 1, j]
                b[idx] += self.solver.inv_dx * self.solver.velocity_x[i + 1, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.solver.is_empty(i + 1, j):  # homogeneous Dirichlet
                    A[idx, idx + self.solver.w_grid] -= dt_inv_dx_sqrd * inv_rho

            if not self.solver.is_colliding(i - 1, j):  # homogeneous Neumann
                inv_rho = self.solver.volume_x[i, j] / self.solver.mass_x[i, j]
                b[idx] -= self.solver.inv_dx * self.solver.velocity_x[i, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.solver.is_empty(i - 1, j):  # homogeneous Dirichlet
                    A[idx, idx - self.solver.w_grid] -= dt_inv_dx_sqrd * inv_rho

            if not self.solver.is_colliding(i, j + 1):  # homogeneous Neumann
                inv_rho = self.solver.volume_y[i, j + 1] / self.solver.mass_y[i, j + 1]
                b[idx] += self.solver.inv_dx * self.solver.velocity_y[i, j + 1]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.solver.is_empty(i, j + 1):  # homogeneous Dirichlet
                    A[idx, idx + 1] -= dt_inv_dx_sqrd * inv_rho

            if not self.solver.is_colliding(i, j - 1):  # homogeneous Neumann
                inv_rho = self.solver.volume_y[i, j] / self.solver.mass_y[i, j]
                b[idx] -= self.solver.inv_dx * self.solver.velocity_y[i, j]
                diagonal += dt_inv_dx_sqrd * inv_rho
                if not self.solver.is_empty(i, j - 1):  # homogeneous Dirichlet
                    A[idx, idx - 1] -= dt_inv_dx_sqrd * inv_rho

            A[idx, idx] += diagonal

    @ti.kernel
    def apply_pressure(self, pressure: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.solver.dt[None] * self.solver.inv_dx
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            idx = i * self.solver.w_grid + j
            if self.solver.is_interior(i - 1, j) or self.solver.is_interior(i, j):
                if not (self.solver.is_colliding(i - 1, j) or self.solver.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - self.solver.w_grid]
                    inv_rho = self.solver.volume_x[i, j] / self.solver.mass_x[i, j]
                    self.solver.velocity_x[i, j] += inv_rho * coefficient * pressure_gradient
                else:
                    self.solver.velocity_x[i, j] = 0
            if self.solver.is_interior(i, j - 1) or self.solver.is_interior(i, j):
                if not (self.solver.is_colliding(i, j - 1) or self.solver.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - 1]
                    inv_rho = self.solver.volume_y[i, j] / self.solver.mass_y[i, j]
                    self.solver.velocity_y[i, j] += inv_rho * coefficient * pressure_gradient
                else:
                    self.solver.velocity_y[i, j] = 0

    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.w_cells),
            num_rows=self.w_cells,
            num_cols=self.w_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.w_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system:
        solver = SparseCG(A.build(), b, atol=1e-5, max_iter=500)
        p, _ = solver.solve()

        # Correct pressure:
        self.apply_pressure(p)
