from taichi.linalg import SparseMatrixBuilder, SparseCG

import taichi as ti


@ti.data_oriented
class HeatSolver:
    def __init__(self, solver) -> None:
        self.w_cells = solver.w_grid * solver.w_grid
        self.solver = solver

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            idx = (i * self.solver.w_grid) + j  # raveled index
            b[idx] = self.solver.temperature_c[i, j]  # right-hand side

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact with fixed
            # temperature bodies (like a heated pan (-> boundary cells in our case) or air (-> empty cells)),
            # i.e we keep the currently recorded cell temperatures for empty (air) cells.
            if not self.solver.is_interior(i, j):
                A[idx, idx] += 1.0
                continue

            # Compute (1 / dx^2) * ((dt * dx^d) / (m_c * c_c)) [Jiang 2016, Ch. 5.8],
            # NOTE: dx^d is cancelled out by 1 / dx^2 because d == 2.
            dt_inv_mass_capacity = self.solver.dt[None] / (self.solver.mass_c[i, j] * self.solver.capacity_c[i, j])
            inv_dx_sqrd = self.solver.inv_dx * self.solver.inv_dx
            diagonal = 1.0  # to keep max_num_triplets as low as possible

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that are corresponding
            # to insulated objects, i.e we set the conductivity to zero for faces adjacent to insulated cells
            # (or by simply just not incorporating them).
            if not self.solver.is_insulated(i + 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]
                if self.solver.is_empty(i + 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx + self.solver.w_grid] -= dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i + 1, j] * self.solver.temperature_c[i + 1, j]

            if not self.solver.is_insulated(i - 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i, j]
                if self.solver.is_empty(i - 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx - self.solver.w_grid] -= dt_inv_mass_capacity * self.solver.conductivity_x[i, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_x[i, j] * self.solver.temperature_c[i - 1, j]

            if not self.solver.is_insulated(i, j + 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]
                if self.solver.is_empty(i, j + 1):  # non-homogeneous Dirichlet
                    A[idx, idx + 1] -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j + 1] * self.solver.temperature_c[i, j + 1]

            if not self.solver.is_insulated(i, j - 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j]
                if self.solver.is_empty(i, j - 1):  # non-homogeneous Dirichlet
                    A[idx, idx - 1] -= dt_inv_mass_capacity * self.solver.conductivity_y[i, j]
                    b[idx] += inv_dx_sqrd * self.solver.conductivity_y[i, j] * self.solver.temperature_c[i, j - 1]

            A[idx, idx] += diagonal  # add value from variable, to keep max_num_triplets as low as possible

    @ti.kernel
    def fill_temperature_field(self, T: ti.types.ndarray()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
            self.solver.temperature_c[i, j] = T[(i * self.solver.w_grid) + j]

    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.w_cells),
            num_rows=self.w_cells,
            num_cols=self.w_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.w_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        solver = SparseCG(A.build(), b, atol=1e-5, max_iter=500)
        T, _ = solver.solve()

        self.fill_temperature_field(T)
