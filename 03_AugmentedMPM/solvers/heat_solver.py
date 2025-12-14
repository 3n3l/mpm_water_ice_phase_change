from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from parsing import should_use_direct_solver

import taichi as ti


@ti.data_oriented
class HeatSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.n_grid = mpm_solver.n_grid
        self.mpm_solver = mpm_solver
        self.inv_dx = mpm_solver.inv_dx
        self.dt = mpm_solver.dt

        self.classification_c = mpm_solver.classification_c
        self.temperature_c = mpm_solver.temperature_c
        self.capacity_c = mpm_solver.capacity_c
        self.mass_c = mpm_solver.mass_c

        self.classification_x = mpm_solver.classification_x
        self.classification_y = mpm_solver.classification_y
        self.conductivity_x = mpm_solver.conductivity_x
        self.conductivity_y = mpm_solver.conductivity_y

        self.should_use_direct_solver = should_use_direct_solver

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            idx = (i * self.n_grid) + j  # raveled index
            b[idx] = self.temperature_c[i, j]  # right-hand side

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact with fixed
            # temperature bodies (like a heated pan (-> boundary cells in our case) or air (-> empty cells)),
            # i.e we keep the currently recorded cell temperatures for empty (air) cells.
            if not self.mpm_solver.is_interior(i, j):
                A[idx, idx] += 1.0
                continue

            # Compute (1 / dx^2) * ((dt * dx^d) / (m_c * c_c)) [Jiang 2016, Ch. 5.8],
            # NOTE: dx^d is cancelled out by 1 / dx^2 because d == 2.
            dt_inv_mass_capacity = self.dt / (self.mass_c[i, j] * self.capacity_c[i, j])
            inv_dx_sqrd = self.inv_dx * self.inv_dx
            diagonal = 1.0  # to keep max_num_triplets as low as possible

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that are corresponding
            # to insulated objects, i.e we set the conductivity to zero for faces adjacent to insulated cells
            # (or by simply just not incorporating them).
            if not self.mpm_solver.is_insulated(i + 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.conductivity_x[i + 1, j]
                if self.mpm_solver.is_empty(i + 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx + self.n_grid] -= dt_inv_mass_capacity * self.conductivity_x[i + 1, j]
                    b[idx] += inv_dx_sqrd * self.conductivity_x[i + 1, j] * self.temperature_c[i + 1, j]

            if not self.mpm_solver.is_insulated(i - 1, j):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.conductivity_x[i, j]
                if self.mpm_solver.is_empty(i - 1, j):  # non-homogeneous Dirichlet
                    A[idx, idx - self.n_grid] -= dt_inv_mass_capacity * self.conductivity_x[i, j]
                    b[idx] += inv_dx_sqrd * self.conductivity_x[i, j] * self.temperature_c[i - 1, j]

            if not self.mpm_solver.is_insulated(i, j + 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.conductivity_y[i, j + 1]
                if self.mpm_solver.is_empty(i, j + 1):  # non-homogeneous Dirichlet
                    A[idx, idx + 1] -= dt_inv_mass_capacity * self.conductivity_y[i, j + 1]
                    b[idx] += inv_dx_sqrd * self.conductivity_y[i, j + 1] * self.temperature_c[i, j + 1]

            if not self.mpm_solver.is_insulated(i, j - 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.conductivity_y[i, j]
                if self.mpm_solver.is_empty(i, j - 1):  # non-homogeneous Dirichlet
                    A[idx, idx - 1] -= dt_inv_mass_capacity * self.conductivity_y[i, j]
                    b[idx] += inv_dx_sqrd * self.conductivity_y[i, j] * self.temperature_c[i, j - 1]

            A[idx, idx] += diagonal  # add value from variable, to keep max_num_triplets as low as possible

    @ti.kernel
    def fill_temperature_field(self, T: ti.types.ndarray()):  # pyright: ignore
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            self.temperature_c[i, j] = T[(i * self.n_grid) + j]

    def solve(self):
        # TODO: max_num_triplets could be optimized to N * 5?
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        if self.should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            T = solver.solve(b)
            self.fill_temperature_field(T)
            # FIXME: remove this debugging statements or move to test file
            # solver_succeeded, temperature = solver.info(), T.to_numpy()
            # assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            # assert not np.any(np.isnan(temperature)), "NAN VALUE IN NEW TEMPERATURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b, atol=1e-6, max_iter=500)
            T, _ = solver.solve()

        self.fill_temperature_field(T)
