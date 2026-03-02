import utils  # import first to append parent directory to path

from _common.constants import Classification

import taichi as ti

ti.init(arch=ti.cpu)


n_grid = 7
dx = 1 / n_grid
inv_dx = float(n_grid)
additional_offset = 0.5
boundary_width = 1

rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
particle_vol = (dx * 0.5) ** 2
mass_p = particle_vol * rho_0

# NOTE: x, y axes are in the correct order here!
classification_c = ti.field(dtype=ti.int32, shape=(n_grid, n_grid))
classification_y = ti.field(dtype=ti.int32, shape=(n_grid + 1, n_grid))
classification_x = ti.field(dtype=ti.int32, shape=(n_grid, n_grid + 1))

mass_c = ti.field(dtype=ti.float32, shape=(n_grid, n_grid))
mass_y = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
mass_x = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))

volume_y = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
volume_x = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))

# NOTE: log which faces contributed to the particle velocity in G2P
contributed_c = ti.field(dtype=ti.float32, shape=(n_grid, n_grid))
contributed_y = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
contributed_x = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))


@ti.func
def is_valid(i: int, j: int) -> bool:
    return i >= 0 and i <= n_grid - 1 and j >= 0 and j <= n_grid - 1


@ti.func
def is_colliding(i: int, j: int) -> bool:
    return is_valid(i, j) and classification_c[i, j] == Classification.Colliding


@ti.func
def is_interior(i: int, j: int) -> bool:
    return is_valid(i, j) and classification_c[i, j] == Classification.Interior


@ti.kernel
def particle_to_grid(position_p: ti.template()):  # pyright: ignore
    # NOTE: the solvers are inverted as their axes are treated as if they were (x, y),
    #       here we treat them as (y, x).
    base_x = ti.floor((position_p * inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
    base_y = ti.floor((position_p * inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
    base_c = ti.floor((position_p * inv_dx - ti.Vector([2.0, 2.0])), dtype=ti.i32)

    dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.5, 0.0])
    dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.0, 0.5])
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5])

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
        weight_x = w_x[i][0] * w_x[j][1]
        weight_y = w_y[i][0] * w_y[j][1]
        weight_c = w_c[i][0] * w_c[j][1]

        mass_x[base_x + offset] += weight_x * mass_p
        mass_y[base_y + offset] += weight_y * mass_p
        mass_c[base_c + offset] += weight_c * mass_p


@ti.kernel
def classify_cells():
    for i, j in classification_c:
        is_colliding = boundary_width > i or n_grid - boundary_width <= i
        is_colliding |= boundary_width > j or n_grid - boundary_width <= j
        if is_colliding:
            classification_c[i, j] = Classification.Colliding
            continue

        # A cell is interior if the cell and all of its surrounding faces have mass.
        cell_is_interior = mass_c[i, j] > 0
        cell_is_interior &= mass_y[i, j] > 0 and mass_y[i + 1, j] > 0
        cell_is_interior &= mass_x[i, j] > 0 and mass_x[i, j + 1] > 0

        if cell_is_interior:
            classification_c[i, j] = Classification.Interior
            continue

        # All remaining cells are empty.
        classification_c[i, j] = Classification.Empty


@ti.kernel
def compute_volumes():
    # # FIXME: this seems to be wrong, the paper has a sum over CDFs
    # control_volume = 0.5 * self.dx * self.dx
    # # for i, j in self.classification_c:
    # for i, j in ti.ndrange(self.w_grid + 1, self.w_grid + 1):
    #     # if self.classification_c[i, j] == Classification.Interior:
    #     if self.is_interior(i, j):
    #     # if not self.is_colliding(i, j):
    #         self.volume_x[i + 1, j] += control_volume
    #         self.volume_y[i, j + 1] += control_volume
    #         self.volume_x[i, j] += control_volume
    #         self.volume_y[i, j] += control_volume

    directional = [
        0.0416670,  # i = -2
        0.4583300,  # i = -1
        0.4583300,  # i = 0
        0.0416670,  # i = 1
        0.0,  # i = 2
    ]
    orthogonal = [
        0.0026042,  # i = -2
        0.1979125,  # i = -1
        0.5989600,  # i = 0
        0.1979125,  # i = 1
        0.0026042,  # i = 2
    ]

    # directional = [
    #     0.0,  # i = 2
    #     0.0416670,  # i = 1
    #     0.4583300,  # i = 0
    #     0.4583300,  # i = -1
    #     0.0416670,  # i = -2
    # ]
    # orthogonal = [
    #     0.0026042,  # i = 2
    #     0.1979125,  # i = 1
    #     0.5989600,  # i = 0
    #     0.1979125,  # i = -1
    #     0.0026042,  # i = -2
    # ]

    # NOTE: 4x4 grid, strict
    # for i, j in self.classification_c:
    for i, j in ti.ndrange(n_grid + 1, n_grid + 1):
        # # if self.is_colliding(i,j):
        # if not is_interior(i, j):
        #     continue

        for k, l in ti.static(ti.ndrange(5, 5)):
            if is_interior(i - 2 + k, j - 2 + l):
                volume_x[i, j] += directional[k] * orthogonal[l]
                volume_y[i, j] += directional[l] * orthogonal[k]
                # self.volume_x[i, j] += self.dx * directional[k] * orthogonal[l]
                # self.volume_y[i, j] += self.dx * directional[l] * orthogonal[k]
                # self.volume_x[i, j] += self.integral_cubic_kernel()
                # self.volume_y[i, j] += self.dx * directional[l] * orthogonal[k]


@ti.kernel
def grid_to_particle(position_p: ti.template()):  # pyright: ignore
    base_x = ti.floor((position_p * inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
    base_y = ti.floor((position_p * inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
    base_c = ti.floor((position_p * inv_dx - ti.Vector([1.0, 1.0])), dtype=ti.i32)

    dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.5, 0.0])
    dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.0, 0.5])
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5])

    # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
    w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
    w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
    w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

    for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
        offset = ti.Vector([i, j])
        c_weight = w_c[i][0] * w_c[j][1]
        x_weight = w_x[i][0] * w_x[j][1]
        y_weight = w_y[i][0] * w_y[j][1]
        contributed_c[base_c + offset] += c_weight
        contributed_x[base_x + offset] += x_weight
        contributed_y[base_y + offset] += y_weight


def main():
    # positions = [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.9, 0.9), (1.0, 1.0)]
    # positions = [(0.32, 0.52), (0.52, 0.32), (0.52, 0.52)]
    # positions = [(0.5, 0.5), (0.52, 0.52), (0.48, 0.48)]
    positions = [(0.525, 0.525)]
    # positions = [(0.48, 0.48)]
    # positions = [(0.5, 0.5)]

    for x, y in positions:
        position_p = ti.Vector([x, y])
        classification_c.fill(Classification.Empty)
        classification_x.fill(Classification.Empty)
        classification_y.fill(Classification.Empty)

        mass_c.fill(0.0)
        mass_x.fill(0.0)
        mass_y.fill(0.0)

        volume_x.fill(0.0)
        volume_y.fill(0.0)

        contributed_c.fill(0.0)
        contributed_x.fill(0.0)
        contributed_y.fill(0.0)

        print()
        particle_to_grid(position_p)
        classify_cells()
        compute_volumes()
        grid_to_particle(position_p)

        ############################################################################################
        ### Particle-to-Grid
        ############################################################################################
        print(f"-> P2G, CUBIC @ {position_p}:")
        print("mass_c:")
        utils.print_mass(mass_c.to_numpy())

        print()
        print("mass_x:")
        utils.print_mass(mass_x.to_numpy())

        print()
        print("mass_y:")
        utils.print_mass(mass_y.to_numpy())

        ############################################################################################
        ### Classification
        ############################################################################################
        print(f"\n-> CLASSIFICATION @ {position_p}:")
        print("classification_c:")
        utils.print_classification(classification_c.to_numpy())

        ############################################################################################
        ### Control Volumes
        ############################################################################################
        print(f"\n-> Control Volumes @ {position_p}:")

        print()
        print("volume_x:")
        utils.print_mass(volume_x.to_numpy())

        print()
        print("volume_y:")
        utils.print_mass(volume_y.to_numpy())

        ############################################################################################
        ### Grid-to-Particle
        ############################################################################################
        print(f"\n-> G2P, QUADRATIC @ {position_p}:")
        print("mass_c:")
        utils.print_mass(contributed_c.to_numpy())

        print()
        print("mass_x:")
        utils.print_mass(contributed_x.to_numpy())

        print()
        print("mass_y:")
        utils.print_mass(contributed_y.to_numpy())
        print()

    # print(utils.print_green("HOW IT SHOULD BE"))


if __name__ == "__main__":
    main()
