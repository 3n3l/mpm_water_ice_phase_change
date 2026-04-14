"""This shows the ASYMMETRICAL MELTING, that happened before reworking the interpolation."""

import utils  # import first to append parent directory to path

from _common.constants import Classification

import taichi as ti

ti.init(arch=ti.cpu)


n_grid = 6
dx = 1 / n_grid
inv_dx = float(n_grid)
additional_offset = 0.5
boundary_width = 0

rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
particle_vol = (dx * 0.5) ** 2
mass_p = particle_vol * rho_0

# NOTE: x, y axes are in the correct order here!
classification_c = ti.field(dtype=ti.int8, shape=(n_grid, n_grid))
classification_y = ti.field(dtype=ti.int8, shape=(n_grid + 1, n_grid))
classification_x = ti.field(dtype=ti.int8, shape=(n_grid, n_grid + 1))

mass_c = ti.field(dtype=ti.float32, shape=(n_grid, n_grid))
mass_y = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
mass_x = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))

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


@ti.kernel
def particle_to_grid(position_p: ti.template()):  # pyright: ignore

    # NOTE: ASYMMETRICAL MELTING

    # Lower left corner of the interpolation grid:
    base_x = ti.floor((position_p * inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
    base_y = ti.floor((position_p * inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
    base_c = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

    # Distance between lower left corner and particle position:
    dist_x = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
    dist_y = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)

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

    for i, j in ti.static(ti.ndrange(4, 4)):  # Loop over 4x4 grid node neighborhood
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
        # A cell is marked as colliding if all of its surrounding faces are colliding.
        cell_is_colliding = classification_y[i, j] == Classification.Colliding
        cell_is_colliding &= classification_y[i + 1, j] == Classification.Colliding
        cell_is_colliding &= classification_x[i, j] == Classification.Colliding
        cell_is_colliding &= classification_x[i, j + 1] == Classification.Colliding

        if cell_is_colliding:
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
def grid_to_particle(position_p: ti.template()):  # pyright: ignore

    # NOTE: ASYMMETRICAL MELTING

    # Lower left corner of the interpolation grid:
    base_y = ti.floor((position_p * inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
    base_x = ti.floor((position_p * inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
    base_c = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

    # Distance between lower left corner and particle position:
    dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.5, 0.0])
    dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.0, 0.5])
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)

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

        contributed_c.fill(0.0)
        contributed_x.fill(0.0)
        contributed_y.fill(0.0)

        print()
        particle_to_grid(position_p)
        classify_cells()
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

    print(utils.print_red("ASYMMETRICAL MELTING"))


if __name__ == "__main__":
    main()
