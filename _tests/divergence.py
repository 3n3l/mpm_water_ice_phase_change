import math
import utils  # import first to append parent directory to path

from _common.configurations import Configuration, Rectangle, Circle
from _common.samplers import PoissonDiskSampler
from _common.solvers import StaggeredSolver

from _common.simulation import BaseSimulation
from _common.parsers import parser, add_configuration
from _common.constants import Water

from A_APIC.apic import APIC
from C_Augmented.augmented_mpm import AugmentedMPM

import taichi as ti
import numpy as np

OFFSET = 0.0234375
MAX_ITERATIONS = 300
LOWER_BOUND = -1e-6
UPPER_BOUND = 1e-6


configurations = [
    Configuration(
        name="Dam Break",
        information="Water",
        dt=1e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Spherefall, Water",
        information="Water",
        dt=1e-3,
        geometries=[
            Circle(
                material=Water,  # pyright: ignore
                center=(0.5, 0.4),
                velocity=(0, -3),
                radius=0.1,
            ),
        ],
    ),
]


def print_wrt_bound(value: float) -> str:
    if value < LOWER_BOUND or value > UPPER_BOUND:
        return utils.print_red(str(value))
    else:
        return utils.print_green(str(value))


class TestRenderer(BaseSimulation):
    def __init__(
        self,
        solver: StaggeredSolver,
        configurations: list[Configuration],
        poisson_disk_sampler: PoissonDiskSampler,
        radius: float,
    ) -> None:
        super().__init__(
            solver=solver,
            configurations=configurations,
            sampler=poisson_disk_sampler,
            prefix="",
            radius=radius,
            name="",
        )
        self.divergence_sum = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.max_divergence = 0
        self.min_divergence = 0

    # @ti.kernel
    # def compute_divergence(self, divergence: ti.types.ndarray()):  # pyright: ignore
    #     for i, j in self.solver.mass_c:
    #         divergence[i, j] = 0
    #         if self.solver.is_interior(i, j):
    #             divergence[i, j] += self.solver.velocity_x[i + 1, j]
    #             divergence[i, j] -= self.solver.velocity_x[i, j]
    #             divergence[i, j] += self.solver.velocity_y[i, j + 1]
    #             divergence[i, j] -= self.solver.velocity_y[i, j]
    #             # avg[i, j] += div[i, j]

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray(), avg: ti.types.ndarray()):  # pyright: ignore
        # for i, j in ti.ndrange(self.solver.w_grid, self.solver.w_grid):
        # for i, j in self.solver.mass_c:
        for i, j in ti.ndrange(self.solver.n_grid, self.solver.n_grid):
            div[i, j] = 0
            if self.solver.is_interior(i, j):
                div[i, j] += self.solver.velocity_x[i + 1, j]
                div[i, j] -= self.solver.velocity_x[i, j]
                div[i, j] += self.solver.velocity_y[i, j + 1]
                div[i, j] -= self.solver.velocity_y[i, j]
                avg[i, j] += div[i, j]

    def run(self) -> None:
        self.divergence_sum.fill(0)
        self.max_divergence = 0
        self.min_divergence = 0

        for i in range(1, MAX_ITERATIONS + 1):
            self.substep()
            self.compute_divergence(self.divergence, self.divergence_sum)

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            divergence = self.divergence.to_numpy()
            abs_curr_min = np.min(divergence)
            if abs_curr_min < self.min_divergence:
                self.min_divergence = np.min(divergence)
            abs_curr_max = np.abs(np.max(divergence))
            if abs_curr_max > np.abs(self.max_divergence):
                self.max_divergence = np.max(divergence)


def main() -> None:
    add_configuration(configurations)
    arguments = parser.parse_args()

    print("DEBUG:", arguments.debug)

    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug, verbose=arguments.verbose)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug, verbose=arguments.verbose)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug, verbose=arguments.verbose)

    max_particles, n_grid = 300_000, 128
    radius = 1 / (4 * float(n_grid))  # 4 particles per cell
    vol_0 = math.pi * (radius**2)

    ampm_solver = AugmentedMPM(max_particles=max_particles, n_grid=n_grid, vol_0=vol_0)
    apic_solver = APIC(max_particles=max_particles, n_grid=n_grid, vol_0=vol_0)
    solvers = {"our Method": ampm_solver, "APIC": apic_solver}

    for solver_name, solver_object in solvers.items():
        results = []
        all_tests_succeeded = True
        poisson_disk_sampler = PoissonDiskSampler(solver=solver_object)
        test_renderer = TestRenderer(
            poisson_disk_sampler=poisson_disk_sampler,
            configurations=configurations,
            solver=solver_object,
            radius=radius,
        )

        for configuration in configurations:
            print(f"NOW RUNNING: {configuration.name} with {solver_name}")
            test_renderer.load_configuration(configuration)
            test_renderer.run()

            average_divergence = test_renderer.divergence_sum.to_numpy() / MAX_ITERATIONS
            min_average, max_average = np.min(average_divergence), np.max(average_divergence)
            min_spiking, max_spiking = test_renderer.min_divergence, test_renderer.max_divergence
            test_succeeded = min_average > LOWER_BOUND and max_average < UPPER_BOUND
            all_tests_succeeded &= test_succeeded
            result = (
                f"{configuration.name} {configuration.information}\n"
                f"-> AVERAGE DIVERGENCE: min, max = {print_wrt_bound(min_average)}, {print_wrt_bound(max_average)}\n"
                f"-> SPIKING DIVERGENCE: min, max = {print_wrt_bound(min_spiking)}, {print_wrt_bound(max_spiking)}\n"
                f"-> {utils.print_green("PASSED!") if test_succeeded else utils.print_red("DID NOT PASS!")}\n"
            )
            results.append(result)

        print(f"\n\niterations = {MAX_ITERATIONS}, lower bound = {LOWER_BOUND}, upper bound = {UPPER_BOUND}\n")
        print(*results, sep="\n", end="\n\n")
        print("\033[92m:)))))))))\033[0m" if all_tests_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
