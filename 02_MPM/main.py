import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.simulation import GGUI_Simulation, GUI_Simulation
from _common.samplers import PoissonDiskSampler
from _common.parsers import parser, add_configuration
from _common.presets import snow_presets

from snow_mpm import MPM

import taichi as ti


def main():
    configurations = snow_presets
    add_configuration(configurations)
    arguments = parser.parse_args()
    print(parser.epilog)

    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug, verbose=arguments.verbose)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug, verbose=arguments.verbose)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug, verbose=arguments.verbose)

    initial_configuration = arguments.configuration % len(configurations)
    name = "Material Point Method for Snow Simulation"
    prefix = "MPM"

    max_particles, n_grid = 300_000, 128
    radius = 1 / (6 * float(n_grid))  # 6 particles per cell
    vol_0 = math.pi * (radius**2)

    mpm_solver = MPM(max_particles, n_grid, vol_0)
    poisson_disk_sampler = PoissonDiskSampler(solver=mpm_solver, r=radius, k=10)
    if arguments.gui.lower() == "ggui":
        renderer = GGUI_Simulation(
            initial_configuration=initial_configuration,
            sampler=poisson_disk_sampler,
            configurations=configurations,
            solver=mpm_solver,
            res=(720, 720),
            prefix=prefix,
            radius=radius,
            name=name,
        )
        renderer.run()
    elif arguments.gui.lower() == "gui":
        renderer = GUI_Simulation(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=poisson_disk_sampler,
            solver=mpm_solver,
            prefix=prefix,
            radius=radius,
            name=name,
            res=720,
        )
        renderer.run()

    print("\n", "#" * 100, sep="")
    print("###", name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
