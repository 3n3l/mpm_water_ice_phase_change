import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.simulation import GGUI_Simulation, GUI_Simulation
from _common.parsers.parsing import parser, add_configuration
from _common.samplers import PoissonDiskSampler
from _common.presets import water_presets

from apic import APIC

import taichi as ti


def main():
    configurations = water_presets
    add_configuration(configurations)
    arguments = parser.parse_args()
    print(parser.epilog)

    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    initial_configuration = arguments.configuration % len(configurations)
    name = f"Affine Particle-In-Cell Method"
    prefix = f"APIC"

    max_particles, n_grid = 300_000, 128
    radius = 1 / (6 * float(n_grid))  # 6 particles per cell
    vol_0 = math.pi * (radius**2)

    solver = APIC(max_particles=max_particles, n_grid=n_grid, vol_0=vol_0)
    sampler = PoissonDiskSampler(solver=solver, r=radius, k=30)
    if arguments.gui.lower() == "ggui":
        simulation = GGUI_Simulation(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=sampler,
            solver=solver,
            prefix=prefix,
            res=(720, 720),
            radius=radius,
            name=name,
        )
        simulation.run()
    elif arguments.gui.lower() == "gui":
        simulation = GUI_Simulation(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=sampler,
            prefix=prefix,
            solver=solver,
            radius=radius,
            name=name,
            res=720,
        )
        simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
