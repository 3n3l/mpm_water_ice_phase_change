import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.simulation import GGUI_Simulation, GUI_Simulation
from _common.samplers import BasePoissonDiskSampler

from parsing import arguments, should_use_implicit_update
from presets import configuration_list
from mpm import MPM

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=True)
        # ti.init(arch=ti.cpu, debug=arguments.debug)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    initial_configuration = arguments.configuration % len(configuration_list)
    name = "Material Point Method for Snow Simulation"
    prefix = "MPM"

    max_particles, n_grid = 300_000, 128
    radius = 1 / (6 * float(n_grid))  # 6 particles per cell
    vol_0 = math.pi * (radius**2)

    mpm_solver = MPM(max_particles, n_grid, vol_0)
    poisson_disk_sampler = BasePoissonDiskSampler(solver=mpm_solver, r=radius, k=10)
    if arguments.gui.lower() == "ggui":
        renderer = GGUI_Simulation(
            initial_configuration=initial_configuration,
            sampler=poisson_disk_sampler,
            configurations=configuration_list,
            solver=mpm_solver,
            res=(720, 720),
            prefix=prefix,
            radius=radius,
            name=name,
        )
        renderer.run()
    # elif arguments.gui.lower() == "gui":
    #     renderer = GUI_Simulation
    #         initial_configuration=initial_configuration,
    #         sampler=poisson_disk_sampler,
    #         configurations=configuration_list,
    #         name=simulation_name,
    #         solver=mpm_solver,
    #         res=720,
    #     )
    #     renderer.run()

    print("\n", "#" * 100, sep="")
    print("###", name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
