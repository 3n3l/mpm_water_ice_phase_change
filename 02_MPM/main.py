import sys, os

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

    dt = 1e-4
    max_particles = 100_000
    n_grid = 128 * arguments.quality

    mpm_solver = MPM(max_particles, n_grid, dt)
    poisson_disk_sampler = BasePoissonDiskSampler(solver=mpm_solver, r=0.002, k=30)

    simulation_name = "Material Point Method for Snow Simulation"
    initial_configuration = arguments.configuration % len(configuration_list)
    if arguments.gui.lower() == "ggui":
        renderer = GGUI_Simulation(
            initial_configuration=initial_configuration,
            sampler=poisson_disk_sampler,
            configurations=configuration_list,
            name=simulation_name,
            solver=mpm_solver,
            res=(720, 720),
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
    print("###", simulation_name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
