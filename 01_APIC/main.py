import sys, os
tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.simulation import GGUI_Simulation, GUI_Simulation
from parsing import arguments, should_use_cuda_backend, should_use_collocated
from presets import configuration_list
from sampler import PoissonDiskSampler
from apic import APIC

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    ti.init(arch=ti.cuda if should_use_cuda_backend else ti.cpu, debug=True)

    initial_configuration = arguments.configuration % len(configuration_list)
    simulation_name = f"Affine Particle-In-Cell Method"

    # FIXME: changing configuration isn't working

    # The radius for the particles and the Poisson-Disk Sampler:
    # TODO: this could be computed from radius, this should just be n_pc * n_grid^2?!
    n_pc = 8
    max_particles = 500_000
    n_grid = 128 * arguments.quality
    radius = 1 / (n_pc * float(n_grid))  # dx / 4
    dt = 1e-3 / arguments.quality

    solver = APIC(max_particles, n_grid, dt)
    sampler = PoissonDiskSampler(apic_solver=solver, r=radius, k=30)

    simulation = GGUI_Simulation(
        initial_configuration=initial_configuration,
        configurations=configuration_list,
        sampler=sampler,
        solver=solver,
        name=simulation_name,
        res=(720, 720),
        radius=radius,
    )
    simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", simulation_name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
