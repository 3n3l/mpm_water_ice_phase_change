import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.simulation import GGUI_Simulation, GUI_Simulation
from _common.samplers import BasePoissonDiskSampler

from parsing import arguments, should_use_cuda_backend, should_use_collocated
from _common.presets import water_presets
from apic import APIC

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    ti.init(arch=ti.cuda if should_use_cuda_backend else ti.cpu, debug=False)

    initial_configuration = arguments.configuration % len(water_presets)
    name = f"Affine Particle-In-Cell Method"
    prefix = f"APIC"

    max_particles, n_grid = 300_000, 128
    radius = 1 / (6 * float(n_grid))  # 6 particles per cell
    vol_0 = math.pi * (radius**2)

    solver = APIC(max_particles=max_particles, n_grid=n_grid, vol_0=vol_0)
    sampler = BasePoissonDiskSampler(solver=solver, r=radius, k=30)
    simulation = GGUI_Simulation(
        initial_configuration=initial_configuration,
        configurations=water_presets,
        sampler=sampler,
        solver=solver,
        prefix=prefix,
        res=(720, 720),
        radius=radius,
        name=name,
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
