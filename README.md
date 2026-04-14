### Material Point Method for Two-Way Simulation of Water and Ice with Phase Change
[MLS-MPM](https://dl.acm.org/doi/10.1145/3197517.3201293) implementation of [Augmented MPM for phase-change and varied materials](https://dl.acm.org/doi/10.1145/2601097.2601176), written in [Taichi](https://www.taichi-lang.org/).




### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate MPM
```
You also need to install [cuSPARSE libraries](https://pypi.org/project/nvidia-cusparse-cu12/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for the CUDA backend,
and [Vulkan Drivers](https://developer.nvidia.com/vulkan-driver) for the GGUI frontend. Both of these are optional, but result in better performance and visibility.




### Simulation
```bash
python C_Augmented/main.py --arch=CPU     # runs the simulation on the CPU
python C_Augmented/main.py --arch=CUDA    # runs the simulation on the GPU
```

If Vulkan is not available you can resort to Taichi's older GUI system with
```bash
python C_Augmented/main.py --gui=GUI --configuration=3
```
Keep in mind that the simulation starts paused, pause/unpause is toggled with space.




### Options
```bash
usage: main.py [-h] [-a [{CPU,CUDA}]] [-g [{GGUI,GUI}]] [-s [{Direct,Iterative}]] [-d] [-v] [-c [CONFIGURATION]]

options:
  -h, --help            show this help message and exit
  -a, --arch [{CPU,CUDA}]
                        Choose the Taichi architecture to run on.
  -g, --gui [{GGUI,GUI}]
                        Use GGUI (depends on Vulkan) or GUI system for the simulation.
  -s, --solverType [{Direct,Iterative}]
                        Choose whether to use a direct or iterative solver for the pressure and heat systems.
  -d, --debug           Turn on debugging.
  -v, --verbose         Turn on verbose logging.
  -c, --configuration [CONFIGURATION]
                        Available Configurations:
                        [0] -> Spherefall, Ice
                        [1] -> Dropping Cube, Ice
                        [2] -> Waterjet
                        [3] -> Waterjet & Pool
                        [4] -> Dam Break
                        [5] -> Dam Break, Centered
                        [6] -> Spherefall, Water
                        [7] -> Pool
                        [8] -> Melting Ice Cube, Floating
                        [9] -> Melting Ice Ball, Floating
                        [10] -> Melting Ice Ball
                        [11] -> Melting Ice Cube
                        [12] -> Pool & Ice Cubes
                        [13] -> Freezing Pool
                        [14] -> Waterjet & Ice Cubes
                        [15] -> Waterjet & Smash
                        [16] -> Spherefall, Water vs. Ice
```




### Implemented Methods

#### 01. Affine Particle-in-Cell Method (APIC) for Water
TODO


#### 02. Moving Least Squares Material Point Method (MLS-MPM) for Ice
TODO


#### 03. Augmented MLS-MPM for Water & Ice with Phase Change
TODO
