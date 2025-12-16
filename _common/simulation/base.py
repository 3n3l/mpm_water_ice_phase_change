from _common.configurations import Configuration
from _common.samplers import BasePoissonDiskSampler
from _common.solvers import BaseSolver

from abc import abstractmethod
from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class BaseSimulation:
    def __init__(
        self,
        configurations: list[Configuration],
        sampler: BasePoissonDiskSampler,
        solver: BaseSolver,
        prefix: str,
        name: str,
        initial_configuration: int = 0,
    ) -> None:
        """Constructs a Renderer object, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        # State.
        self.is_paused = True
        self.should_write_to_disk = False
        self.should_create_video = True
        self.should_create_gif = False
        self.is_showing_settings = not self.is_paused  # wether the settings are showing
        self.should_show_settings = True  # wether the settings should be shown

        # Name and video/gif prefix
        self.video_prefix = prefix
        self.name = name

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # Solvers and samplers
        self.sampler = sampler
        self.solver = solver

        # Load the initial configuration and reset the solver to this configuration.
        self.current_frame = 0
        self.configurations = configurations
        self.configuration_id = initial_configuration
        self.load_configuration(configurations[self.configuration_id])

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def has_loadable_geometry(self):
        if len(self.subsequent_geometries) <= 0:
            return False

        return self.current_frame == self.subsequent_geometries[0].frame_threshold

    def substep(self) -> None:
        self.current_frame += 1

        # Load all remaining geometries with a satisfied frame threshold:
        while self.has_loadable_geometry():
            self.sampler.add_geometry(self.subsequent_geometries.pop(0))

        self.solver.substep()

    def dump_frames(self) -> None:
        """
        Creates an output directory, a VideoManager in this directory and then dumps frames to this directory.
        """
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        title = f"{self.video_prefix}_{date}"
        output_dir = f"{self.parent_dir}/{title}"
        os.makedirs(output_dir)
        self.video_manager = ti.tools.VideoManager(
            output_dir=output_dir,
            video_filename=title,
            automatic_build=False,
            framerate=60,
        )

    def create_video(self) -> None:
        """
        Converts stored frames in the before created output directory to a video.
        """
        self.video_manager.make_video(gif=self.should_create_gif, mp4=self.should_create_video)

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.configuration = configuration
        self.reset()

    def reset(self) -> None:
        """
        Reset the simulation.
        """
        self.solver.reset(self.configuration)
        self.current_frame = 0

        # We copy this, so we can pop from this list and check the length:
        self.subsequent_geometries = self.configuration.subsequent_geometries.copy()

        # Load all the initial geometries into the solver:
        for geometry in self.configuration.initial_geometries:
            self.sampler.add_geometry(geometry)
