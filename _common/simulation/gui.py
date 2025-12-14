from _common.samplers import BasePoissonDiskSampler
from _common.configurations import Configuration
from _common.simulation import BaseSimulation
from _common.constants import ColorHEX
from _common.solvers import BaseSolver

from abc import abstractmethod

import taichi as ti


@ti.data_oriented
class GUI_Simulation(BaseSimulation):
    def __init__(
        self,
        name: str,
        res: int,
        solver: BaseSolver,
        configurations: list[Configuration],
        sampler: BasePoissonDiskSampler,
        initial_configuration: int = 0,
    ) -> None:
        """Constructs a  GUI renderer, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configuration: the one configuration for the solver
        """
        super().__init__(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=sampler,
            solver=solver,
        )

        # GUI.
        self.gui = ti.GUI(name, res=res, background_color=ColorHEX.Background)

    def run(self) -> None:
        """Runs this simulation."""
        while self.gui.running:
            if self.gui.get_event(ti.GUI.PRESS):
                if self.gui.event.key == "r":  # pyright: ignore
                    self.reset()
                elif self.gui.event.key == ti.GUI.SPACE:  # pyright: ignore
                    self.is_paused = not self.is_paused
                elif self.gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:  # pyright: ignore
                    break
            if not self.is_paused:
                self.substep()
            self.render()

    @abstractmethod
    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""
        pass
