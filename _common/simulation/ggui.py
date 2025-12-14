from _common.constants import ColorRGB, State, Simulation
from _common.configurations import Configuration
from _common.samplers import BasePoissonDiskSampler
from _common.simulation import BaseSimulation
from _common.solvers import BaseSolver

from typing import Callable

import taichi as ti


class DrawingOption:
    """
    This holds name, state and a callable for drawing a chosen foreground/background.
    """

    def __init__(self, name: str, is_active: bool, call_draw: Callable) -> None:
        self.is_active = is_active
        self.draw = call_draw
        self.name = name


@ti.data_oriented
class GGUI_Simulation(BaseSimulation):
    def __init__(
        self,
        configurations: list[Configuration],
        sampler: BasePoissonDiskSampler,
        res: tuple[int, int],
        solver: BaseSolver,
        prefix: str,
        name: str,
        initial_configuration: int = 0,
        radius=0.0015,
    ) -> None:
        super().__init__(
            initial_configuration=initial_configuration,
            configurations=configurations,
            prefix=prefix,
            sampler=sampler,
            solver=solver,
            name=name,
        )

        # GGUI.
        self.window = ti.ui.Window(name, res, fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.radius = radius

        # Fields that hold certain colors, must be update in each draw call.
        self.temperature_colors_p = ti.Vector.field(3, dtype=ti.f32, shape=self.solver.max_particles)
        # TODO: also move the phase colors here, then only update the phase colors when drawing the phase?!

        # Scratch field, will be updated with the visible cells of the offset fields of the solver.
        self.scratch_field = ti.field(dtype=ti.i32, shape=(self.solver.n_grid, self.solver.n_grid))

        # Construct a vector field as a heat map:
        self.heat_map_length = len(ColorRGB.HeatMap)
        self.heat_map = ti.Vector.field(3, dtype=ti.f32, shape=self.heat_map_length)
        for i, color in enumerate(ColorRGB.HeatMap):
            self.heat_map[i] = color

        # Values to control the drawing of the temperature:
        # TODO: these should be moved somewhere else
        self.should_normalize_temperature = False

        # Foreground Options:
        self.foreground_options = [
            DrawingOption("Temperature", False, self.draw_temperature_p),
            DrawingOption("Nothing", False, lambda: None),
            DrawingOption("Phase", True, self.draw_phase_p),
        ]

        # Background Options:
        self.background_options = [
            DrawingOption("Classification", False, lambda: self.show_contour(self.solver.classification_c)),
            DrawingOption("Temperature", False, lambda: self.show_contour(self.solver.temperature_c)),
            DrawingOption("Background", True, lambda: self.canvas.set_background_color(ColorRGB.Background)),
            DrawingOption("Mass", False, lambda: self.show_contour(self.solver.mass_c)),
        ]

    def show_configurations(self) -> None:
        """
        Show all possible configurations inside own subwindow, choosing one will
        load that configuration and reset the solver.
        """
        prev_configuration_id = self.configuration_id
        # with self.gui.sub_window("Configurations", 0.01, 0.01, 0.48, 0.48) as subwindow:
        with self.gui.sub_window("Configurations", 0.01, 0.01, 0.65, 0.64) as subwindow:
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if subwindow.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                _id = self.configuration_id
                configuration = self.configurations[_id]
                self.load_configuration(configuration)
                self.is_paused = True

    def show_foreground_options(self) -> None:
        """
        Show the foreground drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Foreground", 0.67, 0.01, 0.32, 0.24) as subwindow:
            for option in self.foreground_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.foreground_options:
                        _option.is_active = False
                    option.is_active = True

    def show_background_options(self) -> None:
        """
        Show the background drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Background", 0.67, 0.26, 0.32, 0.24) as subwindow:
            for option in self.background_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.background_options:
                        _option.is_active = False
                    option.is_active = True

    def show_parameters(self) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        """
        pass
        # # with self.gui.sub_window("Parameters", 0.01, 0.51, 0.98, 0.48) as subwindow:
        # with self.gui.sub_window("Parameters", 0.01, 0.66, 0.98, 0.33) as subwindow:
        #     self.solver.theta_c_ice[None] = subwindow.slider_float(
        #         text="Critical Compression  [Ice]",
        #         old_value=self.solver.theta_c_ice[None],
        #         minimum=1e-2,
        #         maximum=1e-1,
        #     )
        #     self.solver.theta_s_ice[None] = subwindow.slider_float(
        #         text="Critical Stretch      [Ice]",
        #         old_value=self.solver.theta_s_ice[None],
        #         minimum=1e-3,
        #         maximum=1e-2,
        #     )
        #     self.solver.zeta_ice[None] = subwindow.slider_int(
        #         text="Hardening Coefficient [Ice]",
        #         old_value=self.solver.zeta_ice[None],
        #         minimum=1,
        #         maximum=20,
        #     )
        #     self.solver.E_ice[None] = subwindow.slider_float(
        #         text="Young's Modulus       [Ice]",
        #         old_value=self.solver.E_ice[None],
        #         minimum=4.8e4,
        #         maximum=5.5e5,
        #     )
        #     self.solver.nu_ice[None] = subwindow.slider_float(
        #         text="Poisson's Ratio       [Ice]",
        #         old_value=self.solver.nu_ice[None],
        #         minimum=0.1,
        #         maximum=0.4,
        #     )
        #     self.solver.ambient_temperature[None] = subwindow.slider_int(
        #         text="Ambient Temperature",
        #         old_value=int(self.solver.ambient_temperature[None]),  # pyright: ignore
        #         minimum=-273,
        #         maximum=273,
        #     )
        #     self.solver.boundary_temperature[None] = subwindow.slider_int(
        #         text="Boundary Temperature",
        #         old_value=int(self.solver.boundary_temperature[None]),  # pyright: ignore
        #         minimum=-273,
        #         maximum=273,
        #     )
        #
        # E = self.solver.E_ice[None]
        # nu = self.solver.nu_ice[None]
        # self.solver.lambda_0_ice[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        # self.solver.mu_0_ice[None] = E / (2 * (1 + nu))

    def show_buttons(self) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        """
        with self.gui.sub_window("Settings", 0.67, 0.51, 0.32, 0.14) as subwindow:
            if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
                # This button toggles between saving frames and not saving frames.
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.dump_frames()
                else:
                    self.create_video()
            if subwindow.button(" Reset Particles "):
                self.reset()
            if subwindow.button(" Start Simulation"):
                self.is_paused = False

    def show_settings(self) -> None:
        """
        Show settings in a GGUI subwindow, this should be called once per generated frames
        and will only show these settings if the simulation is paused at the moment.
        """
        if not self.is_paused or not self.should_show_settings:
            self.is_showing_settings = False
            return  # don't bother

        self.is_showing_settings = True
        self.show_foreground_options()
        self.show_background_options()
        self.show_configurations()
        self.show_parameters()
        self.show_buttons()

    def handle_events(self) -> None:
        """
        Handle key presses arising from window events.
        """
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in ["h"]:
                self.should_show_settings = not self.should_show_settings
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    @ti.kernel
    def update_temperature_p(self):
        max_temperature = Simulation.MaximumTemperature
        min_temperature = Simulation.MininumTemperature

        # Get min, max values for normalization:
        if self.should_normalize_temperature:
            for p in self.temperature_colors_p:
                if self.solver.state_p[p] == State.Hidden:
                    continue  # ignore uninitialized particles
                temperature = self.solver.temperature_p[p]
                if temperature > max_temperature:
                    max_temperature = temperature
                elif temperature < min_temperature:
                    min_temperature = temperature

        # Affine combination of the colors based on min, max values and the temperature:
        max_index, min_index = self.heat_map_length - 1, 0
        factor = ti.cast(max_index, ti.f32)
        for p in self.temperature_colors_p:
            if self.solver.state_p[p] == State.Hidden:
                continue  # ignore uninitialized particles
            t = self.solver.temperature_p[p]
            a = (t - min_temperature) / (max_temperature - min_temperature)
            color1 = self.heat_map[ti.max(min_index, ti.floor(factor * a, ti.i8))]
            color2 = self.heat_map[ti.min(max_index, ti.ceil(factor * a, ti.i8))]
            self.temperature_colors_p[p] = ((1 - a) * color1) + (a * color2)

    def draw_temperature_p(self) -> None:
        """
        Draw the temperature for each particle.
        """
        # self.update_temperature_p()
        self.canvas.circles(
            per_vertex_color=self.temperature_colors_p,
            centers=self.solver.position_p,
            radius=self.radius,
        )

    def draw_phase_p(self) -> None:
        """
        Draw the phase for each particle.
        """
        self.canvas.circles(
            per_vertex_color=self.solver.color_p,
            centers=self.solver.position_p,
            radius=self.radius,
        )

    @ti.kernel
    def update_scratch_field(self, scalar_field: ti.template()): # pyright: ignore
        for i,j in ti.ndrange(self.solver.n_grid, self.solver.n_grid):
            self.scratch_field[i,j] = scalar_field[i,j]

    def show_contour(self, scalar_field) -> None:
        """
        Show the contour of a given scalar field.
        """
        # Because the fields of the solver are offset to move the boundary beyond the visible
        # portion of the window, we need to remove the invisible parts, otherwise the contour
        # will be offset.
        self.update_scratch_field(scalar_field)
        self.canvas.contour(self.scratch_field, cmap_name="magma", normalize=True)

    def render(self) -> None:
        """
        Renders the simulation with the data from the MLS-MPM solver.
        """
        # Draw chosen foreground/brackground, NOTE: foreground must be drawn last.
        for option in self.background_options + self.foreground_options:
            if option.is_active:
                option.draw()

        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())

        self.window.show()

    def run(self) -> None:
        """Runs this simulation."""
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
            self.render()
