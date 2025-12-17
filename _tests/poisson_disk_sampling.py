import utils

from _common.configurations import Rectangle, Circle
from _common.samplers import PoissonDiskSampler
from _common.constants import Ice, ColorRGB, State

import taichi as ti

ti.init(arch=ti.cpu, debug=True)

max_particles = 50_000
position_p = ti.Vector.field(2, dtype=float, shape=max_particles)
state_p = ti.field(dtype=float, shape=max_particles)

n_particles = ti.field(dtype=int, shape=())

state_p.fill(State.Hidden)
position_p.fill([42, 42])
n_particles[None] = 0


@ti.data_oriented
class PoissonDiskSampler(PoissonDiskSampler):
    def __init__(
        self,
        position_p: ti.template(),  # pyright: ignore
        state_p: ti.template(),  # pyright: ignore
        max_p: int,
        r: float,
        k: int,
    ) -> None:
        super().__init__(position_p, state_p, max_p, r, k)
        self._head = n_particles


@ti.kernel
def naive_add_rectangle(n_new: int, rectangle: ti.template()):  # pyright: ignore
    for p in ti.ndrange((n_particles[None], n_particles[None] + n_new)):
        x = ti.random() * rectangle.width + rectangle.x
        y = ti.random() * rectangle.height + rectangle.y
        position_p[p] = [x, y]
    n_particles[None] += n_new


@ti.kernel
def naive_add_circle(n_new: int, circle: ti.template()):  # pyright: ignore
    for p in ti.ndrange((n_particles[None], n_particles[None] + n_new)):
        t = 2 * ti.math.pi * ti.random()
        r = circle.radius * ti.math.sqrt(ti.random())
        x = (r * ti.sin(t)) + circle.x
        y = (r * ti.cos(t)) + circle.y
        position_p[p] = [x, y]
    n_particles[None] += n_new


def main() -> None:
    window = ti.ui.Window(
        "Poisson Disk Sampling [LEFT] vs. Naive Implementation [RIGHT]",
        res=(720, 720),
        fps_limit=60,
    )
    canvas = window.get_canvas()

    ### Naive Sampling
    n_samples = 5_000
    naive_add_circle(
        n_samples,
        Circle(
            material=Ice,  # pyright: ignore
            radius=0.1,
            velocity=(0, 0),
            center=(0.75, 0.75),
        ),
    )
    naive_add_rectangle(
        n_samples,
        Rectangle(
            material=Ice,  # pyright: ignore
            size=(0.2, 0.2),
            velocity=(0, 0),
            lower_left=(0.65, 0.15),
        ),
    )

    ### Poisson Disk Sampling
    pds = PoissonDiskSampler(position_p, state_p, max_particles, r=0.002, k=30)
    pds.add_geometry(
        Circle(
            material=Ice,  # pyright: ignore
            radius=0.1,
            velocity=(0, 0),
            center=(0.25, 0.75),
        )
    )
    pds.add_geometry(
        Rectangle(
            material=Ice,  # pyright: ignore
            size=(0.2, 0.2),
            velocity=(0, 0),
            lower_left=(0.15, 0.15),
        )
    )

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                window.running = False

        canvas.set_background_color(ColorRGB.Background)
        canvas.circles(color=Ice.Color, centers=position_p, radius=0.001)
        window.show()


if __name__ == "__main__":
    main()
