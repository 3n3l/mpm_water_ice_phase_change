from _common.configurations import Configuration

from argparse import ArgumentParser, RawTextHelpFormatter


epilog = "\n\033[91m>>> Press R to [R]eset, [S]PACE to pause/unpause the [S]imulation!\033[0m\n"
parser = ArgumentParser(prog="main.py", epilog=epilog, formatter_class=RawTextHelpFormatter)


def add_configuration(configurations: list[Configuration]):
    help = f"Available Configurations:\n{'\n'.join([f'[{i}] -> {c.name}' for i, c in enumerate(configurations)])}"
    parser.add_argument(
        "-c",
        "--configuration",
        default=0,
        nargs="?",
        help=help,
        type=int,
    )


# TODO: the quality variable isn't used at all at the moment
# quality_help = "Choose a quality multiplicator for the simulation (higher is better)."
# parser.add_argument(
#     "-q",
#     "--quality",
#     default=1,
#     nargs="?",
#     help=quality_help,
#     type=int,
# )

solver_type_help = "Choose the Taichi architecture to run on."
parser.add_argument(
    "-a",
    "--arch",
    default="CPU",
    nargs="?",
    choices=["CPU", "CUDA"],
    help=solver_type_help,
)

ggui_help = "Use GGUI (depends on Vulkan) or GUI system for the simulation."
parser.add_argument(
    "-g",
    "--gui",
    default="GGUI",
    nargs="?",
    choices=["GGUI", "GUI"],
    help=ggui_help,
)

solver_type_help = "Choose whether to use a direct or iterative solver for the pressure and heat systems."
parser.add_argument(
    "-s",
    "--solverType",
    default="Iterative",
    nargs="?",
    choices=["Direct", "Iterative"],
    help=solver_type_help,
)

solver_type_help = "Turn on debugging."
parser.add_argument(
    "-d",
    "--debug",
    default=False,
    action="store_true",
    help=solver_type_help,
)
