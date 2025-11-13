"""
Small CLI utils for the project.

Important: this module avoids consuming sys.argv on import so other modules
can import it without interfering with the main program's argparse.
"""
import argparse
import os
from typing import Optional, List

parser = argparse.ArgumentParser(usage="python main.py")
parser.add_argument("--from_scratch", action="store_true", dest="from_scratch")
parser.add_argument(
    "-i",
    "--input",
    action="store",
    dest="input",
    default="image",
    choices=["image", "mordred_descriptor"],
)
parser.add_argument(
    "-s",
    "--solvent",
    action="store",
    dest="solvent",
    default="all",
    choices=[
        "all",
        "ethanol",
        "methanol",
        "water",
        "ethyl acetate",
        "acetone",
        "hexane",
        "acetonitrile",
        "diethyl ether",
        "toluene",
        "benzene",
        "pentane",
        "tetrahydrofuran",
        "dimethylsulfoxide",
        "isopropanol",
        "dimethylformamide",
        "cyclohexane",
        "heptane",
        "best_single",
    ],
)
parser.add_argument(
    "-j",
    "--join_mode",
    action="store",
    dest="mode",
    default="concat",
    choices=["concat", "one_hot", "drop"],
)
parser.add_argument("--no_augs", action="store_true", dest="no_augs", default=False)
parser.add_argument("--robot_test", action="store_true", dest="robot_test", default=False)
parser.add_argument(
    "--gpu_idx",
    action="store",
    dest="gpu_idx",
    default="0",
    choices=["0", "1", "2", "3", "4", "5"],
)
parser.add_argument(
    "-m",
    "--model",
    action="store",
    dest="model",
    default="resnet18",
    choices=["resnet18", "convnext_tiny_in22k", "vit_small_patch16_224"],
)


def get_args(argv: Optional[List[str]] = None):
    """Parse and return args; pass argv=[] to parse defaults without consuming sys.argv."""
    return parser.parse_args(argv)


# parse arguments only when run as a program; on import parse defaults (do not consume sys.argv)
if __name__ == "__main__":
    args = get_args()
else:
    args = get_args([])

# set visible CUDA devices from parsed args (safe: uses default when imported)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

__all__ = ["parser", "get_args", "args"]