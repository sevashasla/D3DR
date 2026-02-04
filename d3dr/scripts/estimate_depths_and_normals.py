import argparse
import datetime
import os
import time
from pathlib import Path

time_start = time.time()
print("Start at:", datetime.datetime.now().strftime("%H:%M:%S"))

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input folder")
parser.add_argument(
    "--output_dir", type=str, default=None, help="output folder"
)

parser.add_argument(
    "--dn_splatter_path",
    type=str,
    help="dir with dn-splatter",
    default="/home/skorokho/coding/voi_gs/dn-splatter",
)
parser.add_argument("--run_depth", type=int, default=1, help="do depth?")
parser.add_argument("--run_normals", type=int, default=1, help="do normals?")
parser.add_argument(
    "--normal", type=str, help="", choices=["omni", "dsine"], default="dsine"
)

args = parser.parse_args()
dn_splatter_path = os.path.normpath(args.dn_splatter_path)

if args.output_dir is None:
    args.output_dir = args.input_dir
extra_dir = Path(args.output_dir) / "extra"
depth_dir = Path(args.output_dir) / "extra/depths"
normal_dir = Path(args.output_dir) / "extra/normals"
os.makedirs(args.output_dir, exist_ok=True)

if args.run_normals:
    normal_dir.mkdir(exist_ok=True, parents=True)
    print("RUN NORMALS")
    if args.normal == "omni":
        os.system(
            f"python3 d3dr/scripts/normals_from_pretrain.py "
            f"--data-dir {args.input_dir} --resolution=low --save_path {str(normal_dir)} "
        )
    else:
        os.system(
            f"python3 d3dr/scripts/normals_from_pretrain.py "
            f"--data-dir {args.input_dir} --model-type dsine --save_path {str(normal_dir)}"
        )

if args.run_depth:
    depth_dir.mkdir(exist_ok=True, parents=True)
    print("RUN MONO DEPTH")
    os.system(
        f"python3 d3dr/scripts/depth_from_pretrain.py "
        f"--data-dir {args.input_dir} --save_path {str(depth_dir)} "
    )

print("End at:", datetime.datetime.now().strftime("%H:%M:%S"))
print(f"took {(time.time() - time_start) / 60:.2f} minutes")
