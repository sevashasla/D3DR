import os
import argparse
import random
import re
import glob

def make_textures(args):
    random.seed(args.random_state)
    with open(args.mtl_path) as f:
        input_file_data = f.read().splitlines()
    
    unknown_texture_folder = os.path.join(args.texture_path, "unknown")

    output_file_data = []
    for line in input_file_data:
        if line.startswith("map_Kd"):
            continue
        if line.startswith("newmtl"):
            mat_name = line[len("newmtl"):]
            # Borrowed from blenderproc
            # https://github.com/DLR-RM/BlenderProc/blob/8ee2265dbb36099aef1f8e77acda3a2042c249b6/blenderproc/python/loader/SceneNetLoader.py#L80C20-L95C91
            if "." in mat_name:
                mat_name = mat_name[:mat_name.find(".")]
            mat_name = mat_name.replace(" ", "")
            mat_name = mat_name.replace("_", "")
            # remove all digits from the string
            mat_name = ''.join([i for i in mat_name if not i.isdigit()])
            image_paths = glob.glob(os.path.join(args.texture_path, mat_name, "*"))
            if not image_paths:
                if not os.path.exists(unknown_texture_folder):
                    raise FileNotFoundError(f"The unknown texture folder does not exist: "
                                            f"{unknown_texture_folder}, check if it was set correctly "
                                            f"via the config.")
                image_paths = glob.glob(os.path.join(unknown_texture_folder, "*"))
                if not image_paths:
                    raise FileNotFoundError(f"The unknown texture folder did not contain any "
                                            f"textures: {unknown_texture_folder}")
            image_paths.sort()
            image_path = random.choice(image_paths)
            output_file_data.append(line)
            output_file_data.append(f"map_Kd {image_path}")
        else:
           output_file_data.append(line)
    with open(args.mtl_path, "w") as f:
        f.write("\n".join(output_file_data))
    # print("\n".join(output_file_data))    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtl_path", type=str, default="/home/sevashasla/Documents/blender_data/SceneNetData/1Bathroom/1_labels.mtl")
    parser.add_argument("--texture_path", type=str, default="/home/sevashasla/Documents/blender_data/texture_library")
    parser.add_argument("--random_state", type=int, default=0)

    args = parser.parse_args()
    if args.texture_path is None:
        grandgrandparent = os.path.dirname(os.path.dirname(os.path.dirname(args.mtl_path)))
        args.texture_path = os.path.join(grandgrandparent, "texture_library")
    print(args)
    make_textures(args)

if __name__ == "__main__":
    main()

