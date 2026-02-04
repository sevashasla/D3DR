import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_path", type=str)
    parser.add_argument("--images_path", type=str)

    args = parser.parse_args()
    with open(args.poses_path) as f:
        data = json.load(f)

    # process poses
    for frame in data['frames']:
        old_name = frame['file_path']
        file_number = int(os.path.basename(frame['file_path'])[6:-4])
        file_dir = os.path.dirname(frame['file_path'])
        frame['file_path'] = os.path.join(file_dir, f"color_{file_number:05}.png")
        print("rename:", old_name, frame['file_path'])

    with open(args.poses_path, "w") as f:
        json.dump(data, f)
    
    # process images names
    for image_name in os.listdir(args.images_path):
        file_number = int(image_name[6:-4])
        file_dir = os.path.dirname(image_name)
        new_image_name = os.path.join(file_dir, f"color_{file_number:05}.png")
        os.rename(os.path.join(args.images_path, image_name), os.path.join(args.images_path, new_image_name))

if __name__ == "__main__":
    main()