import argparse

import open3d as o3d
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Extract point cloud from Gaussian splat checkpoint"
    )
    parser.add_argument("--ckpt_path", help="Path to checkpoint file")
    parser.add_argument("--output_file", help="Path to output PLY file")
    parser.add_argument(
        "--num_points",
        type=int,
        default=100000,
        help="Number of points to sample",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for sampling"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    st = torch.load(args.ckpt_path, map_location="cpu")
    means = st["pipeline"]["_model.gauss_params.means"]

    random_ids = torch.randperm(means.shape[0])[: args.num_points]
    means_sampled = means[random_ids]
    means_np = means_sampled.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_np)
    o3d.io.write_point_cloud(args.output_file, pcd)
    print(f"Saved {args.num_points} points to {args.output_file}")


if __name__ == "__main__":
    main()
