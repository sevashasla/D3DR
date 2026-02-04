import torch
from torch import Tensor

from typing import Optional, Tuple, List, Dict
import open3d as o3d

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_utils import normalize_with_norm
from d3dr.utils.normal_utils import normal_from_depth_image

class MultiViewConsistency:
    """
    Makes a set images multiview consistent using their depths, images, and poses.
    """
    def __init__(
            self, 
            pixel_offset: float = 0.5,
            voxel_size: float = 0.01,
            N: Optional[int] = None,
            device: str = "cpu",
            N_: int = 1,
        ):
        self.pixel_offset = pixel_offset
        self.voxel_size = voxel_size
        self.N = N
        self.N_ = N_
        self.device = device
    
    @torch.no_grad()
    def make_consistent(
        self,
        images: List[torch.Tensor],
        depths: List[Tensor], 
        masks: List[Tensor], 
        normals: Optional[List[Tensor]],
        cameras: Cameras,
        return_mesh: bool = False,
    ):
        # generate point clouds
        pcs = []
        colors_pcs = []
        normals_pcs = []
        x_flat_ids_s = []
        y_flat_ids_s = []
        for i in range(len(images)):
            pc, color, normal, x_flat_ids, y_flat_ids = self.im2pc(
                image=images[i],
                depth=depths[i].squeeze(-1), 
                mask=masks[i].squeeze(-1), 
                normal=None,
                camera=cameras[i][0],
            )

            pcs.append(pc)
            colors_pcs.append(color)
            normals_pcs.append(normal)
            x_flat_ids_s.append(x_flat_ids)
            y_flat_ids_s.append(y_flat_ids)

        # average colors
        colors_av = self.average_colors(
            pcs,
            colors_pcs,
        )

        # assign colors av back
        for i in range(len(images)):
            images[i][y_flat_ids_s[i], x_flat_ids_s[i], :] = colors_av[i]

        if return_mesh:
            mesh = self.generate_voxel_scene(
                scene_pts=torch.cat(pcs, dim=0),
                colors_pts=torch.cat(colors_pcs, dim=0),
            )
            return images, mesh
        return images

    @torch.no_grad()
    def _generate_rays(
        self,
        mask: Tensor,
        camera: Cameras,
    ):
        '''
        mask: (H, W) Mask (bool)
        cameras: (,) Camera
        '''
        H, W = mask.shape

        y, x = torch.meshgrid(
            torch.arange(0, H, self.N_, device=self.device),
            torch.arange(0, W, self.N_, device=self.device),
            indexing="ij",
        )

        x_flat_ids = x.flatten()
        y_flat_ids = y.flatten()

        # take only where mask is True
        take_ids = mask[y_flat_ids, x_flat_ids]
        x_flat_ids = x_flat_ids[take_ids]
        y_flat_ids = y_flat_ids[take_ids]
        
        x_flat = (x_flat_ids.to(torch.float32) + self.pixel_offset)
        y_flat = (y_flat_ids.to(torch.float32) + self.pixel_offset)
        
        z_dir = -torch.ones_like(x_flat) # (H*W)
        x_dir = (x_flat - camera.cx) / camera.fx # (H*W)
        y_dir = (camera.cy - y_flat) / camera.fy # (H*W)
        directions = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (H*W, 3)
        return directions, x_flat_ids, y_flat_ids

    @torch.no_grad()
    def im2pc(
            self,
            image: torch.Tensor,
            depth: torch.Tensor, 
            mask: torch.Tensor, 
            normal: torch.Tensor,
            camera: Cameras,
        ):
        """
        Convert image and depth to point cloud.
        Args:
            image: (H, W, 3) RGB image (float) OpenGL style
            depth: (H, W) Depth map (float)
            mask: (H, W) Mask (bool)
            normal: (H, W, 3) Normal map (float), OpenGL style
            cameras: (,) Camera
        Returns:
            points: (H*W, 3) Point cloud
            colors: (H*W, 3) RGB colors
        """

        # points, _ = normalize_with_norm(points, -1)
        directions, x_flat_ids, y_flat_ids = self._generate_rays(mask, camera)
        z_flat = depth[y_flat_ids, x_flat_ids].flatten().to(torch.float32)

        points = directions * z_flat.unsqueeze(-1)  # (H*W, 3)

        rots = camera.camera_to_worlds[:3, :3]  # (3, 3)
        trans = camera.camera_to_worlds[:3, 3].unsqueeze(0)  # (1, 3)
        points_3d = torch.matmul(points, rots.T) # (H*W, 3)
        points_3d += trans

        colors_pc = image[y_flat_ids, x_flat_ids, :]
        if not normal is None:
            normal = normal[y_flat_ids, x_flat_ids, :]
            normal = torch.matmul(normal, rots.T)  # (H*W, 3)
        
        return points_3d, colors_pc, normal, x_flat_ids, y_flat_ids
    
    @torch.no_grad()
    def average_colors(
        self,
        pcs, 
        colors_pcs,
    ):
        """
        pcs: array of Float Tensor (N_i, 3) M
        colors_pcs: array of Float Tensor (N_i, 3) M
        """

        bboxes_min = [pc.amin(dim=0) for pc in pcs] # (M, 3)
        bboxes_max = [pc.amax(dim=0) for pc in pcs] # (M, 3)
        bbox_min = torch.stack(bboxes_min, dim=0).amin(dim=0) # (3,)
        bbox_max = torch.stack(bboxes_max, dim=0).amax(dim=0) # (3,)

        if not self.N is None:
            voxel_size = (bbox_max - bbox_min).prod().item() / self.N
        else:
            voxel_size = self.voxel_size
        
        ids_all = []
        for pc in pcs:
            ids = (pc - bbox_min) / voxel_size
            ids = torch.floor(ids).long() # (N_i, 3)
            ids_all.append(ids)
        
        ids_all = torch.cat(ids_all, dim=0) # (N, 3)
        ids_unique, ids_inv = torch.unique(ids_all, dim=0, return_inverse=True) # (N', 3), (N,)

        colors_all = torch.cat(colors_pcs, dim=0) # (N, 3)

        colors_mean = torch.zeros_like(ids_unique, dtype=torch.float32) # (N', 3)
        for i in range(3):
            colors_mean[:, i].scatter_reduce_(0, ids_inv, colors_all[:, i], reduce="mean", include_self=False) # (N', 3)

        colors_all_av = colors_mean[ids_inv]
        colors_final = colors_all_av.split([len(c) for c in colors_pcs])

        return colors_final
    
    @torch.no_grad()
    def average_colors_2(self):
        # Use Radius graph instead of voxel-based graph https://github.com/rusty1s/pytorch_cluster
        raise NotImplementedError("Not implemented yes :(")

    def generate_voxel_scene( 
        self,
        scene_pts: torch.FloatTensor, 
        colors_pts: torch.FloatTensor,
    ):
        
        vs = self.voxel_size
        bb_scene = (scene_pts.min(dim=0).values, scene_pts.max(dim=0).values)
                
        # create a mesh from scene_pts
        N_voxels = torch.ceil((bb_scene[1] - bb_scene[0]) / vs).to(torch.long)
        
        voxels_ids = torch.floor((scene_pts - bb_scene[0]) / vs)
        voxels_ids = torch.stack([voxels_ids[:, i].clip(0, N_voxels[i] - 1).to(torch.long) for i in range(3)], axis=1)
        voxels_ids_uniq, inverse_indices = torch.unique(voxels_ids, return_inverse=True, dim=0)
        
        ids_x = voxels_ids_uniq[:, 0]
        ids_y = voxels_ids_uniq[:, 1]
        ids_z = voxels_ids_uniq[:, 2]

        # find voxels representations
        voxels_x = bb_scene[0][0] + torch.arange(0, N_voxels[0], device=self.device) * vs
        voxels_y = bb_scene[0][1] + torch.arange(0, N_voxels[1], device=self.device) * vs
        voxels_z = bb_scene[0][2] + torch.arange(0, N_voxels[2], device=self.device) * vs

        voxels_x = voxels_x[ids_x]
        voxels_y = voxels_y[ids_y]
        voxels_z = voxels_z[ids_z]

        # generate cubes
        vectors = torch.cat([
            torch.stack([voxels_x + vs, voxels_y,       voxels_z], dim=1),
            torch.stack([voxels_x + vs, voxels_y + vs,  voxels_z], dim=1),
            torch.stack([voxels_x,      voxels_y + vs,  voxels_z], dim=1),
            torch.stack([voxels_x,      voxels_y,       voxels_z], dim=1),
            torch.stack([voxels_x + vs, voxels_y,       voxels_z + vs], dim=1),
            torch.stack([voxels_x + vs, voxels_y + vs,  voxels_z + vs], dim=1),
            torch.stack([voxels_x,      voxels_y + vs,  voxels_z + vs], dim=1),
            torch.stack([voxels_x,      voxels_y,       voxels_z + vs], dim=1),
        ], dim=0)

        # update the cubes sizes because of their widths
        # find centers of the cubes
        cubes_centers = torch.stack([ch for ch in vectors.chunk(8, dim=0)], dim=0).mean(dim=0)
        cubes_centers = cubes_centers.repeat(8, 1)
        # find final widths in every cube as mean of its gaussians
        colors = torch.zeros((voxels_x.shape[0], 3), device=self.device, dtype=torch.float32)
        for i in range(3):
            colors[:, i].scatter_reduce_(dim=0, index=inverse_indices, src=colors_pts[:, i], reduce="max", include_self=False)

        # generate faces
        N = ids_x.shape[0]
        arn = torch.arange(N, device=self.device)
        faces = torch.cat([
            torch.stack([arn + (1 - 1) * N, arn + (2 - 1) * N, arn + (5 - 1) * N], dim=1),
            torch.stack([arn + (2 - 1) * N, arn + (6 - 1) * N, arn + (5 - 1) * N], dim=1),
            torch.stack([arn + (2 - 1) * N, arn + (3 - 1) * N, arn + (6 - 1) * N], dim=1),
            torch.stack([arn + (3 - 1) * N, arn + (7 - 1) * N, arn + (6 - 1) * N], dim=1),
            torch.stack([arn + (3 - 1) * N, arn + (4 - 1) * N, arn + (7 - 1) * N], dim=1),
            torch.stack([arn + (4 - 1) * N, arn + (8 - 1) * N, arn + (7 - 1) * N], dim=1),
            torch.stack([arn + (4 - 1) * N, arn + (1 - 1) * N, arn + (8 - 1) * N], dim=1),
            torch.stack([arn + (1 - 1) * N, arn + (5 - 1) * N, arn + (8 - 1) * N], dim=1),
            torch.stack([arn + (1 - 1) * N, arn + (4 - 1) * N, arn + (2 - 1) * N], dim=1),
            torch.stack([arn + (4 - 1) * N, arn + (3 - 1) * N, arn + (2 - 1) * N], dim=1),
            torch.stack([arn + (5 - 1) * N, arn + (6 - 1) * N, arn + (8 - 1) * N], dim=1),
            torch.stack([arn + (6 - 1) * N, arn + (7 - 1) * N, arn + (8 - 1) * N], dim=1),
        ], dim=0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vectors.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.repeat(8, 1).cpu().numpy())
        return mesh
