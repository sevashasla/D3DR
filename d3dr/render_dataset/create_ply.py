"""
Processed blender scene and creates a PLY file with points sampled from the scene.

Three sampling strategies are implemented:
1. Randomly sample an object based on the object's area and then sample a face based on the face's area.
2. Randomly sample an object and then randomly sample a face.
3. Randomly sample a point in the bounding box of the scene and find the closest face to the point.
"""

import sys
import numpy as np
import mathutils
import bmesh

# blender importing sucks
sys.path.append("/home/skorokho/.local/lib/python3.11/site-packages/")
from tqdm import tqdm

def get_vertices_and_polygons_mesh(obj, mesh, shift=0):
    vertices = [obj.matrix_world @ v.co for v in mesh.verts]
    to_idx = lambda x: [v.index for v in x]
    polygons = [(np.array(to_idx(p.verts)) + shift).tolist() for p in mesh.faces]
    return vertices, polygons

class BlenderPlyCreator:
    def __init__(self, bpy_data):
        self.bpy_data = bpy_data

        self.objects = []
        self.objects_areas = []
        self.meshes = []
        self.texture_images = []
        self.objects_faces_areas = []
        for obj in self.bpy_data.objects:
            if (
                not obj.hide_render and obj.type == 'MESH' and \
                obj.data.uv_layers.active and \
                not obj.active_material is None and \
                not obj.active_material.node_tree is None
            ):
                # get image of the texture
                texture_node = next((n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE'), None)
                if not texture_node or not texture_node.image:
                    self.texture_images.append(None)
                else:
                    image = texture_node.image
                    self.texture_images.append(image)
                
                # save the object
                self.objects.append(obj)

                # save the mesh
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.faces.ensure_lookup_table()
                self.meshes.append(bm)

                # save the area
                curr_area = self._calculate_area(obj)
                self.objects_areas.append(curr_area)
                
                # save the faces areas
                self.objects_faces_areas.append([])
                for face in bm.faces:
                    self.objects_faces_areas[-1].append(face.calc_area())
                self.objects_faces_areas[-1] = np.array(self.objects_faces_areas[-1])
                self.objects_faces_areas[-1] /= np.sum(self.objects_faces_areas[-1])

        self.objects_areas = np.array(self.objects_areas)
        self.objects_areas /= np.sum(self.objects_areas)
        self._build_bvh_tree_and_bounding_box()

    def _calculate_area(self, obj):
        # save the mesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)

        curr_volume = bm.calc_volume(signed=False)
        curr_area = curr_volume ** (2/3)
        return curr_area

    def _find_obj_by_face_index(self, face_index):
        obj_index = np.where(self.objects_faces_sizes <= face_index)[0].max().item()
        return obj_index

    def _build_bvh_tree_and_bounding_box(self):
        all_vertices = []
        all_polygons = []
        self.objects_faces_sizes = [0] # to find the object and face by the index

        for obj, mesh in zip(self.objects, self.meshes):
            curr_vertices, curr_polygons = get_vertices_and_polygons_mesh(obj, mesh, len(all_vertices))
            all_vertices.extend(curr_vertices)
            all_polygons.extend(curr_polygons)
            self.objects_faces_sizes.append(len(curr_polygons))
        
        # when I get index I will find by 
        # obj_index = np.where(self.<...> <= index).max()
        self.objects_faces_sizes = np.array(self.objects_faces_sizes)
        self.objects_faces_sizes = np.cumsum(self.objects_faces_sizes)

        # find the bb
        all_vertices_np = np.array(all_vertices)
        self.bb_min = np.min(all_vertices_np, axis=0)
        self.bb_max = np.max(all_vertices_np, axis=0)

        # build the bvh tree
        self.bhv_tree = mathutils.bvhtree.BVHTree.FromPolygons(all_vertices, all_polygons)
        return self.bb_min, self.bb_max, self.bhv_tree

        
    def _process_one_uv(self, uv_coord):
        if uv_coord > 1:
            return uv_coord - int(uv_coord)
        elif uv_coord < 0:
            return 1 + uv_coord - int(uv_coord)
        else:
            return uv_coord

    def _process_uv(self, uv):
        return mathutils.Vector([self._process_one_uv(uv[0]), self._process_one_uv(uv[1])])

    def sample(self, obj_idx, face_idx):
        obj = self.objects[obj_idx]
        bm = self.meshes[obj_idx]
        face = bm.faces[face_idx]
        image = self.texture_images[obj_idx]

        # random barycentric coordinates
        r1, r2 = np.random.uniform(size=2)
        r1_05 = r1 ** 0.5
        u = 1 - r1_05
        v = r1_05 * (1 - r2)
        w = 1 - u - v
        point = (face.verts[0].co * u +
                 face.verts[1].co * v +
                 face.verts[2].co * w)
        
        # Transform to world space
        world_point = obj.matrix_world @ point
        normal = obj.matrix_world.to_3x3() @ face.normal

        if image is None:
            return world_point, normal, (0.5, 0.5, 0.5)
        
        # Get UV coordinates
        uv_layer = bm.loops.layers.uv.active
        uv_coords = \
            self._process_uv(face.loops[0][uv_layer].uv) * u + \
            self._process_uv(face.loops[1][uv_layer].uv) * v + \
            self._process_uv(face.loops[2][uv_layer].uv) * w
        
        # Map UV coordinates to pixel color in the texture
        pixel_x = int(uv_coords.x * image.size[0])
        pixel_x = min(max(pixel_x, 0), image.size[0] - 1)
        pixel_y = int(uv_coords.y * image.size[1])
        pixel_y = min(max(pixel_y, 0), image.size[1] - 1)
        color = image.pixels[(pixel_y * image.size[0] + pixel_x) * 4 : (pixel_y * image.size[0] + pixel_x) * 4 + 3]
        return world_point, normal, color

    def create_ply(self, filename, num_points=100_000, seed=42):
        np.random.seed(seed)
        self.num_points = num_points
        points = []
        normals = []
        colors = []
        for i in tqdm(range(num_points)):
            random_strategy = np.random.randint(0, 3)
            if random_strategy == 0:
                # random object by area
                random_obj_idx = np.random.choice(len(self.objects), p=self.objects_areas)
                # random face by area
                random_face_idx = np.random.choice(len(self.objects_faces_areas[random_obj_idx]), p=self.objects_faces_areas[random_obj_idx])
            elif random_strategy == 1:
                random_obj_idx = np.random.choice(len(self.objects))
                random_face_idx = np.random.choice(len(self.objects_faces_areas[random_obj_idx]))
            else:
                random_point = np.random.uniform(self.bb_min, self.bb_max)
                _, _, random_face_idx, _ = self.bhv_tree.find_nearest(random_point)
                random_obj_idx = self._find_obj_by_face_index(random_face_idx)
                random_face_idx -= self.objects_faces_sizes[random_obj_idx + 1]

            point, normal, color = self.sample(random_obj_idx, random_face_idx)
            points.append(point)
            normals.append(normal)
            colors.append(color)

        # save everything to a PLY file
        with open(filename, "w") as f:
            # Header
            f.write(
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )

            for point, normal, color in zip(points, normals, colors):
                x, y, z = point
                nx, ny, nz = normal
                try:
                    r, g, b = color
                except Exception as e:
                    print(e)
                    r, g, b = 0.5, 0.5, 0.5
                r = int(255 * r)
                g = int(255 * g)
                b = int(255 * b)
                f.write(f"{x:8f} {y:8f} {z:8f} {nx:8f} {ny:8f} {nz:8f} {r} {g} {b}\n")

    def __call__(self, filename, num_points=100_000, seed=42):
        self.create_ply(filename, num_points, seed)
