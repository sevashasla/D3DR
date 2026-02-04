import json
import os
from copy import deepcopy

import bpy
import mathutils
import numpy as np

# EMPTY LOOK AT LOCATION IS Vector((-0.016989149153232574, -0.08221031725406647, 0.9884341955184937))

VIEWS = 10
N_LIGHTS = (1, 4)
RESOLUTION = 400
OBJ_SCALE = 0.1
RESULTS_PATH = "/home/sevashasla/Documents/blender_data/bathroom_2_scene"
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = "PNG"
SMALLER_R = 0.8
NAME_LOOK_AT = "EmptyLookAt"

fp = bpy.path.abspath(RESULTS_PATH)

RESULT_OBJ_SCENE_PATH = os.path.join(RESULTS_PATH, "obj_scene")
RESULT_SCENE_PATH = os.path.join(RESULTS_PATH, "scene")
RESULT_OBJ_PATH = os.path.join(RESULTS_PATH, "obj")

os.makedirs(fp, exist_ok=True)
os.makedirs(RESULT_OBJ_SCENE_PATH, exist_ok=True)
os.makedirs(RESULT_SCENE_PATH, exist_ok=True)
os.makedirs(RESULT_OBJ_PATH, exist_ok=True)

np.random.seed(41)


def get_vertices_and_polygons(obj, shift=0):
    mesh = obj.to_mesh()
    vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
    polygons = [(np.array(p.vertices) + shift).tolist() for p in mesh.polygons]
    return vertices, polygons


def build_bvh_tree(consider_collections=None, exclude_collections=None):
    if consider_collections is None:
        consider_collections = [
            collection.name for collection in bpy.data.collections
        ]
    if exclude_collections is None:
        exclude_collections = []
    consider_collections = set(consider_collections)
    exclude_collections = set(exclude_collections)
    take_collections = consider_collections - exclude_collections

    all_vertices = []
    all_polygons = []

    for obj in bpy.context.scene.objects:
        if (
            obj.type == "MESH"
            and obj.users_collection[0].name in take_collections
        ):
            curr_vertices, curr_polygons = get_vertices_and_polygons(
                obj, len(all_vertices)
            )
            all_vertices.extend(curr_vertices)
            all_polygons.extend(curr_polygons)

    bhv_tree = mathutils.bvhtree.BVHTree.FromPolygons(
        all_vertices, all_polygons
    )
    return bhv_tree


# --------------------------------------------------------------------------------------


def bounding_box(collection_name=None):
    min_corner = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    max_corner = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

    if collection_name is None:
        go_through = bpy.data.objects
    else:
        go_through = bpy.data.collections[collection_name].all_objects

    for obj in go_through:
        # Only consider mesh objects (or modify this as needed)
        if obj.type == "MESH":
            # Get the bounding box in local coordinates and convert to world coordinates
            bbox_corners = [
                obj.matrix_world @ mathutils.Vector(corner)
                for corner in obj.bound_box
            ]

            # Update the min and max corner of the bounding box
            for corner in bbox_corners:
                min_corner = mathutils.Vector(
                    (min(min_corner[i], corner[i]) for i in range(3))
                )
                max_corner = mathutils.Vector(
                    (max(max_corner[i], corner[i]) for i in range(3))
                )

    return min_corner, max_corner


# --------------------------------------------------------------------------------------


def add_light(name, in_bb, except_bb=None):
    # create random light parameters
    point_light_loc = np.random.uniform(*in_bb)
    if except_bb is not None:
        except_bb = (np.array(except_bb[0]), np.array(except_bb[1]))
        while np.all(except_bb[0] <= point_light_loc) and np.all(
            point_light_loc <= except_bb[1]
        ):
            point_light_loc = np.random.uniform(*in_bb)

    light_data = bpy.data.lights.new(name=name + "_data", type="POINT")
    light_data.energy = np.random.uniform(30, 100)

    # create light
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    light_object.location = point_light_loc
    bpy.context.collection.objects.link(light_object)


# --------------------------------------------------------------------------------------


def parent_obj_to_camera(b_camera, point_look_at):
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = point_look_at
    #    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


# --------------------------------------------------------------------------------------


def set_camera(position, rotation_angle_z_local, cam=None):
    if cam is None:
        cam = bpy.data.objects["Camera"]
    cam.rotation_euler[2] = rotation_angle_z_local
    cam.location = position


# --------------------------------------------------------------------------------------


# only for debugging
def draw_sphere(location):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=location)


# --------------------------------------------------------------------------------------


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


# --------------------------------------------------------------------------------------


def render_scene(out_data, store_path, bhv_tree):
    # I will use quite a lot of global variables
    # here so I decided to write them down
    global VIEWS, radius, b_empty, EMPTY_LOCATION, tree
    direction_angle_z = 0
    DIRECTION_ANGLE_Z_RANGE = (-45, 45)
    DIRECTION_ANGLE_Z_MAW = 0.3  # moving average weight

    ROTATION_CAMERA_Z_RANGE = (-15, 15)

    B_EMPTY_RADIUS = radius / 10
    B_EMPTY_MAW = 0.3

    cam = bpy.data.objects["Camera"]  # it's kinda global
    scene = bpy.context.scene  # it's kinda global
    b_empty.location = EMPTY_LOCATION

    for i in range(VIEWS):
        direction_angle_xy = 2 * np.pi * (i / VIEWS)

        direction_angle_z += (1 - DIRECTION_ANGLE_Z_MAW) * (
            np.random.uniform(*DIRECTION_ANGLE_Z_RANGE) - direction_angle_z
        )

        curr_direction = mathutils.Vector(
            (
                np.cos(direction_angle_xy),
                np.sin(direction_angle_xy),
                np.sin(direction_angle_z),
            )
        )
        curr_direction.normalize()

        b_empty_location_new_hat = (
            EMPTY_LOCATION
            + B_EMPTY_RADIUS
            * mathutils.Vector(np.random.uniform(-1, 1, 3).tolist())
        )
        b_empty.location += (1 - B_EMPTY_MAW) * (
            b_empty_location_new_hat - b_empty.location
        )

        camera_rotation_z = np.deg2rad(
            np.random.uniform(*ROTATION_CAMERA_Z_RANGE)
        )

        # shoot rays from the center and find the first hit. It will be the radius
        _, _, _, curr_distance = bhv_tree.ray_cast(
            b_empty.location, curr_direction
        )
        if curr_distance:
            curr_distance = min(curr_distance, radius)
        else:
            curr_distance = radius
        position = b_empty.location + curr_direction * curr_distance

        set_camera(position, camera_rotation_z, cam=cam)
        # render a frame
        scene.render.filepath = store_path + "/color_" + str(i)

        tree.nodes["Depth Output"].base_path = store_path
        tree.nodes["Depth Output"].file_slots[0].path = "/depth_" + str(i)

        tree.nodes["Normal Output"].base_path = store_path
        tree.nodes["Normal Output"].file_slots[0].path = "/normal_" + str(i)

        bpy.ops.render.render(write_still=True)  # render still

        frame_data = {
            "file_path": scene.render.filepath,
            "transform_matrix": listify_matrix(cam.matrix_world),
        }
        out_data["frames"].append(frame_data)

        with open(os.path.join(store_path, "transforms.json"), "w") as out_file:
            json.dump(out_data, out_file, indent=4)

    b_empty.location = EMPTY_LOCATION


# --------------------------------------------------------------------------------------


class RenderSceneContextManager:
    def __init__(self):
        self.visible_objects_state = {
            obj.name: obj.hide_render for obj in bpy.data.objects
        }

    def __enter__(self):
        """
        hide object elements
        """
        # make the object invisible during rendering
        for obj in bpy.data.objects:
            if obj.users_collection[0].name == "ObjectCollection":
                obj.hide_render = True

    def __exit__(self, exc_type, exc_val, exc_traceback):
        global b_empty
        if exc_type is not None:
            print(exc_val)

        # revert it back
        for obj in bpy.data.objects:
            if obj.users_collection[0].name == "ObjectCollection":
                obj.hide_render = self.visible_objects_state[obj.name]

        b_empty.location = EMPTY_LOCATION


# --------------------------------------------------------------------------------------


class RenderObjectContextManager:
    def __init__(self, add_lights=True):
        self.add_lights = add_lights
        self.visible_objects_state = {
            obj.name: obj.hide_render for obj in bpy.data.objects
        }

    def __enter__(self):
        """
        add here
        1) shift object to the center
        2) add lights
        3) hide all other objects during rendering
        """
        global min_corner_obj, max_corner_obj
        global out_data_object_scene
        # now hide everything except the object
        for obj in bpy.data.objects:
            if obj.users_collection[0].name != "ObjectCollection":
                obj.hide_render = True
        # and except the camera, of couse
        bpy.data.objects["Camera"].hide_render = False

        # place it to the center
        self.object_center_initial = (min_corner_obj + max_corner_obj) / 2
        self.object_rotation_initial = out_data_object_scene["euler_rotation"]
        self.radius_object = (
            np.linalg.norm(max_corner_obj - min_corner_obj).item() / 2
        )
        corner = mathutils.Vector(
            (self.radius_object, self.radius_object, self.radius_object)
        )
        for obj in bpy.data.collections["ObjectCollection"].objects:
            if obj.parent is None:
                obj.location = obj.location + (-self.object_center_initial)
                obj.rotation_euler[0] = 0
                obj.rotation_euler[1] = 0
                obj.rotation_euler[2] = 0

        bpy.context.view_layer.update()

        new_min_corner_obj, new_max_corner_obj = bounding_box(
            "ObjectCollection"
        )
        self.object_center_new = (new_min_corner_obj + new_max_corner_obj) / 2

        # add lights
        if self.add_lights:
            self.num_lights = int((3 * self.radius_object)) + np.random.randint(
                *N_LIGHTS
            )

            for light_idx in range(self.num_lights):
                add_light(
                    f"light_{light_idx}",
                    (
                        self.object_center_new - corner * 1.2,
                        self.object_center_new + corner * 1.2,
                    ),
                    except_bb=(new_min_corner_obj, new_max_corner_obj),
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        global b_empty
        if exc_type is not None:
            print(exc_val)

        if self.add_lights:
            for light_idx in range(self.num_lights):
                bpy.data.objects.remove(
                    bpy.data.objects[f"light_{light_idx}"], do_unlink=True
                )

        for obj in bpy.data.objects:
            if obj.users_collection[0].name != "ObjectCollection":
                obj.hide_render = self.visible_objects_state[obj.name]

        for obj in bpy.data.collections["ObjectCollection"].objects:
            if obj.parent is None:
                obj.location = obj.location - (-self.object_center_initial)
                obj.rotation_euler[0] = self.object_rotation_initial[0]
                obj.rotation_euler[1] = self.object_rotation_initial[1]
                obj.rotation_euler[2] = self.object_rotation_initial[2]

        b_empty.location = EMPTY_LOCATION


# --------------------------------------------------------------------------------------


def render_object(out_data, store_path, object_center_new, radius_object):
    # I will use quite a lot of global variables
    # here so I decided to write them down
    global VIEWS, b_empty, EMPTY_LOCATION, tree
    direction_angle_z = 0
    DIRECTION_ANGLE_Z_RANGE = (-45, 45)
    DIRECTION_ANGLE_Z_MAW = 0.3  # moving average weight

    ROTATION_CAMERA_Z_RANGE = (-15, 15)

    B_EMPTY_RADIUS = radius_object / 10
    B_EMPTY_MAW = 0.3

    camera_radius = 2.0 * radius_object
    CAMERA_RADIUS_RATIO_RANGE = (1.5, 3.0)
    CAMERA_RADIUS_MAW = 0.3

    cam = bpy.data.objects["Camera"]  # it's kinda global
    scene = bpy.context.scene  # it's kinda global
    b_empty.location = object_center_new

    for i in range(VIEWS):
        direction_angle_xy = 2 * np.pi * (i / VIEWS)

        direction_angle_z += (1 - DIRECTION_ANGLE_Z_MAW) * (
            np.random.uniform(*DIRECTION_ANGLE_Z_RANGE) - direction_angle_z
        )

        curr_direction = mathutils.Vector(
            (
                np.cos(direction_angle_xy),
                np.sin(direction_angle_xy),
                np.sin(direction_angle_z),
            )
        )
        curr_direction.normalize()

        b_empty_location_new_hat = (
            object_center_new
            + B_EMPTY_RADIUS
            * mathutils.Vector(np.random.uniform(-1, 1, 3).tolist())
        )
        b_empty.location += (1 - B_EMPTY_MAW) * (
            b_empty_location_new_hat - b_empty.location
        )

        camera_rotation_z = np.deg2rad(
            np.random.uniform(*ROTATION_CAMERA_Z_RANGE)
        )

        # shoot rays from the center and find the first hit. It will be the radius
        camera_radius_new_hat = (
            np.random.uniform(*CAMERA_RADIUS_RATIO_RANGE) * radius_object
        )
        camera_radius += (1 - CAMERA_RADIUS_MAW) * (
            camera_radius_new_hat - camera_radius
        )

        position = b_empty.location + curr_direction * camera_radius

        set_camera(position, camera_rotation_z, cam=cam)

        # render a frame
        scene.render.filepath = store_path + "/color_" + str(i)

        tree.nodes["Depth Output"].base_path = store_path
        tree.nodes["Depth Output"].file_slots[0].path = "/depth_" + str(i)

        tree.nodes["Normal Output"].base_path = store_path
        tree.nodes["Normal Output"].file_slots[0].path = "/normal_" + str(i)

        bpy.ops.render.render(write_still=True)  # render still

        frame_data = {
            "file_path": scene.render.filepath,
            "transform_matrix": listify_matrix(cam.matrix_world),
        }
        out_data["frames"].append(frame_data)

        with open(os.path.join(store_path, "transforms.json"), "w") as out_file:
            json.dump(out_data, out_file, indent=4)

    b_empty.location = EMPTY_LOCATION


# --------------------------------------------------------------------------------------


def render_mask(out_data, store_path):
    global tree

    scene.render.filepath = store_path + "/useless_color"
    tree.nodes["Depth Output"].base_path = store_path
    tree.nodes["Depth Output"].file_slots[0].path = "/useless_depth"
    tree.nodes["Normal Output"].base_path = store_path
    tree.nodes["Normal Output"].file_slots[0].path = "/useless_normal"

    for i in range(len(out_data["frames"])):
        cam.matrix_world = mathutils.Matrix(
            out_data["frames"][i]["transform_matrix"]
        )

        tree.nodes["Mask Output"].base_path = store_path
        tree.nodes["Mask Output"].file_slots[0].path = "/mask_" + str(i)

        # render a frame
        bpy.ops.render.render(write_still=True)  # render still


# --------------------------------------------------------------------------------------
# PREPARE THE SCENE
# --------------------------------------------------------------------------------------

for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

bhv_tree_object_scene = build_bvh_tree()
bhv_tree_scene = build_bvh_tree(exclude_collections=["ObjectCollection"])

min_corner, max_corner = bounding_box(None)
print(np.prod(max_corner - min_corner))
min_corner_obj, max_corner_obj = bounding_box("ObjectCollection")

# we will mostly look at this point
# point_look_at = (min_corner_obj + max_corner_obj) / 2
b_empty = bpy.data.objects[NAME_LOOK_AT]
EMPTY_LOCATION = b_empty.location.copy()

# create a camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.angle = 1.0
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
# b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

# shoot N_shoot rays to understand the median distance
max_distance = np.linalg.norm(max_corner - min_corner).item()
N_shoot = 16
distances = []
for i in range(N_shoot):
    angle = i / N_shoot * 2 * np.pi
    direction = mathutils.Vector((np.cos(angle), np.sin(angle), 0))
    _, _, _, curr_distance = bhv_tree_object_scene.ray_cast(
        b_empty.location, direction
    )
    distances.append(max_distance if curr_distance is None else curr_distance)

radius = np.median(distances).item()

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

bpy.context.scene.render.use_persistent_data = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if "Custom Outputs" not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.label = "Custom Outputs"
    render_layers.name = "Custom Outputs"

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
    depth_file_output.name = "Depth Output"
    if FORMAT == "OPEN_EXR":
        links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.inputs["From Min"].default_value = 0
        map.inputs["From Max"].default_value = 8
        map.inputs["To Min"].default_value = 1
        map.inputs["To Max"].default_value = 0
        links.new(render_layers.outputs["Depth"], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.name = "Normal Output"
    links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])

# Background
# bpy.context.scene.render.dither_intensity = 0.0
# bpy.context.scene.render.film_transparent = True

world = bpy.context.scene.world

# If no world is set, create a new world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

# Set the background color to black (R, G, B)
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")

# If the background node exists, set the color to black
if bg_node:
    bg_node.inputs["Color"].default_value = (
        0,
        0,
        0,
        1,
    )  # Black with full opacity

# Set the strength of the background to 0 if you want no light emitted from it
bpy.context.scene.render.film_transparent = False

scene = bpy.context.scene
scene.camera = cam
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

# Data to store in JSON file
out_data_object_scene = {
    "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
}
out_data_object_scene["frames"] = []
# save for other
out_data_object = deepcopy(out_data_object_scene)
out_data_scene = deepcopy(out_data_object_scene)

out_data_object_scene["object_center"] = list(
    ((min_corner_obj + max_corner_obj) / 2)
)

all_rotations = []
for obj in bpy.data.collections["ObjectCollection"].objects:
    if obj.parent is None:
        all_rotations.append(obj.rotation_euler)
if len(all_rotations) != 1:
    raise RuntimeError("Wrong size of all_rotations!")

out_data_object_scene["euler_rotation"] = list(all_rotations[0])

print(out_data_object_scene)

## ----- rendering -----

bpy.context.scene.render.engine = "BLENDER_EEVEE"

render_scene(
    out_data_object_scene, RESULT_OBJ_SCENE_PATH, bhv_tree_object_scene
)

with RenderSceneContextManager() as cm:
    render_scene(out_data_scene, RESULT_SCENE_PATH, bhv_tree_scene)

with RenderObjectContextManager() as cm:
    render_object(
        out_data_object, RESULT_OBJ_PATH, cm.object_center_new, cm.radius_object
    )

## ----- render masks -----

for obj in bpy.data.collections["ObjectCollection"].all_objects:
    if obj.type == "MESH":
        obj.pass_index = 1

bpy.context.scene.render.engine = "CYCLES"  # the default is 'BLENDER_EEVEE'
bpy.context.scene.cycles.samples = 16  # Reduce the sample count

bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

if "Mask Output" not in tree.nodes:
    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    mask_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask_output.label = "Mask Output"
    mask_output.name = "Mask Output"
    render_layers = tree.nodes["Custom Outputs"]
    links.new(render_layers.outputs["IndexOB"], id_mask.inputs["ID value"])
    links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])

cam.constraints.clear()

render_mask(out_data_object_scene, RESULT_OBJ_SCENE_PATH)

with RenderObjectContextManager() as cm:
    render_mask(out_data_object, RESULT_OBJ_PATH)
