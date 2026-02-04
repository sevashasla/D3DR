# ./blender -b /scratch/izar/skorokho/data/my_blend_3/bathroom_1.blend --python ~/coding/voi_gs/render_dataset/render_indoor.py
# "out_root": "/home/skorokho/data/blender_data/",
import sys, os
import json
import bpy
import bmesh
import mathutils
import numpy as np
import warnings
from copy import deepcopy

import logging

# --------------------------------------------------------------------------------------

sys.path.append(os.path.dirname(__file__))
from utils import (
    build_bvh_tree,
    bounding_box,
    get_radius_scene,
    render_scene,
    render_object,
    render_mask,
    get_real_obj_location_and_euler,
    create_camera_constraint,
    _calculate_dist,

    RenderSceneContextManager,
    RenderObjectContextManager,
)

from create_ply import BlenderPlyCreator

# --------------------------------------------------------------------------------------

SCENE_NAME = os.path.basename(sys.argv[2])
RENDER_SCENE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "render_scenes.json")
with open(RENDER_SCENE_CONFIG_PATH) as f:
    data_store = json.load(f)
features_by_name = {el['name']: el for el in data_store['scenes']}
RESULTS_PATH = os.path.join(data_store['out_root'], features_by_name[SCENE_NAME]["result_end"])
N_LIGHTS = features_by_name[SCENE_NAME].get("num_lights", 2)
N_LIGHTS_FACES = features_by_name[SCENE_NAME].get("num_lights_faces", 0)

DO_RENDER_SCENE = features_by_name[SCENE_NAME].get("do_render_scene", True)
DO_RENDER_OBJ_SCENE = features_by_name[SCENE_NAME].get("do_render_obj_scene", True)
DO_RENDER_OBJ = features_by_name[SCENE_NAME].get("do_render_obj", True)
DO_RENDER_EVAL = features_by_name[SCENE_NAME].get("do_render_eval", True)

DO_RENDER_OBJ_SCENE_MASK = features_by_name[SCENE_NAME].get("do_render_obj_scene_mask", DO_RENDER_OBJ_SCENE)
DO_RENDER_OBJ_MASK = features_by_name[SCENE_NAME].get("do_render_obj_mask", DO_RENDER_OBJ)
DO_RENDER_EVAL_MASK = features_by_name[SCENE_NAME].get("do_render_eval_mask", DO_RENDER_EVAL)

DO_PLY_OBJ_SCENE = features_by_name[SCENE_NAME].get("do_ply_obj_scene", DO_RENDER_OBJ_SCENE)
DO_PLY_SCENE = features_by_name[SCENE_NAME].get("do_ply_scene", DO_RENDER_SCENE)
DO_PLY_OBJ = features_by_name[SCENE_NAME].get("do_ply_obj", DO_RENDER_OBJ)

NUM_STEPS = features_by_name[SCENE_NAME].get("num_steps", 256)

NUM_PLY_POINTS_OBJ = features_by_name[SCENE_NAME].get("num_ply_pts_obj", 10_000)
NUM_PLY_POINTS_SCENE = features_by_name[SCENE_NAME].get("num_ply_pts_scene", 50_000)
NUM_PLY_POINTS_OBJ_SCENE = features_by_name[SCENE_NAME].get("num_ply_pts_obj_scene", 5_000)

RADIUS_OBJ_MULT_FOR_CAMERA = features_by_name[SCENE_NAME].get("radius_obj_mult_for_camera", 2.0)

BETAS = features_by_name[SCENE_NAME].get("betas", 0.3)

PATH_POSES = features_by_name[SCENE_NAME].get("path_poses", None)
GENERATE_PATH_OF = features_by_name[SCENE_NAME].get("generate_path_of", [])
PATH_POSES_OBJ = None if "obj" in GENERATE_PATH_OF else os.path.join(PATH_POSES, "obj/transforms.json") 
PATH_POSES_SCENE = None if "scene" in GENERATE_PATH_OF else os.path.join(PATH_POSES, "scene/transforms.json") 
PATH_POSES_OBJ_SCENE = None if "obj_scene" in GENERATE_PATH_OF else os.path.join(PATH_POSES, "obj_scene/transforms.json") 
PATH_POSES_OBJ_SCENE_EVAL = None if "obj_scene_eval" in GENERATE_PATH_OF else os.path.join(PATH_POSES, "obj_scene_eval/transforms.json") 

CORNER_LIGHT_MULTIPLIER = features_by_name[SCENE_NAME].get("corner_light_multiplier", 1.2)
RANGE_LIGHTS = features_by_name[SCENE_NAME].get("range_lights", (5, 15))

assert features_by_name[SCENE_NAME]["type"] == "indoor", f"Scene {SCENE_NAME} is not indoor"

IMPORTANT_LOGS = data_store["important_logs_path"]
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s]: %(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    filename=IMPORTANT_LOGS, 
    level=logging.INFO, 
    filemode='a',
    encoding='utf-8',
)
logger.info(f"Start rendering {SCENE_NAME}")

# if not all([
#     DO_PLY_OBJ, DO_PLY_OBJ_SCENE, DO_PLY_SCENE, 
#     DO_RENDER_OBJ, DO_RENDER_OBJ_SCENE, DO_RENDER_SCENE, 
#     DO_RENDER_OBJ_MASK, DO_RENDER_OBJ_SCENE_MASK
# ]):
#     logger.info("Exit because everything is done")

VIEWS = features_by_name[SCENE_NAME].get("num_views", 250)
RESOLUTION = 800
FORMAT = 'PNG'
COLOR_DEPTH = 8
# FORMAT = 'OPEN_EXR'
# COLOR_DEPTH = 16
SMALLER_R = 0.8
NAME_LOOK_AT = "EmptyLookAt"

fp = bpy.path.abspath(RESULTS_PATH)

used_s = set([
    "num_lights",
    "num_lights_faces",
    "do_render_scene", "do_render_obj_scene", "do_render_obj",
    "do_render_obj_scene_mask",
    "do_render_obj_mask",
    "do_ply_scene", "do_ply_obj_scene", "do_ply_obj",
    "num_ply_pts_obj", "num_ply_pts_scene", "num_ply_pts_obj_scene",
    "radius_obj_mult_for_camera",
    "num_views",
    "name", "type", "result_end", "corner_light_multiplier",
    "range_lights", "path_poses_scene", "path_poses", "generate_path_of", 
   "betas", "do_render_eval", "do_render_eval_mask", "num_steps"
])

if len(set(features_by_name[SCENE_NAME].keys()) - used_s) != 0:
    raise RuntimeError(f"Keys are not used: {set(features_by_name[SCENE_NAME].keys()) - used_s}")

RESULT_OBJ_SCENE_PATH = os.path.join(RESULTS_PATH, "obj_scene")
RESULT_OBJ_SCENE_EVAL_PATH = os.path.join(RESULTS_PATH, "obj_scene_eval")
RESULT_SCENE_EVAL_PATH = os.path.join(RESULTS_PATH, "scene_eval")
RESULT_SCENE_PATH = os.path.join(RESULTS_PATH, "scene")
RESULT_OBJ_PATH = os.path.join(RESULTS_PATH, "obj")

os.makedirs(fp, exist_ok=True)
os.makedirs(RESULT_OBJ_SCENE_PATH, exist_ok=True)
os.makedirs(RESULT_OBJ_SCENE_EVAL_PATH, exist_ok=True)
os.makedirs(RESULT_SCENE_EVAL_PATH, exist_ok=True)
os.makedirs(RESULT_SCENE_PATH, exist_ok=True)
os.makedirs(RESULT_OBJ_PATH, exist_ok=True)


# --------------------------------------------------------------------------------------
# PREPARE THE GPU (from chatgpt)
# --------------------------------------------------------------------------------------

# Ensure we access the preferences for Cycles addon
cycles_prefs = bpy.context.preferences.addons.get("cycles")

# Enable GPU compute device
if cycles_prefs is not None:
    cycles_prefs = cycles_prefs.preferences
    cycles_prefs.compute_device_type = "OPTIX"

    # Fetch and enable all GPU devices
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = True

    # Set the rendering device to GPU
    bpy.context.scene.cycles.device = "GPU"
else:
    print("Cycles addon is not available. Ensure it is enabled.")

bpy.context.scene.cycles.feature_set = 'SUPPORTED' 

# --------------------------------------------------------------------------------------
# PREPARE THE SCENE 
# --------------------------------------------------------------------------------------

np.random.seed(42)

for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

bhv_tree_object_scene = build_bvh_tree(bpy.data)
bhv_tree_scene = build_bvh_tree(bpy.data, exclude_collections=["ObjectCollection"])

min_corner, max_corner = bounding_box(bpy.data, None)
print(np.prod(max_corner - min_corner))
print("bb scene", min_corner, max_corner)
min_corner_obj, max_corner_obj = bounding_box(bpy.data, "ObjectCollection")
print("bb obj", min_corner_obj, max_corner_obj)
center_obj = (min_corner_obj + max_corner_obj) / 2

# we will mostly look at this point
#point_look_at = (min_corner_obj + max_corner_obj) / 2
b_empty = bpy.data.objects[NAME_LOOK_AT]
EMPTY_LOCATION = b_empty.location.copy()

# create a camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.angle = 1.0
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)

create_camera_constraint(cam, b_empty)

# shoot N_shoot rays to understand the median distance
radius = get_radius_scene(min_corner, max_corner, bhv_tree_object_scene, b_empty)

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

bpy.context.scene.render.use_persistent_data = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapRange")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.inputs['From Min'].default_value = 0
      map.inputs['From Max'].default_value = 8
      map.inputs['To Min'].default_value = 1
      map.inputs['To Max'].default_value = 0
      links.new(render_layers.outputs['Depth'], map.inputs[0])

      links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
#bpy.context.scene.render.dither_intensity = 0.0
#bpy.context.scene.render.film_transparent = True

world = bpy.context.scene.world

# If no world is set, create a new world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

# Set the background color to black (R, G, B)
world.use_nodes = True
bg_node = world.node_tree.nodes.get('Background')

# If the background node exists, set the color to black
if bg_node:
    bg_node.inputs['Color'].default_value = (0, 0, 0, 1)  # Black with full opacity

# Set the strength of the background to 0 if you want no light emitted from it
bpy.context.scene.render.film_transparent = False

scene = bpy.context.scene
scene.camera = cam
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

# Data to store in JSON file

default_transforms = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'frames': [],
}

def maybe_get_data(path, example):
    if path is None:
        return example
    else:
        with open(path) as f:
            return json.load(f)

# save for other
if PATH_POSES_OBJ_SCENE is None:
    out_data_object_scene = deepcopy(default_transforms)
    out_data_object_scene['object_center'], out_data_object_scene['euler_rotation'] = get_real_obj_location_and_euler(bpy.data)
    out_data_object_scene['object_center'] = list(out_data_object_scene['object_center'])
    out_data_object_scene['euler_rotation'] = list(out_data_object_scene['euler_rotation'])
else:
    with open(PATH_POSES_OBJ_SCENE) as f:
        out_data_object_scene = json.load(f)

# maybe take object poses
out_data_object = maybe_get_data(PATH_POSES_OBJ, default_transforms)
out_data_scene = maybe_get_data(PATH_POSES_SCENE, default_transforms)
out_data_obj_scene_eval = maybe_get_data(PATH_POSES_OBJ_SCENE_EVAL, default_transforms)

## ----- rendering -----

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = NUM_STEPS


if DO_RENDER_OBJ_SCENE:
    render_scene(
        bpy.data, bpy.context, bpy.ops, tree,
        out_data_object_scene, RESULT_OBJ_SCENE_PATH, bhv_tree_object_scene, radius, 
        b_empty, EMPTY_LOCATION, VIEWS, center_location=EMPTY_LOCATION
    )

if DO_RENDER_EVAL:
    # render around the object
    min_dist_obj = _calculate_dist(
        bpy.data.objects['Camera'].data.angle_x, 
        (min_corner_obj, max_corner_obj),
        BETAS[0],
    )
    max_dist_obj = _calculate_dist(
        bpy.data.objects['Camera'].data.angle_x, 
        (min_corner_obj, max_corner_obj),
        BETAS[1],
    )

    render_scene(
        bpy.data, bpy.context, bpy.ops, tree,
        out_data_obj_scene_eval, RESULT_OBJ_SCENE_EVAL_PATH, bhv_tree_scene, min(radius, max_dist_obj), 
        b_empty, EMPTY_LOCATION, VIEWS, center_location=center_obj, min_dist=min_dist_obj
    )

if DO_PLY_OBJ_SCENE:
    ply_saver = BlenderPlyCreator(bpy.data)
    ply_saver.create_ply(os.path.join(RESULT_OBJ_SCENE_PATH, "sparce_pc.ply"), num_points=NUM_PLY_POINTS_OBJ_SCENE)

with RenderSceneContextManager(bpy.data, b_empty, EMPTY_LOCATION) as cm:
    if DO_RENDER_SCENE:
        render_scene(
            bpy.data, bpy.context, bpy.ops, tree,
            out_data_scene, RESULT_SCENE_PATH, bhv_tree_scene, radius, 
            b_empty, EMPTY_LOCATION, VIEWS, center_location=EMPTY_LOCATION
        )

    if DO_RENDER_EVAL:
        # save this for object
        render_scene(
            bpy.data, bpy.context, bpy.ops, tree,
            out_data_obj_scene_eval, RESULT_SCENE_EVAL_PATH, bhv_tree_scene, min(radius, max_dist_obj), 
            b_empty, EMPTY_LOCATION, VIEWS, center_location=center_obj, min_dist=min_dist_obj
        )

    if DO_PLY_SCENE:
        ply_saver = BlenderPlyCreator(bpy.data)
        ply_saver.create_ply(os.path.join(RESULT_SCENE_PATH, "sparce_pc.ply"), num_points=NUM_PLY_POINTS_SCENE)

with RenderObjectContextManager(
        bpy.data, bpy.context, 
        b_empty, EMPTY_LOCATION, 
        min_corner_obj, max_corner_obj,
        out_data_object_scene,
        num_lights=N_LIGHTS,
        num_lights_faces=N_LIGHTS_FACES,
        corner_light_multiplier=CORNER_LIGHT_MULTIPLIER,
        range_lights=RANGE_LIGHTS,
    ) as cm:
    if DO_RENDER_OBJ:
        center_in_000 = (max_corner_obj - min_corner_obj)/2 
        
        # this logic is not perfect as well, because 
        # the it is not clear why the object would have
        # a movement_vector from corner. But ok...
        render_object(
            bpy.data, bpy.context, bpy.ops, tree,
            out_data_object, RESULT_OBJ_PATH, center_in_000 + cm.object_center_new, cm.radius_object,
            b_empty, EMPTY_LOCATION, VIEWS, RADIUS_OBJ_MULT_FOR_CAMERA
        )
    if DO_PLY_OBJ:
        ply_saver = BlenderPlyCreator(bpy.data)
        ply_saver.create_ply(os.path.join(RESULT_OBJ_PATH, "sparce_pc.ply"), num_points=NUM_PLY_POINTS_OBJ)

bpy.context.view_layer.update() # for some reason this doesn't help...

bpy.ops.object.select_all(action='DESELECT')  # Deselect everything
bpy.ops.object.select_all(action='SELECT')    # Reselect everything


## ----- render masks -----
if DO_RENDER_OBJ_SCENE_MASK or DO_RENDER_OBJ_MASK or DO_RENDER_EVAL_MASK:
    for obj in bpy.data.collections["ObjectCollection"].all_objects:
        if obj.type == 'MESH':
            obj.pass_index = 1

    bpy.context.scene.render.engine = 'CYCLES'  # the default is 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.cycles.samples = 1  # Reduce the sample count

    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

    if not 'Mask Output' in tree.nodes:
        id_mask = tree.nodes.new('CompositorNodeIDMask')
        id_mask.index = 1
        mask_output = tree.nodes.new(type="CompositorNodeOutputFile")
        mask_output.label = 'Mask Output'
        mask_output.name = 'Mask Output'
        render_layers = tree.nodes['Custom Outputs']
        links.new(render_layers.outputs['IndexOB'], id_mask.inputs['ID value'])
        links.new(id_mask.outputs['Alpha'], mask_output.inputs[0])

    cam.constraints.clear()

    if DO_RENDER_OBJ_SCENE_MASK:
        render_mask(
            bpy.data, bpy.context, bpy.ops, tree,
            out_data_object_scene, RESULT_OBJ_SCENE_PATH, VIEWS=VIEWS,
        )
    
    if DO_RENDER_EVAL_MASK:
        render_mask(
            bpy.data, bpy.context, bpy.ops, tree,
            out_data_obj_scene_eval, RESULT_OBJ_SCENE_EVAL_PATH, VIEWS=VIEWS,
        )

    if DO_RENDER_OBJ_MASK:
        with RenderObjectContextManager(
            bpy.data, bpy.context, 
            b_empty, EMPTY_LOCATION, 
            min_corner_obj, max_corner_obj,
            out_data_object_scene,
            corner_light_multiplier=CORNER_LIGHT_MULTIPLIER,
            range_lights=RANGE_LIGHTS,
        ) as cm:
            render_mask(
                bpy.data, bpy.context, bpy.ops, tree,
                out_data_object, RESULT_OBJ_PATH, VIEWS=VIEWS,
            )
