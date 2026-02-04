import sys, os
import json
import bpy
import bmesh
import mathutils
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(__file__))
from utils import (
    build_bvh_tree,
    bounding_box,
    get_radius_scene,
    render_scene,
    render_object,
    render_mask,

    RenderSceneContextManager,
    RenderObjectContextManager,
)

SCENE_NAME = os.path.basename(sys.argv[2])
RENDER_SCENE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "render_scenes.json")
with open(RENDER_SCENE_CONFIG_PATH) as f:
    data_store = json.load(f)
features_by_name = {el['name']: el for el in data_store['scenes']}
RESULTS_PATH = os.path.join(data_store['out_root'], features_by_name[SCENE_NAME]["result_end"])
N_LIGHTS = features_by_name[SCENE_NAME]["num_lights"]

VIEWS = 50
RESOLUTION = 800
COLOR_DEPTH = 8
FORMAT = 'PNG'
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

np.random.seed(42)

# --------------------------------------------------------------------------------------
# PREPARE THE SCENE 
# --------------------------------------------------------------------------------------

for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

bhv_tree_object_scene = build_bvh_tree(bpy.data)
bhv_tree_scene = build_bvh_tree(bpy.data, exclude_collections=["ObjectCollection"])

min_corner, max_corner = bounding_box(bpy.data, None)
print(np.prod(max_corner - min_corner))
min_corner_obj, max_corner_obj = bounding_box(bpy.data, "ObjectCollection")

# we will mostly look at this point
#point_look_at = (min_corner_obj + max_corner_obj) / 2
b_empty = bpy.data.objects[NAME_LOOK_AT]
EMPTY_LOCATION = b_empty.location.copy()

# create a camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.angle = 1.0
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
#b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

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
out_data_object_scene = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}
out_data_object_scene['frames'] = []
# save for other
out_data_object = deepcopy(out_data_object_scene)
out_data_scene = deepcopy(out_data_object_scene)

out_data_object_scene['object_center'] = list(((min_corner_obj + max_corner_obj) / 2))

all_rotations = []
for obj in bpy.data.collections["ObjectCollection"].objects:
    if obj.parent is None:
        all_rotations.append(obj.rotation_euler)
if len(all_rotations) != 1:
    raise RuntimeError("Wrong size of all_rotations!")

out_data_object_scene['euler_rotation'] = list(all_rotations[0])

print(out_data_object_scene)

## ----- rendering -----

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 128

render_scene(
    bpy.data, bpy.context, bpy.ops, tree,
    out_data_object_scene, RESULT_OBJ_SCENE_PATH, bhv_tree_object_scene, radius, 
    b_empty, EMPTY_LOCATION, VIEWS,
)

with RenderSceneContextManager(bpy.data, b_empty, EMPTY_LOCATION) as cm:
    render_scene(
        bpy.data, bpy.context, bpy.ops, tree,
        out_data_scene, RESULT_SCENE_PATH, bhv_tree_scene, radius, 
        b_empty, EMPTY_LOCATION, VIEWS,
    )

with RenderObjectContextManager(
        bpy.data, bpy.context, 
        b_empty, EMPTY_LOCATION, 
        min_corner_obj, max_corner_obj,
        out_data_object_scene,
        num_lights=N_LIGHTS,
    ) as cm:
    render_object(
        bpy.data, bpy.context, bpy.ops, tree,
        out_data_object, RESULT_OBJ_PATH, cm.object_center_new, cm.radius_object,
        b_empty, EMPTY_LOCATION, VIEWS
    )

## ----- render masks -----

for obj in bpy.data.collections["ObjectCollection"].all_objects:
   if obj.type == 'MESH':
       obj.pass_index = 1

bpy.context.scene.render.engine = 'CYCLES'  # the default is 'BLENDER_EEVEE'
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

render_mask(
    bpy.data, bpy.context, bpy.ops, tree,
    out_data_object_scene, RESULT_OBJ_SCENE_PATH
)

with RenderObjectContextManager(
    bpy.data, bpy.context, 
    b_empty, EMPTY_LOCATION, 
    min_corner_obj, max_corner_obj,
    out_data_object_scene
) as cm:
    render_mask(
        bpy.data, bpy.context, bpy.ops, tree,
        out_data_object, RESULT_OBJ_PATH
    )
