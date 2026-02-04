import numpy as np
import mathutils
import json
import os
from copy import deepcopy
import math

# --------------------------------------------------------------------------------------

def create_camera_constraint(cam, track_to):
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.target = track_to

# --------------------------------------------------------------------------------------

def get_vertices_and_polygons(obj, shift=0):
    mesh = obj.to_mesh()
    vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
    polygons = [(np.array(p.vertices) + shift).tolist() for p in mesh.polygons]
    return vertices, polygons        

# bpy_data_collections = bpy.data.collections
# bpy_data_objects = bpy.data.objects
def build_bvh_tree(bpy_data, consider_collections=None, exclude_collections=None):
    if consider_collections is None:
        consider_collections = [collection.name for collection in bpy_data.collections]
    if exclude_collections is None:
        exclude_collections = []
    consider_collections = set(consider_collections)
    exclude_collections = set(exclude_collections)
    take_collections = consider_collections - exclude_collections
    
    all_vertices = []
    all_polygons = []

    for obj in bpy_data.objects:
        if obj.type == 'MESH' and ((len(obj.users_collection) == 0) or (obj.users_collection[0].name in take_collections)):
            curr_vertices, curr_polygons = get_vertices_and_polygons(obj, len(all_vertices))
            all_vertices.extend(curr_vertices)
            all_polygons.extend(curr_polygons)
    
    bhv_tree = mathutils.bvhtree.BVHTree.FromPolygons(all_vertices, all_polygons)
    return bhv_tree

# --------------------------------------------------------------------------------------

def bounding_box(bpy_data, collection_name=None):
    min_corner = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_corner = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    if collection_name is None:
        go_through = bpy_data.objects
    else:
        go_through = bpy_data.collections[collection_name].objects    
    
    for obj in go_through:
        # Only consider mesh objects (or modify this as needed)
        if obj.type == 'MESH':
            # Get the bounding box in local coordinates and convert to world coordinates
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            # Update the min and max corner of the bounding box
            for corner in bbox_corners:
                min_corner = mathutils.Vector((min(min_corner[i], corner[i]) for i in range(3)))
                max_corner = mathutils.Vector((max(max_corner[i], corner[i]) for i in range(3)))
    
    return min_corner, max_corner

# --------------------------------------------------------------------------------------

def add_lights_faces(bpy_data, bpy_context, name, in_bb, num_lights=6, range_lights=(5, 15)):
    import warnings
    # create random light parameters
    xmin, ymin, zmin = list(in_bb[0])
    xmax, ymax, zmax = list(in_bb[1])
    locations = [
        np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin]),
        np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmax]),
        np.array([(xmin + xmax) / 2, ymin, (zmin + zmax) / 2]),
        np.array([(xmin + xmax) / 2, ymax, (zmin + zmax) / 2]),
        np.array([xmin, (ymin + ymax) / 2, (zmin + zmax) / 2]),
        np.array([xmax, (ymin + ymax) / 2, (zmin + zmax) / 2]),
    ]
    locations_ids = np.random.choice(len(locations), num_lights, replace=False)
    locations = [locations[i] for i in locations_ids]
    names = []
    for i, point_light_loc in enumerate(locations):
        curr_name = name + f"{i}_face"
        names.append(curr_name)
        light_data = bpy_data.lights.new(name=curr_name + "_data", type='POINT')
        light_data.energy = np.random.uniform(*range_lights)
        
        # create light
        light_object = bpy_data.objects.new(name=curr_name, object_data=light_data)
        light_object.location = point_light_loc
        bpy_context.collection.objects.link(light_object)

    return names

def add_light(bpy_data, bpy_context, name, in_bb, except_bb=None):
    # create random light parameters
    point_light_loc = np.random.uniform(*in_bb)
    if not except_bb is None:
        except_bb = (np.array(except_bb[0]), np.array(except_bb[1]))
        while np.all(except_bb[0] <= point_light_loc) and np.all(point_light_loc <= except_bb[1]):
            point_light_loc = np.random.uniform(*in_bb)

    light_data = bpy_data.lights.new(name=name + "_data", type='POINT')
    light_data.energy = np.random.uniform(50, 80)
    
    # create light
    light_object = bpy_data.objects.new(name=name, object_data=light_data)
    light_object.location = point_light_loc
    bpy_context.collection.objects.link(light_object)

    return name

# --------------------------------------------------------------------------------------

def parent_obj_to_camera(bpy_data, bpy_context, point_look_at):
    b_empty = bpy_data.objects.new("Empty", None)
    b_empty.location = point_look_at
    scn = bpy_context.scene
    scn.collection.objects.link(b_empty)
    return b_empty

# --------------------------------------------------------------------------------------

def set_camera(bpy_data, position, rotation_angle_z_local, cam=None):
    if cam is None:
        cam = bpy_data.objects["Camera"]
    cam.rotation_euler[2] = rotation_angle_z_local    
    cam.location = position   

# --------------------------------------------------------------------------------------

# only for debugging
def draw_sphere(bpy_ops, location):
    bpy_ops.mesh.primitive_uv_sphere_add(radius=0.1, location=location)


# --------------------------------------------------------------------------------------

def _calculate_dist(
        # self, 
        camera_angle_x,
        bb_obj,
        beta,
    ):
    # generate poses
    obj_width = max(list(bb_obj[1] - bb_obj[0]))

    # (w / 2)/ (d * tan(fov / 2)) = sqrt(beta) => 
    # d = (w / 2) / (tan(fov / 2) * sqrt(beta))
    min_dist = (0.5 * obj_width / (math.tan(camera_angle_x / 2) * math.sqrt(beta)))
    return min_dist

# --------------------------------------------------------------------------------------

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

# --------------------------------------------------------------------------------------

def get_radius_scene(min_corner, max_corner, bhv_tree_object_scene, b_empty, N_shoot=32):
    max_distance = np.linalg.norm(max_corner - min_corner).item()
    distances = []
    for i in range(N_shoot):
        angle = i / N_shoot * 2 * np.pi
        direction = mathutils.Vector((np.cos(angle), np.sin(angle), 0))
        _, _, _, curr_distance = bhv_tree_object_scene.ray_cast(b_empty.location, direction)
        distances.append(max_distance if curr_distance is None else curr_distance)

    return np.median(distances).item()

# --------------------------------------------------------------------------------------

def force_update_obj(bpy_data, bpy_context):
    for obj in bpy_data.collections["ObjectCollection"].objects:
        obj.update_tag()
    bpy_context.view_layer.depsgraph.update()
    bpy_context.view_layer.update()

def render_scene(
        bpy_data, bpy_context, bpy_ops, tree,
        out_data, store_path, bhv_tree, radius,
        b_empty, EMPTY_LOCATION, VIEWS, 
        center_location, min_dist=0.0,
    ):
    direction_angle_z = 0
    DIRECTION_ANGLE_Z_RANGE = (-10, 46)
    DIRECTION_ANGLE_Z_MAW = 0.2 # moving average weight
    CURR_DISTANCE_RANGE = (0.4, 1.0)

    ROTATION_CAMERA_Z_RANGE = (-15, 15)

    # B_EMPTY_RADIUS = radius / 10
    B_EMPTY_RADIUS = 0
    B_EMPTY_MAW = 0.2

    cam = bpy_data.objects["Camera"] # it's kinda global
    scene = bpy_context.scene  # it's kinda global
    b_empty.location = center_location

    use_given_poses = len(out_data["frames"]) > 0
    print("USE GIVEN POSES:", use_given_poses)
    # VIEWS = len(out_data["frames"]) if use_given_poses else VIEWS
    if use_given_poses:
        cam.constraints.clear()
        
    for i in range(VIEWS):
        # whether to use given poses or generate them
        if use_given_poses:
            cam.matrix_world = mathutils.Matrix(out_data["frames"][i]["transform_matrix"])
        else:
            
            # maybe it will be adjusted later
            direction_angle_xy = 2 * np.pi * (i / VIEWS)
            curr_distance = -1.0

            while curr_distance < min_dist:    
                direction_angle_z += (1 - DIRECTION_ANGLE_Z_MAW) *\
                    (np.random.uniform(*DIRECTION_ANGLE_Z_RANGE) - direction_angle_z)
                    
                curr_direction = mathutils.Vector((
                    np.cos(np.deg2rad(direction_angle_z)) * np.cos(direction_angle_xy), 
                    np.cos(np.deg2rad(direction_angle_z)) * np.sin(direction_angle_xy), 
                    np.sin(np.deg2rad(direction_angle_z))
                ))
                curr_direction.normalize()
                    
                b_empty_location_new_hat = center_location + \
                    B_EMPTY_RADIUS * mathutils.Vector(np.random.uniform(-1, 1, 3).tolist())
                b_empty.location += (1 - B_EMPTY_MAW) * (b_empty_location_new_hat - b_empty.location)
                
                camera_rotation_z = np.deg2rad(np.random.uniform(*ROTATION_CAMERA_Z_RANGE))
                
                # shoot rays from the center and find the first hit. It will be the radius
                _, _, _, curr_distance = bhv_tree.ray_cast(b_empty.location, curr_direction)
                if (curr_distance):
                    curr_distance = min(curr_distance, radius)
                else:
                    curr_distance = radius
                
                curr_distance_new = np.random.uniform(*CURR_DISTANCE_RANGE) * curr_distance
                curr_distance = curr_distance_new if curr_distance_new >= min_dist else curr_distance

                direction_angle_xy = 2 * np.pi * np.random.uniform()

            position = b_empty.location + curr_direction * curr_distance

            set_camera(bpy_data, position, camera_rotation_z, cam=cam)
        
        # render a frame
        scene.render.filepath = store_path + f'/color_{i:05}'
        
        tree.nodes['Depth Output'].base_path = store_path
        tree.nodes['Depth Output'].file_slots[0].path = f"/depth_{i:05}"
        
        tree.nodes['Normal Output'].base_path = store_path
        tree.nodes['Normal Output'].file_slots[0].path = f"/normal_{i:05}"

        bpy_ops.render.render(write_still=True)  # render still
    
        if not use_given_poses:
            frame_data = {
                'file_path': scene.render.filepath,
                'transform_matrix': listify_matrix(cam.matrix_world)
            }
            out_data['frames'].append(frame_data)

    with open(os.path.join(store_path, "transforms.json"), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    if use_given_poses:
        create_camera_constraint(cam, b_empty)

    b_empty.location = EMPTY_LOCATION

# --------------------------------------------------------------------------------------

class RenderSceneContextManager:
    def __init__(self, bpy_data, b_empty, EMPTY_LOCATION):
        self.bpy_data = bpy_data
        self.b_empty = b_empty
        self.EMPTY_LOCATION = EMPTY_LOCATION
        self.visible_objects_state = {obj.name: obj.hide_render for obj in self.bpy_data.objects}

    def __enter__(self):
        '''
        hide object elements
        '''
        # make the object invisible during rendering
        for obj in self.bpy_data.collections["ObjectCollection"].objects:
            obj.hide_render = True


    def __exit__(self, exc_type, exc_val, exc_traceback):
        if not exc_type is None:
            print(exc_val)

        # revert it back
        for obj in self.bpy_data.collections["ObjectCollection"].objects:
            obj.hide_render = self.visible_objects_state[obj.name]

        self.b_empty.location = self.EMPTY_LOCATION

# --------------------------------------------------------------------------------------

def get_real_obj_location_and_euler(bpy_data):
    locations = []
    for obj in bpy_data.collections["ObjectCollection"].objects:
        if obj.parent is None:
            locations.append(obj.location)
    assert len(locations) == 1

    euler_angles = []
    for obj in bpy_data.collections["ObjectCollection"].objects:
        if obj.parent is None:
            euler_angles.append(obj.rotation_euler)
    assert len(euler_angles) == 1
    return deepcopy(locations[0]), deepcopy(euler_angles[0])

# --------------------------------------------------------------------------------------

class RenderObjectContextManager:
    def __init__(
            self, 
            bpy_data, bpy_context, 
            b_empty, EMPTY_LOCATION, 
            min_corner_obj, max_corner_obj,
            out_data_object_scene, 
            add_lights=True,
            num_lights=None,
            num_lights_faces=0,
            corner_light_multiplier=1.5,
            range_lights=(5, 15),
        ):
        self.bpy_data = bpy_data
        self.b_empty = b_empty
        self.bpy_context = bpy_context
        self.EMPTY_LOCATION = EMPTY_LOCATION
        self.visible_objects_state = {obj.name: obj.hide_render for obj in bpy_data.objects}
        self.used_names = []

        self.min_corner_obj = min_corner_obj
        self.max_corner_obj = max_corner_obj
        self.out_data_object_scene = out_data_object_scene

        self.add_lights = add_lights
        self.num_lights = num_lights
        self.num_lights_faces = num_lights_faces
        self.corner_light_multiplier = corner_light_multiplier
        self.range_lights = range_lights

    def __enter__(self):   
        '''
        add here
        1) shift object to the center
        2) add lights
        3) hide all other objects during rendering
        '''
        # now hide everything except the object
        for obj in self.bpy_data.objects:
            if len(obj.users_collection) > 0 and obj.users_collection[0].name != "ObjectCollection":
                obj.hide_render = True
        # and except the camera, of couse
        self.bpy_data.objects["Camera"].hide_render = False

        # place it to the center
        self.object_center_initial, self.object_rotation_initial = get_real_obj_location_and_euler(self.bpy_data)

        print("Object Center:", self.object_center_initial)
        print("Object Rotation:", self.object_rotation_initial)

        self.radius_object = np.linalg.norm(self.max_corner_obj - self.min_corner_obj).item() / 2
        corner = mathutils.Vector((self.radius_object, self.radius_object, self.radius_object))
        for obj in self.bpy_data.collections["ObjectCollection"].objects:
            if obj.parent is None:
                obj.location = obj.location + ( - self.object_center_initial)
                obj.rotation_euler[0] = 0; obj.rotation_euler[1] = 0; obj.rotation_euler[2] = 0

        # self.bpy_context.view_layer.update()
        force_update_obj(self.bpy_data, self.bpy_context)

        new_min_corner_obj, new_max_corner_obj = bounding_box(self.bpy_data, "ObjectCollection")
        self.object_center_new = mathutils.Vector((0,0,0))

        # add lights
        if self.add_lights:
            if self.num_lights is None:
                self.num_lights = int((3 * self.radius_object)) + 1
            print("NUM LIGHTS:", self.num_lights)

            for light_idx in range(self.num_lights):
                name = add_light(
                    self.bpy_data, self.bpy_context,
                    f"light_{light_idx}", 
                    (self.object_center_new - corner * self.corner_light_multiplier, self.object_center_new + corner * self.corner_light_multiplier),
                    except_bb=(new_min_corner_obj, new_max_corner_obj)
                )
                self.used_names.append(name)
            
            if self.num_lights_faces > 0:
                many_names_6 = add_lights_faces(
                    self.bpy_data, self.bpy_context,
                    "light", 
                    (self.object_center_new - corner * self.corner_light_multiplier, self.object_center_new + corner * self.corner_light_multiplier),
                    num_lights=self.num_lights_faces,
                    range_lights=self.range_lights,
                )
                self.used_names.extend(many_names_6)

        # turn off the HDRI
        world_nodes = self.bpy_context.scene.world.node_tree
        self.background_node = None
        for node in world_nodes.nodes:
            if node.type == 'BACKGROUND':
                self.background_node = node
                break

        if self.background_node:
            self.original_strength = self.background_node.inputs['Strength'].default_value
            self.background_node.inputs['Strength'].default_value = 0.0

        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        global b_empty
        if not exc_type is None:
            print(exc_val)

        if self.add_lights:
            for name in self.used_names:
                self.bpy_data.objects.remove(self.bpy_data.objects[name], do_unlink=True)

        for obj in self.bpy_data.objects:
            if len(obj.users_collection) > 0 and obj.users_collection[0].name != "ObjectCollection":
                obj.hide_render = self.visible_objects_state[obj.name]

        for obj in self.bpy_data.collections["ObjectCollection"].objects:
            if obj.parent is None:
                obj.location = obj.location - ( - self.object_center_initial)
                obj.rotation_euler[0] = self.object_rotation_initial[0]
                obj.rotation_euler[1] = self.object_rotation_initial[1] 
                obj.rotation_euler[2] = self.object_rotation_initial[2]


        if self.background_node:
            self.background_node.inputs['Strength'].default_value = self.original_strength
        
        self.b_empty.location = self.EMPTY_LOCATION
        force_update_obj(self.bpy_data, self.bpy_context)

# --------------------------------------------------------------------------------------

def render_object(
        bpy_data, bpy_context, bpy_ops, tree,
        out_data, store_path, object_center_new, radius_object,
        b_empty, EMPTY_LOCATION, VIEWS,
        RADIUS_OBJ_MULT_FOR_CAMERA,
    ):
    direction_angle_z = 0
    DIRECTION_ANGLE_Z_RANGE = (-45, 60)
    DIRECTION_ANGLE_Z_MAW = 0.2 # moving average weight

    ROTATION_CAMERA_Z_RANGE = (-15, 15)

    B_EMPTY_RADIUS = radius_object / 10
    B_EMPTY_MAW = 0.2

    camera_radius = RADIUS_OBJ_MULT_FOR_CAMERA * radius_object
    CAMERA_RADIUS_RATIO_RANGE = (1.5, 3.0)
    CAMERA_RADIUS_MAW = 0.2

    cam = bpy_data.objects["Camera"] # it's kinda global
    scene = bpy_context.scene  # it's kinda global
    b_empty.location = object_center_new

    use_given_poses = len(out_data["frames"]) > 0
    print("USE GIVEN POSES:", use_given_poses)
    
    if use_given_poses:
        cam.constraints.clear()

    for i in range(VIEWS):
        if use_given_poses:
            cam.matrix_world = mathutils.Matrix(out_data["frames"][i]["transform_matrix"])
        else:
            direction_angle_xy = 2 * np.pi * (i / VIEWS)
            
            direction_angle_z += (1 - DIRECTION_ANGLE_Z_MAW) *\
                (np.random.uniform(*DIRECTION_ANGLE_Z_RANGE) - direction_angle_z)
                
            curr_direction = mathutils.Vector((
                np.cos(np.deg2rad(direction_angle_z)) * np.cos(direction_angle_xy), 
                np.cos(np.deg2rad(direction_angle_z)) * np.sin(direction_angle_xy), 
                np.sin(np.deg2rad(direction_angle_z))
            ))
            curr_direction.normalize()
                
            b_empty_location_new_hat = object_center_new + \
                B_EMPTY_RADIUS * mathutils.Vector(np.random.uniform(-1, 1, 3).tolist())
            b_empty.location += (1 - B_EMPTY_MAW) * (b_empty_location_new_hat - b_empty.location)
            
            camera_rotation_z = np.deg2rad(np.random.uniform(*ROTATION_CAMERA_Z_RANGE))
            
            # shoot rays from the center and find the first hit. It will be the radius
            camera_radius_new_hat = np.random.uniform(*CAMERA_RADIUS_RATIO_RANGE) * radius_object
            camera_radius += (1 - CAMERA_RADIUS_MAW) * (camera_radius_new_hat - camera_radius)
            
            position = b_empty.location + curr_direction * camera_radius

            set_camera(bpy_data, position, camera_rotation_z, cam=cam)
        
        # render a frame
        scene.render.filepath = store_path + f'/color_{i:05}'
        
        tree.nodes['Depth Output'].base_path = store_path
        tree.nodes['Depth Output'].file_slots[0].path = f"/depth_{i:05}"
        
        tree.nodes['Normal Output'].base_path = store_path
        tree.nodes['Normal Output'].file_slots[0].path = f"/normal_{i:05}"

        bpy_ops.render.render(write_still=True)  # render still

        frame_data = {
            'file_path': scene.render.filepath,
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        if not use_given_poses:
            out_data['frames'].append(frame_data)

    with open(os.path.join(store_path, "transforms.json"), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    if use_given_poses:
        create_camera_constraint(cam, b_empty)

    b_empty.location = EMPTY_LOCATION

# --------------------------------------------------------------------------------------

def render_mask(
        bpy_data, bpy_context, bpy_ops, tree,
        out_data, store_path, VIEWS
    ):

    cam = bpy_data.objects["Camera"] # it's kinda global
    scene = bpy_context.scene  # it's kinda global

    scene.render.filepath = store_path + '/useless_color'
    tree.nodes['Depth Output'].base_path = store_path
    tree.nodes['Depth Output'].file_slots[0].path = "/useless_depth"
    tree.nodes['Normal Output'].base_path = store_path
    tree.nodes['Normal Output'].file_slots[0].path = "/useless_normal"

    for i in range(
        min(len(out_data["frames"]), VIEWS)
    ):    
        cam.matrix_world = mathutils.Matrix(out_data["frames"][i]["transform_matrix"])

        tree.nodes['Mask Output'].base_path = store_path
        tree.nodes['Mask Output'].file_slots[0].path = f"/mask_{i:05}"

        # render a frame
        bpy_ops.render.render(write_still=True)  # render still    

# --------------------------------------------------------------------------------------
