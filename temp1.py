import bpy
import numpy as np
import json
# from pathlib import Path
import os
from tqdm import tqdm
import multiprocessing

IMAGE_SIZE = 512
RADIUS = 10


def get_k(focal_length, width, height):
    return np.array([[focal_length, 0, width/2],
                    [0, focal_length, height/2],
                    [0, 0, 1]])


def set_camera_location(theta, phi, radius):
    def rot_phi(ph):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(ph), np.sin(ph), 0],
            [0, -np.sin(ph), np.cos(ph), 0],
            [0, 0, 0, 1]]).astype(np.float32)

    def rot_theta(th):
        return np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).astype(np.float32)

    # def extri_spherical(theta, phi, radius, mesh_scale=1):
    extri = rot_theta(theta / 180 * np.pi)
    extri = rot_phi(phi / 180 * np.pi) @ extri
    trans = np.array([0, 0, -radius, 1]).reshape(-1, 1)
    extri = np.concatenate((extri[:, :3], trans), axis=1)
    pose = np.linalg.inv(extri)
    pose = np.array([[-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]]) @ pose
    x, y, z = pose[:3, 3]
    return x, y, z, pose

def edit_mat(obj, new_mat, new_name, old_mat, color=None):
    new_mat.name = new_name
    old_color = old_mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value
    if new_name.endswith('carpaint'):
        if color is not None:
            new_mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value = color
        else:
            new_mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value = old_color
    # elif new_name.endswith('carpaint'):
    #     new_mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value = old_color

    for s in obj.material_slots:
        if s.material.name == old_mat.name:
            s.material = bpy.data.materials[new_name]


def load_obj(obj_path):
    # base_dir = '/media/zsh/data2/datasets/cars_selected'
    # sections = os.listdir(base_dir)
    # import obj
    # base_dir = Path('/mnt/data1/datasets/cars_selected/')
    base_dir = '/mnt/data1/datasets/cars_selected/'
    car_name = 'Volkswagen_1600_(Typ3)_variant_1965'

    # obj_path = '/mnt/data1/datasets/cars_selected/November_2020/test_ok/Volkswagen_1600_(Typ3)_variant_1965/Volkswagen_1600_(Typ3)_variant_1965.obj'
    # obj_path = '/media/zsh/data2/datasets/cars_selected/October_2020/test_ok/Mercedes-Benz_280_(W108)_SEL_1972/Mercedes-Benz_280_(W108)_SEL_1972.obj'
    # obj_path = '/media/zsh/data2/datasets/cars_selected/October_2020/test_ok/Toyota_Land-Cruiser_(J70)_DoubleCab_Pickup_(VDJ79)_2012/Toyota_Land-Cruiser_(J70)_DoubleCab_Pickup_(VDJ79)_2012.obj'
    # delete all things left
    for ob in bpy.data.objects:
        if ob.name != 'Starlight Sun':
            # ob.select_set(True)
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.ops.outliner.orphans_purge()
    for m in bpy.data.materials:
        bpy.data.materials.remove(m)

    # enable psa add-on
    if not bpy.data.worlds['World'].psa_general_settings.enabled:
        bpy.data.worlds['World'].psa_general_settings.enabled = True
    bpy.data.objects['Starlight Sun'].rotation_euler[2] = -37 / 180 * np.pi
    bpy.data.objects['Starlight Sun'].rotation_euler[0] = -11 / 180 * np.pi
    # import obj
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='Y', axis_up='Z')
    bpy.ops.object.select_all(action='DESELECT')

    # join
    ob = bpy.data.objects[0]
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

    ## recenter
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    ## rescale
    scale = 0.009
    bpy.context.object.scale[0] = scale
    bpy.context.object.scale[1] = scale
    bpy.context.object.scale[2] = scale

    # material edit
    carpaint_path = '/home/zsh/projects/blender/material/red_carpaint.blend'
    with bpy.data.libraries.load(carpaint_path, link=False) as (data_src, data_dst):
        data_dst.materials = data_src.materials
    carpaint_mat = data_dst.materials[0]
    old_mat = bpy.data.materials['carpaint']
    edit_mat(ob, carpaint_mat, 'new_carpaint', old_mat, color=None)

    glass_path = '/home/zsh/projects/blender/material/glass.blend'
    with bpy.data.libraries.load(glass_path, link=False) as (data_src, data_dst):
        data_dst.materials = data_src.materials
    data_dst.materials = data_src.materials
    glass_mat = data_dst.materials[0]
    old_mats = [bpy.data.materials['windowglass'], bpy.data.materials['mirror'], bpy.data.materials['clearglass'], ]
    for mat in old_mats:
        edit_mat(ob, glass_mat, f'new_{mat.name}', old_mat, color=None)

    # carpaint_mat.name = 'new_carpaint'
    # origin_color = bpy.data.materials['carpaint'].node_tree.nodes['Principled BSDF'].inputs[0].default_value
    # bpy.data.materials["carpaint"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = origin_color



def scene_setup(save_dir):
    # render settings
    context = bpy.context
    context.scene.render.engine = 'CYCLES'
    context.scene.cycles.device = 'GPU'
    context.scene.cycles.adaptive_threshold = 0.05
    context.scene.cycles.samples = 1024

    # render resolution
    data = bpy.data
    data.scenes['Scene'].render.resolution_x = IMAGE_SIZE
    data.scenes['Scene'].render.resolution_y = IMAGE_SIZE
    data.scenes['Scene'].render.resolution_percentage = 100
    # context.scene.render.use_antialiasing = True
    data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
    data.scenes['Scene'].render.film_transparent = True

    # add camera
    location = 0, 0, 10
    w, h, focal_length = 36, 36, 50
    scene = bpy.context.scene
    bpy.ops.object.camera_add(align='WORLD', location=location, rotation=(0, 0, 0))
    # data.objects['Camera'].sensor_width = w
    # data.objects['Camera'].lens = focal_length
    scene.camera = data.objects['Camera']
    camera_angle_x = np.arctan(w//2 / focal_length) * 2
    focal = (IMAGE_SIZE // 2) / np.tan(camera_angle_x / 2)
    k = get_k(focal, IMAGE_SIZE, IMAGE_SIZE)

    # add light
    # bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=location, rotation=(0, 0, 0))
    # light_dl = bpy.data.lights['Sun']
    # light_dl.energy = 3
    # light_dl.color = (0,1,0)
    # light = bpy.data.objects['Sun']
    # bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 10, 0), rotation=(-90, 0, 0))
    # light_dl = bpy.data.lights['Sun.001']
    # light_dl.energy = 2
    # light_dl.color = (1,1,1)

    # set location
    frames = []
    for theta in range(-180, 180, 10):
        for phi in range(0, -46, -15):
            x, y, z, pose = set_camera_location(theta, phi, radius=RADIUS)
            scene.camera.location = x, y, z
            # light.location = x, y, z
            rot_y = theta * np.pi / 180
            rot_x = phi * np.pi / 180
            scene.camera.rotation_mode = 'XYZ'
            scene.camera.rotation_euler[0] = rot_x + np.pi / 2  # important! nerf's rot_x is reversed
            scene.camera.rotation_euler[2] = rot_y + np.pi
            # light.rotation_mode = 'XYZ'
            # light.rotation_euler[0] = rot_x + np.pi / 2
            # light.rotation_euler[2] = rot_y + np.pi


            # file_name = '/media/zsh/data2/datasets/blender_render_10/test/1.png'
            file_name = f'{phi}_{theta}.png'
            os.makedirs(os.path.join(save_dir, 'images',), exist_ok=True)
            save_path = os.path.join(save_dir, 'images', file_name)
            bpy.data.scenes['Scene'].render.filepath = save_path
            bpy.ops.render.render(write_still=True)

            data_frame = {'file_path': save_path,
                          'file_name': file_name,
                          'transform_matrix': pose.tolist(),
                          'intrinsic_matrix': k.tolist(), }
            frames.append(data_frame)

    meta = {'camera_angle_x': camera_angle_x,
            'frames': frames,}
    meta_path = os.path.join(save_dir, 'transforms_train.json')
    with open(meta_path, 'w') as fp:
        json.dump(meta, fp, indent=4)


def render():
    base_dir = '/media/zsh/data2/datasets/cars_selected'
    data_dir = '/media/zsh/data2/datasets/blender_render_10_newmat_1'

    sections = sorted(os.listdir(base_dir))
    # bpy.data.worlds['World'].psa_general_settings.enabled = True
    for sec in sections:
        print(f'now processing section:{sec}')
        obj_dir = os.path.join(base_dir, sec, 'test_ok')
        for obj_name in sorted(os.listdir(obj_dir)):
            print(f'now processing obj:{obj_name}')
            obj_path = os.path.join(obj_dir, obj_name, f'{obj_name}.obj')
            load_obj(obj_path)
            save_dir = os.path.join(data_dir, sec, obj_name)
            os.makedirs(save_dir, exist_ok=True)
            scene_setup(save_dir)

def render_one():
    obj_path = '/media/zsh/data2/datasets/cars_selected/December_2020/test_ok/Chevrolet_Blazer_K5_1976/Chevrolet_Blazer_K5_1976.obj'
    # obj_path = '/mnt/data2/datasets/cars_selected/December_2020/test_ok/Chevrolet_Blazer_K5_1976/Chevrolet_Blazer_K5_1976.obj'
    data_dir = './trans_ill_newmat'
    load_obj(obj_path)
    save_dir = data_dir
    os.makedirs(save_dir, exist_ok=True)
    scene_setup(save_dir)

def process_one(section, obj_name, obj_dir):
    data_dir = '/media/zsh/data2/datasets/blender_render_10_newmat'
    obj_path = os.path.join(obj_dir, obj_name, f'{obj_name}.obj')
    load_obj(obj_path)
    save_dir = os.path.join(data_dir, section, obj_name)
    os.makedirs(save_dir, exist_ok=True)
    scene_setup(save_dir)

def multiprocess_render():
    base_dir = '/media/zsh/data2/datasets/cars_selected'
    sections = sorted(os.listdir(base_dir))
    # bpy.data.worlds['World'].psa_general_settings.enabled = True
    for sec in sections:
        obj_dir = os.path.join(base_dir, sec, 'test_ok')
        pool = multiprocessing.Pool(processes=10)
        for car_name in sorted(os.listdir(obj_dir)):
            pool.apply_async(process_one, (sec, car_name, obj_dir))
        pool.close()
    pool.join()


if __name__ == '__main__':
    render()


