"""Builds the authored test scenes (run inside Blender):

    blender -b --python scenes/make_scenes.py                 # all of them
    blender -b --python scenes/make_scenes.py -- ripple       # only the ones named

cornell.blend    an enclosed box lit only by an emissive panel: diffuse bounce light, a mirror ball,
                 no lamp objects at all (so the renderer gets no analytic light hints)
glassware.blend  clear, tinted and frosted glass on a white floor under area, spot and point lights
ripple.blend     a 40-frame animation in which meshes change shape: a sphere with a wave travelling over
                 it, a cylinder that bends over, a cube that tumbles across, a camera that drifts

Render settings follow the user's stage.blend (Cycles on the CPU, 64 samples, noise threshold 0.01,
OpenImageDenoise), so Blender is configured alike in every comparison.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))


def reset(width, height, view='AgX', exposure=0.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    c = s.cycles
    c.device, c.samples, c.use_adaptive_sampling, c.adaptive_threshold = 'CPU', 64, True, 0.01
    c.use_denoising, c.denoiser, c.max_bounces = True, 'OPENIMAGEDENOISE', 12
    s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = width, height, 100
    s.render.image_settings.file_format = 'PNG'
    s.view_settings.view_transform, s.view_settings.look, s.view_settings.exposure = view, 'None', exposure
    return s


def world(scene, color, strength):
    w = bpy.data.worlds.new('World')
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (*color, 1.0)
    bg.inputs['Strength'].default_value = strength
    scene.world = w


def material(name, base=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5, transmission=0.0, ior=1.5, emission=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*base, 1.0)
    b.inputs['Metallic'].default_value = metallic
    b.inputs['Roughness'].default_value = roughness
    b.inputs['Transmission Weight'].default_value = transmission
    b.inputs['IOR'].default_value = ior
    if emission:
        b.inputs['Emission Color'].default_value = (*emission, 1.0)
        b.inputs['Emission Strength'].default_value = strength
    return m


def finish(obj, mat, smooth=False):
    obj.data.materials.append(mat)
    if smooth:
        for p in obj.data.polygons:
            p.use_smooth = True
    return obj


def plane(name, size, location, rotation, mat):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)
    o = bpy.context.object
    o.name = name
    return finish(o, mat)


def camera(scene, location, target, lens):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.data.lens = lens
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam


def light(name, kind, location, target, energy, color=(1, 1, 1), **props):
    data = bpy.data.lights.new(name, kind)
    data.energy, data.color = energy, color
    for k, v in props.items():
        setattr(data, k, v)
    o = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(o)
    o.location = location
    o.rotation_euler = (Vector(target) - Vector(location)).to_track_quat('-Z', 'Y').to_euler()


def cornell():
    s = reset(900, 900, view='Standard')
    world(s, (0, 0, 0), 0.0)
    white, red, green = material('White', (0.73, 0.73, 0.73), roughness=0.9), material('Red', (0.63, 0.06, 0.05), roughness=0.9), material('Green', (0.12, 0.45, 0.14), roughness=0.9)
    h = math.pi / 2
    plane('Floor', 2, (0, 0, 0), (0, 0, 0), white)
    plane('Ceiling', 2, (0, 0, 2), (math.pi, 0, 0), white)
    plane('Back', 2, (0, 1, 1), (h, 0, 0), white)
    plane('Left', 2, (-1, 0, 1), (0, h, 0), red)
    plane('Right', 2, (1, 0, 1), (0, -h, 0), green)
    plane('Panel', 0.6, (0, 0, 1.995), (math.pi, 0, 0), material('Lamp', (0, 0, 0), emission=(1.0, 0.86, 0.68), strength=28.0))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.34, 0.32, 0.6), rotation=(0, 0, math.radians(17)), scale=(0.6, 0.6, 1.2))
    finish(bpy.context.object, white).name = 'Tall box'
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.36, location=(0.42, -0.28, 0.36))
    finish(bpy.context.object, material('Mirror', (0.92, 0.88, 0.72), metallic=1.0, roughness=0.12), smooth=True).name = 'Ball'
    camera(s, (0, -3.9, 1.0), (0, 0, 1.0), 50)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'cornell.blend'))


def glassware():
    s = reset(1200, 800)
    world(s, (0.02, 0.022, 0.03), 1.0)
    plane('Floor', 60, (0, 0, 0), (0, 0, 0), material('Paper', (0.82, 0.82, 0.8), roughness=0.6))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.7, location=(-1.15, 0.25, 0.7))
    finish(bpy.context.object, material('Clear glass', (1, 1, 1), roughness=0.0, transmission=1.0, ior=1.5), smooth=True).name = 'Sphere'
    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=32, major_radius=0.62, minor_radius=0.24, location=(0.95, -0.45, 0.24))
    finish(bpy.context.object, material('Amber glass', (1.0, 0.52, 0.12), roughness=0.04, transmission=1.0, ior=1.5), smooth=True).name = 'Ring'
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.42, depth=1.5, location=(0.35, 1.05, 0.75))
    o = finish(bpy.context.object, material('Frosted blue', (0.35, 0.55, 1.0), roughness=0.25, transmission=1.0, ior=1.45), smooth=True)
    o.name = 'Column'
    for p in o.data.polygons:   # flat caps, smooth side
        p.use_smooth = abs(p.normal.z) < 0.5
    bpy.ops.mesh.primitive_cube_add(size=0.62, location=(-0.25, -1.15, 0.31), rotation=(0, 0, math.radians(28)))
    finish(bpy.context.object, material('Red block', (0.75, 0.07, 0.06), roughness=0.5)).name = 'Block'
    light('Key', 'AREA', (-4.0, -3.0, 5.5), (0, 0, 0.5), 900, (1.0, 0.96, 0.9), shape='SQUARE', size=3.0)
    light('Spot', 'SPOT', (4.5, 3.0, 5.0), (0.3, 0.2, 0.4), 2600, (0.85, 0.92, 1.0), spot_size=math.radians(38), spot_blend=0.35, shadow_soft_size=0.12)
    light('Spark', 'POINT', (-0.2, -2.6, 0.9), (0, 0, 0.5), 110, (1.0, 0.7, 0.4), shadow_soft_size=0.08)
    camera(s, (5.4, -5.4, 2.7), (0, 0, 0.55), 50)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'glassware.blend'))


def ripple():
    s = reset(640, 400)
    s.frame_start, s.frame_end = 1, 40
    world(s, (0.3, 0.35, 0.45), 1.0)
    plane('Floor', 12, (0, 0, 0), (0, 0, 0), material('Ground', (0.6, 0.6, 0.6), roughness=0.5))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0, location=(-1.6, 0, 1.1))
    o = finish(bpy.context.object, material('Red', (0.7, 0.1, 0.1), roughness=0.3), smooth=True)
    o.name = 'Ripple'
    wave = o.modifiers.new('Wave', 'WAVE')   # animates with the frame number on its own
    wave.height, wave.width, wave.speed, wave.use_normal = 0.25, 0.8, 0.12, True
    bpy.ops.mesh.primitive_cube_add(size=1.2, location=(1.8, 0.5, 0.6))
    o = finish(bpy.context.object, material('Blue', (0.1, 0.2, 0.7), roughness=0.4))
    o.name = 'Mover'
    o.keyframe_insert('location', frame=1)
    o.keyframe_insert('rotation_euler', frame=1)
    o.location, o.rotation_euler = (1.2, -1.0, 1.1), (0.3, 0.2, 1.2)
    o.keyframe_insert('location', frame=40)
    o.keyframe_insert('rotation_euler', frame=40)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.25, depth=2.4, location=(0.3, 1.6, 1.2))
    o = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.object.mode_set(mode='OBJECT')
    finish(o, material('Green', (0.1, 0.6, 0.2), roughness=0.5), smooth=True).name = 'Bender'
    bend = o.modifiers.new('Bend', 'SIMPLE_DEFORM')
    bend.deform_method, bend.deform_axis, bend.angle = 'BEND', 'X', 0.0
    bend.keyframe_insert('angle', frame=1)
    bend.angle = math.radians(80)
    bend.keyframe_insert('angle', frame=40)
    light('Sun', 'SUN', (3.0, -3.0, 6.0), (0.4, 0.2, 0.0), 3.0)
    camera(s, (0, -8, 3.2), (0, 0, 0.6), 50)
    cam = s.camera
    cam.keyframe_insert('location', frame=1)
    cam.location = (1.5, -7.5, 3.6)
    cam.keyframe_insert('location', frame=40)
    s.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'ripple.blend'))


SCENES = dict(cornell=cornell, glassware=glassware, ripple=ripple)
for name in (sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else SCENES):
    SCENES[name]()
