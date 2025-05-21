# Recorded script from Mayavi2
from numpy import array
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# ------------------------------------------- 
module_manager = engine.scenes[0].children[0].children[0]
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 0
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 0
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
scene = engine.scenes[0]
scene.scene.camera.position = [5.769073860860264, -40.05813403086964, 15.705330884874934]
scene.scene.camera.focal_point = [7.0, 6.307637390964858, 9.136774258918834]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.0014194707320588987, 0.1402305572451933, 0.9901178596094224]
scene.scene.camera.clipping_range = [31.244705261444583, 68.8224099258276]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 0
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
camera_light = engine.scenes[0].scene.light_manager.lights[0]
camera_light.elevation = 0.0
camera_light.azimuth = 0.0
camera_light1 = engine.scenes[0].scene.light_manager.lights[1]
camera_light1.elevation = 0.0
camera_light1.azimuth = 0.0
camera_light1.intensity = 1.0
camera_light1.activate = False
camera_light2 = engine.scenes[0].scene.light_manager.lights[2]
camera_light2.elevation = 0.0
camera_light2.azimuth = 0.0
camera_light2.intensity = 1.0
camera_light2.activate = False
scene.scene.light_manager.light_mode = 'vtk'
camera_light1.activate = True
camera_light2.activate = True
camera_light.elevation = 45.0
camera_light.azimuth = 45.0
camera_light1.elevation = -30.0
camera_light1.azimuth = -60.0
camera_light1.intensity = 0.6
camera_light2.elevation = -30.0
camera_light2.azimuth = 60.0
camera_light2.intensity = 0.5
scene.scene.light_manager.light_mode = 'raymond'
camera_light.elevation = 0.0
camera_light.azimuth = 0.0
camera_light1.elevation = 0.0
camera_light1.azimuth = 0.0
camera_light1.intensity = 1.0
camera_light1.activate = False
camera_light2.elevation = 0.0
camera_light2.azimuth = 0.0
camera_light2.intensity = 1.0
camera_light2.activate = False
scene.scene.light_manager.light_mode = 'vtk'
from tvtk.pyface.light_manager import CameraLight
camera_light4 = CameraLight()
scene.scene.light_manager.lights[4:4] = [camera_light4]
scene.scene.light_manager.number_of_lights = 5

camera_light5 = CameraLight()
scene.scene.light_manager.lights[5:5] = [camera_light5]
scene.scene.light_manager.number_of_lights = 6

camera_light6 = CameraLight()
scene.scene.light_manager.lights[6:6] = [camera_light6]
scene.scene.light_manager.number_of_lights = 7

camera_light7 = CameraLight()
scene.scene.light_manager.lights[7:7] = [camera_light7]
scene.scene.light_manager.number_of_lights = 8
scene.scene.light_manager.lights[7:8] = []
scene.scene.light_manager.number_of_lights = 7
scene.scene.light_manager.lights[6:7] = []
scene.scene.light_manager.number_of_lights = 6
camera_light1.activate = True
camera_light2.activate = True
camera_light.elevation = 45.0
camera_light.azimuth = 45.0
camera_light1.elevation = -30.0
camera_light1.azimuth = -60.0
camera_light1.intensity = 0.6
camera_light2.elevation = -30.0
camera_light2.azimuth = 60.0
camera_light2.intensity = 0.5
scene.scene.light_manager.light_mode = 'raymond'
camera_light5.intensity = 0.9
camera_light5.intensity = 0.8
camera_light5.intensity = 0.7
scene.scene.camera.position = [6.198684295694855, -40.067726179151634, 15.718129946219019]
scene.scene.camera.focal_point = [7.429610434834592, 6.298045242682867, 9.149573320262915]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.0014194707320588987, 0.1402305572451933, 0.9901178596094224]
scene.scene.camera.clipping_range = [31.244705261444583, 68.8224099258276]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
scene.scene.camera.position = [5.574233877749423, -39.4374780431701, 19.112773732318644]
scene.scene.camera.focal_point = [7.429610434834592, 6.298045242682867, 9.149573320262915]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [-0.0073918254126496795, 0.2131322619907986, 0.9769953939583107]
scene.scene.camera.clipping_range = [30.37617702456188, 70.23363190431918]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.interaction_state = 0
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([2147483647, 2147483647], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_normalized_viewport_size = array([0., 0.])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1], dtype=int64)
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.75713542, 0.05061816])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.1, 0.8])
