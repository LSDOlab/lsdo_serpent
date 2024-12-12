import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update(plt.rcParamsDefault)
import lsdo_geo
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import pickle
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from lsdo_serpent.core.fish_post_processor import fish_post_processor
import modopt

recorder = csdl.Recorder(inline=True, debug=False)
recorder.start()

# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import NBodyPendulumModel, NBodyPendulumStateData
# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import PendulumBody
# from mbd_trial.framework_trial.examples.n_body_flexible_pendulum.n_body_flexible_pendulum_model import PendulumSystem

from examples.advanced_examples.dynamic_beam_optimization.robotic_fish_optimization_model import Body, System, SerpentV1DynamicsModel, StateData, evaluate_actuator_loads


plot = False
make_video = True
save_stuff = True
run_stuff = True
run_jax = True
run_optimization = False

run_stuff_twice = False
if run_stuff:
    run_stuff_twice = False  # Set to True if you want to run stuff twice to see the simulation time without compilation time
scale_flexible_states = False

# region Import and Setup
def import_geometry() -> lfs.Function:
    with open("examples/advanced_examples/dynamic_beam_optimization/fishy_volume_geometry_fine.pickle", 'rb') as handle:
    # with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_mini_v1_volume_geometry_2.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        # fishy.coefficients = csdl.Variable(value=fishy.coefficients.value, name='fishy_coefficients')   # Remake variables because can't pickle variables
        fishy.coefficients = csdl.Variable(value=fishy.coefficients, name='fishy_coefficients')   # Remake variables because can't pickle variables
        return fishy

fishy_geometry = import_geometry()
new_geometry_function_space = lfs.BSplineSpace(num_parametric_dimensions=3, degree=3, coefficients_shape=(15,11,11))
# fishy_geometry.coefficients.reshape((51,15,15,3))
fishy_geometry = fishy_geometry.refit(new_function_space=new_geometry_function_space, grid_resolution=(65,35,35))
# fishy_geometry.plot()
fishy_geometry.name = 'fishy'
# fishy_geometry.plot()

# region Create Beam Representation

points_to_project = np.array(
    [
        [0.278, 0., 0.],        # head front
        [0.218, 0., 0.],        # head back
        [0.18, 0., 0.],         # module 1 middle
        [0.15, 0., 0.],         # module 1 back
        [0.126, 0., 0.],        # module 2 front
        [0.092, 0., 0.],        # module 2 middle
        [0.058, 0., 0.],        # module 2 back
        [0.034, 0., 0.],        # module 3 front
        [0.000, 0., 0.],        # module 3 middle
        [-0.034, 0., 0.],       # module 3 back
        [-0.053, 0., 0.],       # tail front
        [-0.1, 0., 0.],       # tail middle 1
        [-0.14, 0., 0.],       # tail middle 2
        [-0.181, 0., 0.],       # tail back
    ]
)
beam_parametric_mesh = fishy_geometry.project(points_to_project, plot=False)
parametric_mesh = beam_parametric_mesh
solver_geometry_representations = {}
solver_geometry_representations['beam'] = lsdo_geo.Mesh(geometry=fishy_geometry, parametric_coordinates=parametric_mesh)

# Create parametric grid for u=0, v=0-1, w=0-1 with 5 nodes in each direction
num_nodes = 25
u = np.linspace(0.5, 0.5, num_nodes*num_nodes).reshape((-1,1))  # u is constant at 0
v = np.linspace(0, 1, num_nodes)
w = np.linspace(0, 1, num_nodes)
v, w = np.meshgrid(v, w, indexing='ij')
v = v.reshape((-1,1))
w = w.reshape((-1,1))
silicone_parametric_mesh = np.hstack([u, v, w]).reshape((num_nodes,num_nodes,3))

num_fr4_nodes = 11
# fr4_left_line = fishy_geometry.project(np.linspace(np.array([0., -3.33360111e-02, -0.000381]), np.array([0., 3.33549866e-02, -0.000381]), num_fr4_nodes), 
#                                        projection_tolerance=2.e-4)
# fr4_right_line = fishy_geometry.project(np.linspace(np.array([0., -3.33360111e-02, 0.000381]), np.array([0., 3.33549866e-02, 0.000381]), num_fr4_nodes), 
#                                        projection_tolerance=2.e-4)
fr4_left_line = fishy_geometry.project(np.linspace(np.array([0., -3.30360111e-02, -0.000381]), np.array([0., 3.30549866e-02, -0.000381]), num_fr4_nodes), 
                                       projection_tolerance=2.e-4)
fr4_right_line = fishy_geometry.project(np.linspace(np.array([0., -3.30360111e-02, 0.000381]), np.array([0., 3.30549866e-02, 0.000381]), num_fr4_nodes), 
                                       projection_tolerance=2.e-4)
# fr4_parametric_mesh = np.concatenate([fr4_left_line.reshape((1,-1,3)), fr4_right_line.reshape((1,-1,3))], axis=0)
fr4_parametric_mesh = np.linspace(fr4_left_line, fr4_right_line, num_fr4_nodes)

solver_geometry_representations['cross_section'] = {
    'silicone':lsdo_geo.Mesh(geometry=fishy_geometry, parametric_coordinates=silicone_parametric_mesh),
    'fr4':lsdo_geo.Mesh(geometry=fishy_geometry, parametric_coordinates=fr4_parametric_mesh),
}

# solver_geometry_representations['cross_section']['fr4'].evaluate(fishy_geometry, plot=True)

# endregion Create Beam Representation

# region Fluid Mesh Parametric Coordinates Definition
# num_panels_per_dimension = 5    # NOTE: Probably want to individually manipulate this for each direction.
# side_num_chordwise = 7  # not including front/back contribution
# side_num_chordwise = 13  # not including front/back contribution    
# side_num_chordwise = 15  # not including front/back contribution    NOTE: Been running with this
side_num_chordwise = 17  # not including front/back contribution
# side_num_chordwise = 21  # not including front/back contribution
# side_num_chordwise = 25  # not including front/back contribution
# side_num_chordwise = 34  # not including front/back contribution
# side_num_spanwise = 5  # not including top/bottom contribution      
side_num_spanwise = 7  # not including top/bottom contribution
# side_num_spanwise = 9  # not including top/bottom contribution
# side_num_spanwise = 11  # not including top/bottom contribution     NOTE: Been running with this
# side_num_spanwise = 13  # not including top/bottom contribution
# side_num_spanwise = 17  # not including top/bottom contribution
num_chordwise = side_num_chordwise + 3 + 5
num_spanwise = side_num_spanwise + side_num_spanwise//2
parametric_grid_1 = np.zeros((3, side_num_spanwise, 3))  # First dimension can be arbitrarily set
parametric_grid_2 = np.zeros((5, side_num_spanwise, 3))  # First dimension can be arbitrarily set
parametric_grid_3 = np.zeros((side_num_chordwise, side_num_spanwise//2, 3))  # Second dimension can be arbitrarily set
parametric_grid_4 = np.zeros((side_num_chordwise, side_num_spanwise//2, 3))  # Second dimension can be arbitrarily set
parametric_grid_5 = np.zeros((side_num_chordwise, side_num_spanwise, 3))
parametric_grid_6 = np.zeros((side_num_chordwise, side_num_spanwise, 3))
# parametric_mesh_2, parametric_mesh_1 = \
#     np.meshgrid(np.linspace(0., 1., num_panels_per_dimension), np.linspace(0., 1., num_panels_per_dimension))
# parametric_grid_1 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_2 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_3 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_4 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_5 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_6 = np.zeros((num_panels_per_dimension**2, 3))
# parametric_grid_5 = np.zeros((num_panels_per_dimension, num_panels_per_dimension, 3))
# parametric_grid_6 = np.zeros((num_panels_per_dimension, num_panels_per_dimension, 3))

parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(1., 0., 3))
parametric_grid_1[:,:,1] = parametric_mesh_2
parametric_grid_1[:,:,2] = parametric_mesh_1
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(0., 1., 5))
parametric_grid_2[:,:,0] = 1.
parametric_grid_2[:,:,1] = parametric_mesh_2
parametric_grid_2[:,:,2] = parametric_mesh_1
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise//2), np.linspace(0., 1., side_num_chordwise))
parametric_grid_3[:,:,0] = parametric_mesh_1
# parametric_grid_3[:,1] = np.zeros(parametric_mesh_1.shape)
parametric_grid_3[:,:,2] = parametric_mesh_2
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise//2), np.linspace(0., 1., side_num_chordwise))
parametric_grid_4[:,:,0] = parametric_mesh_1
parametric_grid_4[:,:,1] = 1.
parametric_grid_4[:,:,2] = parametric_mesh_2
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(0., 1., side_num_chordwise + 2)[1:-1])
parametric_grid_5[:,:,0] = parametric_mesh_1
parametric_grid_5[:,:,1] = parametric_mesh_2
# parametric_grid_5[:,:,2] = np.zeros(parametric_mesh_1.shape)
parametric_mesh_2, parametric_mesh_1 = \
    np.meshgrid(np.linspace(0., 1., side_num_spanwise), np.linspace(1., 0., side_num_chordwise + 2)[1:-1])
parametric_grid_6[:,:,0] = parametric_mesh_1
parametric_grid_6[:,:,1] = parametric_mesh_2
parametric_grid_6[:,:,2] = np.ones_like(parametric_mesh_1)


# panel_method_parametric_grids = [parametric_grid_1, parametric_grid_2, parametric_grid_3, parametric_grid_4, parametric_grid_5, parametric_grid_6]

# region Stich together parametric mesh

# num_chordwise = side_num_chordwise + side_num_chordwise + 5 + 4   # left, right, front, back (middle of back (TE) gets added twice)
# num_spanwise = side_num_spanwise  # top, bottom, left, right
# panel_method_parametric_mesh = np.zeros((num_chordwise, side_num_spanwise, 3))

# panel_method_parametric_mesh[:2] = parametric_grid_1[1:]  # Only later half chordwise, All spanwise
# panel_method_parametric_mesh[2:2 + side_num_chordwise] = parametric_grid_5
# panel_method_parametric_mesh[2 + side_num_chordwise: 2 + side_num_chordwise + 5] = parametric_grid_2
# panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise] = \
#     parametric_grid_6 
# panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:] = parametric_grid_1[:2]

# NOTE: Try just evaluating a line on the top (and bottom). For the front faces, use the same endpoint (front top point).
# -- This will create a series of triangular panels for the front face, but I think it might be necessary.
num_chordwise = side_num_chordwise + side_num_chordwise + 5 + 4   # left, right, front, back (middle of back (TE) gets added twice)
num_spanwise = side_num_spanwise + 2
panel_method_parametric_mesh = np.zeros((num_chordwise, num_spanwise, 3))

panel_method_parametric_mesh[:2,1:-1] = parametric_grid_1[1:]  # Only later half chordwise, All spanwise
panel_method_parametric_mesh[2:2 + side_num_chordwise,1:-1] = parametric_grid_5
panel_method_parametric_mesh[2 + side_num_chordwise: 2 + side_num_chordwise + 5,1:-1] = parametric_grid_2
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,1:-1] = \
    parametric_grid_6 
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,1:-1] = parametric_grid_1[:2]

# # - Insert top and bottom
top_line = np.linspace(np.array([0., 1., 0.5]), np.array([1., 1., 0.5]), side_num_chordwise + 2)[1:-1]
bottom_line = np.linspace(np.array([0., 0., 0.5]), np.array([1., 0., 0.5]), side_num_chordwise + 2)[1:-1]
top_line_reversed = top_line[::-1]
bottom_line_reversed = bottom_line[::-1]

panel_method_parametric_mesh[:2,0] = bottom_line[0]
panel_method_parametric_mesh[:2,-1] = top_line[0]
panel_method_parametric_mesh[2:2 + side_num_chordwise, 0] = bottom_line
panel_method_parametric_mesh[2:2 + side_num_chordwise, -1] = top_line
panel_method_parametric_mesh[2 + side_num_chordwise : 2 + side_num_chordwise + 5,0] = bottom_line[-1]
panel_method_parametric_mesh[2 + side_num_chordwise : 2 + side_num_chordwise + 5,-1] = top_line[-1]
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,0] = bottom_line_reversed
panel_method_parametric_mesh[2 + side_num_chordwise + 5 : 2 + side_num_chordwise + 5 + side_num_chordwise,-1] = top_line_reversed
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,0] = bottom_line[0]
panel_method_parametric_mesh[2 + side_num_chordwise + 5 + side_num_chordwise:,-1] = top_line[0]

# panel_mesh_this_timestep = fishy_geometry.evaluate(panel_method_parametric_mesh, plot=True)
# endregion Stich together parametric mesh
# endregion Fluid Mesh Parametric Coordinates Definition

# endregion Import and Setup


# # region Geometry Parameterization

# region -Projections for Design Variables (Parameterization Solver Inputs)
fishy_nose_parametric = fishy_geometry.project(points=np.array([[0.3, 0., 0.]]), plot=False)
fishy_tail_tip_parametric = fishy_geometry.project(points=np.array([[-0.2, 0., 0.]]), plot=False)

fishy_left_parametric = fishy_geometry.project(points=np.array([[0., 0., -0.05]]), plot=False)
fishy_right_parametric = fishy_geometry.project(points=np.array([[0., 0., 0.05]]), plot=False)

fishy_top_parametric = fishy_geometry.project(points=np.array([[0., 0.1, 0.]]), plot=False)
fishy_bottom_parametric = fishy_geometry.project(points=np.array([[0., -0.09, 0.]]), plot=False)

fishy_right_channel_edge_parametric = fishy_geometry.project(points=np.array([[0., 0., -0.02]]), plot=False)
# endregion -Projections for Design Variables (Parameterization Solver Inputs)
# endregion Import and Setup

# region Geometry Parameterization
# region -Create Parameterization Objects
# num_ffd_sections = 2
# ffd_block = lsdo_geo.construct_ffd_block_around_entities(entities=fishy, num_coefficients=(num_ffd_sections,2,2), degree=(1,1,1))
ffd_min_x = np.min(fishy_geometry.coefficients.value[:,:,:,0])
ffd_max_x = np.max(fishy_geometry.coefficients.value[:,:,:,0])
ffd_min_y = np.min(fishy_geometry.coefficients.value[:,:,:,1])
ffd_max_y = np.max(fishy_geometry.coefficients.value[:,:,:,1])
ffd_min_z = np.min(fishy_geometry.coefficients.value[:,:,:,2])
ffd_max_z = np.max(fishy_geometry.coefficients.value[:,:,:,2])
ffd_x_values = np.array([ffd_min_x, ffd_max_x])
ffd_y_values = np.array([ffd_min_y, ffd_max_y])
# ffd_z_values = np.array([ffd_min_z, -0.0015, 0.0015, ffd_max_z])
ffd_z_values = np.array([ffd_min_z, -0.003, 0.003, ffd_max_z])

x,y,z = np.meshgrid(ffd_x_values, ffd_y_values, ffd_z_values, indexing='ij')
ffd_corners = np.array([x.flatten(), y.flatten(), z.flatten()]).T.reshape(
    (len(ffd_x_values), len(ffd_y_values), len(ffd_z_values), 3))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,3,2))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,5,2), degree=(1,2,1))
ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy_geometry, corners=ffd_corners, num_coefficients=(2,7,2), degree=(1,2,1))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,9,2), degree=(1,2,1))
# ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=fishy, corners=ffd_corners, num_coefficients=(2,11,2))
# plotting_elements = ffd_block.plot(show=False)

ffd_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=2,
    parameterized_points_shape=ffd_block.coefficients.shape,
    name='ffd_sectional_parameterization',
)
num_ffd_sections = len(ffd_z_values)
# plotting_elements = fishy_geometry.plot(show=True, opacity=0.3, color='#FFCD00')

length_stretch = csdl.Variable(value=0., name='length_stretch')
width_stretch = csdl.Variable(value=0., name='width_stretch')
height_stretch = csdl.Variable(value=0., name='height_stretch')

# width_shape_variable = csdl.Variable(shape=(1,), value=-0., name='width_shape_deltas')
width_shape_variables = csdl.Variable(shape=(ffd_block.coefficients.shape[1]//2 + 1,), value=0., name='width_shape_deltas')
# width_shape_variables = width_shape_variables.set(csdl.slice[:], width_shape_variable)

# width_shape_variables.set_value(np.array([5.731138044206841364e-3, -6.657115938444321479e-3, -7.965134789907922119e-3, -8.002576986651789070e-3]))
# width_shape_variables.set_value(np.array([1.314760350121188814e-2, 1.243780499900765157e-2, -6.013704500299079214e-3, 1.971819727180312043e-2]))

width_shape_deltas = csdl.Variable(shape=(ffd_block.coefficients.shape[1],), value=0.)
width_shape_deltas = width_shape_deltas.set(csdl.slice[0:width_shape_variables.size], width_shape_variables)
width_shape_deltas = width_shape_deltas.set(csdl.slice[width_shape_variables.size:], width_shape_variables[-2::-1])
# deltas_sum = csdl.sum(width_shape_deltas)

# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
length_stretches = csdl.expand(length_stretch, (num_ffd_sections,))
width_stretches = csdl.expand(width_stretch, (num_ffd_sections,))
width_stretches = width_stretches.set(csdl.slice[1:3], 0.)
width_stretches = width_stretches.set(csdl.slice[0], -width_stretch)
height_stretches = csdl.expand(height_stretch, (num_ffd_sections,))

ffd_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs()
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=0, stretch=length_stretches)
ffd_sectional_parameterization_inputs.add_sectional_translation(axis=2, translation=width_stretches)
ffd_sectional_parameterization_inputs.add_sectional_stretch(axis=1, stretch=height_stretches)
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(ffd_sectional_parameterization_inputs, plot=False)

width_shape_deltas_expanded = csdl.expand(width_shape_deltas, ffd_block_coefficients.shape[:2], 'i->ji')
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,-1,2], ffd_block_coefficients[:,:,-1,2] + width_shape_deltas_expanded)
ffd_block_coefficients = ffd_block_coefficients.set(csdl.slice[:,:,0,2], ffd_block_coefficients[:,:,0,2] - width_shape_deltas_expanded)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
# fishy_geometry.coefficients = fishy_coefficients.reshape(fishy_geometry.coefficients.shape)

fishy_geometry.coefficients = fishy_coefficients
fishy_geometry.coefficients = fishy_geometry.coefficients.reshape((15,11,11,3))

# endregion -Evaluate Parameterization For Solver

# region -Evaluate Parameterization Solver
# region -Evaluate Parameterization Solver Inputs
fishy_nose = fishy_geometry.evaluate(fishy_nose_parametric)
fishy_tail_tip = fishy_geometry.evaluate(fishy_tail_tip_parametric)
computed_fishy_length = csdl.norm(fishy_nose - fishy_tail_tip)

fishy_left = fishy_geometry.evaluate(fishy_left_parametric)
fishy_right = fishy_geometry.evaluate(fishy_right_parametric)
computed_fishy_width = csdl.norm(fishy_left - fishy_right)

fishy_top = fishy_geometry.evaluate(fishy_top_parametric)
fishy_bottom = fishy_geometry.evaluate(fishy_bottom_parametric)
computed_fishy_height = csdl.norm(fishy_top - fishy_bottom)
# endregion -Evaluate Parameterization Solver Inputs

# region Geometric Design Variables
length = csdl.Variable(value=computed_fishy_length.value, name='length')
# width = csdl.Variable(value=computed_fishy_width.value, name='width')
height = csdl.Variable(value=computed_fishy_height.value, name='height')
# height.set_value(7.931587973291041926e-2)
# height.set_value(7.731425247084128927e-2)

# length = csdl.Variable(value=1.1, name='length')
# width = csdl.Variable(value=0.02, name='width')
# height = csdl.Variable(value=0.07, name='height')

# endregion Geometric Design Variables

geometry_parameterization_solver = lsdo_geo.ParameterizationSolver()

geometry_parameterization_solver.add_parameter(length_stretch)
# geometry_parameterization_solver.add_parameter(width_stretch)
geometry_parameterization_solver.add_parameter(height_stretch)

geometric_parameterization_variables = lsdo_geo.GeometricVariables()
geometric_parameterization_variables.add_variable(computed_fishy_length, length)
# geometric_parameterization_variables.add_variable(computed_fishy_width, width)
geometric_parameterization_variables.add_variable(computed_fishy_height, height)

geometry_parameterization_solver.evaluate(geometric_variables=geometric_parameterization_variables)

# cam = dict(
#     position=(0.515331, -0.0733514, 0.0547675),
#     focal_point=(0.0564735, 0.0511758, -0.0318653),
#     viewup=(-0.257268, -0.965993, -0.0258928),
#     roll=-178.723,
#     distance=0.483283,
#     clipping_range=(1.06092e-3, 1.06092),
# )
# ffd_block_plot = ffd_block.plot(plot_embedded_points=False, show=False)
# plotting_elements = fishy_geometry.plot(opacity=1., additional_plotting_elements=[ffd_block_plot], color='#C69214', show=False)
# import vedo
# plotter = vedo.Plotter(axes=0)
# plotter.show(plotting_elements, camera=cam)
# exit()

fishy_left = fishy_geometry.evaluate(fishy_left_parametric)
fishy_right = fishy_geometry.evaluate(fishy_right_parametric)
computed_fishy_width = csdl.norm(fishy_left - fishy_right)
# endregion Geometry Parameterization

'''
# recommended generalized_alpha_spectral_radius values:
# 0.82 for 1,2 bodies
# 0.7 for 3,4 bodies
# 0.6 for 5,6 bodies
# Not sure after that
'''
fishy_body = Body(geometry=fishy_geometry, name='fishy', representations=solver_geometry_representations)
fishy = System(bodies=[fishy_body])

# Create a two-body pendulum model
# generalized_alpha_spectral_radius = 0.1
# generalized_alpha_spectral_radius = 0.2
# generalized_alpha_spectral_radius = 0.5
# generalized_alpha_spectral_radius = 0.6
# generalized_alpha_spectral_radius = 0.7
# generalized_alpha_spectral_radius = 0.75
# generalized_alpha_spectral_radius = 0.82
# generalized_alpha_spectral_radius = 0.98
generalized_alpha_spectral_radius = 1.
# time_step = 1.e-5
# time_step = 1.e-4
# time_step = 5.e-4
# time_step = 1.e-3
# time_step = 5.e-3
# time_step = 1.e-2
# time_step = 0.0166
# time_step = 0.5
# time_step = 0.25
# time_step = 0.1
# time_step = 0.075
time_step = 0.05
# time_step = 0.02
# time_step = 0.04
model = SerpentV1DynamicsModel(system=fishy)

def generate_initial_states(swim_speed):
    '''
    This function helps generates intial states for the fishy that satisfy the constraints.
    '''
    num_rigid_body_states = 6
    num_beam_nodes = solver_geometry_representations['beam'].shape[0]
    num_flexible_states = num_beam_nodes*6
    initial_states = {}
    initial_state_derivatives = {}

    initial_states['fishy'] = {}
    initial_states['fishy']['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,), value=0.)

    initial_state_derivatives['fishy'] = {}
    initial_state_derivatives['fishy']['rigid_body_states'] = csdl.Variable(shape=(2*num_rigid_body_states,), value=0.)
    initial_state_derivatives['fishy']['rigid_body_states'] = initial_state_derivatives['fishy']['rigid_body_states'].set(csdl.slice[0], swim_speed)

    initial_states['fishy']['flexible_states'] = csdl.Variable(shape=(num_flexible_states,), value=0.)
    initial_state_derivatives['fishy']['flexible_states'] = csdl.Variable(shape=(2*num_flexible_states,), value=0.)

    # initial_states['lagrange_multipliers'] = np.zeros(2*num_bodies)
    # initial_state_derivatives['lagrange_multipliers'] = np.zeros(2*num_bodies)

    initial_state_data = StateData(states=initial_states, state_derivatives=initial_state_derivatives)
    return initial_state_data
        
# swim_speed = csdl.Variable(value=1.e-2)
# swim_speed = csdl.Variable(value=0.02)
# swim_speed = csdl.Variable(value=0.05)
# swim_speed = csdl.Variable(value=0.1)
swim_speed = csdl.Variable(value=0.13)
# swim_speed = csdl.Variable(value=0.17)
# swim_speed = csdl.Variable(value=0.2)
# swim_speed = csdl.Variable(value=0.3)
# swim_speed = csdl.Variable(value=0.5)
# swim_speed = csdl.Variable(value=1.)
initial_state_data = generate_initial_states(swim_speed)

# temp variables because geometry hasn't been pasted in yet
length = csdl.Variable(value=0.45)

# actuation_frequency = swim_speed/length*2.5 # traveling wave travels 50% faster than fluid
# actuation_frequency = swim_speed/length*1.5 # traveling wave travels 50% faster than fluid
# actuation_frequency = swim_speed/length*4 # traveling wave travels 50% faster than fluid
# actuation_frequency = csdl.Variable(value=0.5)
# actuation_frequency = csdl.Variable(value=1.)
actuation_frequency = csdl.Variable(value=7.124027530754508675e-01)
actuation_period = 1/actuation_frequency
num_actuation_cycles = 2
num_time_steps_per_cycle = int(actuation_period.value/time_step)
num_time_steps = num_time_steps_per_cycle*num_actuation_cycles
time = csdl.Variable(shape=(num_time_steps,), value=0.)
for i in csdl.frange(time.size):
    time = time.set(csdl.slice[i], i*time_step)
# t_final = num_actuation_cycles*actuation_period
print('num_time_steps: ', num_time_steps)

# max_pressure = 0.
# max_pressure = 2.e4 # Pa / around 3 psi
# max_pressure = 3.e4 # Pa / around 3 psi
# max_pressure = 5.e4 # Pa / around 7.5 psi?
# max_pressure = 2.e5 # Pa / Artificially high rn to see a displacement
max_pressure = 3.e5 # Pa / Artificially high rn to see a displacement
# max_pressure = 1.997650626835765664e+5
# max_pressure = 5.e5 # Pa / Artificially high rn to see a displacement       # This is what I was using
# max_pressure = 1.e6 # Pa / Artificially high rn to see a displacement
# max_pressure = 2.e6 # Pa / Artificially high rn to see a displacement
# max_pressure = 4.e6 # Pa / Artificially high rn to see a displacement
max_pressure = csdl.Variable(value=max_pressure)

fishy_cross_section = solver_geometry_representations['cross_section']['silicone'].evaluate(fishy_geometry)
actuator_loads, actuator_load_derivatives = evaluate_actuator_loads(time, actuation_frequency, max_pressure=max_pressure, cross_section_mesh=fishy_cross_section, 
                                   generalized_alpha_spectral_radius=generalized_alpha_spectral_radius, num_actuators=3)
external_fluid_load = csdl.Variable(shape=(num_time_steps, 6), value=0.)

if run_stuff and not run_jax:
    recorder.inline = True
else:
    recorder.inline = False

from time import time as timer
# region MBD Section
t1 = timer()
t, states, state_derivatives, lagrange_multipliers = model.evaluate(initial_state_data, time, generalized_alpha_spectral_radius=generalized_alpha_spectral_radius,
                                                                    actuator_loads=actuator_loads, external_fluid_load=external_fluid_load)

flexible_states_output = states['fishy']['flexible_states']
flexible_state_derivatives_output = state_derivatives['fishy']['flexible_states']
states_constraint = flexible_states_output[0] - flexible_states_output[flexible_states_output.shape[0]-1]
state_derivatives_constraint = flexible_state_derivatives_output[0] - flexible_state_derivatives_output[flexible_state_derivatives_output.shape[0]-1]

periodicity_enforcer = csdl.nonlinear_solvers.GaussSeidel(tolerance=1.e-6)
periodicity_enforcer.add_state(initial_state_data.states['fishy']['flexible_states'], states_constraint)
                               #initial_value=initial_state_data.states['fishy']['flexible_states'])
periodicity_enforcer.add_state(initial_state_data.state_derivatives['fishy']['flexible_states'], state_derivatives_constraint)
                                #initial_value=initial_state_data.state_derivatives['fishy']['flexible_states'])
periodicity_enforcer.run()
print('initial_state_data.states[\'fishy\'][\'flexible_states\'].value: ',
       initial_state_data.states['fishy']['flexible_states'].value[0])
print('initial_state_data.state_derivatives[\'fishy\'][\'flexible_states\'].value: ', 
      initial_state_data.state_derivatives['fishy']['flexible_states'].value[0])
# endregion MBD Section

# region Geometry Reconstruction
panel_mesh = csdl.Variable(
    shape=(1, num_time_steps, num_chordwise, num_spanwise, 3),
    value=0.)
for i in csdl.frange(t.size):
    for body in model.system.bodies:
        rigid_body_states = states[body.name]['rigid_body_states'][i,:]
        flexible_states = states[body.name]['flexible_states'][i,:].reshape((points_to_project.shape[0],6))

        # b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(9,))
        # geometry_b_spline = lfs.Function(space=b_spline_function_space, coefficients=points_to_project[:,0])
        # displacement_b_spline = lfs.Function(space=b_spline_function_space, coefficients=flexible_states[:,2])
        # plotting_elements = b_spline.plot(show=False)
        # b_spline.plot(point_types=['coefficients'], plot_types=['point_cloud'], color='orange', additional_plotting_elements=plotting_elements)
        # import matplotlib.pyplot as plt
        # plt.plot(beam_parametric_mesh[:,0], flexible_states[:,2])
        # plt.plot(points_to_project[:,0], flexible_states[:,2])
        # plt.show()

        body.geometry = body.design_geometry.copy()
        body.apply_rigid_body_motion(rigid_body_states)
        body.apply_flexible_motion(flexible_states, rigid_body_states)

        panel_mesh_this_time_step = body.geometry.evaluate(panel_method_parametric_mesh, plot=False)
        panel_mesh = panel_mesh.set(csdl.slice[0, i], panel_mesh_this_time_step)
# endregion Geometry Reconstruction


# region Fluid Simulation
# panel_mesh = panel_mesh[:,:,:,::-1]
# panel_mesh = panel_mesh[:,:,::-1,:]
panel_mesh_velocities = (panel_mesh[:,1:] - panel_mesh[:,:-1])/time_step
panel_mesh = (panel_mesh[:,:-1] + panel_mesh[:,1:])/2
coll_vel = (panel_mesh_velocities[:,:,:-1,:-1,:] + panel_mesh_velocities[:,:,1:,:-1,:] + 
            panel_mesh_velocities[:,:,1:,1:,:] + panel_mesh_velocities[:,:,:-1,1:,:])/4.
panel_mesh_free_stream = csdl.Variable(shape=panel_mesh_velocities.shape, value=0.)
# panel_mesh_free_stream = panel_mesh_free_stream.set(csdl.slice[:,:,:,:,0], 
#                     csdl.expand(state_derivatives['fishy']['rigid_body_states'][:-1,0], panel_mesh_free_stream.shape[:-1], 'j->ijkl'))
panel_mesh_free_stream = panel_mesh_free_stream.set(csdl.slice[:,:,:,:,0], swim_speed)

panel_mesh_list = [panel_mesh]
panel_mesh_velocities_list = [panel_mesh_free_stream]
coll_vel_list = [coll_vel]
output_dict, panel_mesh_dict, wake_panel_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    panel_mesh_list, 
    panel_mesh_velocities_list, 
    coll_vel_list,
    dt=time_step, # leads to more consistent scaling on the order of 1e-2 I think
    free_wake=False,    # This makes practically no difference
)

num_boundary_layer_per_panel = 2
boundary_layer_mesh = csdl.Variable(value=np.zeros((1, num_time_steps-1, int(num_boundary_layer_per_panel*panel_mesh.shape[2]-1), panel_mesh.shape[3], 3)))
boundary_layer_mesh = boundary_layer_mesh.set(csdl.slice[:,:,::2,:,:], value=panel_mesh)
boundary_layer_mesh = boundary_layer_mesh.set(csdl.slice[:,:,1::2,:,:], value=(panel_mesh[:,:,1:,:,:] + panel_mesh[:,:,:-1,:,:])/2)

panel_forces, _ = fish_post_processor(panel_mesh_dict, output_dict, boundary_layer_mesh, mu)
# panel_forces = output_dict['surface_0']['panel_forces']
panel_forces_output = csdl.sum(panel_forces[0,panel_forces.shape[1]//num_actuation_cycles:-1,:,:,:], axes=(1,2))   # leads to more consistent scaling on the order of 1e-1 I think
# panel_forces_output = csdl.sum(panel_forces[0,:-1], axes=(1,2))   # leads to more consistent scaling on the order of 1e-1 I think
# net_thrust_drag = csdl.average(panel_forces_output)     # leads to more consistent scaling on the order of 1e-1 I think
net_force = csdl.average(panel_forces_output[:,0])
print(panel_forces_output.value)
# endregion Fluid Simulation

# region Objective and Constraints
# flexible_states_output = states['fishy']['flexible_states']
# flexible_state_derivatives_output = state_derivatives['fishy']['flexible_states']

# states_constraint = flexible_states_output[0] - flexible_states_output[flexible_states_output.shape[0]-1]
# state_derivatives_constraint = flexible_state_derivatives_output[0] - flexible_state_derivatives_output[flexible_state_derivatives_output.shape[0]-1]

# objective = csdl.vdot(states_constraint, states_constraint) + csdl.vdot(state_derivatives_constraint, state_derivatives_constraint)

# objective = -swim_speed


num_beam_nodes = solver_geometry_representations['beam'].shape[0]
structural_state_derivatives = state_derivatives['fishy']['flexible_states'][:,:num_beam_nodes*6].reshape((num_time_steps, num_beam_nodes, 6))
structural_states = states['fishy']['flexible_states'][:,:num_beam_nodes*6].reshape((num_time_steps, num_beam_nodes, 6))
theta_y = (structural_states[:-1,:,4] + structural_states[1:,:,4])/2
theta_dot_y = (structural_state_derivatives[:-1,:,4] + structural_state_derivatives[1:,:,4])/2
applied_moment_y = actuator_loads[:,:,4]
applied_moment_dot_y = actuator_load_derivatives[:,:,4]
total_work = csdl.sum((applied_moment_y*theta_dot_y) + (applied_moment_dot_y*theta_y)*time_step)
average_power = total_work/time[-1]
cost_of_transport = average_power/swim_speed
objective = cost_of_transport# + 1.e-2*net_force**2

# initial_state_data.states['fishy']['flexible_states'].set_as_design_variable(scaler=1.e2)
# initial_state_data.state_derivatives['fishy']['flexible_states'].set_as_design_variable(scaler=1.e2)

max_pressure.set_as_design_variable(scaler=1.e-5, upper=1.e6, lower=5.e4)
actuation_frequency.set_as_design_variable(scaler=1., lower=0.4, upper=2.)
# swim_speed.set_as_design_variable(scaler=1.e1, lower=0.02, upper=0.5)

# width_shape_variables.set_as_design_variable(lower=-(computed_fishy_width.value*0.4)/2, upper=(computed_fishy_width.value)/2, scaler=1.e2)
# height.set_as_design_variable(lower=height.value*0.6, upper=height.value*2., scaler=1.e2)

objective.set_as_objective(scaler=1.e-2)

# net_force.set_as_constraint(scaler=1.e6, equals=0.)
net_force.set_as_constraint(scaler=1.e2, equals=0.)
# endregion Objective and Constraints


# Optimized design with no height variable and speed = 0.13 m/s
# max_pressure.set_value(2.602878263871166187e+5)
# actuation_frequency.set_value(8.676421592495715540e-01)
# width_shape_variables.set_value(np.array([8.019665177287379487e-3, 8.019665177287379487e-3, -4.763237667354227600e-3, -3.704929654351165058e-3]))

# A converged solution with speed = 0.13 m/s
# max_pressure.set_value(8.221922247972699438e+5)
# actuation_frequency.set_value(7.904851738897465951e-01)
# width_shape_variables.set_value(np.array([-4.600999670188704038e-3, -6.347761159866847080e-3, -7.834255362725039973e-3, -7.852778177542345528e-3]))
# height.set_value(4.002249106560485714e-2)

# Manual guess
# max_pressure.set_value(5.602878263871166187e+5)
# actuation_frequency.set_value(8.676421592495715540e-01)
# width_shape_variables.set_value(np.array([8.019665177287379487e-3, 8.019665177287379487e-3, -4.763237667354227600e-3, -3.704929654351165058e-3]))

# A converged solution with speed = 0.3 m/s
# max_pressure.set_value(8.740307698945173343e+5)
# actuation_frequency.set_value(1.443402491567813417e+00)
# width_shape_variables.set_value(np.array([-8.014792589434250880e-3, -8.015297737154035707e-3, -8.013851904173168172e-3, 2.001162727618399639e-2]))
# height.set_value(3.991623273868089239e-2)

# A converged solution with speed = 0.3 m/s and actually correct geometry mappings
max_pressure.set_value(2.980301246807620874e+05)
actuation_frequency.set_value(7.671249518924843080e-01)
width_shape_variables.set_value(np.array([1.723869007938495512e-2, -8.019335325554517446e-03, -8.019541514525408976e-03, -6.837124935639948831e-03]))
height.set_value(3.991630648526205416e-02)


t2 = timer()
print(f'Python time: {t2-t1}')
jax_inputs = []
# jax_inputs.append(pendulum_system.pendulums[0].mass)
jax_inputs.append(length)
jax_inputs.append(swim_speed)
# print('-----------------')
# recorder.count_origins(mode='line')
# print('-----------------')
# recorder.count_origins()
# print('-----------------')
# exit()
additional_outputs = []
for body in model.system.bodies:
    additional_outputs.append(states[body.name]['rigid_body_states'])
    additional_outputs.append(state_derivatives[body.name]['rigid_body_states'])
    additional_outputs.append(states[body.name]['flexible_states'])
    additional_outputs.append(state_derivatives[body.name]['flexible_states'])
additional_outputs = additional_outputs + lagrange_multipliers['physical_constraints']
additional_outputs = additional_outputs + lagrange_multipliers['structural_constraints']
additional_outputs = additional_outputs + [panel_forces_output, net_force]
additional_outputs = additional_outputs + [panel_mesh, output_dict['surface_0']['Cp'],
                                           output_dict['surface_0']['panel_forces'],
                                           wake_panel_mesh_dict['surface_0']['mesh'],
                                           mu, sigma, mu_wake,
                                           fishy_geometry.coefficients]


jax_sim = csdl.experimental.JaxSimulator(
    recorder = recorder,
    # additional_inputs = list(initial_state_data.states.values()),
    additional_inputs=jax_inputs,
    # additional_outputs = list(states.values()['rigid_body_states']) + list(state_derivatives.values()['rigid_body_states']) \
    #                     + list(states.values()['flexible_states']) + list(state_derivatives.values()['flexible_states']) \
    #                     + lagrange_multipliers,
    additional_outputs=additional_outputs,
    gpu=False
)
# recorder.print_largest_variables(n = 50)
# exit()

if run_stuff and run_jax and not run_optimization:

    t3 = timer()
    jax_sim.run()
    t4 = timer()
    print(f'JAX 1st time: {t4-t3}')
    print('net_force: ', net_force.value)

if run_optimization:
    
    # states_constraint.set_as_constraint(scaler=1.e2)
    # state_derivatives_constraint.set_as_constraint(scaler=1.e2)

    t1 = timer()
    optimization_problem = modopt.CSDLAlphaProblem(problem_name='dynamic_fishy_optimization', simulator=jax_sim)
    t2 = timer()
    print('compile time?: ', t2 - t1)
    optimizer = modopt.PySLSQP(optimization_problem, solver_options={'maxiter':100, 'acc':1.e-3}, readable_outputs=['x'])

    t3 = timer()
    print('optimizer setup time: ', t3 - t2)
    optimizer.solve()
    t4 = timer()
    print('solve time: ', t4 - t3)
    optimizer.print_results()
    exit()

recorder.inline = True
# # swim_speeds = np.linspace(0.02, 0.3, 5)
# swim_speeds = np.linspace(0.02, 0.5, 5)
# net_panel_forces = np.zeros_like(swim_speeds)
# for i, set_swim_speed in enumerate(swim_speeds):
#     swim_speed.set_value(set_swim_speed)

#     jax_sim[swim_speed] = set_swim_speed
#     jax_sim.run()

#     net_panel_force = np.average(jax_sim[panel_forces_output][:,0])
#     print(f'Net Panel Force: {net_panel_force}')
#     net_panel_forces[i] = net_panel_force

# import matplotlib.pyplot as plt
# plt.plot(swim_speeds, net_panel_forces)
# plt.show()
# # # exit()
    
plt.plot(t.value[-panel_forces_output.shape[0]-1:-1], panel_forces_output.value)
# plt.plot(t.value[:-2], panel_forces_output.value)
plt.show()

from lsdo_serpent.utils.plot import plot_wireframe, plot_pressure_distribution, plot_transient_pressure_distribution
if True:
    wake_mesh = wake_panel_mesh_dict['surface_0']['mesh']
    plot_wireframe(panel_mesh.value, wake_mesh.value, mu.value, mu_wake.value, num_time_steps, side_view=False, interactive=False, backend='cv', name='fish_demo')

Cp = output_dict['surface_0']['Cp']
plot_transient_pressure_distribution(panel_mesh.value, Cp.value, side_view=False, backend='cv', interactive=False)
# x_force = output_dict['surface_0']['panel_forces'][:,:,:,:,0]
# plot_transient_pressure_distribution(panel_mesh.value, x_force.value, side_view=False, backend='cv', interactive=False)


if save_stuff and run_stuff:
    for body in model.system.bodies:
        states[body.name]['rigid_body_states'] = states[body.name]['rigid_body_states'].value
        state_derivatives[body.name]['rigid_body_states'] = state_derivatives[body.name]['rigid_body_states'].value
        states[body.name]['flexible_states'] = states[body.name]['flexible_states'].value
        state_derivatives[body.name]['flexible_states'] = state_derivatives[body.name]['flexible_states'].value

    for key, lagrange_multiplier_list in lagrange_multipliers.items():
        for i, lagrange_multiplier in enumerate(lagrange_multiplier_list):
            lagrange_multipliers[key][i] = lagrange_multiplier.value


    file_path = 'examples/advanced_examples/dynamic_beam_optimization/'
    file_name = file_path + "t.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(t.value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "states.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "state_derivatives.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(state_derivatives, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = file_path + "lagrange_multipliers.pickle"
    with open(file_name, 'wb+') as handle:
        pickle.dump(lagrange_multipliers, handle, protocol=pickle.HIGHEST_PROTOCOL)

if run_stuff and run_jax and run_stuff_twice:
    t5 = timer()
    jax_sim.run()
    t6 = timer()
    print(f'JAX Nth time: {t6-t5}')

if run_stuff:
    print(f'Python time: {t2-t1}')
    if run_jax:
        print(f'JAX 1st time: {t4-t3}')
        if run_stuff_twice:
            print(f'JAX Nth time: {t6-t5}')

load_stuff = not run_stuff
if load_stuff:
    file_path = 'examples/advanced_examples/dynamic_beam_optimization/'
    file_name = file_path + "t.pickle"
    with open(file_name, 'rb') as handle:
        t = pickle.load(handle)
    file_name = file_path + "states.pickle"
    with open(file_name, 'rb') as handle:
        states = pickle.load(handle)
    file_name = file_path + "state_derivatives.pickle"
    with open(file_name, 'rb') as handle:
        state_derivatives = pickle.load(handle)
    file_name = file_path + "lagrange_multipliers.pickle"
    with open(file_name, 'rb') as handle:
        lagrange_multipliers = pickle.load(handle)

if save_stuff or load_stuff:
    # Convert the states and state derivatives to csdl.Variables
    for body in model.system.bodies:
        states[body.name]['rigid_body_states'] = csdl.Variable(value=states[body.name]['rigid_body_states'])
        state_derivatives[body.name]['rigid_body_states'] = csdl.Variable(value=state_derivatives[body.name]['rigid_body_states'])
        states[body.name]['flexible_states'] = csdl.Variable(value=states[body.name]['flexible_states'])
        state_derivatives[body.name]['flexible_states'] = csdl.Variable(value=state_derivatives[body.name]['flexible_states'])

    # for key, lagrange_multiplier_list in lagrange_multipliers.items():
        # for i, lagrange_multiplier in enumerate(lagrange_multiplier_list):
        #     lagrange_multipliers[key][i] = csdl.Variable(value=lagrange_multiplier)
    # lagrange_multipliers['physical_constraints'] = [csdl.Variable(value=lagrange_multiplier) for lagrange_multiplier in lagrange_multipliers['physical_constraints']]

# r0 = f(x0)
# r1 = f(x1)

# X0[0] = x0
# X01[1] = x1
# R01 = F(X01)
# r1 = F01[1]
# r0 = F01[0]



state_velocities = {}
state_accelerations = {}
for body in model.system.bodies:
    state_velocities[body.name] = state_derivatives[body.name]['rigid_body_states'].value[:,:6]
    state_accelerations[body.name] = state_derivatives[body.name]['rigid_body_states'].value[:,6:]

if plot:

    # calculate energy through the simulation
    energy = np.zeros(t.size-1)
    for body in model.system.bodies:
        kinetic_energy = 0.5*body.mass.value*np.linalg.norm(state_velocities[body.name][:,:3], axis=1)**2
        # potential_energy = body.mass.value*model.g*states[body.name]['rigid_body_states'].value[:,2]
        energy += kinetic_energy[:-1]# + potential_energy[:-1]

    # plot the results
    # plt.figure()
    # for i, body in enumerate(model.system.bodies):
    #     plt.plot(t,states[body.name]['rigid_body_states'].value[:-1,3],linewidth=2, label=f'Angular Displacement {i+1}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('$\\theta_{1}$,$\\theta_{2}$')
    # plt.legend()
    # plt.title('Angular Displacements')
    # plt.grid(True)

    # plt.figure()
    # for i, body in enumerate(model.system.bodies):
    #     plt.plot(t,state_velocities[body.name][:-1,3],linewidth=2, label=f'Angular Velocity {i+1}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('$\dot{\\theta_{1}}$,$\dot{\\theta_{2}}$')
    # plt.legend()
    # plt.title('Angular Velocities')
    # plt.grid(True)

    # plt.figure()
    # for i, body in enumerate(model.system.bodies):
    #     plt.plot(t,state_accelerations[body.name][:-1,3],linewidth=2, label=f'Angular Acceleration {i+1}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('$\ddot{\\theta_{1}}$,$\ddot{\\theta_{2}}$')
    # plt.legend()
    # plt.title('Angular Accelerations')
    # plt.grid(True)

    # plt.figure()
    # for i, body in enumerate(model.system.bodies):
    #     plt.plot(states[body.name]['rigid_body_states'].value[:-1,1],states[body.name]['rigid_body_states'].value[:-1,2],linewidth=2, label=f'Mass {i+1}')
    # plt.xlabel('$x_{1},x_{2}$')
    # plt.ylabel('$y_{1},y_{2}$')
    # plt.legend()
    # plt.grid(True)
    # plt.title('Mass Trajectories')

    plt.figure()
    for i, body in enumerate(model.system.bodies):
        plt.plot(t[:-1],states[body.name]['flexible_states'].value[:-1,2],linewidth=2, label='Z-Displacements')
    plt.xlabel('Time (s)')
    plt.ylabel('$Z-Displacement$ (m)')
    plt.legend()
    plt.title('Z-Displacement vs. Time')
    plt.grid(True)

    # for i, body in enumerate(model.pendulum_system.pendulums):  # whatever, number of bodies = number of physical connections
    #     plt.figure()
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,0],linewidth=2, label=f'Lagrange Multiplier {2*i+1}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,1],linewidth=2, label=f'Lagrange Multiplier {2*i+2}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,2],linewidth=2, label=f'Lagrange Multiplier {2*i+3}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,3],linewidth=2, label=f'Lagrange Multiplier {2*i+4}')
    #     plt.plot(t,lagrange_multipliers['physical_constraints'][i].value[:-1,4],linewidth=2, label=f'Lagrange Multiplier {2*i+5}')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('$\lambda_{1}$,$\lambda_{2}$')
    #     plt.legend()
    #     plt.title('Lagrange Multipliers For Body ' + str(i+1))
    #     plt.grid(True)

    plt.figure()
    plt.plot(t[:-1],energy,'k-',linewidth=2, label='Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Energy')
    plt.grid(True)

    plt.show()

if make_video:

    import vedo
    folder_path = 'examples/advanced_examples/dynamic_beam_optimization/videos/'
    file_name = 'fishy.mp4'
    video = vedo.Video(folder_path + file_name, duration=time.value[-1], backend="cv")

    # ax = vedo.Axes(xrange=(-bounding_size,bounding_size), yrange=(-bounding_size,lengths['pendulum1']*0.8), htitle=__doc__)
    # ax = vedo.Axes(yrange=(-bounding_size,bounding_size), zrange=(-bounding_size-bounding_size/2,bounding_size-bounding_size/2), htitle=__doc__)

    camera = {
        'pos': (3, 2, 2),
        'focalPoint': (0, 0, 0),
        # 'focalPoint': (0, 0, 0),
        'viewup': (0, 1, 0),
        'distance': 3,
    }

    # for body in model.system.bodies:
    #     for function in body.geometry.functions.values():
    #         function.coefficients = function.coefficients.value
    model.system.bodies[0].geometry.coefficients = model.system.bodies[0].geometry.coefficients.value

    # for i in range(len(t)):
    skipping_rate = 1
    # skipping_rate = len(t)//250
    print(f'skipping_rate: {skipping_rate}')
    # angular_momentum_values = []
    for i in np.arange(0, t.size, skipping_rate):
        if i % 10 == 0:
            print(f'{i}/{t.size}')
            # print('------------------')

        plotting_elements = []
        for body in model.system.bodies:
            rigid_body_states = states[body.name]['rigid_body_states'].value[i,:]

            scaling_factor = 1
            if scale_flexible_states:
                scaling_factor = 100000
                # scaling_factor = 10
            flexible_states = states[body.name]['flexible_states'].value[i,:].reshape((-1,6))*scaling_factor

            b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(points_to_project.shape[0],))
            geometry_b_spline = lfs.Function(space=b_spline_function_space, coefficients=points_to_project[:,0])
            displacement_b_spline = lfs.Function(space=b_spline_function_space, coefficients=flexible_states[:,2])
            # plotting_elements = b_spline.plot(show=False)
            # b_spline.plot(point_types=['coefficients'], plot_types=['point_cloud'], color='orange', additional_plotting_elements=plotting_elements)
            # import matplotlib.pyplot as plt
            # plt.plot(beam_parametric_mesh[:,0], flexible_states[:,2])
            # plt.plot(points_to_project[:,0], flexible_states[:,2])
            # plt.show()


            body.geometry = body.design_geometry.copy()
            body.apply_rigid_body_motion(rigid_body_states)
            beam_mesh = body.representations['beam'].evaluate(body.geometry).value
            body.apply_flexible_motion(flexible_states,
                                       rigid_body_states)
            # print('rigid body states:', rigid_body_states)
            # print('flexible states: \n', flexible_states)
            # print('new lagrange_multipliers: ', lagrange_multipliers['structural_constraints'][1][i,:])

            plotting_elements = body.geometry.plot(show=False, opacity=0.5)

            # flexible_displacements = flexible_states[:,:3]
            # rigid_body_rotations = rigid_body_states[3:]
            # flexible_displacements_rotated = lsdo_geo.rotate(flexible_displacements, axis_origin=np.zeros(3), axis_vector=np.array([0,0,1]), angles=rigid_body_rotations[2])
            # flexible_displacements_rotated = lsdo_geo.rotate(flexible_displacements_rotated, axis_origin=np.zeros(3), axis_vector=np.array([0,1,0]), angles=rigid_body_rotations[1])
            # flexible_displacements_rotated = lsdo_geo.rotate(flexible_displacements_rotated, axis_origin=np.zeros(3), axis_vector=np.array([1,0,0]), angles=rigid_body_rotations[0])
            # deformed_beam_mesh = beam_mesh + flexible_displacements_rotated
            deformed_beam_mesh = beam_mesh + flexible_states[:,:3]
            # plotting_elements = body.geometry.plot_meshes(meshes=[beam_mesh], mesh_opacity=0.5, function_opacity=0.01, additional_plotting_elements=plotting_elements, show=False)
            # plotting_elements = body.geometry.plot_meshes(meshes=[deformed_beam_mesh], mesh_opacity=0.8, mesh_color='gold', function_opacity=0.5, additional_plotting_elements=plotting_elements, show=False)
            beam_plot = vedo.Line(deformed_beam_mesh).linewidth(5).color('gold')
            plotting_elements.append(beam_plot)
            beam_plot_points = vedo.Points(deformed_beam_mesh).c('red').point_size(10)
            plotting_elements.append(beam_plot_points)

        # plotting_elements = []
        # for j in range(num_bodies):
        #     # pendulum = vedo.Cylinder(pos=(pendulum_tops[j,0], pendulum_tops[j,1], 0), r=0.05, height=lengths[f'pendulum{j+1}'], c=(0,0,1))
        #     pendulum_line = vedo.Line(pendulum_tops[j], pendulum_bottoms[j]).linewidth(5).color(pendulum_colors[j])
        #     pendulum_top_and_bot = vedo.Points([pendulum_tops[j], pendulum_bottoms[j]]).c(dot_colors[j]).point_size(10)
        #     # pendulum.rotate_z(states[bodies[j]][i,2], rad=True)
        #     plotting_elements.append(pendulum_line)
        #     plotting_elements.append(pendulum_top_and_bot)

        # plotter = vedo.Plotter(size=(3200,2000),offscreen=True)
        plotter = vedo.Plotter(offscreen=True)
        # plotter = vedo.Plotter(offscreen=True)
        # plotter.show(plotting_elements, camera=camera, axes=1)
        # plotter.show(plotting_elements, axes=0)
        plotter.show(plotting_elements, axes=0, viewup='y')
        # plotter.show(plotting_elements, ax, viewup='z')

        video.add_frame()
    video.close()
