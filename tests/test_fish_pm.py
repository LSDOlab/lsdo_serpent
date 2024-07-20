import numpy as np 
import csdl_alpha as csdl
import pickle
from lsdo_serpent import MESH_PATH
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from lsdo_serpent.utils.plot import plot_wireframe, plot_pressure_distribution

# mesh import
file_name = str(MESH_PATH) + '/structured_fish_mesh.pickle'

file = open(file_name, 'rb')
mesh = pickle.load(file) # (nt, nc, ns, 3)
file.close()

num_nodes = 1
dt = 0.01
nt = mesh.shape[0]
mesh = mesh.reshape((num_nodes,) + mesh.shape) # (nn, nt, nc, ns, 3)
mesh_velocity = np.zeros_like(mesh) # NOTE: TEMPORARY
mesh_velocity[:,:,:,:,0] = 1.

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    mesh_list, mesh_velocity_list, dt=dt, free_wake=False
)

coll_points = mesh_dict['surface_0']['panel_center']
Cp = output_dict['surface_0']['Cp']
wake_mesh = wake_mesh_dict['surface_0']['mesh']
panel_forces = output_dict['surface_0']['panel_forces']

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocity], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, wake_mesh, coll_points, Cp, panel_forces], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
wake_mesh = jax_sim[wake_mesh]
coll_points = jax_sim[coll_points]
Cp = jax_sim[Cp]
panel_forces = jax_sim[panel_forces]
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]

if False:
    plot_pressure_distribution(mesh, Cp)

if True:
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, interactive=False, backend='cv', name='fish_demo')