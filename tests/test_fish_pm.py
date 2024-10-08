import numpy as np 
import csdl_alpha as csdl
import pickle
from lsdo_serpent import MESH_PATH
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from lsdo_serpent.utils.plot import plot_wireframe, plot_pressure_distribution, plot_transient_pressure_distribution

V_inf = 1.
# ==== mesh import ====
# file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh.pickle'
file_name = str(MESH_PATH) + '/structured_fish_mesh.pickle'

file = open(file_name, 'rb')
mesh = pickle.load(file) # (nt, nc, ns, 3)
file.close()

# ==== mesh velocity import ====
# file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh_velocities.pickle'
file_name = str(MESH_PATH) + '/structured_fish_mesh_velocities.pickle'

file = open(file_name, 'rb')
mesh_velocity = pickle.load(file) # (nt, nc, ns, 3)
file.close()

num_nodes = 1
dt = 0.01
nt = mesh.shape[0]
mesh = mesh.reshape((num_nodes,) + mesh.shape) # (nn, nt, nc, ns, 3)
mesh_velocity = mesh_velocity.reshape((num_nodes,) + mesh_velocity.shape)

# We use the computed fish velocities for the collocation point velocities
# NOTE: we want collocation velocities at the panel centers; we get this by averaging the velocities as such
coll_vel = (mesh_velocity[:,:,:-1,:-1,:] + mesh_velocity[:,:,1:,:-1,:] + mesh_velocity[:,:,1:,1:,:] + mesh_velocity[:,:,:-1,1:,:])/4.

# here we set up the free-stream velocity grid for each MESH NODE 
mesh_free_stream = np.zeros_like(mesh_velocity)
mesh_free_stream[:,:,:,:,0] = V_inf


recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_free_stream)
coll_vel = csdl.Variable(value=coll_vel)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]
coll_vel_list = [coll_vel]
# exit()
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    mesh_list, 
    mesh_velocity_list, 
    coll_vel_list,
    dt=dt, 
    free_wake=False
)
# OUTPUTS NEEDED TO COMPUTE FISH STATE
panel_forces = output_dict['surface_0']['panel_forces']

# OUTPUTS NEEDED FOR VISUALIZATION
Cp = output_dict['surface_0']['Cp']
wake_mesh = wake_mesh_dict['surface_0']['mesh']

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocity], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, wake_mesh, Cp, panel_forces], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
wake_mesh = jax_sim[wake_mesh]
Cp = jax_sim[Cp]
panel_forces = jax_sim[panel_forces]
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]

print('fishy done')

F_net_time = np.sum(panel_forces.reshape(panel_forces.shape[1:]), axis=(1,2))
F_avg = np.average(F_net_time, axis=0)

if False:
    plot_pressure_distribution(mesh, Cp, interactive=True)

if True:
    plot_transient_pressure_distribution(mesh, Cp, side_view=True, backend='cv', interactive=False)

if False:
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, side_view=True, interactive=False, backend='cv', name='fish_demo')