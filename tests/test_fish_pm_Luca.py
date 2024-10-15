import numpy as np 
import csdl_alpha as csdl
import pickle
from lsdo_serpent import MESH_PATH
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from lsdo_serpent.core.fish_post_processor import fish_post_processor

from lsdo_serpent.utils.plot import plot_wireframe, plot_pressure_distribution, plot_transient_pressure_distribution
import matplotlib.pyplot as plt
import time

V_inf = 1.
# ==== mesh import ====
file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh.pickle'
# file_name = str(MESH_PATH) + '/structured_fish_mesh.pickle'
# file_name = str(MESH_PATH) + '/panel_mesh.pickle'

file = open(file_name, 'rb')
mesh = pickle.load(file)[:5,:] # (nt, nc, ns, 3)
file.close()

# ==== mesh velocity import ====
file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh_velocities.pickle'
# file_name = str(MESH_PATH) + '/structured_fish_mesh_velocities.pickle'
# file_name = str(MESH_PATH) + '/panel_mesh_velocities.pickle'

file = open(file_name, 'rb')
mesh_velocity = pickle.load(file)[:5,:] # (nt, nc, ns, 3)
file.close()

num_nodes = 1
actuation_frequency = 1
num_steps_per_cycle = 33
num_cycles = 3
final_time = (1/actuation_frequency)*num_cycles
num_time_steps = num_cycles*num_steps_per_cycle
dt = final_time/num_time_steps
print(dt)
# dt = 0.01
nt = mesh.shape[0]
mesh = mesh.reshape((num_nodes,) + mesh.shape) # (nn, nt, nc, ns, 3)
# mesh_velocity = (mesh[:,1:] - mesh[:,:-1])/dt
# mesh = (mesh[:,1:] + mesh[:,:-1])/2 # kind of like a midpoint rule I guess
mesh_velocity = mesh_velocity.reshape((num_nodes,) + mesh_velocity.shape)

# We use the computed fish velocities for the collocation point velocities
# NOTE: we want collocation velocities at the panel centers; we get this by averaging the velocities as such
coll_vel = (mesh_velocity[:,:,:-1,:-1,:] + mesh_velocity[:,:,1:,:-1,:] + mesh_velocity[:,:,1:,1:,:] + mesh_velocity[:,:,:-1,1:,:])/4.

# here we set up the free-stream velocity grid for each MESH NODE 
mesh_free_stream = np.zeros_like(mesh_velocity)
mesh_free_stream[:,:,:,:,0] = V_inf


recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

dummy_val = csdl.Variable(value=1.)
mesh = csdl.Variable(value=mesh) * dummy_val
mesh_velocity = csdl.Variable(value=mesh_free_stream) * dummy_val
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
panel_forces_PM = output_dict['surface_0']['panel_forces']

end_force_PM = csdl.sum(panel_forces_PM, axes=(2,3))[0,-2,0]

# OUTPUTS NEEDED FOR VISUALIZATION
Cp = output_dict['surface_0']['Cp']
wake_mesh = wake_mesh_dict['surface_0']['mesh']

num_BL_per_panel = 2
BL_mesh = csdl.Variable(value=np.zeros((num_nodes, nt, int(num_BL_per_panel*mesh.shape[2]-1), mesh.shape[3], 3)))
BL_mesh = BL_mesh.set(csdl.slice[:,:,::2,:,:], value=mesh)
BL_mesh = BL_mesh.set(csdl.slice[:,:,1::2,:,:], value=(mesh[:,:,1:,:,:] + mesh[:,:,:-1,:,:])/2)

panel_forces = fish_post_processor(mesh_dict, output_dict, BL_mesh, mu)

end_force = csdl.sum(panel_forces, axes=(2,3))[0,-2,0]

outputs = [end_force_PM, end_force]

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[dummy_val], # list of outputs (put in csdl variable)
    additional_outputs=outputs, # list of outputs (put in csdl variable)
)
# jax_sim.run()
start = time.time()
asdf = jax_sim.check_totals(step_size=1.e-3)
end = time.time()
print('check totals time:', end-start)
# asdf = jax_sim.compute_totals()


print('fishy done')

if False:
    plot_pressure_distribution(mesh, Cp, interactive=True)

if False:
    plot_transient_pressure_distribution(mesh, Cp, side_view=True, backend='cv', interactive=False)

if False:
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, side_view=True, interactive=False, backend='cv', name='fish_demo')