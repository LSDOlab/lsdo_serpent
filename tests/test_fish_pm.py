import numpy as np 
import csdl_alpha as csdl
import pickle
from lsdo_serpent import MESH_PATH
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from lsdo_serpent.utils.plot import plot_wireframe, plot_pressure_distribution
import matplotlib.pyplot as plt

scale = 1.

# ==== mesh import ====
# file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh.pickle'
file_name = str(MESH_PATH) + '/structured_fish_mesh.pickle'
# file_name = str(MESH_PATH) + '/panel_mesh.pickle'

file = open(file_name, 'rb')
mesh = pickle.load(file) # (nt, nc, ns, 3)
file.close()

# ==== mesh velocity import ====
# file_name = str(MESH_PATH) + '/coarse_structured_fish_mesh_velocities.pickle'
file_name = str(MESH_PATH) + '/structured_fish_mesh_velocities.pickle'
# file_name = str(MESH_PATH) + '/panel_mesh_velocities.pickle'

file = open(file_name, 'rb')
mesh_velocity = pickle.load(file) # (nt, nc, ns, 3)
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

print('coll_vel', coll_vel[0,0])

# here we set up the free-stream velocity grid for each MESH NODE 
mesh_free_stream = np.zeros_like(mesh_velocity)
# mesh_free_stream[:,:,:,:,0] = 0.01
mesh_free_stream[:,:,:,:,0] = 0.25
# mesh_free_stream[:,:,:,:,0] = 1.
# mesh_free_stream[:,:,:,:,0] = 2.
# mesh_free_stream[:,:,:,:,0] = 5.
# mesh_free_stream[:,:,:,:,0] = 10.
# mesh_free_stream[:,:,:,:,0] = 25.


recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_free_stream)
coll_vel = csdl.Variable(value=-coll_vel)

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

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocity], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, wake_mesh, Cp, panel_forces], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
wake_mesh = jax_sim[wake_mesh]
Cp = jax_sim[Cp]
# panel_forces = jax_sim[panel_forces]
# panel_forces *= 1e3 # scale density to water
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]

print('fishy done')
# x_forces_across_time = np.sum(panel_forces[0,:-1,:,:,0], axis=(1,2))
# forces_across_time = np.sum(panel_forces[0,:-1,:,:,:], axis=(1,2))
# total_force_x = np.sum(x_forces_across_time)
# print('panel_forces', x_forces_across_time)
# print('panel_forces', total_force_x)
# print('panel_forces', forces_across_time/num_time_steps)
# print('panel_forces', np.sum(forces_across_time, axis=0)/num_time_steps)
# plt.plot(x_forces_across_time)
# plt.show()

# free_stream_velocities = np.linspace(0.1, 1.5, 15)
free_stream_velocities = np.linspace(0.1, 5, 15)
# free_stream_velocities = [10]
output_forces = np.zeros_like(free_stream_velocities)
for i, cruise_speed in enumerate(free_stream_velocities):
    jax_sim[mesh_velocity][:,:,:,:,0] = cruise_speed
    jax_sim.run()
    panel_forces_output = jax_sim[panel_forces]*1000# / (1/2*1000*cruise_speed**2*0.025)
    panel_forces_output = np.sum(panel_forces_output[0,:-1,:,:,0])
    output_forces[i] = panel_forces_output
    print(output_forces[i])
    print('cruise_speed', cruise_speed)

plt.plot(free_stream_velocities, output_forces)
plt.show()


if True:
    plot_pressure_distribution(mesh, Cp, interactive=True)

if True:
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, side_view=True, interactive=False, backend='cv', name='fish_demo')