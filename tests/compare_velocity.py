import numpy as np
import pickle
from lsdo_serpent import MESH_PATH

mesh_file = str(MESH_PATH) + '/structured_fish_mesh.pickle'
mesh_vel_file = str(MESH_PATH) + '/panel_mesh_velocities.pickle'

datafile = open(mesh_file, 'rb')
mesh = pickle.load(datafile)
datafile.close()

datafile = open(mesh_vel_file, 'rb')
mesh_vel = pickle.load(datafile)
datafile.close()

dt = 1/33
mesh_vel_FD = (mesh[1:,:,:,:] - mesh[:-1,:,:,:]) / dt
mesh_vel_avg = (mesh[1:,:,:,:] + mesh[:-1,:,:,:]) / 2

error = (mesh_vel_FD - mesh_vel_avg)/(mesh_vel_avg+1.e-12)