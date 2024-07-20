import numpy as np
import pickle
from lsdo_serpent import MESH_PATH

file_name = str(MESH_PATH) + '/structured_fish_mesh.pickle'

file = open(file_name, 'rb')
data = pickle.load(file)
file.close()