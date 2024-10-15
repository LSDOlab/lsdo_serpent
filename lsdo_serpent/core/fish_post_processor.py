import numpy as np 
import csdl_alpha as csdl

from lsdo_serpent.core.laminar_separation_model import laminar_separation_model
from VortexAD.core.panel_method.perturbation_velocity_comp import least_squares_velocity

def fish_post_processor(mesh_dict, output_dict, BL_mesh, mu):
    surface_names = list(mesh_dict.keys())
    surface_name = surface_names[0]
    mesh = mesh_dict[surface_name]['mesh']
    num_nodes, nt = mesh.shape[0], mesh.shape[1]
    nc, ns = mesh.shape[2], mesh.shape[3]
    LE_ind = int((nc-1)/2)
    body_vel = output_dict[surface_name]['body_vel']

    Cf_list, lam_list = laminar_separation_model(mesh=mesh, Ue=body_vel, BL_mesh=BL_mesh)

    Cf_lower, Cf_upper = Cf_list[0], Cf_list[1]
    lam_lower, lam_upper = lam_list[0], lam_list[1]

    Cf = csdl.Variable(value=np.zeros((num_nodes, nt, nc-1, ns-1)))
    Cf = Cf.set(csdl.slice[:,:,:LE_ind,:], value=Cf_lower[:,:,::-1,:])
    Cf = Cf.set(csdl.slice[:,:,LE_ind:,:], value=Cf_upper)

    lam = csdl.Variable(value=np.zeros((num_nodes, nt, nc-1, ns-1)))
    lam = lam.set(csdl.slice[:,:,:LE_ind,:], value=lam_lower[:,:,::-1,:])
    lam = lam.set(csdl.slice[:,:,LE_ind:,:], value=lam_upper)

    # Cp_PM = output_dict[surface_name]['Cp']

    # V_sq = body_vel**2
    # del_H = csdl.Variable(value=np.zeros(body_vel.shape))
    # del_H = del_H.set(csdl.slice[:,:,LE_ind:-1,:], value=(V_sq[:,:,(LE_ind-1):-2,:] - V_sq[:,:,(LE_ind+1):,:])/2)
    # del_H = del_H.set(csdl.slice[:,:,:LE_ind,:], value=-V_sq[:,:,:LE_ind,:])

    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])


    nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity']
    coll_vel = mesh_dict[surface_name]['coll_point_velocity']
    total_vel = nodal_cp_velocity+coll_vel
    Q_inf_norm = csdl.norm(total_vel, axes=(4,))
    rho = 1000.

    Cp_PM = output_dict[surface_name]['Cp']
    panel_area = mesh_dict[surface_name]['panel_area']
    panel_normal = mesh_dict[surface_name]['panel_normal']
    panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
    dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp_PM
    dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijkl->ijkla') * panel_normal

    friction_drag_scalar = 0.5*rho*Q_inf_norm**2*Cf*panel_area
    friction_drag = csdl.expand(friction_drag_scalar, panel_normal.shape, 'ijkl->ijkla') * panel_x_dir

    panel_forces = csdl.Variable(value=np.zeros(panel_normal.shape))
    panel_forces = panel_forces.set(csdl.slice[:,:,:LE_ind,:,:], value=dF[:,:,:LE_ind,:,:]-friction_drag[:,:,:LE_ind,:,:])
    panel_forces = panel_forces.set(csdl.slice[:,:,LE_ind:,:,:], value=dF[:,:,LE_ind:,:,:]+friction_drag[:,:,LE_ind:,:,:])

    return panel_forces