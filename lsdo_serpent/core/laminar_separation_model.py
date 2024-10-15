import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.csdl_switch import switch_func

def laminar_separation_model(mesh, Ue, BL_mesh):
    '''
    The lists are ordered as TE lower to LE, then LE to TE_upper
    '''
    nu = 1.002e-6 # m^2/s
    rho = 1000. # kg/m^3

    nn, nt, nc, ns = mesh.shape[:-1]
    nc_BL = BL_mesh.shape[2]
    num_panels_chordwise = nc-1

    LE_ind = int((num_panels_chordwise)/2)
    LE_ind_BL = int((nc_BL-1)/2)
    num_BL_vel_chord = int((nc+1)/2) # extrapolating to edges in chordwise direction (not the panels)
    nc_BL_one_way = int((nc_BL+1)/2)

    BL_per_panel = int((nc_BL_one_way-1)/(num_BL_vel_chord-1))
    # BL_per_panel = 1

    # setting up lower surface
    lower_mesh = mesh[:,:,:(LE_ind+1),:,:] 
    lower_mesh = lower_mesh[:,:,::-1,:,:] # reversing the mesh bc we want to go from LE to TE
    lower_Ue = Ue[:,:,:LE_ind,:]
    lower_Ue = lower_Ue[:,:,::-1,:] # reversing bc we want to go from LE to TE

    lower_BL_vel = csdl.Variable(value=np.zeros((nn, nt, num_BL_vel_chord, ns-1)))
    lower_BL_vel = lower_BL_vel.set(csdl.slice[:,:,1:-1,:], value=(lower_Ue[:,:,1:,:]+lower_Ue[:,:,:-1,:])/2.)
    lower_BL_vel = lower_BL_vel.set(csdl.slice[:,:,0,:], value=(Ue[:,:,LE_ind-1,:]+Ue[:,:,LE_ind,:])/2.)
    lower_BL_vel = lower_BL_vel.set(csdl.slice[:,:,-1,:], value=(Ue[:,:,0,:]+Ue[:,:,-1,:])/2.) # same for upper and lower

    BL_mesh_lower_rev = BL_mesh[:,:,:(LE_ind_BL+1),:,:][:,:,::-1,:,:] # check if there needs to be a +1 on the indexing
    BL_mesh_lower = (BL_mesh_lower_rev[:,:,:,:-1,:]+BL_mesh_lower_rev[:,:,:,1:,:])/2 # going from ns to ns-1
    dx_lower = csdl.norm(BL_mesh_lower[:,:,1:,:,:] - BL_mesh_lower[:,:,:-1,:,:], axes=(4,))

    mesh_lower_span_center = (lower_mesh[:,:,:,:-1,:]+lower_mesh[:,:,:,1:,:])/2.
    BL_lower_vel_slope_panel = (lower_BL_vel[:,:,1:,:] - lower_BL_vel[:,:,:-1,:]) / \
                    csdl.norm(mesh_lower_span_center[:,:,1:,:,:] - mesh_lower_span_center[:,:,:-1,:,:], axes=(4,))

    # setting up upper surface
    upper_mesh = mesh[:,:,LE_ind:,:,:]
    upper_Ue = Ue[:,:,LE_ind:,:]

    upper_BL_vel = csdl.Variable(value=np.zeros((nn, nt, num_BL_vel_chord, ns-1)))
    upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,1:-1,:], value=(upper_Ue[:,:,1:,:]+upper_Ue[:,:,:-1,:])/2.)
    upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,0,:], value=(Ue[:,:,LE_ind-1,:]+Ue[:,:,LE_ind,:])/2.)
    upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,-1,:], value=(Ue[:,:,0,:]+Ue[:,:,-1,:])/2.) # same for upper and lower

    BL_mesh_upper = (BL_mesh[:,:,LE_ind_BL:,:-1,:]+BL_mesh[:,:,LE_ind_BL:,1:,:])/2 # going from ns to ns-1
    dx_upper= csdl.norm(BL_mesh_upper[:,:,1:,:,:] - BL_mesh_upper[:,:,:-1,:,:], axes=(4,))

    mesh_upper_span_center = (upper_mesh[:,:,:,:-1,:]+upper_mesh[:,:,:,1:,:])/2.
    BL_upper_vel_slope_panel = (upper_BL_vel[:,:,1:,:] - upper_BL_vel[:,:,:-1,:]) / \
                    csdl.norm(mesh_upper_span_center[:,:,1:,:,:] - mesh_upper_span_center[:,:,:-1,:,:], axes=(4,))


    # upper_BL_vel = csdl.Variable(value=np.zeros(lower_BL_vel.shape))
    # upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,1:-1,:], value=(upper_Ue[:,:,1:,:]+upper_Ue[:,:,:-1,:])/2.)
    # upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,0,:], value=(Ue[:,:,LE_ind-1,:]+Ue[:,:,LE_ind,:])/2.)
    # upper_BL_vel = upper_BL_vel.set(csdl.slice[:,:,-1,:], value=(Ue[:,:,0,:]+Ue[:,:,-1,:])/2.) # same for upper and lower

    # getting edge velocities for upper and lower surface
    Ue_lower = csdl.Variable(value=np.zeros((nn, nt, nc_BL_one_way, ns-1)))
    Ue_lower = Ue_lower.set(csdl.slice[:,:,::BL_per_panel,:], value=lower_BL_vel)

    Ue_upper = csdl.Variable(value=np.zeros((nn, nt, nc_BL_one_way, ns-1)))
    Ue_upper = Ue_upper.set(csdl.slice[:,:,::BL_per_panel,:], value=upper_BL_vel)

    for ind in range(BL_per_panel):
        interp_value_lower = lower_BL_vel[:,:,:-1,:]+BL_lower_vel_slope_panel*\
            csdl.norm(BL_mesh_lower[:,:,ind:-1:BL_per_panel,:,:]-mesh_lower_span_center[:,:,:-1,:,:]+1.e-12, axes=(4,))
        Ue_lower = Ue_lower.set(csdl.slice[:,:,ind:-1:BL_per_panel,:], value=interp_value_lower)

        interp_value_upper = upper_BL_vel[:,:,:-1,:]+BL_upper_vel_slope_panel*\
            csdl.norm(BL_mesh_upper[:,:,ind:-1:BL_per_panel,:,:]-mesh_upper_span_center[:,:,:-1,:,:]+1.e-12, axes=(4,))
        Ue_upper = Ue_upper.set(csdl.slice[:,:,ind:-1:BL_per_panel,:], value=interp_value_upper)

    # dUe_dx for upper and lower surfaces
    dUe_dx_lower = csdl.Variable(value=np.zeros(Ue_lower.shape))
    dUe_dx_lower = dUe_dx_lower.set(csdl.slice[:,:,1:-1,:], value=(Ue_lower[:,:,2:,:] - Ue_lower[:,:,:-2,:])/(dx_lower[:,:,1:,] + dx_lower[:,:,:-1,:]))
    dUe_dx_lower = dUe_dx_lower.set(csdl.slice[:,:,0,:], value=(-3*Ue_lower[:,:,0,:] + 4*Ue_lower[:,:,1,:] - Ue_lower[:,:,2,:]) / (dx_lower[:,:,0,:] + dx_lower[:,:,1,:]))
    dUe_dx_lower = dUe_dx_lower.set(csdl.slice[:,:,-1,:], value=(Ue_lower[:,:,-3,:] - 4*Ue_lower[:,:,-2,:] + 3*Ue_lower[:,:,-1,:]) / (dx_lower[:,:,-2,:] + dx_lower[:,:,-1,:]))


    dUe_dx_upper = csdl.Variable(value=np.zeros(Ue_lower.shape))
    dUe_dx_upper = dUe_dx_upper.set(csdl.slice[:,:,1:-1,:], value=(Ue_upper[:,:,2:,:] - Ue_upper[:,:,:-2,:])/(dx_upper[:,:,1:,] + dx_upper[:,:,:-1,:]))
    dUe_dx_upper = dUe_dx_upper.set(csdl.slice[:,:,0,:], value=(-3*Ue_upper[:,:,0,:] + 4*Ue_upper[:,:,1,:] - Ue_upper[:,:,2,:]) / (dx_upper[:,:,0,:] + dx_upper[:,:,1,:]))
    dUe_dx_upper = dUe_dx_upper.set(csdl.slice[:,:,-1,:], value=(Ue_upper[:,:,-3,:] - 4*Ue_upper[:,:,-2,:] + 3*Ue_upper[:,:,-1,:]) / (dx_upper[:,:,-2,:] + dx_upper[:,:,-1,:]))

    # integration for Thwaites method
    vel_integrand_lower = (Ue_lower[:,:,:-1,:]**5 + Ue_lower[:,:,1:,:]**5)/2*dx_lower
    vel_integrand_upper = (Ue_upper[:,:,:-1,:]**5 + Ue_upper[:,:,1:,:]**5)/2*dx_upper

    vel_integration_lower = csdl.Variable(value = np.zeros(Ue_lower.shape))
    dx_cumsum_lower = csdl.Variable(value=np.zeros(Ue_lower.shape))

    vel_integration_upper = csdl.Variable(value = np.zeros(Ue_upper.shape))
    dx_cumsum_upper = csdl.Variable(value=np.zeros(Ue_upper.shape))

    for ind in range(1, nc_BL_one_way):
        dx_cumsum_lower = dx_cumsum_lower.set(
            csdl.slice[:,:,ind,:],
            value=csdl.sum(dx_lower[:,:,:ind,:], axes=(2,))
        )
        vel_integration_lower = vel_integration_lower.set(
            csdl.slice[:,:,ind,:],
            value=csdl.sum(vel_integrand_lower[:,:,:ind,:], axes=(2,))
        )

        dx_cumsum_upper = dx_cumsum_upper.set(
            csdl.slice[:,:,ind,:],
            value=csdl.sum(dx_upper[:,:,:ind,:], axes=(2,))
        )
        vel_integration_upper = vel_integration_upper.set(
            csdl.slice[:,:,ind,:],
            value=csdl.sum(vel_integrand_upper[:,:,:ind,:], axes=(2,))
        )

    # ==== computing BL profiles ====
    # theta
    theta_lower_0 = (0.075*nu/((dUe_dx_lower[:,:,0,:])**2)**0.5)**0.5
    theta_lower_integration = (0.45*nu/Ue_lower[:,:,1:,:]**6*vel_integration_lower[:,:,1:,:])**0.5
    theta_lower = csdl.Variable(value=np.zeros(Ue_lower.shape))
    theta_lower = theta_lower.set(csdl.slice[:,:,0,:], value=theta_lower_0)
    theta_lower = theta_lower.set(csdl.slice[:,:,1:,:], value=theta_lower_integration)

    theta_upper_0 = (0.075*nu/((dUe_dx_upper[:,:,0,:])**2)**0.5)**0.5
    theta_upper_integration = (0.45*nu/Ue_upper[:,:,1:,:]**6*vel_integration_upper[:,:,1:,:])**0.5
    theta_upper = csdl.Variable(value=np.zeros(Ue_upper.shape))
    theta_upper = theta_upper.set(csdl.slice[:,:,0,:], value=theta_upper_0)
    theta_upper = theta_upper.set(csdl.slice[:,:,1:,:], value=theta_upper_integration)

    # lambda (nondim pressure gradient)
    lam_lower = theta_lower**2/nu*dUe_dx_lower
    lam_upper = theta_upper**2/nu*dUe_dx_upper

    # H & l
    H_func_list_lower = [
            0.0731/(0.14 + lam_lower) + 2.088,
            2.61 - 3.75*lam_lower + 5.24*lam_lower**2,
        ]
    H_lower = switch_func(lam_lower, H_func_list_lower, [0], scale=100.)

    H_func_list_upper = [
            0.0731/(0.14 + lam_upper) + 2.088,
            2.61 - 3.75*lam_upper + 5.24*lam_upper**2,
        ]
    H_upper = switch_func(lam_upper, H_func_list_upper, [0], scale=100.)

    l_func_list_lower = [
        0.22 +1.402*lam_lower + 0.018*lam_lower/(0.107+lam_lower),
        0.22 + 1.57*lam_lower-1.8*lam_lower**2
    ]
    l_lower = switch_func(lam_lower, l_func_list_lower, [0], scale=100.)
    l_func_list_upper = [
        0.22 +1.402*lam_upper + 0.018*lam_upper/(0.107+lam_upper),
        0.22+1.57*lam_upper-1.8*lam_upper**2
    ]
    l_upper = switch_func(lam_upper, l_func_list_upper, [0], scale=100.)

    # Cf & delta star
    Cf_lower = 2*nu/Ue_lower/theta_lower*l_lower
    delta_star_lower = H_lower*theta_lower

    Cf_upper = 2*nu/Ue_upper/theta_upper*l_upper
    delta_star_upper = H_upper*theta_upper

    # extrapolating to the coarse potential flow grid with IDW
    # NOTE: we extrapolate Cf and lambda
    
    for i in range(BL_per_panel):
        pass
    
    # inv = inviscid
    Cf_lower_inv = (Cf_lower[:,:,:-1:BL_per_panel,:] + Cf_lower[:,:,BL_per_panel::BL_per_panel,:])/2
    lam_lower_inv = (lam_lower[:,:,:-1:BL_per_panel,:] + lam_lower[:,:,BL_per_panel::BL_per_panel,:])/2

    Cf_upper_inv = (Cf_upper[:,:,:-1:BL_per_panel,:] + Cf_upper[:,:,BL_per_panel::BL_per_panel,:])/2
    lam_upper_inv = (lam_upper[:,:,:-1:BL_per_panel,:] + lam_upper[:,:,BL_per_panel::BL_per_panel,:])/2
    
    # smoothing out negative Cf
    zero_lower = csdl.Variable(value=np.zeros(Cf_lower_inv.shape))
    Cf_lower_inv_smooth = csdl.maximum(
        Cf_lower_inv,
        zero_lower,
        rho=1000.
    )

    zero_upper = csdl.Variable(value=np.zeros(Cf_upper_inv.shape))
    Cf_upper_inv_smooth = csdl.maximum(
        Cf_upper_inv,
        zero_upper,
        rho=1000.
    )

    Cf_PM = [Cf_lower_inv_smooth, Cf_upper_inv_smooth]
    # Cf_PM = [Cf_lower_inv, Cf_upper_inv]
    lam_PM = [lam_lower_inv, lam_upper_inv]

    # TODO: check outputs via plots 

    return Cf_PM, lam_PM
