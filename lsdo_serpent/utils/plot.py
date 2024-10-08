import vedo
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show, Mesh
import numpy as np
from vedo import *
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

def plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, interactive = False, plot_mirror = True, wake_color='cyan', rotor_wake_color='red', surface_color='gray', cmap='jet', absolute=True, side_view=False, name='sample_gif', backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(-.25,.5),
        yrange=(-0.5, 0.5),
        zrange=(-.25, .25),
    )
    video = Video(name+".mp4", fps=5, backend=backend)
    # first get min and max mu value:
    min_mu_b = np.min(mu)
    max_mu_b = np.max(mu)
    min_mu_w = np.min(mu_wake)
    max_mu_w = np.max(mu_wake)

    min_mu = np.min((min_mu_b, min_mu_w))
    max_mu = np.max((max_mu_b, max_mu_w))

    for i in range(nt-1):

        # min_mu_b = np.min(mu[:,i,:])
        # max_mu_b = np.max(mu[:,i,:])
        # min_mu_w = np.min(mu_wake[:,i,:])
        # max_mu_w = np.max(mu_wake[:,i,:])

        # min_mu = np.min((min_mu_b, min_mu_w))
        # max_mu = np.max((max_mu_b, max_mu_w))
        vp = Plotter(
            bg='white',
            # bg2='white',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1,
            size=(2500,2500))

        # Any rendering loop goes here, e.g.
        draw_scalarbar = True
        color = wake_color
        mesh_points = mesh[:, i, :] # does not vary with time here
        nx = mesh_points.shape[1]
        ny = mesh_points.shape[2]
        connectivity = []
        for k in range(nx-1):
            for j in range(ny-1):
                connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
            vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
        mu_color = np.reshape(mu[:,i,:], (-1,1))
        
        vps.cmap(cmap, mu_color, on='cells', vmin=min_mu, vmax=max_mu)
        vp += vps
        vp += __doc__
        wake_points = wake_mesh[:,i,:(i+1),:]
        # mu_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_mu_w'][i, 0:i, :], (-1,1))
        # if absolute:
        #     mu_w = np.absolute(mu_w)
        # wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
        nx = wake_points.shape[1]
        ny = wake_points.shape[2]
        connectivity = []
        for k in range(nx-1):
            for j in range(ny-1):
                connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
        vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity], c=color, alpha=1)
        if i > 0:
            mu_wake_color = np.reshape(mu_wake[:,i,:(i)*(ny-1)], (-1,1))
            vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
        if draw_scalarbar:
            vps.add_scalarbar()
            draw_scalarbar = False
        vps.linewidth(1)
        vp += vps
        vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        if side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
    #     vp.close_window()
    # vp.close_window()
    video.close()  # merge all the recorded frames


def plot_pressure_distribution(mesh, Cp, surface_color='white', cmap='jet', interactive=False, top_view=False, front_top_view=False):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(-.25,.5),
        yrange=(-0.5, 0.5),
        zrange=(-.25, .25),
    )
    vp = Plotter(
        bg='white',
        # bg2='white',
        # axes=0,
        #  pos=(0, 0),
        offscreen=False,
        interactive=1,
        size=(2500,2500))

    # Any rendering loop goes here, e.g.
    draw_scalarbar = True
    # color = wake_color
    mesh_points = mesh[:, :, :] # does not vary with time here
    nx = mesh_points.shape[2]
    ny = mesh_points.shape[3]
    connectivity = []
    for k in range(nx-1):
        for j in range(ny-1):
            connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
        vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
        # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.)
    Cp_color = np.reshape(Cp[:,-2,:,:], (-1,1))
    Cp_min, Cp_max = np.min(Cp[:,-2,:,:]), np.max(Cp[:,-2,:,:])
    # Cp_min, Cp_max = -5., 3.
    # Cp_min, Cp_max = -4., 1.
    vps.cmap(cmap, Cp_color, on='cells', vmin=Cp_min, vmax=Cp_max)
    # vps.cmap(cmap, Cp_color, on='cells', vmin=-0.4, vmax=1)
    vps.add_scalarbar()
    vp += vps
    vp += __doc__
    # wake_points = wake_mesh[:,i,:(i+1),:]
    # # mu_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_mu_w'][i, 0:i, :], (-1,1))
    # # if absolute:
    # #     mu_w = np.absolute(mu_w)
    # # wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
    # nx = wake_points.shape[1]
    # ny = wake_points.shape[2]
    # connectivity = []
    # for k in range(nx-1):
    #     for j in range(ny-1):
    #         connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
    # vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity], c=color, alpha=1)
    # mu_wake_color = np.reshape(mu_wake[:,i,:(i)*(ny-1)], (-1,1))
    # vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
    # if draw_scalarbar:
    #     vps.add_scalarbar()
    #     draw_scalarbar = False
    # vps.linewidth(1)
    # vp += vps
    # vp += __doc__
    # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
    # video.action(cameras=[cam1, cam1])
    # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
    #         axes=False, interactive=False)  # render the scene
    # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
    #         axes=False, interactive=False, zoom=True)  # render the scene
    if top_view:
        vp.show(axs, elevation=0, azimuth=0, roll=90,
                axes=False, interactive=interactive)  # render the scene
    elif front_top_view:
        vp.show(axs, elevation=0, azimuth=-45, roll=90,
                axes=False, interactive=interactive)  # render the scene
    else:
        vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                axes=False, interactive=interactive)  # render the scene
    # video.add_frame()  # add individual frame
    # # time.sleep(0.1)
    # # vp.interactive().close()
    # vp.close_window()
        
def plot_transient_pressure_distribution(mesh, Cp, name='unsteady_Cp', surface_color='white', cmap='jet', interactive=False, side_view=False, top_view=False, front_top_view=False, backend='imageio'):
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(-.25,.5),
        yrange=(-0.5, 0.5),
        zrange=(-.25, .25),
    )
    video = Video(name+".mp4", fps=5, backend=backend)

    nt = mesh.shape[1]
    Cp_min = np.min(Cp[:,:-1,:])
    Cp_max = np.max(Cp[:,:-1,:])
    for t in range(nt-1):
        vp = Plotter(
            bg='white',
            # bg2='white',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1,
            size=(2500,2500))

        # Any rendering loop goes here, e.g.
        draw_scalarbar = True
        # color = wake_color
        mesh_points = mesh[:, t, :] # does not vary with time here
        nx = mesh_points.shape[1]
        ny = mesh_points.shape[2]
        connectivity = []
        for k in range(nx-1):
            for j in range(ny-1):
                connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
            vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.).linecolor('black')
            # vps = Mesh([np.reshape(mesh_points, (-1, 3)), connectivity], c=surface_color, alpha=1.)
        Cp_color = np.reshape(Cp[:,t,:,:], (-1,1))
        # Cp_min, Cp_max = np.min(Cp[:,-2,:,:]), np.max(Cp[:,-2,:,:])
        # Cp_min, Cp_max = -5., 3.
        # Cp_min, Cp_max = -4., 1.
        vps.cmap(cmap, Cp_color, on='cells', vmin=Cp_min, vmax=Cp_max)
        # vps.cmap(cmap, Cp_color, on='cells', vmin=-0.4, vmax=1)
        vps.add_scalarbar()
        vp += vps
        vp += __doc__
        # wake_points = wake_mesh[:,i,:(i+1),:]
        # # mu_w = np.reshape(sim['system_model.wig.wig.wig.operation.prob.' + 'op_' + surface_name+'_mu_w'][i, 0:i, :], (-1,1))
        # # if absolute:
        # #     mu_w = np.absolute(mu_w)
        # # wake_points = np.concatenate((np.reshape(mesh_points[-1,:,:],(1,ny,3)), wake_points))
        # nx = wake_points.shape[1]
        # ny = wake_points.shape[2]
        # connectivity = []
        # for k in range(nx-1):
        #     for j in range(ny-1):
        #         connectivity.append([k*ny+j,(k+1)*ny+j,(k+1)*ny+j+1,k*ny+j+1])
        # vps = Mesh([np.reshape(wake_points, (-1, 3)), connectivity], c=color, alpha=1)
        # mu_wake_color = np.reshape(mu_wake[:,i,:(i)*(ny-1)], (-1,1))
        # vps.cmap(cmap, mu_wake_color, on='cells', vmin=min_mu, vmax=max_mu)
        # if draw_scalarbar:
        #     vps.add_scalarbar()
        #     draw_scalarbar = False
        # vps.linewidth(1)
        # vp += vps
        # vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        # vp.show(axs, elevation=-60, azimuth=45, roll=-45,
        #         axes=False, interactive=False)  # render the scene
        # vp.show(axs, elevation=-60, azimuth=-90, roll=90,
        #         axes=False, interactive=False, zoom=True)  # render the scene
        if top_view:
            vp.show(axs, elevation=0, azimuth=0, roll=90,
                    axes=False, interactive=interactive)  # render the scene
        elif front_top_view:
            vp.show(axs, elevation=0, azimuth=-45, roll=90,
                    axes=False, interactive=interactive)  # render the scene
        if side_view:
            vp.show(axs, elevation=-90, azimuth=0, roll=0,
                    axes=False, interactive=interactive)  # render the scene
        else:
            vp.show(axs, elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=interactive)  # render the scene
        video.add_frame()  # add individual frame
        # # time.sleep(0.1)
        # # vp.interactive().close()
        # vp.close_window()
    video.close()



# plot the lift distribution across the half span:
def plot_lift_spanwise(nt, n_avg, var, num_span, num_chordwise_vlm, num_props, rpos):

    fig = plt.figure(figsize=(8,3))
    xpos = np.linspace(0,num_span,num_span)

    data = np.zeros((num_span))
    for i in range(nt - n_avg - 1, nt - 1):
        temp = np.zeros(num_span)
        for j in range(num_chordwise_vlm - 1):
            temp[:] += var[i,j*num_span:(j+1)*num_span,0].flatten()
        data[:] += temp

    print(data/n_avg)

    plt.plot(xpos/max(xpos), data/n_avg, label='_nolegend_')
    plt.scatter(xpos/max(xpos), data/n_avg, label='_nolegend_')
    for i in range(int(num_props/2)): plt.axvline(x=rpos[i], color='black', linestyle='dashed', linewidth=2)
    plt.xlim([0,1])

    plt.xlabel('Spanwise location')
    plt.ylabel('Lift (N/m)')
    plt.legend(['Rotor locations'], frameon=False)
    # plt.savefig('lift_distribution_pos_y.png', transparent=True, bbox_inches="tight", dpi=400)
    plt.show()