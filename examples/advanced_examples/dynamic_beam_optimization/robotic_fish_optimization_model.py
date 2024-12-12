'''
This is a CSDL model of a two-body pendulum system. The model is constructed using individual residual models for each body
and a time discretization model preppended to each residual model (which computes the derivatives of the states and lagrange multipliers
as a function of the states and lagrange multipliers). The residual models are then combined into a single residual model and a Newton solver
(which is implemented in Python) will converge the residual model at each time step. For this Newton solver:
residual: [df/dx + lambda*dc/dx, c] = 0
Jacobian: KKT matrix which will be assembled with the help of CSDL AD.

In addition to the residual and time discretization models, we also need a constraints model that will compute the constraints
and append the lagrange multiplier term (lambda*dc/dx) to the residuals.

NOTE: In future iteration, decouple time stepping of each residual models by representing states functionally (as a function of time and space)
NOTE: As a first iteration, we take a SAND approach to this inner optimization/solver, where the optimizer picks all of the states
and lagrange multipliers as design variables and the residual model is only used for single evaluation (not driving residual to 0).
NOTE: In the approach from the note above, we do not need to feed the lagrange multipliers into the residual models.

TODO: Figure out how to automate input creation, design variable addition, output registration, and connection.
TODO: Automate the creation of this model altogether.
TODO: Allow use of geometry to specify constraints.
TODO: Think about whether self.bodies should be a dictionary of [body_name, body_residual_model] or perhaps [body_name, body_component/obj].
TODO: As the first note says, decouple time stepping (SIFR)

TODO: Add mapping from structural rotations to geometry

TODO: Speed up simulation!!
    -- Stack all constraints at the end?
'''


import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sps
import lsdo_geo
import lsdo_function_spaces as lfs

from mbd_trial.framework_trial.time_discretizations.generalized_alpha import GeneralizedAlphaModel
from mbd_trial.framework_trial.time_discretizations.generalized_alpha import GeneralizedAlphaStateData
# from mbd_trial.framework_trial.models.gravitational_potential_energy_model import compute_gravitational_potential_energy
from mbd_trial.framework_trial.models.rigid_body_kinetic_energy_model import compute_rigid_body_kinetic_energy
# from mbd_trial.framework_trial.models.pendulum_residual import PendulumResidualModel
from mbd_trial.framework_trial.models.residual_intertial_term import ResidualInertialTerm
from mbd_trial.framework_trial.models.rigid_body_gravity_residual_3d import RigidBodyGravityResidual3D
import aframe

from dataclasses import dataclass
from typing import Union

from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer


@dataclass
class StateData:
    states:dict[str,csdl.Variable]
    state_derivatives:dict[str,csdl.Variable]


@dataclass
class Body:
    geometry : Union[lsdo_geo.Geometry,lfs.FunctionSet,lfs.Function]
    body_counter = 0
    name : str = None
    mass : csdl.Variable = None
    center_of_mass : csdl.Variable = None
    representations : dict[str,lsdo_geo.Mesh] = None        # NOTE: I'm thinking of setting this using an add_analysis method or modularizing MBD

    # @property
    # def beam_mesh(self) -> csdl.Variable:
    #     # return self.geometry.evaluate_representations(self.beam_representation, plot=False)
    #     return self.design_geometry.evaluate_representations(self.beam_representation, plot=False)

    def __post_init__(self):
        if self.name is None:
            self.name = f'body{Body.body_counter}'
            Body.body_counter += 1

        self.design_geometry = self.geometry.copy()
            
        # self.geometry.add_representation(representation=self.beam_representation)
        # self.design_geometry.add_representation(representation=self.beam_representation)
        # self.beam_mesh = self.design_geometry.evaluate_representations(self.beam_representation, plot=False)

        # if self.material is None:
        #     # E = 79.e9
        #     # E = 5.e6
        #     # E = 6.e5    # Young's modulus of soft material
        #     # E = 1e5
        #     # E = 3.e4      # Young's modulus of Jello
        #     # E = 1e3
        #     # E = 1
        #     E = 70e9     # Young's modulus of aluminum
        #     nu = 0.3
        #     G = E/(2*(1+nu))
        #     # density = 1
        #     # density = 1e2
        #     # density = 1080  # Density of soft material
        #     # density = 1270  # Density of Jello
        #     density = 2700    # Density of aluminum
        #     self.material = aframe.Material(E=E, G=G, density=density)
        # skin_thickness = csdl.Variable(value=0.001)
        # self.skin_thickness = skin_thickness
        # if self.mass is None:
        #     self.mass = (2*(height*self.length) + 2*(height*width) + 2*(self.length*width))*skin_thickness*self.material.density # hardcoding pendulum dimensions for now to get the idea across

        beam_mesh = self.representations['beam'].evaluate(self.geometry)

        E_dragon = csdl.Variable(value=6.e+5, name='E_dragon')  # This is what I was using
        # E_dragon = csdl.Variable(value=2.e+6, name='E_dragon')
        # E_dragon = csdl.Variable(value=2.3794898134622741e+5, name='E_dragon')
        # E_dragon = csdl.Variable(value=1.e5, name='E_dragon')
        nu_dragon = csdl.Variable(value=0.49, name='nu_dragon')
        # rho_dragon = 1080
        # rho_dragon = 500
        # rho_dragon = 250
        rho_dragon = 100    # This is what I was using
        # rho_dragon = 50    # decreasing this increases ability for fluid to propgate with wave cleanly
        # NOTE: If the wave is already propogating cleanly, then decreasing this sliightly decreases the thrust because the tail doesn't bend as much
        # E_fr4 = csdl.Variable(value=0.223121251595327195e+9, name='E_fr4')
        # E_fr4 = csdl.Variable(value=1.e+9, name='E_fr4')
        E_fr4 = csdl.Variable(value=24e+9, name='E_fr4')
        # E_fr4 = csdl.Variable(value=50e+9, name='E_fr4')
        nu_fr4 = csdl.Variable(value=0.12, name='nu_fr4')
        # rho_fr4 = 1850
        rho_fr4 = 5000        # This makes practically no difference
        silicone = aframe.Material(E=E_dragon, G=E_dragon/(2*(1+nu_dragon)), density=rho_dragon)
        softer_silicone = aframe.Material(E=E_dragon/60, G=E_dragon/(2*(1+nu_dragon)), density=rho_dragon/5)
        pla = aframe.Material(E=1.e8, G=1.e8/(2*(1+0.3)), density=1000000)
        # softer_silicone = aframe.Material(E=E_dragon/10, G=E_dragon/(2*(1+nu_dragon)), density=rho_dragon)
        fr4 = aframe.Material(E=E_fr4, G=E_fr4/(2*(1+nu_fr4)), density=rho_fr4)
        self.materials = [silicone, fr4]
        cross_section_meshes = [mesh.evaluate(self.geometry) for mesh in self.representations['cross_section'].values()]
        cross_section = aframe.CSComposite(meshes=cross_section_meshes, materials=self.materials)

        rigid_cross_section = aframe.CSCircle(radius=0.002)
        pla_cross_section = aframe.CSCircle(radius=0.02)

        # self.beam = aframe.Beam(name=self.name, mesh=beam_mesh, material=self.materials[0], cs=cross_section, z=False)
        
        head_beam = aframe.Beam(name='head', mesh=beam_mesh[:2], material=pla, cs=pla_cross_section, z=False)
        actuator_1_beam = aframe.Beam(name='actuator_1', mesh=beam_mesh[1:4], material=self.materials[0], cs=cross_section, z=False)
        rigid_couple_1 = aframe.Beam(name='rigid_couple_1', mesh=beam_mesh[3:5], material=fr4, cs=rigid_cross_section, z=False)
        actuator_2_beam = aframe.Beam(name='actuator_2', mesh=beam_mesh[4:7], material=self.materials[0], cs=cross_section, z=False)
        rigid_couple_2 = aframe.Beam(name='rigid_couple_2', mesh=beam_mesh[6:8], material=fr4, cs=rigid_cross_section, z=False)
        actuator_3_beam = aframe.Beam(name='actuator_3', mesh=beam_mesh[7:10], material=self.materials[0], cs=cross_section, z=False)
        rigid_couple_3 = aframe.Beam(name='rigid_couple_3', mesh=beam_mesh[9:11], material=fr4, cs=rigid_cross_section, z=False)
        tail_beam = aframe.Beam(name='tail', mesh=beam_mesh[10:], material=softer_silicone, cs=cross_section, z=False)
        # self.beam.fix(0)
        # head_beam.fix(0)

        # beam = aframe.Beam(name=body.name, mesh=body.beam_mesh, material=body.material, cs=body.cross_section, z=True)
        frame = aframe.Frame()
        # frame.add_beam(self.beam)
        frame.add_beam(head_beam)
        frame.add_beam(actuator_1_beam)
        frame.add_beam(rigid_couple_1)
        frame.add_beam(actuator_2_beam)
        frame.add_beam(rigid_couple_2)
        frame.add_beam(actuator_3_beam)
        frame.add_beam(rigid_couple_3)
        frame.add_beam(tail_beam)

        frame.add_joint([head_beam, actuator_1_beam], [1, 0])
        frame.add_joint([actuator_1_beam, rigid_couple_1], [2, 0])
        frame.add_joint([rigid_couple_1, actuator_2_beam], [1, 0])
        frame.add_joint([actuator_2_beam, rigid_couple_2], [2, 0])
        frame.add_joint([rigid_couple_2, actuator_3_beam], [1, 0])
        frame.add_joint([actuator_3_beam, rigid_couple_3], [2, 0])
        frame.add_joint([rigid_couple_3, tail_beam], [1, 0])
        self.frame = frame
        # frame.solve() # This is for testing purposes
        # frame.add_beam(beam)
        
        # gravitational_acceleration = csdl.Variable(value=np.array([0, 0, -self.g, 0, 0, 0]))    # gravitational work
        # # Defining flexible states/residual in local frame, so rotate gravitational acceleration to local frame
        # gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), 
        #                                              angles=-state_data[body.name]['rigid_body_states'].states[5]).reshape((6,))
        # gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), 
        #                                              angles=-state_data[body.name]['rigid_body_states'].states[4]).reshape((6,))
        # gravitational_acceleration = lsdo_geo.rotate(gravitational_acceleration, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), 
        #                                              angles=-state_data[body.name]['rigid_body_states'].states[3]).reshape((6,))
        # frame.add_acc(gravitational_acceleration)

        # create the global stiffness/mass matrices
        dim, num = frame._utils()
        frame.dim = dim
        # frame._mass_properties()
        self.K, self.M = frame._global_matrices()
        # solve the system of equations
        # alpha, beta = 0.04, 0.1
        # alpha, beta = 2., 10.
        # alpha, beta = 1.6, 4.
        # alpha, beta = 0.8, 2.
        alpha, beta = 0.4, 1.   # This is what I was using
        # alpha, beta = 0.2, 0.5
        # alpha, beta = 0.1, 0.25
        self.D = alpha*self.K + beta*self.M

        # Set up mapping from beam displacements to geometry
        # - Fit B-spline to beam mesh
        # beam_mesh = beam_mesh.value
        # b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=beam_mesh.shape[0])
        # b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=beam_mesh.shape[0])
        b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=beam_mesh.shape[0])
        # beam_b_spline_coefficients = np.array([
        #     [0.278, 0., 0.],        # head front
        #     [0.218, 0., 0.],        # head back (module 1 front)
        #     [0.218, 0., 0.],        # head back (module 1 front)
        #     [(0.218+0.15)/2, 0., 0.],        # module 1 middle
        #     [0.15, 0., 0.],         # module 1 back
        #     [0.15, 0., 0.],         # module 1 back
        #     [0.126, 0., 0.],        # module 2 front
        #     [0.126, 0., 0.],        # module 2 front
        #     [(0.126+0.058)/2, 0., 0.],        # module 2 middle
        #     [0.058, 0., 0.],        # module 2 back
        #     [0.058, 0., 0.],        # module 2 back
        #     [0.034, 0., 0.],        # module 3 front
        #     [0.034, 0., 0.],        # module 3 front
        #     [(0.034-0.034)/2, 0., 0.],        # module 3 middle
        #     [-0.034, 0., 0.],       # module 3 back
        #     [-0.034, 0., 0.],       # module 3 back
        #     [-0.053, 0., 0.],       # tail front
        #     [-0.181, 0., 0.],       # tail back
        # ])
        # b_spline_function_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, 
        #                                            coefficients_shape=beam_b_spline_coefficients.shape[0])
        # beam_b_spline = lfs.Function(space=b_spline_function_space, coefficients=beam_b_spline_coefficients)
        beam_b_spline = b_spline_function_space.fit_function(beam_mesh, parametric_coordinates=np.linspace(0, 1, num=beam_mesh.shape[0]))

        # - Create mapping from beam displacements to grid mesh displacements
        # -- Compute displacement portion of matrix mappings (precomputing matrix for efficiency)
        self.beam_displacement_to_geometry_displacement_mapping = {}
        self.beam_evaluation_maps = {}
        self.wireframe_coordinates_with_respect_to_beam_coordinates = {}
        self.wireframe_to_geometry_displacement_mapping = {}
        for function_index, function in enumerate([self.geometry]):
            # - Create grid mesh over geometry
            # fitting_resolution = 5  # For now to go faster
            fitting_resolution = (31,23,23)
            # fitting_resolution = (61,33,33)
            # fitting_resolution = 25
            # fitting_resolution = 35
            parametric_geometry_wireframe = function.space.generate_parametric_grid(grid_resolution=fitting_resolution)
            geometry_wireframe = function.evaluate(parametric_geometry_wireframe)

            # - Project grid mesh onto B-spline
            closest_parametric_coordinates = beam_b_spline.project(geometry_wireframe.value, plot=False, grid_search_density_parameter=20.)
            beam_to_wireframe_displacement_mapping = b_spline_function_space.compute_basis_matrix(closest_parametric_coordinates)
            
            self.beam_evaluation_maps[function_index] = beam_to_wireframe_displacement_mapping
            closest_physical_coordinates = beam_b_spline.evaluate(closest_parametric_coordinates)
            self.wireframe_coordinates_with_respect_to_beam_coordinates[function_index] = geometry_wireframe - closest_physical_coordinates

            geometry_to_wireframe_displacement_mapping = function.space.compute_basis_matrix(parametric_geometry_wireframe)
            # import scipy.sparse.linalg as spsl
            self.wireframe_to_geometry_displacement_mapping[function_index] = np.linalg.pinv(geometry_to_wireframe_displacement_mapping.toarray())
            # self.wireframe_to_geometry_displacement_mapping[function_index] = spsl.pinv(geometry_to_wireframe_displacement_mapping)
            # self.wireframe_to_geometry_displacement_mapping[function_index] = spsl.inv(geometry_to_wireframe_displacement_mapping.transpose() @ geometry_to_wireframe_displacement_mapping) @ geometry_to_wireframe_displacement_mapping.transpose()
            self.beam_displacement_to_geometry_displacement_mapping[function_index] = self.wireframe_to_geometry_displacement_mapping[function_index] @ beam_to_wireframe_displacement_mapping
        self.beam_displacement_to_geometry_displacement_mapping = np.vstack(list(self.beam_displacement_to_geometry_displacement_mapping.values()))
        self.beam_evaluation_maps = sps.vstack(list(self.beam_evaluation_maps.values()))
        if len(self.wireframe_coordinates_with_respect_to_beam_coordinates) > 1:
            self.wireframe_coordinates_with_respect_to_beam_coordinates = csdl.vstack(list(self.wireframe_coordinates_with_respect_to_beam_coordinates.values()))
        else:
            self.wireframe_coordinates_with_respect_to_beam_coordinates = self.wireframe_coordinates_with_respect_to_beam_coordinates[0]



        # -- Apply displacements from rotations (using small angle approximation? Not necessary, but more efficient)
        # --- I will use small angle approximation, so the mapping can be pre-computed as a matrix for efficiency?
        # TODO: Come back to this
        # closest_points = beam_b_spline.evaluate(closest_parametric_coordinates)
        # offsets = geometry_wireframe.value - closest_points

        grid_resolution = (15,15,15)
        if (isinstance(self.geometry, lfs.FunctionSet)):
            parametric_grid = self.design_geometry.generate_parametric_grid(grid_resolution=grid_resolution)
        else:
            parametric_grid = self.design_geometry.space.generate_parametric_grid(grid_resolution=grid_resolution)
        grid = self.design_geometry.evaluate(parametric_grid)
        if (isinstance(self.geometry, lfs.FunctionSet)):
            grid = grid.reshape((len(self.design_geometry.functions),) + grid_resolution + (3,))
        else:
            grid = grid.reshape((1,) + grid_resolution + (3,))
        u_vectors = grid[:,1:] - grid[:,:-1]
        v_vectors = grid[:,:,1:] - grid[:,:,:-1]
        w_vectors = grid[:,:,:,1:] - grid[:,:,:,:-1]

        self.design_element_centers = grid[:,:-1,:-1,:-1] + u_vectors[:,:,:-1,:-1]/2 + v_vectors[:,:-1,:,:-1]/2 + w_vectors[:,:-1,:-1,:]/2

        element_areas_1 = csdl.cross(u_vectors[:,:,:-1], v_vectors[:,:-1,:], axis=4)
        element_areas_2 = csdl.cross(u_vectors[:,:,1:], v_vectors[:,1:,:], axis=4)
        # element_areas = (csdl.norm(element_areas_1 + element_areas_2, axes=(3,)))/2
        element_areas = element_areas_1 + element_areas_2/2
        element_volumes_1 = csdl.einsum(element_areas[:,:,:,:-1], w_vectors[:,:-1,:-1,:,:], action='ijklm,ijklm->ijkl')
        element_volumes_2 = csdl.einsum(element_areas[:,:,:,1:], w_vectors[:,1:,1:,:,:], action='ijklm,ijklm->ijkl')
        element_volumes = (element_volumes_1 + element_volumes_2)/2

        self.element_masses = element_volumes*self.materials[0].density  # Assume all of the mass is from silicone. This also ignores chambers for simplicity
        self.element_masses_expanded = csdl.expand(self.element_masses, self.design_element_centers.shape, 'ijkl->ijklm')

        self.mass = csdl.sum(self.element_masses, axes=(0,1,2,3))
        self.mass.add_name(f'{self.name}_mass')

        _, self.design_center_of_mass, _, _, _ = self.evaluate_mass_properties(self.design_geometry, properties_to_compute=['center_of_mass'])

        self.design_relative_element_centers = self.design_element_centers - csdl.expand(self.design_center_of_mass, self.design_element_centers.shape, 'm->ijklm')


    def evaluate_mass_properties(self, geometry:lsdo_geo.Geometry=None, properties_to_compute:list[str]=['all']) -> tuple[csdl.Variable, csdl.Variable, csdl.Variable, csdl.Variable]:
        '''
        Evaluates the mass, center of mass, angle of mass, and moment of inertia of the body.

        Parameters
        ----------
        geometry : lsdo_geo.Geometry = None
            The geometry of the body. If None, the geometry of the body is used. Default is None.
        
        Returns
        -------
        mass : csdl.Variable
            The mass of the body.
        center_of_mass : csdl.Variable
            The center of mass of the body.
        moment_of_inertia : csdl.Variable
            The moment of inertia of the body.
        angular_momentum : csdl.Variable
            The angular momentum of the body.
        '''
        if geometry is None:
            geometry = self.geometry

        if ['all'] in properties_to_compute:
            properties_to_compute = ['mass', 'center_of_mass', 'moment_of_inertia', 'angular_momentum', 'change_in_angle_of_mass']
        mass = None
        center_of_mass = None
        moment_of_inertia = None
        angular_momentum = None
        change_in_angle_of_mass = None

        # TODO: Add gauss-quadrature or analytic integration to lsdo_function_spaces so I can properly do center_of_mass = rho*integral(B)*coefficients

        # grid_resolution = (11,11)
        grid_resolution = (15,15,15)
        if (isinstance(geometry, lfs.FunctionSet)):
            parametric_grid = geometry.generate_parametric_grid(grid_resolution=grid_resolution)
        elif (isinstance(geometry, lfs.Function)):
            parametric_grid = geometry.space.generate_parametric_grid(grid_resolution=grid_resolution)

        grid = geometry.evaluate(parametric_grid)

        # material = self.material
        # area_density = material.density*self.skin_thickness

        if (isinstance(geometry, lfs.FunctionSet)):
            grid = grid.reshape((len(geometry.functions),) + grid_resolution + (3,))
        else:
            grid = grid.reshape((1,) + grid_resolution + (3,))
        u_vectors = grid[:,1:] - grid[:,:-1]
        v_vectors = grid[:,:,1:] - grid[:,:,:-1]
        w_vectors = grid[:,:,:,1:] - grid[:,:,:,:-1]

        element_areas_1 = csdl.cross(u_vectors[:,:,:-1,:-1], v_vectors[:,:-1,:,:-1], axis=4)
        element_areas_2 = csdl.cross(u_vectors[:,:,1:,:-1], v_vectors[:,1:,:,:-1], axis=4)
        element_areas = (element_areas_1 + element_areas_2)/2
        element_volumes_1 = csdl.einsum(element_areas, w_vectors[:,:-1,:-1,:,:], action='ijklm,ijklm->ijkl')
        element_volumes_2 = csdl.einsum(element_areas, w_vectors[:,1:,1:,:,:], action='ijklm,ijklm->ijkl')
        element_volumes = (element_volumes_1 + element_volumes_2)/2

        element_centers = grid[:,:-1,:-1,:-1] + u_vectors[:,:,:-1,:-1]/2 + v_vectors[:,:-1,:,:-1]/2 + w_vectors[:,:-1,:-1,:]/2

        if 'mass' in properties_to_compute:
            self.element_masses = element_volumes*self.materials[0].density
            self.element_masses_expanded = csdl.expand(self.element_masses, element_centers.shape, 'ijkl->ijklm')

            total_mass = csdl.sum(self.element_masses)
            mass = total_mass
            mass.add_name(f'{self.name}_mass')
            self.mass = mass

        if 'center_of_mass' in properties_to_compute or 'moment_of_inertia' in properties_to_compute:
            first_moment_of_mass = csdl.sum(element_centers*self.element_masses_expanded, axes=(0,1,2,3))
            center_of_mass = first_moment_of_mass/self.mass
            center_of_mass.add_name(f'{self.name}_center_of_mass')

        if 'angular_momentum' in properties_to_compute:
            element_velocities = element_centers - self.design_element_centers
            element_momentums = element_velocities*self.element_masses_expanded

            angular_momentums = csdl.cross(self.design_relative_element_centers, element_momentums, axis=4)
            angular_momentum = csdl.sum(angular_momentums, axes=(0,1,2,3))

        if 'change_in_angle_of_mass' in properties_to_compute or 'moment_of_inertia' in properties_to_compute:
            relative_element_centers = element_centers - csdl.expand(center_of_mass, element_centers.shape, 'm->ijklm')
        
        if 'change_in_angle_of_mass' in properties_to_compute:
            change_in_angle_of_mass = csdl.sum(self.element_masses_expanded*csdl.cross(relative_element_centers, self.design_relative_element_centers, axis=4), axes=(0,1,2,3))


        if 'moment_of_inertia' in properties_to_compute:
            x_component = relative_element_centers[:,:,:,:,0]
            y_component = relative_element_centers[:,:,:,:,1]
            z_component = relative_element_centers[:,:,:,:,2]
            x_component_squared = x_component**2
            y_component_squared = y_component**2
            z_component_squared = z_component**2

            second_moment_of_mass = csdl.Variable(value=np.zeros((3,3)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,0],
                                                        second_moment_of_mass[0,0] + 
                                                        csdl.sum((y_component_squared + z_component_squared)*self.element_masses, axes=(0,1,2,3)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,1],
                                                        second_moment_of_mass[1,1] + 
                                                        csdl.sum((x_component_squared + z_component_squared)*self.element_masses, axes=(0,1,2,3)))
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,2],
                                                        second_moment_of_mass[2,2] + 
                                                        csdl.sum((x_component_squared + y_component_squared)*self.element_masses, axes=(0,1,2,3)))
            
            xy_term = csdl.sum(x_component*y_component*self.element_masses, axes=(0,1,2,3))
            xz_term = csdl.sum(x_component*z_component*self.element_masses, axes=(0,1,2,3))
            yz_term = csdl.sum(y_component*z_component*self.element_masses, axes=(0,1,2,3))

            term_01 = second_moment_of_mass[0,1] - xy_term
            term_02 = second_moment_of_mass[0,2] - xz_term
            term_12 = second_moment_of_mass[1,2] - yz_term

            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,1], term_01)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,0], term_01)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[0,2], term_02)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,0], term_02)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[1,2], term_12)
            second_moment_of_mass = second_moment_of_mass.set(csdl.slice[2,1], term_12)

            moment_of_inertia = second_moment_of_mass

        return mass, center_of_mass, moment_of_inertia, angular_momentum, change_in_angle_of_mass
        

    def apply_rigid_body_motion(self, rigid_body_states:csdl.Variable, geometry:lsdo_geo.Geometry=None, function_indices:list=None):
        '''
        Applies rigid body motion to the geometry of the body.

        Parameters
        ----------
        rigid_body_states : csdl.Variable
            The state of the body [x, y, z, theta_x, theta_y, theta_z].
        geometry : lsdo_geo.Geometry, optional
            The geometry to apply the rigid body motion to. If None, the geometry of the body is used. Default is None.
        function_indices : list[int], optional
            The indices of the functions to apply the rigid body motion to. If None, all functions are used. Default is None.
        '''
        # num_physical_dimensions = list(self.geometry.num_physical_dimensions.values())[0]
        # geometry_coefficients_reshaped = \
        #     self.geometry.coefficients.reshape((len(self.geometry.coefficients)//num_physical_dimensions, num_physical_dimensions))

        # current_center_of_mass = self.center_of_mass
        # points_centered_at_origin = geometry_coefficients_reshaped - current_center_of_mass
        # points_rotated = csdl.rotate_using_rotation_matrix(
        #     points_centered_at_origin, angle=rigid_body_states[2], cartesian_axis='z', units='radians')

        if geometry is None:
            geometry = self.geometry

        if function_indices is None:
            if (isinstance(geometry, lfs.FunctionSet)):
                function_indices = list(geometry.functions.keys())
                
        design_center_of_mass = self.design_center_of_mass

        self.center_of_mass = rigid_body_states[:3]
        if isinstance(geometry, lfs.FunctionSet):
            for function_index in function_indices:
                function = geometry.functions[function_index]
                function.coefficients = function.coefficients - csdl.expand(design_center_of_mass, function.coefficients.shape, 'i->jki')
                function.coefficients = function.coefficients + csdl.expand(self.center_of_mass, function.coefficients.shape, 'i->jki')
                # function.plot()

            geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5], function_indices=function_indices)
            geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4], function_indices=function_indices)
            geometry.rotate(axis_origin=self.center_of_mass, axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3], function_indices=function_indices)
        elif isinstance(geometry, lfs.Function):
            geometry.coefficients = geometry.coefficients - csdl.expand(design_center_of_mass, geometry.coefficients.shape, 'i->jkli')
            geometry.coefficients = geometry.coefficients + csdl.expand(self.center_of_mass, geometry.coefficients.shape, 'i->jkli')

            geometry_coefficients = geometry.coefficients.reshape((geometry.coefficients.size//3, 3))
            geometry_coefficients = lsdo_geo.rotate(geometry_coefficients, self.center_of_mass, axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5])
            geometry_coefficients = lsdo_geo.rotate(geometry_coefficients, self.center_of_mass, axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4])
            geometry_coefficients = lsdo_geo.rotate(geometry_coefficients, self.center_of_mass, axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3])
            geometry.coefficients = geometry_coefficients.reshape(geometry.coefficients.shape)
        
        # self.geometry.plot()


    def apply_flexible_motion(self, flexible_states:csdl.Variable, rigid_body_states:csdl.Variable=None, 
                              geometry:lsdo_geo.Geometry=None, function_indices:list=None):
        '''
        Applies flexible motion to the geometry of the body.

        Parameters
        ----------
        flexible_states : csdl.Variable -- shape=(num_nodes,6)
            The state of the body [x, y, z, theta_x, theta_y, theta_z] for each structural node.
        rigid_body_states : csdl.Variable = None, -- shape=(6,)
            The state of the body [x, y, z, theta_x, theta_y, theta_z]. This is used to transform flexible states back to global frame.
            If None, it is assumed there are no rigid body rotations.
        geometry : lsdo_geo.Geometry = None
            The geometry to apply the flexible motion to. If None, the geometry of the body is used. Default is None.
        function_indices : list[int], optional = None
            The indices of the functions to apply the flexible motion to. If None, all functions are used. Default is None.
        '''
        num_physical_dimensions = 3

        flexible_states = flexible_states.reshape((-1,6))

        displacements = flexible_states[:,:3]    # first 3 dof are displacements
        rotations = flexible_states[:,3:]        # last 3 dof are rotations

        if geometry is None:
            geometry = self.geometry

        if function_indices is None:
            if isinstance(geometry, lfs.FunctionSet):
                function_indices = list(geometry.functions.keys())

        # geometry = self.design_geometry.copy()

        # # unvectorized version
        # for function_index in function_indices:
        #     function = geometry.functions[function_index]
        #     function_coefficients_shape = function.coefficients.shape
        #     function_coefficients = function.coefficients
        #     function_coefficients_reshaped = function_coefficients.reshape((function_coefficients.size//num_physical_dimensions, num_physical_dimensions))
        #     local_displacements = csdl.matmat(self.beam_to_geometry_displacement_mapping[function_index], displacements)
        #     temp_displacements = lsdo_geo.rotate(local_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5])
        #     temp_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4])
        #     global_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3])
        #     # global_displacements = local_displacements
        #     function_coefficients_reshaped = function_coefficients_reshaped + global_displacements
        #     function.coefficients = function_coefficients_reshaped.reshape(function_coefficients_shape)

        # vectorized version
        if isinstance(geometry, lfs.FunctionSet):
            if len(function_indices) > 0:
                geometry_coefficients = []
                for function_index in function_indices:
                    function = geometry.functions[function_index]
                    function_coefficients_shape = function.coefficients.shape
                    function_coefficients = function.coefficients
                    function_coefficients_reshaped = function_coefficients.reshape((function_coefficients.size//num_physical_dimensions, num_physical_dimensions))
                    geometry_coefficients.append(function_coefficients_reshaped)
                geometry_coefficients = csdl.vstack(geometry_coefficients)
            else:
                function = list(geometry.functions.values())[0]
                geometry_coefficients = function.coefficients.reshape((function.coefficients.size//num_physical_dimensions, num_physical_dimensions))
        elif isinstance(geometry, lfs.Function):
            geometry_coefficients = geometry.coefficients.reshape((geometry.coefficients.size//num_physical_dimensions, num_physical_dimensions))

        # Geometry displacements from beam displacements
        # modified_displacements = csdl.Variable(shape=(20,3), value=0.)
        # beam_mesh = self.representations['beam'].evaluate(self.geometry)
        # actuator_length = csdl.norm(beam_mesh[2] - beam_mesh[1])
        # tail_length = csdl.norm(beam_mesh[8] - beam_mesh[7])
        # actuator_1_mid_displacement = (displacements[1]+displacements[2])/2 + (rotations[1] - rotations[2])*actuator_length/8
        # actuator_2_mid_displacement = (displacements[3]+displacements[4])/2 + (rotations[3] - rotations[4])*actuator_length/8
        # actuator_3_mid_displacement = (displacements[5]+displacements[6])/2 + (rotations[5] - rotations[6])*actuator_length/8
        # tail_mid_displacement = (displacements[7]+displacements[8])/2 + (rotations[7] - rotations[8])*tail_length/8
        # modified_displacements = modified_displacements.set(csdl.slice[0], displacements[0])    # head front
        # modified_displacements = modified_displacements.set(csdl.slice[1:3], displacements[1])  # head back (module 1 front)
        # modified_displacements = modified_displacements.set(csdl.slice[3], actuator_1_mid_displacement) # module 1 middle
        # modified_displacements = modified_displacements.set(csdl.slice[4:6], displacements[2])  # module 1 back
        # modified_displacements = modified_displacements.set(csdl.slice[6:8], displacements[3])  # module 2 front
        # modified_displacements = modified_displacements.set(csdl.slice[8], actuator_2_mid_displacement) # module 2 middle
        # modified_displacements = modified_displacements.set(csdl.slice[9:11], displacements[4]) # module 2 back
        # modified_displacements = modified_displacements.set(csdl.slice[11:13], displacements[5])# module 3 front
        # modified_displacements = modified_displacements.set(csdl.slice[13], actuator_3_mid_displacement) # module 3 middle
        # modified_displacements = modified_displacements.set(csdl.slice[14:16], displacements[6])# module 3 back
        # modified_displacements = modified_displacements.set(csdl.slice[16:18], displacements[7])# tail front
        # modified_displacements = modified_displacements.set(csdl.slice[18], tail_mid_displacement) # tail middle
        # modified_displacements = modified_displacements.set(csdl.slice[19], displacements[8])   # tail back
        local_displacements_from_displacements = csdl.matmat(self.beam_displacement_to_geometry_displacement_mapping, displacements)

        # beam_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(20,2))
        # fitting_values = csdl.Variable(shape=(displacements.shape[0]*2, 3), value=0.)
        # beam_b_spline_displacement_coefficients = beam_b_spline_space.fit(values=fitting_values)

        # # Geometry displacements from beam rotations
        # beam_rotations = csdl.matmat(self.beam_evaluation_maps.toarray(), rotations)

        # # # Unvectorized : Too much memory for Jax because of feedback stacking
        # # rotated_coordinates = self.wireframe_coordinates_with_respect_to_beam_coordinates
        # # for i in csdl.frange(beam_rotations.shape[0]):
        # #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=beam_rotations[i,2])
        # #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))
        # #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=beam_rotations[i,1])
        # #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))
        # #     rotated_point = lsdo_geo.rotate(rotated_coordinates[i], np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=beam_rotations[i,0])
        # #     rotated_coordinates = rotated_coordinates.set(csdl.slice[i], rotated_point.reshape((3,)))

        # cos_beam_rotations = csdl.cos(beam_rotations)
        # sin_beam_rotations = csdl.sin(beam_rotations)

        # rotation_tensor_z = csdl.Variable(value=np.zeros((beam_rotations.shape[0], 3, 3)))
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,0,0], cos_beam_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,0,1], -sin_beam_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,1,0], sin_beam_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,1,1], cos_beam_rotations[:,2])
        # rotation_tensor_z = rotation_tensor_z.set(csdl.slice[:,2,2], 1)

        # rotation_tensor_y = csdl.Variable(value=np.zeros((beam_rotations.shape[0], 3, 3)))
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,0,0], cos_beam_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,0,2], sin_beam_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,2,0], -sin_beam_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,2,2], cos_beam_rotations[:,1])
        # rotation_tensor_y = rotation_tensor_y.set(csdl.slice[:,1,1], 1)

        # rotation_tensor_x = csdl.Variable(value=np.zeros((beam_rotations.shape[0], 3, 3)))
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,1,1], cos_beam_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,1,2], -sin_beam_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,2,1], sin_beam_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,2,2], cos_beam_rotations[:,0])
        # rotation_tensor_x = rotation_tensor_x.set(csdl.slice[:,0,0], 1)

        # rotated_coordinates = csdl.einsum(rotation_tensor_z, self.wireframe_coordinates_with_respect_to_beam_coordinates, action='ijk,ik->ij')
        # rotated_coordinates = csdl.einsum(rotation_tensor_y, rotated_coordinates, action='ijk,ik->ij')
        # rotated_coordinates = csdl.einsum(rotation_tensor_x, rotated_coordinates, action='ijk,ik->ij')
        # # rotated_coordinates = csdl.einsum(rotation_tensor_x, self.wireframe_coordinates_with_respect_to_beam_coordinates, action='ijk,ik->ij')
         
        # wireframe_displacements = rotated_coordinates - self.wireframe_coordinates_with_respect_to_beam_coordinates

        # wireframe_counter = 0
        # coefficients_counter = 0
        # # rotation_displacements_y = csdl.Variable(value=np.zeros(geometry_coefficients.shape[:-1]))
        # # rotation_displacements_z = csdl.Variable(value=np.zeros(geometry_coefficients.shape[:-1]))
        # rotation_displacements = csdl.Variable(value=np.zeros(geometry_coefficients.shape))
        # for function_index in function_indices:
        #     # function_displacements_y = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_y[wireframe_counter:wireframe_counter+625])
        #     # function_displacements_z = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_z[wireframe_counter:wireframe_counter+625])
        #     # function_displacements = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements_from_theta_x[wireframe_counter:wireframe_counter+625])
        #     function_displacements = csdl.matmat(self.wireframe_to_geometry_displacement_mapping[function_index], wireframe_displacements[wireframe_counter:wireframe_counter+625])
        #     num_function_coefficients = geometry.functions[function_index].coefficients.size//num_physical_dimensions

        #     # rotation_displacements_y = rotation_displacements_y.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements_y)
        #     # rotation_displacements_z = rotation_displacements_z.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements_z)
        #     rotation_displacements = rotation_displacements.set(csdl.slice[coefficients_counter:coefficients_counter+num_function_coefficients], function_displacements)
        #     wireframe_counter += 625
        #     coefficients_counter += num_function_coefficients

        # local_displacements_from_rotations = rotation_displacements

        # local_displacements = local_displacements_from_displacements + local_displacements_from_rotations
        local_displacements = local_displacements_from_displacements

        if rigid_body_states is not None:
            temp_displacements = lsdo_geo.rotate(local_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 0., 1.]), angles=rigid_body_states[5])
            temp_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([0., 1., 0.]), angles=rigid_body_states[4])
            global_displacements = lsdo_geo.rotate(temp_displacements, np.array([0., 0., 0.]), axis_vector=np.array([1., 0., 0.]), angles=rigid_body_states[3])
        else:
            global_displacements = local_displacements
        
        counter = 0
        if isinstance(geometry, lfs.FunctionSet):
            for function_index in function_indices:
                # function_coefficient_displacements = global_displacements[counter:counter+function.coefficients.size//num_physical_dimensions]

                function = geometry.functions[function_index]
                function_coefficients_shape = function.coefficients.shape
                function_coefficients_reshaped = function.coefficients.reshape((function.coefficients.size//num_physical_dimensions, num_physical_dimensions))
                # function_coefficients_reshaped = function_coefficients_reshaped + function_coefficient_displacements
                function_coefficients_reshaped = function_coefficients_reshaped + global_displacements[counter:counter+function.coefficients.size//num_physical_dimensions]
                function.coefficients = function_coefficients_reshaped.reshape(function_coefficients_shape)

                counter += function.coefficients.size//num_physical_dimensions
        elif isinstance(geometry, lfs.Function):
            function_coefficients_reshaped = geometry.coefficients.reshape((geometry.coefficients.size//num_physical_dimensions, num_physical_dimensions))
            function_coefficients_reshaped = function_coefficients_reshaped + global_displacements
            geometry.coefficients = function_coefficients_reshaped.reshape(geometry.coefficients.shape)


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:Union[str,lfs.FunctionSet]='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        return self.geometry.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                  color_map=color_map, surface_texture=surface_texture, line_width=line_width,
                                  additional_plotting_elements=additional_plotting_elements, show=show)


@dataclass
class System:
    '''
    bodies : list[Body]
        The list of bodies in the system.
    constraint_pairs : list[tuple[np.ndarray,np.ndarray]]
        The list of constraint pairs in the system represented using their parametric coordinates.
    '''
    bodies: list[Body] = None
    constraint_pairs : list[tuple[np.ndarray,np.ndarray]] = None

    def __post_init__(self):
        if self.bodies is None:
            self.bodies = []
        if self.constraint_pairs is None:
            self.constraint_pairs = []

    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['function'],
              opacity:float=1., color:Union[str,lfs.FunctionSet]='#00629B', color_map:str='jet', surface_texture:str="",
              line_width:float=3., additional_plotting_elements:list=[], show:bool=True) -> list:
        
        plotting_elements = additional_plotting_elements.copy()
        for body in self.bodies:
            plotting_elements = body.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                              color_map=color_map, surface_texture=surface_texture, line_width=line_width,
                                              additional_plotting_elements=plotting_elements, show=False)
        if show:
            lfs.show_plot(plotting_elements=plotting_elements, title='System')


"""
Residuals
"""
class SerpentV1DynamicsModel:
    '''
    The residual model for the system.
    '''
    def __init__(self, system:System) -> None:
        '''
        Creates a residual model.

        Parameters
        ----------
        fish_system : System
            The n-body system.
        '''
        self.system = system
        
        self.num_bodies = len(system.bodies)

        self.time_stepping_order = 2

        self.body_states_at_n = {}
        self.body_state_derivatives_at_n = {}
        # NOTE: Lagrange multipliers are an automatically generated key in these dictionaries at the same level as body states.
        for body in self.system.bodies:
            num_rigid_body_states = 6 # 3D
            num_beam_nodes = body.representations['beam'].shape[0]
            num_flexible_states = num_beam_nodes*6  # 6 dof per node

            self.body_states_at_n[body.name] = {}
            self.body_states_at_n[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,), name=f'{body.name}_rigid_body_states_at_n')
            self.body_states_at_n[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,), name=f'{body.name}_flexible_states_at_n')

            self.body_state_derivatives_at_n[body.name] = {}
            self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states*self.time_stepping_order,),
                                                                                              name=f'{body.name}_rigid_body_state_derivatives_at_n')
            self.body_state_derivatives_at_n[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states*self.time_stepping_order,),
                                                                                            name=f'{body.name}_flexible_state_derivatives_at_n')



    def evaluate(self, initial_states:StateData, time:csdl.Variable,
                 generalized_alpha_spectral_radius:float,
                 actuator_loads:csdl.Variable,
                 external_fluid_load:csdl.Variable) -> dict[str,csdl.Variable]:
        '''
        Solves the MBD problem. Returns history of state data across time.

        Parameters
        ----------
        initial_states : StateData
            The initial states of the system.
        time : csdl.Variable
            The times at the which the simulation will be evaluated/simulated.
        generalized_alpha_spectral_radius : float=0.82
            The spectral radius for the generalized-alpha time discretization.
        '''
        # Evaluate mesh representations
        for body in self.system.bodies:
            for representation in body.representations.values():
                if isinstance(representation, lsdo_geo.Mesh):
                    representation.evaluate(body.geometry, plot=False)
                elif isinstance(representation, dict):
                    for mesh in representation.values():
                        if isinstance(mesh, lsdo_geo.Mesh):
                            mesh.evaluate(body.geometry, plot=False)

        self.time = time
        self.generalized_alpha_spectral_radius = generalized_alpha_spectral_radius

        # set initial states
        for body in self.system.bodies:
            self.body_states_at_n[body.name]['rigid_body_states'] = initial_states.states[body.name]['rigid_body_states']
            self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = initial_states.state_derivatives[body.name]['rigid_body_states']
            self.body_states_at_n[body.name]['flexible_states'] = initial_states.states[body.name]['flexible_states']
            self.body_state_derivatives_at_n[body.name]['flexible_states'] = initial_states.state_derivatives[body.name]['flexible_states']
        # self.lagrange_multipliers_at_n.value = initial_states.states['lagrange_multipliers']
        # self.lagrange_multiplier_derivatives_at_n.value = initial_states.state_derivatives['lagrange_multipliers']

        # Allocate n+1 states. These will be the design variables for the optimizer.
        self.body_states_at_n_plus_1 = {}
        for body in self.system.bodies:
            num_rigid_body_states = 6  # 3D
            num_beam_nodes = body.representations['beam'].shape[0]
            num_flexible_states = num_beam_nodes*6  # 6 dof per node

            self.body_states_at_n_plus_1[body.name] = {}
            self.body_states_at_n_plus_1[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,),
                                                                                         name=f'{body.name}_rigid_body_states_at_n_plus_1', value=0.)
            self.body_states_at_n_plus_1[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,),
                                                                                       name=f'{body.name}_flexible_states_at_n_plus_1', value=0.)
            
        # constraints model        
        # constraints = compute_constraints(states=self.body_states_at_n, lengths=self.lengths)
        # self.num_constraints = len(constraints) # NOTE: This is the number of physical constraints, not the length of the total vector

        # # lagrangian model
        # lagrangian = objective
        # # NOTE: Need to manually compute lagrangian because we need to parameterize lagrange multipliers using generalized alpha model

        # Preallocate states so that the variable can be replaced each time step, and the previous time step can be used as the initial value
        self.lagrange_multipliers_at_n = {}
        self.lagrange_multipliers_at_n['physical_constraints'] = []         # Constraints to keep the bodies from falling apart
        self.lagrange_multipliers_at_n['structural_constraints'] = []   # Constraints to keep the center of mass at the center of the geometry
        self.lagrange_multipliers_at_n_plus_1 = {}
        self.lagrange_multipliers_at_n_plus_1['physical_constraints'] = []         # Constraints to keep the bodies from falling apart
        self.lagrange_multipliers_at_n_plus_1['structural_constraints'] = []    # Constraints to keep the center of mass at the center of the geometry
        self.lagrange_multiplier_derivatives_at_n = {}
        self.lagrange_multiplier_derivatives_at_n['physical_constraints'] = []
        self.lagrange_multiplier_derivatives_at_n['structural_constraints'] = []
        for i in range(self.num_bodies):
            # self.lagrange_multipliers_at_n['physical_constraints'].append(csdl.Variable(shape=(3,), value=0., 
            #                                                name='preallocated_physical_translational_consraint_lagrange_multipliers'))
            # self.lagrange_multipliers_at_n['physical_constraints'].append(csdl.Variable(shape=(2,), value=0., 
            #                                                name='preallocated_physical_rotational_consraint_lagrange_multipliers'))
            # self.lagrange_multiplier_derivatives_at_n['physical_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
            #                                                                 name='preallocated_physical_translational_consraint_lagrange_multiplier_derivatives_at_n'))
            # self.lagrange_multiplier_derivatives_at_n['physical_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*2,), value=0.,
            #                                                                 name='preallocated_physical_rotational_consraint_lagrange_multiplier_derivatives_at_n'))
            
            # # This is the design variable so to speak (implicit variable)
            # self.lagrange_multipliers_at_n_plus_1['physical_constraints'].append(csdl.Variable(shape=(3,), value=0.,
            #                                                             name='preallocated_physical_translational_consraint_lagrange_multiplier_at_n_plus_1'))
            # self.lagrange_multipliers_at_n_plus_1['physical_constraints'].append(csdl.Variable(shape=(2,), value=0.,
            #                                                             name='preallocated_physical_rotational_consraint_lagrange_multiplier_at_n_plus_1'))
            
            self.lagrange_multipliers_at_n['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                              name='preallocated_structural_center_of_mass_lagrange_multipliers'))
            self.lagrange_multipliers_at_n['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                              name='preallocated_structural_angle_of_mass_lagrange_multipliers'))
            self.lagrange_multiplier_derivatives_at_n['structural_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
                                                                            name='preallocated_structural_center_of_mass_lagrange_multiplier_derivatives_at_n'))
            self.lagrange_multiplier_derivatives_at_n['structural_constraints'].append(csdl.Variable(shape=(self.time_stepping_order*3,), value=0.,
                                                                            name='preallocated_structural_angle_of_mass_lagrange_multiplier_derivatives_at_n'))
            
            # This is the design variable so to speak (implicit variable)
            self.lagrange_multipliers_at_n_plus_1['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                                        name='preallocated_center_of_mass_lagrange_multiplier_at_n_plus_1'))
            self.lagrange_multipliers_at_n_plus_1['structural_constraints'].append(csdl.Variable(shape=(3,), value=0.,
                                                                        name='preallocated_center_of_mass_lagrange_multiplier_at_n_plus_1'))
            
        state_history = {}
        state_derivative_history = {}

        # Allocate state history arrays
        for body in self.system.bodies:
            num_rigid_body_states = 6  # 3D
            num_flexible_states = num_beam_nodes*6  # 6 dof per node

            # state_history[body.name] = csdl.Variable(value=np.zeros((len(self.time)+1, num_rigid_body_states)))
            state_history[body.name] = {}
            state_history[body.name]['rigid_body_states'] = csdl.Variable(value=np.zeros((self.time.size, num_rigid_body_states)))
            state_history[body.name]['flexible_states'] = csdl.Variable(value=np.zeros((self.time.size, num_flexible_states)))

            num_rigid_body_state_derivatives = num_rigid_body_states*self.time_stepping_order
            num_flexible_state_derivatives = num_flexible_states*self.time_stepping_order
            # state_derivative_history[body.name] = csdl.Variable(value=np.zeros((len(self.time)+1, num_state_derivatives)))
            state_derivative_history[body.name] = {}
            state_derivative_history[body.name]['rigid_body_states'] = csdl.Variable(value=np.zeros((self.time.size, num_rigid_body_state_derivatives)))
            state_derivative_history[body.name]['flexible_states'] = csdl.Variable(value=np.zeros((self.time.size, num_flexible_state_derivatives)))

            state_history[body.name]['rigid_body_states'] = state_history[body.name]['rigid_body_states'].set(csdl.slice[0,:], initial_states.states[body.name]['rigid_body_states'])
            state_derivative_history[body.name]['rigid_body_states'] = state_derivative_history[body.name]['rigid_body_states'].set(csdl.slice[0,:], initial_states.state_derivatives[body.name]['rigid_body_states'])
            state_history[body.name]['flexible_states'] = state_history[body.name]['flexible_states'].set(csdl.slice[0,:], initial_states.states[body.name]['flexible_states'])
            state_derivative_history[body.name]['flexible_states'] = state_derivative_history[body.name]['flexible_states'].set(csdl.slice[0,:], initial_states.state_derivatives[body.name]['flexible_states'])

        # Allocate lagrange multipliers
        lagrange_multiplier_history = {}
        lagrange_multiplier_history['physical_constraints'] = []
        lagrange_multiplier_history['structural_constraints'] = []
        for i in range(self.num_bodies):
            # lagrange_multiplier_history['physical_constraints'].append(csdl.Variable(name=f'body_{i}_physical_translational_constraint_lagrange_multipliers',
            #                                                            value=np.zeros((self.time.size, 3))))
            # lagrange_multiplier_history['physical_constraints'].append(csdl.Variable(name=f'body_{i}_physical_alignment_constraint_lagrange_multipliers',
            #                                                                          value=np.zeros((self.time.size, 2))))
            lagrange_multiplier_history['structural_constraints'].append(csdl.Variable(name=f'body_{i}_structural_center_of_mass_constraint_lagrange_multipliers',
                                                                                       value=np.zeros((self.time.size, 3))))
            lagrange_multiplier_history['structural_constraints'].append(csdl.Variable(name=f'body_{i}_structural_angle_of_mass_constraint_lagrange_multipliers',
                                                                                       value=np.zeros((self.time.size, 3))))

        # Set initial values
        # for i, lagrange_multipliers in enumerate(self.lagrange_multipliers_at_n):
        #     lagrange_multiplier_history[i][0,:] = initial_states.states['lagrange_multipliers'][i].value

        
        # for i in range(self.time.size-1):
        #     print(i)
        for i in csdl.frange(0, self.time.size-1):
            # solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={'concatenate_ofs':True, 'loop':False})
            # solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={'concatenate_ofs':True, 'loop':True})
            # solver = csdl.nonlinear_solvers.Newton(tolerance=1e-6)
            solver = csdl.nonlinear_solvers.Newton()

            # if i == 20:
            #     print('hi')

            # Allocate n+1 states. These will be the design variables for the optimizer.
            for body in self.system.bodies:
                num_rigid_body_states = 6  # 3D
                num_flexible_states = num_beam_nodes*6  # 6 dof per node

                self.body_states_at_n_plus_1[body.name]['rigid_body_states'] = csdl.Variable(shape=(num_rigid_body_states,),
                                                                                             name=f'{body.name}_rigid_body_states_at_n_plus_1',
                                                                                             value=self.body_states_at_n[body.name]['rigid_body_states'].value)
                                                                                            #  value=0.)
                self.body_states_at_n_plus_1[body.name]['flexible_states'] = csdl.Variable(shape=(num_flexible_states,),
                                                                                           name=f'{body.name}_flexible_states_at_n_plus_1',
                                                                                           value=self.body_states_at_n[body.name]['flexible_states'].value)
                                                                                        #    value=0.)


            state_data = {}
            body_residuals = {}
            # Apply time discretization to states
            time_step = self.time[i+1] - self.time[i]
            for body in self.system.bodies:
                state_data[body.name] = {}
                generalized_alpha_model = GeneralizedAlphaModel(spectral_radius=self.generalized_alpha_spectral_radius, num_states=6)
                body_state_data = generalized_alpha_model.evaluate(states_at_n=self.body_states_at_n[body.name]['rigid_body_states'],
                                    states_at_n_plus_1=self.body_states_at_n_plus_1[body.name]['rigid_body_states'],
                                    state_velocities_at_n=self.body_state_derivatives_at_n[body.name]['rigid_body_states'][0:6],
                                    state_accelerations_at_n=self.body_state_derivatives_at_n[body.name]['rigid_body_states'][6:],
                                    time_step=time_step)
                state_data[body.name]['rigid_body_states'] = body_state_data

                # Apply time discretization to flexible states
                generalized_alpha_model = GeneralizedAlphaModel(spectral_radius=self.generalized_alpha_spectral_radius, num_states=6*num_beam_nodes)
                body_state_data = generalized_alpha_model.evaluate(states_at_n=self.body_states_at_n[body.name]['flexible_states'],
                                    states_at_n_plus_1=self.body_states_at_n_plus_1[body.name]['flexible_states'],
                                    state_velocities_at_n=self.body_state_derivatives_at_n[body.name]['flexible_states'][0:6*num_beam_nodes],
                                    state_accelerations_at_n=self.body_state_derivatives_at_n[body.name]['flexible_states'][6*num_beam_nodes:],
                                    time_step=time_step)
                state_data[body.name]['flexible_states'] = body_state_data

                # print(i)
                # print(state_data[body.name]['flexible_states'].states.value.reshape((num_beam_nodes,6))[:,2])

            
            # Evaluate updated geometry
            for body in self.system.bodies:
                # Start from original geometry each time
                body.geometry = body.design_geometry.copy()

                # Apply rigid body motion
                body.apply_rigid_body_motion(state_data[body.name]['rigid_body_states'].states)
                
                # Apply flexible motion
                body.apply_flexible_motion(state_data[body.name]['flexible_states'].states,
                                           state_data[body.name]['rigid_body_states'].states)
                
                
                # if i > 20:
                #     import matplotlib.pyplot as plt
                #     plt.plot(state_data[body.name]['flexible_states'].states.value.reshape((num_beam_nodes,6))[:,2])
                #     plt.show()

                #     deformed_beam = body.beam_mesh.value + \
                #         state_data[body.name]['flexible_states'].states.value.reshape((num_beam_nodes,6))[:,0:3] + \
                #         state_data[body.name]['rigid_body_states'].states.value[0:3]
                #     body.geometry.plot_meshes(meshes=[deformed_beam])

            # Evaluate rigid body and flexible residuals (Inertial terms and driving gravitational force)
            for body in self.system.bodies:
                body_state_data = state_data[body.name]

                # # residual
                # moment_of_inertia = body.evaluate_mass_properties(properties_to_compute=['moment_of_inertia'])[2]
                # mass_matrix = csdl.Variable(value=np.zeros((6,6)))
                # mass_matrix = mass_matrix.set(csdl.slice[0,0], body.mass)
                # mass_matrix = mass_matrix.set(csdl.slice[1,1], body.mass)
                # mass_matrix = mass_matrix.set(csdl.slice[2,2], body.mass)
                # mass_matrix = mass_matrix.set(csdl.slice[3:,3:], moment_of_inertia)

                # # Evaluate the rigid body dofs portion of the residual
                # inertial_term_model = ResidualInertialTerm()
                # inertial_term = inertial_term_model.evaluate(state_accelerations=body_state_data['rigid_body_states'].state_accelerations,
                #                                              mass_matrix=mass_matrix)

                # # rigid_body_gravity_residual_3d_model = RigidBodyGravityResidual3D(g=self.g)
                # # gravity_residual = rigid_body_gravity_residual_3d_model.evaluate(mass=body.mass)

                # rigid_body_residual = inertial_term# - external_fluid_load[i]

                F = actuator_loads[i].reshape((num_beam_nodes*6,))
                K, M, F = body.frame._boundary_conditions(body.K, body.M, F)
                D = body.D

                # # Evaluate the flexible portion of the residual
                U_dotdot = body_state_data['flexible_states'].state_accelerations
                U_dot = body_state_data['flexible_states'].state_velocities
                U = body_state_data['flexible_states'].states
                flexible_residual = csdl.matvec(M, U_dotdot) + csdl.matvec(D, U_dot) + csdl.matvec(K, U) - F
                # flexible_residual = csdl.matvec(K, U) - F

                # frame.solve(do_residual=True, U=body_state_data['flexible_states'].states, 
                #             U_dot=body_state_data['flexible_states'].state_velocities, 
                #             U_dotdot=body_state_data['flexible_states'].state_accelerations)
                # flexible_residual = frame.residual
                # flexible_residual = flexible_residual.set(csdl.slice[0:3], flexible_residual[0:3] + 10000*self.body_states_at_n_plus_1[body.name]['flexible_states'][0:3]) # TODO: This is a hack to get the optimizer to move the states
                # flexible_residual = flexible_residual.set(csdl.slice[3:6], flexible_residual[3:6] + 10000*self.body_states_at_n_plus_1[body.name]['flexible_states'][3:6]) # TODO: This is a hack to get the optimizer to move the states
                # flexible_residual = flexible_residual + 1.e-6*self.body_states_at_n_plus_1[body.name]['flexible_states']

                # body_residuals[body.name] = {'rigid_body_residual': rigid_body_residual, 'flexible_residual': flexible_residual}
                body_residuals[body.name] = {'flexible_residual': flexible_residual}

            # physical constraints model        
            # physical_constraints = compute_physical_constraints(system=self.system)

            # center of mass constraints
            # structural_constraints = compute_structural_constraints(system=self.system, state_data=state_data)

            lagrange_multiplier_data = {}
            # lagrange_multiplier_data['physical_constraints'] = []
            lagrange_multiplier_data['structural_constraints'] = []
            # Compute lagrange multipliers for the phyiscal constraints and add the lagrange multiplier term to the MBD system
            # body_residuals, solver = add_constraints_to_system(self, body_residuals, solver,
            #                                                             physical_constraints, 'physical_constraints', lagrange_multiplier_data)

            # Compute lagrange multipliers for the center of mass constraints and add the lagrange multiplier term to the MBD system
            # body_residuals, solver = add_constraints_to_system(self, body_residuals, solver, 
            #                                                             structural_constraints, 'structural_constraints', lagrange_multiplier_data,
            #                                                             time_step=time_step)

            # Add the state/residual pairs for the body residuals
            for body, states in self.body_states_at_n_plus_1.items():
                # solver.add_state(states['rigid_body_states'], body_residuals[body]['rigid_body_residual'], initial_value=self.body_states_at_n[body]['rigid_body_states'])
                solver.add_state(states['flexible_states'], body_residuals[body]['flexible_residual'], initial_value=self.body_states_at_n[body]['flexible_states'])

            solver.run()
            # F = np.zeros((num_beam_nodes*6,))
            # F[-4] = 1.
            # U = csdl.solve_linear(K, F)
            # self.body_states_at_n_plus_1['fishy']['flexible_states'] = U

            for body in self.system.bodies:
                state_history[body.name]['rigid_body_states'] = state_history[body.name]['rigid_body_states'].set(csdl.slice[i+1,:], self.body_states_at_n_plus_1[body.name]['rigid_body_states'])
                state_derivative_history[body.name]['rigid_body_states'] = state_derivative_history[body.name]['rigid_body_states'].set(csdl.slice[i,:], self.body_state_derivatives_at_n[body.name]['rigid_body_states'])
                state_history[body.name]['flexible_states'] = state_history[body.name]['flexible_states'].set(csdl.slice[i+1,:], self.body_states_at_n_plus_1[body.name]['flexible_states'])
                state_derivative_history[body.name]['flexible_states'] = state_derivative_history[body.name]['flexible_states'].set(csdl.slice[i,:], self.body_state_derivatives_at_n[body.name]['flexible_states'])

                # Save newly calculated n+1 states and derivatives as n for next time step
                self.body_states_at_n[body.name]['rigid_body_states'] = self.body_states_at_n_plus_1[body.name]['rigid_body_states']
                self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = \
                    self.body_state_derivatives_at_n[body.name]['rigid_body_states'].set(csdl.slice[:6],
                                                                        state_data[body.name]['rigid_body_states'].state_velocities_at_n_plus_1)
                self.body_state_derivatives_at_n[body.name]['rigid_body_states'] = \
                    self.body_state_derivatives_at_n[body.name]['rigid_body_states'].set(csdl.slice[6:],
                                                                            state_data[body.name]['rigid_body_states'].state_accelerations_at_n_plus_1)
                
                self.body_states_at_n[body.name]['flexible_states'] = self.body_states_at_n_plus_1[body.name]['flexible_states']
                self.body_state_derivatives_at_n[body.name]['flexible_states'] = \
                    self.body_state_derivatives_at_n[body.name]['flexible_states'].set(csdl.slice[:6*num_beam_nodes],
                                                                        state_data[body.name]['flexible_states'].state_velocities_at_n_plus_1)
                self.body_state_derivatives_at_n[body.name]['flexible_states'] = \
                    self.body_state_derivatives_at_n[body.name]['flexible_states'].set(csdl.slice[6*num_beam_nodes:],
                                                                            state_data[body.name]['flexible_states'].state_accelerations_at_n_plus_1)
                
            # self.lagrange_multipliers_at_n = self.lagrange_multipliers_at_n_plus_1.copy()
            self.lagrange_multipliers_at_n = {}

            # self.lagrange_multipliers_at_n['physical_constraints'] = self.lagrange_multipliers_at_n_plus_1['physical_constraints'].copy()
            # for j, constraint in enumerate(physical_constraints):
            #     # Save newly calculated n+1 lagrange multipliers and derivatives as n for next time step
            #     self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j] = \
            #         self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j].set(csdl.slice[:constraint.size],
            #                                                         lagrange_multiplier_data['physical_constraints'][j].state_velocities_at_n_plus_1)
            #     self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j] = \
            #         self.lagrange_multiplier_derivatives_at_n['physical_constraints'][j].set(csdl.slice[constraint.size:],
            #                                                         lagrange_multiplier_data['physical_constraints'][j].state_accelerations_at_n_plus_1)

            #     for j, constraint_lagrange_multiplier_data in enumerate(lagrange_multiplier_data['physical_constraints']):
            #         lagrange_multiplier_history['physical_constraints'][j] = lagrange_multiplier_history['physical_constraints'][j].set(
            #                                                                         csdl.slice[i+1,:], constraint_lagrange_multiplier_data.states)

            # self.lagrange_multipliers_at_n['structural_constraints'] = self.lagrange_multipliers_at_n_plus_1['structural_constraints'].copy()
            # for j, constraint in enumerate(structural_constraints):
            #     # Save newly calculated n+1 lagrange multipliers and derivatives as n for next time step
            #     self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j] = \
            #         self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j].set(csdl.slice[:constraint.size],
            #                                                         lagrange_multiplier_data['structural_constraints'][j].state_velocities_at_n_plus_1)
            #     self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j] = \
            #         self.lagrange_multiplier_derivatives_at_n['structural_constraints'][j].set(csdl.slice[constraint.size:],
            #                                                         lagrange_multiplier_data['structural_constraints'][j].state_accelerations_at_n_plus_1)

            #     for j, constraint_lagrange_multiplier_data in enumerate(lagrange_multiplier_data['structural_constraints']):
            #         lagrange_multiplier_history['structural_constraints'][j] = lagrange_multiplier_history['structural_constraints'][j].set(
            #                                                                         csdl.slice[i+1,:], constraint_lagrange_multiplier_data.states)


        return self.time, state_history, state_derivative_history, lagrange_multiplier_history



def evaluate_actuator_loads(time:csdl.Variable, actuation_frequency:csdl.Variable, max_pressure:csdl.Variable, cross_section_mesh:csdl.Variable, 
                         generalized_alpha_spectral_radius:float, num_actuators=3) -> csdl.Variable:
    '''
    Evaluates the fishy loads applied to the system.
    '''
    phase_offset = 2*np.pi/num_actuators
    actuator_1_loads = csdl.cos(2*np.pi*actuation_frequency*time)
    actuator_2_loads = csdl.cos(2*np.pi*actuation_frequency*time - phase_offset)
    actuator_3_loads = csdl.cos(2*np.pi*actuation_frequency*time - 2*phase_offset)

    # Scale by moment produced (geometry dependent)
    num_sections_width = cross_section_mesh.shape[1] - 1
    cross_section_mesh = cross_section_mesh[:,num_sections_width//2:]       # Only consider half of the cross section
    u_vectors = cross_section_mesh[1:, :] - cross_section_mesh[:-1, :]
    v_vectors = cross_section_mesh[:, 1:] - cross_section_mesh[:, :-1]
    u_vectors_0 = u_vectors[:, :-1]
    u_vectors_1 = u_vectors[:, 1:]
    v_vectors_0 = v_vectors[:-1, :]
    v_vectors_1 = v_vectors[1:, :]
    areas_00 = csdl.cross(u_vectors_0, v_vectors_0, axis=2)
    areas_11 = csdl.cross(u_vectors_1, v_vectors_1, axis=2)
    areas = (areas_00 + areas_11) / 2
    areas = csdl.norm(areas, axes=(2,))
    total_area = csdl.sum(areas)        # Total area of the half-cross section

    cross_section_mesh_centers = (cross_section_mesh[1:, 1:] + cross_section_mesh[:-1, :-1]) / 2
    centroid = csdl.sum(cross_section_mesh_centers * csdl.expand(areas, areas.shape+(3,), 'ij->ijk'), axes=(0,1)) / total_area
    moment_arm = centroid[2]  # NOTE: This is the z component of the centroid because that is actuation direction

    max_moment = max_pressure * total_area * moment_arm * 2 # same moment for pressurized side and vacuum side

    # Loads at n, n+1, ...
    actuator_1_loads_at_n = actuator_1_loads * max_moment
    actuator_2_loads_at_n = actuator_2_loads * max_moment
    actuator_3_loads_at_n = actuator_3_loads * max_moment

    # Load derivatives at n, n+1, ...
    actuator_1_load_velocities = -2*np.pi*actuation_frequency*csdl.sin(2*np.pi*actuation_frequency*time)*max_moment
    actuator_2_load_velocities = -2*np.pi*actuation_frequency*csdl.sin(2*np.pi*actuation_frequency*time - phase_offset)*max_moment
    actuator_3_load_velocities = -2*np.pi*actuation_frequency*csdl.sin(2*np.pi*actuation_frequency*time - 2*phase_offset)*max_moment
    actuator_1_load_accelerations = -4*np.pi**2*actuation_frequency**2*csdl.cos(2*np.pi*actuation_frequency*time)*max_moment
    actuator_2_load_accelerations = -4*np.pi**2*actuation_frequency**2*csdl.cos(2*np.pi*actuation_frequency*time - phase_offset)*max_moment
    actuator_3_load_accelerations = -4*np.pi**2*actuation_frequency**2*csdl.cos(2*np.pi*actuation_frequency*time - 2*phase_offset)*max_moment

    # Apply generalized alpha time discretization
    actuator_loads_model = GeneralizedAlphaModel(spectral_radius=generalized_alpha_spectral_radius, num_states=1)
    actuator_1_loads = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    actuator_2_loads = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    actuator_3_loads = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    actuator_1_load_velocities = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    actuator_2_load_velocities = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    actuator_3_load_velocities = csdl.Variable(shape=(time.size-1,), value=-987654321.)
    for i in csdl.frange(time.size-1):
        actuator_1_loads_data = actuator_loads_model.evaluate(actuator_1_loads_at_n[i], actuator_1_loads_at_n[i+1], actuator_1_load_velocities[i], actuator_1_load_accelerations[i], 
                                                       time_step=time[i+1]-time[i])
        actuator_1_loads = actuator_1_loads.set(csdl.slice[i], actuator_1_loads_data.states)

        actuator_2_loads_data = actuator_loads_model.evaluate(actuator_2_loads_at_n[i], actuator_2_loads_at_n[i+1], actuator_2_load_velocities[i], actuator_2_load_accelerations[i],
                                                         time_step=time[i+1]-time[i])
        actuator_2_loads = actuator_2_loads.set(csdl.slice[i], actuator_2_loads_data.states)

        actuator_3_loads_data = actuator_loads_model.evaluate(actuator_3_loads_at_n[i], actuator_3_loads_at_n[i+1], actuator_3_load_velocities[i], actuator_3_load_accelerations[i],
                                                            time_step=time[i+1]-time[i])
        actuator_3_loads = actuator_3_loads.set(csdl.slice[i], actuator_3_loads_data.states)

        actuator_1_load_velocities = actuator_1_load_velocities.set(csdl.slice[i], actuator_1_loads_data.state_velocities)
        actuator_2_load_velocities = actuator_2_load_velocities.set(csdl.slice[i], actuator_2_loads_data.state_velocities)
        actuator_3_load_velocities = actuator_3_load_velocities.set(csdl.slice[i], actuator_3_loads_data.state_velocities)

    # Hardcoding mapping to beam nodes
    beam_loads = csdl.Variable(shape=(time.size-1, 14, 6), value=0.) # hardcoded for 9 beam nodes (num_time_steps, num_beam_nodes, 6)
    beam_loads = beam_loads.set(csdl.slice[:,1,4], actuator_1_loads)      # M_y is 4th load, beam nodes 1,2 correspond to actuator 1
    beam_loads = beam_loads.set(csdl.slice[:,3,4], -actuator_1_loads)      # M_y is 4th load, beam nodes 1,2 correspond to actuator 1
    beam_loads = beam_loads.set(csdl.slice[:,4,4], 1.3*actuator_2_loads)
    beam_loads = beam_loads.set(csdl.slice[:,6,4], -1.3*actuator_2_loads)
    beam_loads = beam_loads.set(csdl.slice[:,7,4], 1.5*actuator_3_loads)
    beam_loads = beam_loads.set(csdl.slice[:,9,4], -1.5*actuator_3_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,4,4], 3*actuator_2_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,6,4], -3*actuator_2_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,7,4], 5*actuator_3_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,9,4], -5*actuator_3_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,4,4], actuator_2_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,6,4], -actuator_2_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,7,4], actuator_3_loads)
    # beam_loads = beam_loads.set(csdl.slice[:,9,4], -actuator_3_loads)

    # build_up_pressure = csdl.Variable(value=(-np.cos(1/2*2*np.pi*time.value[:-1]/time.value[-2]) + 1/2)*max_moment.value)
    # import matplotlib.pyplot as plt
    # plt.plot(time.value[:-1], build_up_pressure.value)
    # plt.show()
    # constant_pressure = csdl.Variable(value=max_moment.value, shape=(time.size-1,))
    # beam_loads = beam_loads.set(csdl.slice[:,8,4], constant_pressure)
    # beam_loads = beam_loads.set(csdl.slice[:,4,4], -constant_pressure)

    beam_load_velocities = csdl.Variable(shape=(time.size-1, 14, 6), value=0.)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,1,4], actuator_1_load_velocities)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,3,4], -actuator_1_load_velocities)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,4,4], 1.3*actuator_2_load_velocities)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,6,4], -1.3*actuator_2_load_velocities)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,7,4], 1.5*actuator_3_load_velocities)
    beam_load_velocities = beam_load_velocities.set(csdl.slice[:,9,4], -1.5*actuator_3_load_velocities)

    return beam_loads, beam_load_velocities




def compute_physical_constraints(system:System) -> list[csdl.Variable]:
    '''
    Computes the body-to-body physical constraints for the system.
    '''
    # num_constraints_per_body = 5
    constraints = []
    for i, body in enumerate(system.bodies):
        constraint_pair = system.constraint_pairs[i]
        # constraint = csdl.Variable(shape=(num_constraints_per_body,), value=0.)
        if i == 0:
            translational_constraint = body.geometry.evaluate(constraint_pair[0]) - constraint_pair[1]
            
            alignment_axis = body.geometry.evaluate(constraint_pair[0], parametric_derivative_orders=(1,0))
            normalized_alignment_axis = alignment_axis/csdl.norm(alignment_axis)
            # alignment_constraint = normalized_alignment_axis[:2] - np.array([1., 0.])   # NOTE: Hardcoding rotation about x axis
            alignment_constraint = normalized_alignment_axis[1:] - np.array([0., 0.])   # NOTE: Hardcoding rotation about x axis
            # NOTE: For alignment constraint, we only want 2 constraints (6 dof, 5 constraints per body). I think this does it.
        else:
            translational_constraint = body.geometry.evaluate(constraint_pair[0]) - \
                                        system.bodies[i-1].geometry.evaluate(constraint_pair[1])
            alignment_axis = body.geometry.evaluate(constraint_pair[0], parametric_derivative_orders=(1,0))
            normalized_alignment_axis = alignment_axis/csdl.norm(alignment_axis)
            # alignment_constraint = normalized_alignment_axis[:2] - np.array([1., 0.])   # NOTE: Hardcoding rotation about x axis
            alignment_constraint = normalized_alignment_axis[1:] - np.array([0., 0.])   # NOTE: Hardcoding rotation about x axis
            # NOTE: For alignment constraint, we only want 2 constraints (6 dof, 5 constraints per body). I think this does it.
        
        # constraint = csdl.vstack([translational_constraint, alignment_constraint])
        # constraint = constraint.set(csdl.slice[:3], translational_constraint)
        # constraint = constraint.set(csdl.slice[3:], alignment_constraint)
        # constraints.append(constraint)
        translational_constraint.add_name(f'{body.name}_translational_physical_constraint')
        alignment_constraint.add_name(f'{body.name}_alignment_physical_constraint')
        constraints.append(translational_constraint)
        constraints.append(alignment_constraint)

    return constraints


def compute_structural_constraints(system:System, state_data:GeneralizedAlphaStateData) -> list[csdl.Variable]:
    '''
    Computes the difference between the state center of mass (rigid body state for each body) and the center of mass as
    defined by the geometry.
    '''
    constraints = []

    for body in system.bodies:
        # local_pendulum = body.copy()
        design_geometry = body.design_geometry.copy()

        body.apply_flexible_motion(flexible_states=state_data[body.name]['flexible_states'].states,
                                             geometry=design_geometry)
        deformed_geometry = design_geometry

        _, center_of_mass, _, _, change_in_angle_of_mass = body.evaluate_mass_properties(geometry=deformed_geometry,
                                                                                        properties_to_compute=['center_of_mass', 'change_in_angle_of_mass'])

        constraint = center_of_mass - body.design_center_of_mass
        constraint.add_name(f'{body.name}_center_of_mass_constraint')
        constraints.append(constraint)

        # angle_of_mass.add_name(f'{body.name}_angle_of_mass')
        # constraint = angle_of_mass - body.design_angle_of_mass
        # constraint.add_name(f'{body.name}_angle_of_mass_constraint')
        # constraints.append(constraint)

        # angular_momentum.add_name(f'{body.name}_angular_momentum')
        # constraint = angular_momentum # Initial "angular momentum" is always 0 because the positions/velocities are relative to the design geometry
        # constraint.add_name(f'{body.name}_angular_momentum_constraint')
        # constraints.append(constraint)

        change_in_angle_of_mass.add_name(f'{body.name}_change_in_angle_of_mass')
        constraint = change_in_angle_of_mass # Initial "angular momentum" is always 0 because the positions/velocities are relative to the design geometry
        constraint.add_name(f'{body.name}_change_in_angle_of_mass_constraint')
        constraints.append(constraint)

    return constraints



def add_constraints_to_system(model:SerpentV1DynamicsModel, body_residuals:dict[str,dict[str,csdl.Variable]], solver:csdl.nonlinear_solvers.Newton, 
                                       constraints:list[csdl.Variable], constraints_name:str, lagrange_multiplier_data:dict[str,list[GeneralizedAlphaStateData]],
                                       time_step:csdl.Variable) \
                                        -> tuple[dict[str,dict[str,csdl.Variable]], csdl.nonlinear_solvers.Newton]:
    for j, constraint in enumerate(constraints):
        model.lagrange_multipliers_at_n_plus_1[constraints_name][j] = csdl.Variable(shape=(constraint.size,), value=0.,
                                                                name=constraints_name+'_lagrange_multiplier_at_n_plus_1')

        generalized_alpha_lagrange_multipliers = GeneralizedAlphaModel(spectral_radius=model.generalized_alpha_spectral_radius, num_states=constraint.size)
        constraint_lagrange_multiplier_data = generalized_alpha_lagrange_multipliers.evaluate(states_at_n=model.lagrange_multipliers_at_n[constraints_name][j],
                                states_at_n_plus_1=model.lagrange_multipliers_at_n_plus_1[constraints_name][j],
                                state_velocities_at_n=model.lagrange_multiplier_derivatives_at_n[constraints_name][j][0:constraint.size],
                                state_accelerations_at_n=model.lagrange_multiplier_derivatives_at_n[constraints_name][j][constraint.size:],
                                time_step=time_step)
        
        lagrange_multiplier_data[constraints_name].append(constraint_lagrange_multiplier_data)
    
        constraint_lagrange_multipliers = constraint_lagrange_multiplier_data.states

        # body_residuals = add_lagrange_multiplier_term_to_residual(body_residuals, constraint, constraint_lagrange_multipliers,
        #                                                           model.body_states_at_n_plus_1, 'rigid_body')
        body_residuals = add_lagrange_multiplier_term_to_residual(body_residuals, constraint, constraint_lagrange_multipliers,
                                                                    model.body_states_at_n_plus_1, 'flexible')

        # Add the state/residual pairs for the constraints
        solver.add_state(model.lagrange_multipliers_at_n_plus_1[constraints_name][j], constraint, initial_value=model.lagrange_multipliers_at_n[constraints_name][j])

    return body_residuals, solver
    

def add_lagrange_multiplier_term_to_residual(body_residuals:dict[str,csdl.Variable], constraint:csdl.Variable, lagrange_multipliers:csdl.Variable,
                                             body_states:dict[str,csdl.Variable], states_name:str) -> dict[str,csdl.Variable]:
    '''
    Adds the dc_dx*lambda term to the body residuals.
    '''
    for body, states in body_states.items():
        current_graph = csdl.get_current_recorder().active_graph
        vecmat = csdl.src.operations.derivatives.reverse.vjp(
            [(constraint,lagrange_multipliers)],
            states[f'{states_name}_states'],
            current_graph,
        )[states[f'{states_name}_states']]
        if vecmat is not None:
            body_residuals[body][f'{states_name}_residual'] = body_residuals[body][f'{states_name}_residual'] + vecmat

    return body_residuals
