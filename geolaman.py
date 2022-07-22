import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
import pygmsh
import gmsh
import meshio
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace,
                         VectorFunctionSpace,TensorFunctionSpace,assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import (LinearProblem, assemble_matrix,assemble_vector,
                                apply_lifting,set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from dolfinx import plot
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import ufl

class point:
    def __init__(self,label,coordinate):
        self.label = label
        self.coordinate = coordinate

class boundary:
    def __init__(self,label,points,fixed_direction=None):
        self.label = label
        self.points = points
        self.mark = None
        self.fixed_direction = fixed_direction

class region:
    def __init__(self,label,density,Young_modulus,Poisson_ratio,
                 cohesion,friction_angle,dilatancy_angle,
                 transition_angle,tension_cutoff_parameter,
                 conductivity,porosity,boundary_points,mesh_size):

        self.label = label
        self.density = density
        self.Young_modulus = Young_modulus                              #elastic model
        self.Poisson_ratio = Poisson_ratio                              #elastic model
        self.cohesion = cohesion                                        #Mohr-Coulomb model
        self.friction_angle = friction_angle                            #Mohr-Coulomb model
        self.dilatancy_angle = dilatancy_angle                          #Mohr-Coulomb model
        self.transition_angle = transition_angle                        #Mohr-Coulomb model
        self.tension_cutoff_parameter = tension_cutoff_parameter        #Mohr-Coulomb model
        self.conductivity = conductivity                                #u-p model
        self.porosity = porosity                                        #u-p model
        self.boundary_points = boundary_points
        self.mesh_size = mesh_size
        self.mark = None

def create_arc_points(label,center,p1,p2,number_of_points):

    radius = np.sqrt((p1.coordinate[0]-center[0])**2+(p1.coordinate[1]-center[1])**2)
    xs = np.linspace(p1.coordinate[0],p2.coordinate[0],2+number_of_points)
    points = []
    for i in range(1,len(xs)-1):
        y = center[1] - np.sqrt(radius**2 - (xs[i]-center[0])**2)
        points.append(point(label+str(i),[xs[i],y]))

    return points

    '''
    #### NO LONGER NEEDED?
    def get_vertex_index_to_cell_indices(self,mesh,subdomains):

        cells_indices = np.where(subdomains.array()==self.mark)[0]
        vertex_index_to_cell_indices = {}
        vertices_indices = []
        for cell_index in cells_indices:
            entity = MeshEntity(mesh,2,cell_index)
            vertices_indices_in_cell = entity.entities(0)
            for vertex_index in vertices_indices_in_cell:
                vertices_indices.append(vertex_index)
                if vertex_index not in vertex_index_to_cell_indices:
                    vertex_index_to_cell_indices.update({vertex_index:[cell_index]})
                else:
                    value = vertex_index_to_cell_indices[vertex_index]
                    vertex_index_to_cell_indices[vertex_index] = value + [cell_index]

        self.vertex_index_to_cell_indices = vertex_index_to_cell_indices
    '''
class column:

    def __init__(self,label,x,depth,segment):

        self.label = label
        self.x = x
        self.depth = depth
        self.segment = segment
        self.nodes = None

    '''
    def get_vertices(self,boundary):


        cell_indices = np.where(self.mesh_objects["boundaries"].array()==boundary.mark)[0]
        vertices = []
        for cell_index in cell_indices:
            line = MeshEntity(self.mesh_objects["mesh"],1,cell_index).entities(0)
            for index in line:
                vertices.append(index)
        return list(set(vertices))

    def initialize_nodes(self):

        vertices = self.get_vertices(self.boundaries["surface"])
        coordinates_x = np.transpose(self.mesh_objects["mesh"].coordinates()[vertices])[0]
        dx = [abs(xp-self.x) for xp in coordinates_x]
        index = vertices[np.argmin(dx)]
        vertex_index_to_cell_indices = {}
        for region in self.regions:
            vertex_index_to_cell_indices.update(region.vertex_index_to_cell_indices)
        all_vertex_indices = list(vertex_index_to_cell_indices.keys())
        all_vertex_positions = self.mesh_objects["mesh"].coordinates()[all_vertex_indices]

        nodes = [index]
        x0,y0 = self.mesh_objects["mesh"].coordinates()[index]
        self.x = x0
        y = y0
        counter, counter_max = 0, 10*int(self.depth/self.segment)
        while y >= y0-self.depth:
            position = [x0,self.mesh_objects["mesh"].coordinates()[nodes[-1]][1]-self.segment]
            dist_squared = np.sum((np.asarray(all_vertex_positions) - position)**2,axis=1)
            index = all_vertex_indices[np.argmin(dist_squared)]
            y = self.mesh_objects["mesh"].coordinates()[index][1]
            nodes.append(index)
            counter += 1
            if counter > counter_max:
                print("... ",self.label," cannot be created with the given parameters.")
                quit()


        self.nodes = nodes
        self.nodal_displacements = {}
        self.nodal_pore_pressures = {}

    '''
class system:

    def __init__(self):

        self.regions = None
        self.boundaries = None
        self.zero_displacement_boundaries = None
        self.pore_pressure_boundaries = None
        self.data_columns = []
        #self.max_displacements = {}

        self.times = None
        self.dt = None
        self.trigger = None
        self.trigger_params = None
        self.gravity_ramp = 10

        self.g = 9.8
        self.water_density = 1000
        self.water_bulk_modulus = 1.96e9

        self.with_surface_load = False
        self.surface_load_force = None
        self.load_surface = None

        self.path_label = None
        self.timestamps = None

    def create_mesh(self):

        self.create_data_path()

        #check if surface boundary is defined
        if "surface" not in [b.label for b in self.boundaries]:
            print("No surface boundary (labelled 'surface') found. Create one to proceed.")
            quit()

        print("Creating mesh ...")
        geom = pygmsh.occ.Geometry()
        model = geom.__enter__()

        points_done = {}
        curves_done = {}
        planes = []
        mark = 1
        for region in self.regions:
            N = len(region.boundary_points)
            curves = []
            for i in range(-1,N-1):
                label = region.boundary_points[i].label+"-"+region.boundary_points[i+1].label
                #print(region.label,label)
                if label not in curves_done.keys():
                    #make endpoints of curve
                    if region.boundary_points[i].label not in points_done.keys():
                        p = model.add_point(region.boundary_points[i].coordinate,region.mesh_size)
                        points_done.update({region.boundary_points[i].label:p})
                    else:
                        p = points_done[region.boundary_points[i].label]

                    if region.boundary_points[i+1].label not in points_done.keys():
                        q = model.add_point(region.boundary_points[i+1].coordinate,region.mesh_size)
                        points_done.update({region.boundary_points[i+1].label:q})
                    else:
                        q = points_done[region.boundary_points[i+1].label]

                    curve = model.add_line(p,q)
                    curves_done.update({label:curve})
                    curves.append(curve)
                else:
                    curves.append(curves_done[label])

            plane = model.add_plane_surface(model.add_curve_loop(curves))
            model.add_physical([plane],region.label)
            planes.append([plane])
            region.mark = mark
            mark += 1

            print("Mesh region for",region.label,": done ...")

        for boundary in self.boundaries:
            segments = []
            N = len(boundary.points)
            for i in range(N-1):
                label = boundary.points[i].label+"-"+boundary.points[i+1].label
                if label not in curves_done.keys():
                    label = boundary.points[i+1].label+"-"+boundary.points[i].label #opposite
                segments.append(curves_done[label])
            model.add_physical(segments,boundary.label)
            boundary.mark = mark
            mark += 1
            print("Marking",boundary.label,": done ...")

        if len(self.regions) > 1:
            for i in range(len(planes)-1):
                #print(i)
                model.boolean_fragments(planes[i],planes[i+1])

        model.synchronize()
        geom.generate_mesh()
        gmsh.write(os.path.join(self.files_path,"mesh.msh"))
        gmsh.clear()
        geom.__exit__()

        '''The following is based on
        https://jorgensd.github.io/dolfinx-tutorial/chapter3/subdomains.html#read-in-msh-files-with-dolfinx
        '''

        msh = meshio.read(os.path.join(self.files_path,"mesh.msh"))

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = self.create_meshio_mesh(msh, "triangle", prune_z=True)
        line_mesh = self.create_meshio_mesh(msh, "line", prune_z=True)
        meshio.write(os.path.join(self.files_path,"mesh.xdmf"), triangle_mesh)
        meshio.write(os.path.join(self.files_path,"facet.xdmf"), line_mesh)

        with XDMFFile(MPI.COMM_WORLD, os.path.join(self.files_path,"mesh.xdmf"), "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")
            self.mesh_subdomains = xdmf.read_meshtags(self.mesh, name="Grid")
            self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        with XDMFFile(MPI.COMM_WORLD, os.path.join(self.files_path,"facet.xdmf"), "r") as xdmf:
            self.mesh_boundaries = xdmf.read_meshtags(self.mesh, name="Grid")

        '''
        mesh_from_file = meshio.read(os.path.join(self.files_path,"mesh.msh"))
        mesh = converts_to_xdmf_file(mesh_from_file, "triangle", False, prune_z=True,)
        meshio.write(os.path.join(self.files_path,"mesh.xdmf"), mesh)
        triangle_mesh = converts_to_xdmf_file(mesh_from_file, "triangle", True, prune_z=True,)
        meshio.write(os.path.join(self.files_path,"physical_region.xdmf"), triangle_mesh)
        line_mesh = converts_to_xdmf_file(mesh_from_file, "line", True, prune_z=True)
        meshio.write(os.path.join(self.files_path,"facet_mesh.xdmf"), line_mesh)

        mesh = Mesh()
        with XDMFFile(os.path.join(self.files_path,"mesh.xdmf")) as infile:
            infile.read(mesh)
        mvc = MeshValueCollection("size_t", mesh, 2)
        with XDMFFile(os.path.join(self.files_path,"facet_mesh.xdmf")) as infile:
            infile.read(mvc, "name_to_read")
        facet_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc)
        with XDMFFile(os.path.join(self.files_path,"physical_region.xdmf")) as infile:
            infile.read(mvc, "name_to_read")
        physical_region = cpp.mesh.MeshFunctionSizet(mesh, mvc)

        File(os.path.join(self.files_path,"mesh.xml")).write(mesh)
        File(os.path.join(self.files_path,"facet_mesh.xml")).write(facet_mesh)
        File(os.path.join(self.files_path,"physical_region.xml")).write(physical_region)

        self.mesh = Mesh(os.path.join(self.files_path,"mesh.xml"))
        self.mesh_subdomains = MeshFunction('size_t', self.mesh, os.path.join(self.files_path,"physical_region.xml"))
        self.mesh_boundaries = MeshFunction('size_t', self.mesh, os.path.join(self.files_path,"facet_mesh.xml"))
        self.dx = Measure("dx",domain=self.mesh,subdomain_data=self.mesh_subdomains)
        self.mesh_objects = {"mesh":self.mesh,"subdomains":self.mesh_subdomains,
                             "boundaries":self.mesh_boundaries}

        #create vertex index to cell index dictionary
        for region in self.regions:
            region.get_vertex_index_to_cell_indices(self.mesh,self.mesh_subdomains)

        print("number of vertices = ",self.mesh.num_vertices())
        print("number of cells = ",self.mesh.num_cells())


        '''
        boundaries = self.boundaries
        self.boundaries = {}
        for boundary in boundaries:
            self.boundaries.update({boundary.label:boundary})
        '''

        self.mark_to_region = {}
        for region in self.regions:
            self.mark_to_region.update({region.mark:region})
        '''

        #creating cell to vertices connection csv file
        self.mesh.topology.create_connectivity(0,self.mesh.topology.dim)
        cell_to_vertices = self.mesh.topology.connectivity(self.mesh.topology.dim,0)
        cell_to_vertices_array = []
        for i in range(len(cell_to_vertices)):
            cell_to_vertices_array.append(cell_to_vertices.links(i))
        cell_to_vertices_array = np.transpose(cell_to_vertices_array)
        df_cell_to_vertices = pd.DataFrame.from_dict({"v1":cell_to_vertices_array[0],
                                                        "v2":cell_to_vertices_array[1],
                                                        "v3":cell_to_vertices_array[2]})
        df_cell_to_vertices.to_csv(os.path.join(self.outputs_path,"cell_to_vertices.csv"))

    def create_data_path(self):

        if self.path_label == None:
            self.path_label = ""

        cwd = os.getcwd()
        self.outputs_path = os.path.join(cwd,"outputs_"+self.path_label)
        self.files_path = os.path.join(cwd,"files_"+self.path_label)
        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)
        if not os.path.exists(self.files_path):
            os.mkdir(self.files_path)

    def create_meshio_mesh(self,mesh, cell_type, prune_z=False):

        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

    def get_vertices(self,boundary):

        cell_indices = self.mesh_boundaries.indices[self.mesh_boundaries.values == boundary.mark]
        vertices = []

    #def initialize_column_nodes(self,surface_vertices):

    def create_columns(self,list_of_column_params):

        print(self.mesh.topology.dim-1)
        #surface_vertices = locate_entities(self.mesh,self.mesh.topology.dim-1,self.boundaries["surface"].mark)
        #print(surface_vertices)
        quit()
        surface_vertices = self.get_vertices(self.boundaries["surface"])

        for column_params in list_of_column_params:
            column_ = column(*column_params)
            print("Creating ",column_.label," ...")
            column_.nodes = self.initialize_column_nodes(surface_vertices)

    def project(self, v, target_func, bcs=[]):
        '''this function is from https://github.com/michalhabera/dolfiny/blob/master/dolfiny/projection.py#L7'''

        # Ensure we have a mesh and attach to measure
        V = target_func.function_space
        dx = ufl.dx(V.mesh)

        # Define variational problem for projection
        w = ufl.TestFunction(V)
        Pv = ufl.TrialFunction(V)
        a = form(ufl.inner(Pv, w) * dx)
        L = form(ufl.inner(v, w) * dx)

        # Assemble linear system
        A = assemble_matrix(a, bcs)
        A.assemble()
        b = assemble_vector(L)
        apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        # Solve linear system
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        solver.solve(b, target_func.vector)

    def epsilon(self,u):
        return ufl.sym(ufl.grad(u))

    def sigma(self,u):
        return self.Lame_lambda*ufl.nabla_div(u)*ufl.Identity(u.geometric_dimension()) + 2*self.Lame_mu*self.epsilon(u)

    def run(self):

        #parameters
        print("Initializing material parameters...")
        P = FunctionSpace(self.mesh, ("DG", 0))
        self.density = Function(P)
        self.Young_modulus = Function(P)
        self.Poisson_ratio = Function(P)
        self.cohesion = Function(P)
        self.friction_angle = Function(P)
        self.porosity = Function(P)
        self.conductivity = Function(P)

        for region in self.regions:
            cells = self.mesh_subdomains.indices[self.mesh_subdomains.values==region.mark]
            self.density.x.array[cells] = np.full(len(cells),region.density)
            self.Young_modulus.x.array[cells] = np.full(len(cells),region.Young_modulus)
            self.Poisson_ratio.x.array[cells] = np.full(len(cells),region.Poisson_ratio)
            self.cohesion.x.array[cells] = np.full(len(cells),region.cohesion)
            self.friction_angle.x.array[cells] = np.full(len(cells),region.friction_angle)
            self.porosity.x.array[cells] = np.full(len(cells),region.porosity)
            self.conductivity.x.array[cells] = np.full(len(cells),region.conductivity)

        #plain strain assumption
        self.Lame_lambda = (self.Young_modulus*self.Poisson_ratio)/((1+self.Poisson_ratio)*(1-2*self.Poisson_ratio))
        self.Lame_mu = 0.5*self.Young_modulus/(1+self.Poisson_ratio)

        print("Initializing finite element solver...")
        V = VectorFunctionSpace(self.mesh,("CG",1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        g = Constant(self.mesh,ScalarType((0,-self.g)))

        #save dof coordinates to csv
        coordinates = V.tabulate_dof_coordinates()
        x0 = np.transpose(coordinates)[0]
        y0 = np.transpose(coordinates)[1]
        df_coordinates = pd.DataFrame.from_dict({"x0":x0,"y0":y0})
        df_coordinates.to_csv(os.path.join(self.outputs_path,"coordinates.csv"))

        #zero displacement boundary condition
        bcs_u = []
        for boundary in self.zero_displacement_boundaries:
            mark = self.boundaries[boundary.label].mark
            fixed_facets = self.mesh_boundaries.indices[self.mesh_boundaries.values==mark]
            fixed_dofs = locate_dofs_topological(V, self.mesh.topology.dim-1, fixed_facets)
            bcs_u.append(dirichletbc(ScalarType((0,0)), fixed_dofs, V))

        a = ufl.inner(self.sigma(u),self.epsilon(v))*ufl.dx
        L = ufl.dot(g,v)*ufl.dx

        print("Running simulation...")
        problem = LinearProblem(a, L, bcs=bcs_u, petsc_options={"ksp_type":"preonly","pc_type":"lu"})
        uh = problem.solve()

        W = TensorFunctionSpace(self.mesh,("DG",0))
        stress = Function(W)
        self.project(self.sigma(uh),stress)
        print(len(stress.x.array)/4)

        xdmf_u = XDMFFile(MPI.COMM_WORLD, os.path.join(self.outputs_path,"displacement.xdmf"), "w")
        xdmf_u.write_mesh(self.mesh)
        xdmf_u.write_function(uh)

        xdmf_sigma = XDMFFile(MPI.COMM_WORLD, os.path.join(self.outputs_path,"stress.xdmf"), "w")
        xdmf_sigma.write_mesh(self.mesh)
        xdmf_sigma.write_function(stress)

        #save displacement data to csv
        uh_sol = np.reshape(uh.x.array,(len(coordinates),2))
        uh_sol_x = np.transpose(uh_sol)[0]
        uh_sol_y = np.transpose(uh_sol)[1]
        df_displacements = pd.DataFrame.from_dict({"t":[0]*len(uh_sol_x),"dx":uh_sol_x,"dy":uh_sol_y})
        df_displacements.to_csv(os.path.join(self.outputs_path,"displacements.csv"))
