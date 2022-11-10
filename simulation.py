import numpy as np
import pandas as pd
import os
import pygmsh
import gmsh
import meshio
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, form,
                         VectorFunctionSpace,TensorFunctionSpace,locate_dofs_topological)
from dolfinx.fem.petsc import (LinearProblem, assemble_matrix, assemble_vector,
                                apply_lifting,set_bc,create_vector)
from dolfinx.io import XDMFFile
from dolfinx import geometry
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import ufl
import geolaman.entities as geo_ent
from geolaman.analysis import factor_of_safety
import matplotlib.pyplot as plt
import matplotlib as mpl

class system:

    def __init__(self):

        self.regions = None
        self.boundaries = None

        self.steps = None

        self.pore_pressure = False
        self.set_pore_pressure_at = []
        self.groundwater_table = None

        self.surface_force = None
        self.surface_force_at = None
        self.volume_force = None

        self.precipitation = None
        self.hydraulic_parameters = {"porosity":0,"dissipation_time":0}
        self.excess_pore_pressure_value = 0.

        self.g = 9.8
        self.water_density = 1000
        self.water_bulk_moulus = 1.96e9

        self.markers = []
        self.columns = []
        self.slip_surfaces = []

        self.path = None
        self.no_print = True

    def create_mesh(self):

        #Path for files
        if self.path == None:
            self.path = os.getcwd()
        self.files_path = os.path.join(self.path,"files")
        if not os.path.exists(self.files_path):
            os.mkdir(self.files_path)
        self.analysis_path = os.path.join(self.path,"analysis")
        if not os.path.exists(self.analysis_path):
            os.mkdir(self.analysis_path)

        if not self.no_print: print("Creating mesh ...")
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
                    #make enpoints of curve
                    if region.boundary_points[i].label not in points_done.keys():
                        p = model.add_point(region.boundary_points[i].coordinate,region.boundary_points[i].mesh_size)
                        points_done.update({region.boundary_points[i].label:p})
                    else:
                        p = points_done[region.boundary_points[i].label]

                    if region.boundary_points[i+1].label not in points_done.keys():
                        q = model.add_point(region.boundary_points[i+1].coordinate,region.boundary_points[i+1].mesh_size)
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

            if not self.no_print: print("Mesh region for",region.label," ...")

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
            if not self.no_print: print("Marking",boundary.label," ...")

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

        boundaries = self.boundaries
        self.boundaries = {}
        for boundary in boundaries:
            self.boundaries.update({boundary.label:boundary})

        self.set_material_parameters()

    def set_material_parameters(self):

        #parameters
        P = FunctionSpace(self.mesh, ("DG",0))
        self.density = Function(P)
        self.Young_modulus = Function(P)
        self.Poisson_ratio = Function(P)
        self.cohesion = Function(P)
        self.friction_angle = Function(P)
        self.porosity = Function(P)
        self.conductivity = Function(P)

        cells_to_region = []
        for region in self.regions:
            cells = self.mesh_subdomains.indices[self.mesh_subdomains.values==region.mark]
            cells_to_region = cells_to_region + [(cell,region.mark) for cell in cells]
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

        Ks = self.Young_modulus/(3*(1-2*self.Poisson_ratio))
        self.stiffness_inv = self.porosity/self.water_bulk_moulus + (1-self.porosity)/Ks


    def create_meshio_mesh(self,mesh, cell_type, prune_z=False):

        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

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

    def initial_boundary_pore_pressure(self,x):
        line_slope = (self.groundwater_table[1][1]-self.groundwater_table[0][1])
        line_slope = line_slope/(self.groundwater_table[1][0]-self.groundwater_table[0][0])
        values = (self.groundwater_table[0][1]+line_slope*(x[0]-self.groundwater_table[0][0]))-x[1]
        values = self.g*self.water_density*values
        return values

    def update_boundary_pore_pressure(self,x):
        return self.initial_boundary_pore_pressure(x) + self.excess_pore_pressure_value

    def save_up_data(self,u,p,t):

        if len(self.markers) > 0:
            for marker in self.markers:
                marker.displacement_data.update({t:u.eval(marker.coordinate,marker.cells_proc)})

        if len(self.columns) > 0:
            for column in self.columns:
                column.displacement_data.update({t:u.eval(column.nodes,column.cells_proc)})

        self.xdmf_u.write_function(u,t)
        if self.pore_pressure:
            self.xdmf_p.write_function(p,t)

    def save_stress_data(self,u,p,t):

        self.project(self.sigma(u),self.stress)
        self.xdmf_sigma.write_function(self.stress,t)
        if len(self.slip_surfaces) > 0:
            for stress_surface in self.slip_surfaces:
                stress_data = self.stress.eval(stress_surface.points,stress_surface.cells_proc)
                normal_stresses, shear_stresses = self.stress_data_calculation(stress_surface,stress_data)
                N = len(normal_stresses)
                if p is not None:
                    pore_pressure_data = p.eval(stress_surface.points,stress_surface.cells_proc)
                    pore_pressure_data = np.transpose(pore_pressure_data)[0]
                else:
                    pore_pressure_data = [0]*N
                df_ = pd.DataFrame.from_dict({"label":[stress_surface.label]*N,"t":[t]*N,"slice":list(range(N)),
                    "normal_stress":normal_stresses,"shear_stress":shear_stresses,"pore_pressure":pore_pressure_data})
                self.slip_surfaces_dataframe = pd.concat([self.slip_surfaces_dataframe,df_],ignore_index=True)

    def stress_data_calculation(self,stress_surface,stress_data):
        normal_stresses = []
        shear_stresses = []
        for i in range(len(stress_data)):
            S = np.reshape(stress_data[i],(2,2))
            #S = S - pore_pressures[i]*np.eye(2)
            a = stress_surface.angles[i]
            R = np.reshape([np.cos(a),np.sin(a),-np.sin(a),np.cos(a)],(2,2))
            S = R*S*R.T
            normal_stresses.append(S[1][1])
            shear_stresses.append(S[0][1])

        return normal_stresses, shear_stresses

    def up_mixed_boundary_conditions(self,VX,V,X):
        bcs = []
        #displacement
        for boundary in self.boundaries.values():
            if boundary.zero_displacement_at != False:
                fixed_facets = self.mesh_boundaries.indices[self.mesh_boundaries.values==boundary.mark]
                if boundary.zero_displacement_at == "xy":
                    u_ = Function(V)
                    u_.x.array[:] = 0
                    fixed_dofs = locate_dofs_topological((VX.sub(0),V), self.mesh.topology.dim-1, fixed_facets)
                    bcs.append(dirichletbc(u_, fixed_dofs, VX.sub(0)))
                elif boundary.zero_displacement_at == "x" or boundary.zero_displacement_at == "y":
                    sub_num = 0 if boundary.zero_displacement_at == "x" else 1
                    Vsub, _ = V.sub(sub_num).collapse()
                    u_ = Function(Vsub)
                    u_.x.array[:] = 0.
                    fixed_dofs = locate_dofs_topological((VX.sub(0).sub(sub_num),Vsub),self.mesh.topology.dim-1,fixed_facets)
                    bcs.append(dirichletbc(u_, fixed_dofs, VX.sub(0).sub(sub_num)))

        if self.groundwater_table is None:
            print("No groundwater table defined")
        else:
            if isinstance(self.groundwater_table,geo_ent.boundary):
                ps = self.groundwater_table.points
                self.groundwater_table = [ps[0].coordinate,ps[1].coordinate]

        #pore_pressure
        fixed_facets = []
        if len(self.set_pore_pressure_at) == 0:
            self.set_pore_pressure_at = list(self.boundaries.values())
        for boundary in self.set_pore_pressure_at:
            fixed_facets += list(self.mesh_boundaries.indices[self.mesh_boundaries.values==boundary.mark])
        fixed_dofs = locate_dofs_topological((VX.sub(1),X), self.mesh.topology.dim-1, fixed_facets)
        fixed_p_for_bc = Function(X)
        fixed_p_for_bc.interpolate(self.initial_boundary_pore_pressure)
        bcs.append(dirichletbc(fixed_p_for_bc,fixed_dofs,VX.sub(1)))

        return bcs, fixed_p_for_bc

    def u_only_boundary_conditions(self,V):
        bcs = []
        for boundary in self.boundaries.values():
            if boundary.zero_displacement_at != False:
                fixed_facets = self.mesh_boundaries.indices[self.mesh_boundaries.values==boundary.mark]
                if boundary.zero_displacement_at == "xy":
                    fixed_dofs = locate_dofs_topological(V, self.mesh.topology.dim-1, fixed_facets)
                    bcs.append(dirichletbc(ScalarType((0,0)), fixed_dofs, V))
                elif boundary.zero_displacement_at == "x" or boundary.zero_displacement_at == "y":
                    sub_num = 0 if boundary.zero_displacement_at == "x" else 1
                    Vsub, _ = V.sub(sub_num).collapse()
                    u_ = Function(Vsub)
                    u_.x.array[:] = 0.
                    fixed_dofs = locate_dofs_topological((V.sub(sub_num),Vsub),self.mesh.topology.dim-1,fixed_facets)
                    bcs.append(dirichletbc(u_, fixed_dofs, V.sub(sub_num)))

        return bcs

    def get_cells_proc(self,entity,points,mesh,bb_tree):
        cells_proc = []
        cell_candidates = geometry.compute_collisions(bb_tree,points)
        colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                cells_proc.append(colliding_cells.links(i)[0])
        points = np.array(points, dtype=np.float64)
        if abs(len(points)-len(cells_proc)) > 0:
            printout = "Number of points and cells for evaluation of "+str(entity.label)+" must be equal. \n"
            printout += "Make sure that its points are inside the mesh."
            raise Exception(printout)

        return cells_proc

    def evolve(self,initialize_only=False):

        bb_tree = geometry.BoundingBoxTree(self.mesh,self.mesh.topology.dim)
        if len(self.markers) > 0:
            for marker in self.markers:
                marker.cells_proc = self.get_cells_proc(marker,[marker.coordinate],self.mesh,bb_tree)

        if len(self.columns) > 0:
            for column in self.columns:
                column.cells_proc = self.get_cells_proc(column,column.nodes,self.mesh,bb_tree)

        if len(self.slip_surfaces) > 0:
            for stress_surface in self.slip_surfaces:
                stress_surface.cells_proc = self.get_cells_proc(stress_surface,stress_surface.points,self.mesh,bb_tree)

        if not self.no_print: print("Initializing system...")
        if self.pore_pressure:
            #Mixed element space if with pore pressure
            V_el = ufl.VectorElement("CG", self.mesh.ufl_cell(), 1)
            X_el = ufl.FiniteElement("CG", self.mesh.ufl_cell(), 1)
            M_el = ufl.MixedElement([V_el, X_el])
            VX = FunctionSpace(self.mesh,M_el)
            (u, p) = ufl.TrialFunctions(VX)
            (v, q) = ufl.TestFunctions(VX)
            V, V_map = VX.sub(0).collapse()
            X, X_map = VX.sub(1).collapse()
            bcs, fixed_p_for_bc = self.up_mixed_boundary_conditions(VX,V,X)
            p_h = Function(X)
            p_h.name = "Pore pressure"
        else:
            V = VectorFunctionSpace(self.mesh,("CG",1))
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            bcs = self.u_only_boundary_conditions(V)
            p_h = None

        #Stress
        W = TensorFunctionSpace(self.mesh,("DG",0))
        self.stress = Function(W)
        self.stress.name = "Stress"
        if len(self.slip_surfaces) > 0:
            self.slip_surfaces_dataframe = pd.DataFrame(columns=["label","t","slice",
                                                "normal_stress","shear_stress","pore_pressure"])

        ug = Function(V) #initial displacement due to gravity
        u_net = Function(V) #displacement relative to ug
        u_net.name = "Displacement"
        u_h = Function(V) #relative to mesh

        g = Constant(self.mesh,ScalarType((0,-self.g)))
        a = ufl.inner(self.sigma(u),self.epsilon(v))*ufl.dx
        L = ufl.dot(self.density*g,v)*ufl.dx
        if self.pore_pressure:
            rho_w = Constant(self.mesh,ScalarType(self.water_density))
            K = Function(V)
            self.project(self.conductivity*rho_w*g,K)
            a += -ufl.inner(p*ufl.Identity(u.geometric_dimension()),self.epsilon(v))*ufl.dx
            a += ufl.dot(self.conductivity*ufl.grad(p),ufl.grad(q))*ufl.dx
            L += ufl.div(K)*q*ufl.dx

        problem = LinearProblem(a,L,bcs,petsc_options={"ksp_type":"preonly","pc_type":"lu"})
        up_h = problem.solve()
        ug.x.array[:] = up_h.x.array[V_map] if self.pore_pressure else up_h.x.array

        self.xdmf_u = XDMFFile(self.mesh.comm, os.path.join(self.analysis_path,"displacements.xdmf"), "w")
        self.xdmf_u.write_mesh(self.mesh)
        self.xdmf_sigma = XDMFFile(MPI.COMM_WORLD, os.path.join(self.analysis_path,"stress.xdmf"), "w")
        self.xdmf_sigma.write_mesh(self.mesh)
        if self.pore_pressure:
            self.xdmf_p = XDMFFile(self.mesh.comm,os.path.join(self.analysis_path,"pore_pressure.xdmf"), "w")
            self.xdmf_p.write_mesh(self.mesh)

        if initialize_only:
            u_h.name = "Displacement"
            if self.pore_pressure:
                u_h.x.array[:] = up_h.x.array[V_map]
                p_h.x.array[:] = up_h.x.array[X_map]
                p_h.name = "Pore pressure"
            else:
                u_h.x.array[:] = up_h.x.array
            self.save_up_data(u_h,p_h,0)
            self.save_stress_data(u_h,p_h,0)
            self.create_dataframe_and_csv_file(initialize_only=initialize_only)

            self.xdmf_u.close()
            self.xdmf_sigma.close()
            self.xdmf_p.close()
            return

        if not self.no_print: print("Running simulation with external trigger...")
        #Main simulation with external trigger
        u_prev = Function(V)
        L = ufl.inner(self.sigma(u_prev),self.epsilon(v))*ufl.dx
        '''NOT L += ufl.inner(self.sigma(u_prev),self.epsilon(v))*ufl.dx
           as the initial stresses are ue to gravity.
           Doing so with u_prev = uh continuously deforms mesh even without
           surface/volume force.
        '''
        if self.pore_pressure:
            up_prev = Function(VX)
            p_excess = Function(X)
            p_excess_prev = Function(X)
            p_prev = Function(X)
            a += ufl.inner(p*ufl.Identity(u.geometric_dimension()),self.epsilon(v))*ufl.dx
            L += ufl.inner(p_excess*ufl.Identity(u.geometric_dimension()),self.epsilon(v))*ufl.dx

            a += ufl.div(u)*q*ufl.dx
            a += self.stiffness_inv*p*q*ufl.dx
            L += self.stiffness_inv*p_excess*q*ufl.dx
            L += ufl.div(u_prev)*q*ufl.dx
            L += self.stiffness_inv*p_prev*q*ufl.dx
            L += self.stiffness_inv*p_excess_prev*q*ufl.dx

        if self.surface_force is not None:
            ds = ufl.Measure("ds",domain=self.mesh,subdomain_data=self.mesh_boundaries)
            Fs = Constant(self.mesh,ScalarType((0,0)))
            L += ufl.dot(Fs,v)*ds(self.boundaries[self.surface_force_at.label].mark)
            Fs_dict = dict(zip(self.steps,self.surface_force))
        if self.volume_force is not None:
            Fv = Constant(self.mesh,ScalarType((0,0)))
            L += ufl.dot(Fv,v)*ufl.dx
            Fv_dict = dict(zip(self.steps,self.volume_force))
        if self.precipitation is not None:
            P_dict = dict(zip(self.steps,self.precipitation))
            dt = self.steps[1]-self.steps[0]
            P_factor = 0
            gamma = self.water_density*self.g
            self.density = self.density*(1-self.hydraulic_parameters["porosity"])+self.hydraulic_parameters["porosity"]*self.water_density
            self.gwt_fluctuations = []

        bilinear_form = form(a)
        linear_form = form(L)
        A = assemble_matrix(bilinear_form, bcs=bcs)
        A.assemble()
        b = create_vector(linear_form)

        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        for t in self.steps:
            if not self.no_print: print("step = ",np.round(t,2))
            if self.surface_force is not None:
                Fs.value = ScalarType(Fs_dict[t])
            if self.volume_force is not None:
                Fv.value = ScalarType(Fv_dict[t])
            if self.precipitation is not None:
                p_excess_prev.x.array[:] = (P_factor*gamma/1000)/self.hydraulic_parameters["porosity"]
                P_factor = P_factor*np.exp(-dt/self.hydraulic_parameters["dissipation_time"])+P_dict[t]
                self.excess_pore_pressure_value = (P_factor*gamma/1000)/self.hydraulic_parameters["porosity"]
                p_excess.x.array[:] = self.excess_pore_pressure_value
                fixed_p_for_bc.interpolate(self.update_boundary_pore_pressure)
                self.gwt_fluctuations.append(self.excess_pore_pressure_value/gamma)

            #update displacement initial condition for this time step
            if not self.pore_pressure:
                u_prev.x.array[:] = up_h.x.array
            else:
                up_prev.x.array[:] = up_h.x.array
                u_prev.x.array[:] = up_prev.x.array[V_map]
                p_prev.x.array[:] = up_prev.x.array[X_map]

            '''NOT u_prev +=  uh
               see https://jorgensd.github.io/dolfinx-tutorial/chapter2/diffusion_code.html
            '''
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)

            apply_lifting(b,[bilinear_form],[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            solver.solve(b, up_h.vector)
            up_h.x.scatter_forward()

            if self.pore_pressure:
                u_h.x.array[:] = up_h.x.array[V_map]
                u_net.x.array[:] = up_h.x.array[V_map]-ug.x.array
                p_h.x.array[:] = up_h.x.array[X_map]
            else:
                u_net.x.array[:] = up_h.x.array-ug.x.array
                u_h.x.array[:] = up_h.x.array
            self.save_up_data(u_net,p_h,t)
            self.save_stress_data(u_h,p_h,t)

        self.create_dataframe_and_csv_file()

    def create_dataframe_and_csv_file(self,initialize_only=False):
        if not self.no_print: print("Creating dataframe ...")
        #Creates dataframe and csv file for data
        if len(self.markers) > 0:
            if not self.no_print: print("... for markers")
            df = self.collate_data_to_dataframe(self.markers)
            df_out = pd.DataFrame()
            for marker in self.markers:
                df_c = df[df.label==marker.label].copy()
                df_c = df_c[["label","t","dx","dy"]]
                df_out = pd.concat([df_out,df_c],ignore_index=True)
            df_out["x0"] = ""
            df_out["y0"] = ""
            for marker in self.markers:
                df_out.loc[(df_out.label==marker.label) & (df_out.t == 0), "x0"] = marker.coordinate[0]
                df_out.loc[(df_out.label==marker.label) & (df_out.t == 0), "y0"] = marker.coordinate[0]
            print(df_out)
            self.markers_dataframe = df_out
            self.markers_dataframe.to_csv(os.path.join(self.analysis_path,"markers_data.csv"))

        if len(self.columns) > 0:
            if not self.no_print: print("... for columns")
            df = self.collate_data_to_dataframe(self.columns)
            df_out = pd.DataFrame()
            for column in self.columns:
                df_c = df[df.label == column.label].copy()
                df_c = df_c[["label","t","node","dx","dy"]]
                df_out = pd.concat([df_out,df_c],ignore_index=True)
            df_out["x0"] = ""
            df_out["y0"] = ""
            for column in self.columns:
                x0 = np.transpose(column.nodes)[0]
                y0 = np.transpose(column.nodes)[1]
                for i in range(len(column.nodes)):
                    df_out.loc[(df_out.label==column.label) & (df_out.t == 0) & (df_out.node == i),"x0"] = x0[i]
                    df_out.loc[(df_out.label==column.label) & (df_out.t == 0) & (df_out.node == i),"y0"] = y0[i]
            df_out["x0"] = pd.to_numeric(df_out["x0"])
            df_out["y0"] = pd.to_numeric(df_out["y0"])
            self.columns_dataframe = df_out
            self.columns_dataframe.to_csv(os.path.join(self.analysis_path,"columns_data.csv"))

        if len(self.slip_surfaces) > 0:
            if not self.no_print: print("... for slip surfaces")
            df = self.slip_surfaces_dataframe
            df["x0"] = ""
            df["y0"] = ""
            df["cohesion"] = ""
            df["friction_angle"] = ""
            for ss in self.slip_surfaces:
                cohesions = self.cohesion.eval(ss.points,ss.cells_proc)
                friction_angles = self.friction_angle.eval(ss.points,ss.cells_proc)
                x0 = np.transpose(ss.points)[0]
                y0 = np.transpose(ss.points)[1]
                for i in range(len(ss.points)):
                    df.loc[(df.label==ss.label) & (df.t == 0) & (df.slice == i),"x0"] = x0[i]
                    df.loc[(df.label==ss.label) & (df.t == 0) & (df.slice == i),"y0"] = y0[i]
                    df.loc[(df.label==ss.label) & (df.t == 0) & (df.slice == i),"cohesion"] = cohesions[i]
                    df.loc[(df.label==ss.label) & (df.t == 0) & (df.slice == i),"friction_angle"] = friction_angles[i]
            df["x0"] = pd.to_numeric(df["x0"])
            df["y0"] = pd.to_numeric(df["y0"])
            df["cohesion"] = pd.to_numeric(df["cohesion"])
            df["friction_angle"] = pd.to_numeric(df["friction_angle"])
            self.slip_surfaces_dataframe = df
            self.slip_surfaces_dataframe.to_csv(os.path.join(self.analysis_path,"slip_surfaces_data.csv"))

        if not initialize_only:
            if self.precipitation is not None:
                self.precipitation_dataframe = pd.DataFrame.from_dict({"t":self.steps,
                                                                   "value":self.precipitation,
                                                                   "gwt_depth_change":self.gwt_fluctuations})

                self.precipitation_dataframe.to_csv(os.path.join(self.analysis_path,"precipitation_data.csv"))

    def collate_data_to_dataframe(self,entities_list):

        df = pd.DataFrame()
        for entity in entities_list:
            entity.create_dataframe()
            df0 = entity.dataframe
            df0["label"] = [entity.label]*len(df0)
            df = pd.concat([df,entity.dataframe],ignore_index=True)
        df = df.sort_values(["t","label"],ignore_index=True)
        return df

    def plot(self,markers=False,columns=False,file_name="",pdf=False,transparent=False):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for region in self.regions:
            points = [p.coordinate for p in region.boundary_points]
            points = points + [points[0]]
            points = np.transpose(points)
            ax.plot(points[0],points[1],color="k",ls=":")

        if markers:
            for marker_ in self.markers:
                ax.scatter(marker_.coordinate[0],marker_.coordinate[1],marker="*",s=25,c="b")
                ax.annotate(marker_.label,(marker_.coordinate[0],marker_.coordinate[1]))

        if columns:
            for column in self.columns:
                points = [column.point]
                points.append([column.point[0],column.point[1]-column.length])
                points = np.transpose(points)
                ax.plot(points[0],points[1],c="r",ls="--")
                ax.annotate(column.label,(column.point[0],column.point[1]))

        plt.gca().set_aspect('equal', adjustable='box')
        filetype = ".pdf" if pdf else ".png"
        if file_name == "":
            path = os.path.join(self.analysis_path,"system_plot"+filetype)
        else:
            path = os.path.join(self.analysis_path,file_name+filetype)
        plt.savefig(path,transparent=transparent)
        plt.close()

    def slip_surfaces_plot(self,every=1,legend=False,file_name="",pdf=False):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for region in self.regions:
            points = [p.coordinate for p in region.boundary_points]
            points = points + [points[0]]
            points = np.transpose(points)
            ax.plot(points[0],points[1],color="k",ls=":")

        colors = plt.cm.jet(np.linspace(0,1,len(self.slip_surfaces[::every])))
        i=0
        for stress_surface in self.slip_surfaces[::every]:
            points = np.transpose(stress_surface.points)
            ax.plot(points[0],points[1],c=colors[i],label=stress_surface.label)
            i += 1
        if legend:
            ax.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        filetype = ".pdf" if pdf else ".png"
        if file_name == "":
            path = os.path.join(self.analysis_path,"system_plot"+filetype)
        else:
            path = os.path.join(self.analysis_path,file_name+filetype)
        plt.savefig(path)
        plt.close()
