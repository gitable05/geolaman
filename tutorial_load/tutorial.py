'''
This tutorial documents simulation of a simple slope 
of a single material acted upon by a surface load or volume force.
We use standard units.
'''

'''
We import the geolaman library
'''
from geolaman.entities import *
from geolaman.simulation import *
from geolaman.analysis import *
import numpy as np

'''
We define the points that would make the slope with the 
geolaman.entities.point class 
Note: points must be unique in label and in coordinates
'''

p0 = point("p0",[0,0],0.5)
p1 = point("p1",[20,0],0.5)
p2 = point("p2",[40,-15],2.)
p3 = point("p3",[50,-15],2.)
p4 = point("p4",[50,-20],5.)
p5 = point("p5",[0,-20],5.)

'''
We define the boundaries of the slope that we will need later on. 
We do this with the geolaman.entities.boundary class.
'''

left_boundary = boundary("left_boundary",[p5,p0],zero_displacement_at="x")
bottom_boundary = boundary("bottom_boundary",[p4,p5],zero_displacement_at="xy")
right_boundary = boundary("right_boundary",[p3,p4],zero_displacement_at="x")
surface = boundary("surface",[p1,p2,p3])
load_surface = boundary("load_surface",[p0,p1])

'''
We define the region that makes up the slope. We do this with
the geolaman.entities.region class. Here, we define the material parameters.
'''

material = region(label="material",
        density = 1900,
        Young_modulus = 2e5,
        Poisson_ratio = 0.49,
        cohesion = 0,
        friction_angle = np.radians(31),
        conductivity = 4.7e-9,
        porosity = 0.3548,
        boundary_points = [p0,p1,p2,p3,p4,p5])

'''
We are now ready to create the mesh and start the simulation.
We instantiate a geolaman.simulation.system class.

We then invoke the function create_mesh() to create the mesh.
'''
system = system()
system.no_print = False #to print out stages of the mesh creation and simulation
system.regions = [material]
system.boundaries = [left_boundary,bottom_boundary,right_boundary,surface,load_surface]
system.create_mesh()

'''
Two folders are made named /files and /analysis.
/files contains the mesh file in xdmf format that can be viewed in Paraview
/analysis will contain results of the simulations

With the mesh created, we create entities to record displacements and stresses
at certain locations in the slope. 

The geolaman.entities.marker to record displacements at a point.
The geolaman.entities.column to record displacements at a column.
The geolaman.entities.slip_surface to record stresses at a slip surface.
'''

#markers
vector_p1p2 = vector_between(p1,p2) #vector between the two points p1 and p2
q = [p1.coordinate[0]+vector_p1p2.vector[0]/2,p1.coordinate[1]+vector_p1p2.vector[1]/2]
marker_1 = marker("M1",[10,0]) #marker M1 created at [10,0]
marker_2 = marker("M2",q)      #marker M2 created at q
system.markers = [marker_1,marker_2] #added to system

#columns
column_1 = column("C1",q,10,1)   #column C1 at p1 with length=10, segment_length=1
column_2 = column("C2",p2,4,0.5)  #column C2 at p2 with length=4, segment_length=0.5
system.columns = [column_1,column_2] #added to system

#plot the markers and columns against the slope --> /analysis/markers_columns_plot.png
system.plot(markers=True,columns=True,file_name="markers_columns_plot") 


R = vector_between([10,0],p2).magnitude                       #radius
dx = 1                                                        #width of slices
arc_params = ([10,0],p2,dx,R)                                 #define arc parameters
points, angles = create_arc_points(*arc_params)               #creates points and angles
slip_surface = slip_surface("S1",points,angles,arc_params)    #create the slip surface "S1" entity 
system.slip_surfaces = [slip_surface]                         #added to system          
system.slip_surfaces_plot(file_name="slip_surface_plot")


#SURFACE FORCE
#The system is now set. We now apply the load on load_surface.
system.steps = np.linspace(0,10,10,endpoint=False)                         #steps of simulations
system.surface_force =  [(0,-10000) for t in system.steps]                 #downward load per step
system.surface_force_at = load_surface                                     #define where the load will act
system.evolve()                                                            #evolve the system


#VOLUME FORCE
#The system is now set. We now apply the volumetric force.
'''system.steps = np.linspace(0,10,10,endpoint=False) 
system.volume_force = [(0,-10000) for t in system.steps]
system.evolve()'''


'''
During the evolution, the displacements in the columns and markers
are recorded at system.columns_dataframe and markers_dataframe. The
stresses at system.slip_surfaces_dataframe.

In /analysis, displacements.xdmf and stress.xdmf can be viewed 
with Paraview to see the displacement and stress of the whole system

Several csv files are also created for the displacements in the
columns and markers, and stresses in the slip surface
'''

#For immediate plotting:
markers_plot(system.markers_dataframe,system.analysis_path)
columns_plot(system.columns_dataframe,system.analysis_path)
#factor_of_safety_plot(system.slip_surfaces_dataframe,system.analysis_path)
#slip_surfaces_stresses_plot(system.slip_surfaces_dataframe,system.analysis_path)