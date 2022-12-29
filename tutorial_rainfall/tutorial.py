'''
This tutorial documents simulation of a simple slope 
of a two regions acted upon by a surface load 
or volume force. We will go straight to the creation of
two regions and creation of the rainfall inputs. More
basic aspects are discussed in the 
tutorial_load/tutorial.py
'''

from geolaman.entities import *
from geolaman.simulation import *
from geolaman.analysis import *
import numpy as np

#points
p0 = point("p0",[0,0],0.5)
p1 = point("p1",[20,0],0.5)
p2 = point("p2",[40,-15],2.)
p3 = point("p3",[50,-15],2.)
p4 = point("p4",[50,-20],2.)
p5 = point("p5",[50,-25],5.)
p6 = point("p6",[0,-25],5.)
p7 = point("p7",[0,-10],2.)

left_boundary = boundary("left_boundary",[p0,p7,p6],zero_displacement_at="x")
bottom_boundary = boundary("bottom_boundary",[p5,p6],zero_displacement_at="xy")
right_boundary = boundary("right_boundary",[p3,p4,p5],zero_displacement_at="x")
surface = boundary("surface",[p0,p1,p2,p3])

'''
In the creation of regions, it is best that order of points when
defined in the boundary_points is consistent across regions. Specially,
those that will be shared between regions. For example, points p4 and p7
will be shared between material1 and material2. Their order 
[...,p4,p7,...] are kept at both the definitions of the regions.
'''

material1 = region(label="material1",
        density = 1900,
        Young_modulus = 2e5,
        Poisson_ratio = 0.49,
        cohesion = 0,
        friction_angle = np.radians(31),
        conductivity = 4.7e-9,
        porosity = 0.3548,
        boundary_points = [p0,p1,p2,p3,p4,p7])

material2 = region(label="material2",
        density = 2500,
        Young_modulus = 2e7,
        Poisson_ratio = 0.25,
        cohesion = 0,
        friction_angle = np.radians(31),
        conductivity = 1e-10,
        porosity = 0.10,
        boundary_points = [p4,p7,p6,p5])

'''
Here, we create the mesh
'''
system = system()
system.no_print = False
system.regions = [material1,material2]
system.boundaries = [left_boundary,bottom_boundary,right_boundary,surface]
system.create_mesh()

'''
Here, we create several markers, columns and slip surfaces
'''
vector_p1p2 = vector_between(p1,p2)
q = [p1.coordinate[0]+vector_p1p2.vector[0]/2,p1.coordinate[1]+vector_p1p2.vector[1]/2]
marker_1 = marker("M1",[10,0])
marker_2 = marker("M2",q)
system.markers = [marker_1,marker_2]

column_1 = column("C1",p1,10,1)
column_2 = column("C2",p2,4,0.5)
system.columns = [column_1,column_2]

#plot the markers and columns against the slope --> /analysis/markers_columns_plot.png
#system.plot(markers=True,columns=True,file_name="markers_columns_plot") 

q1,q2 = [15,0],[40,-15]
R = vector_between(q1,q2).magnitude
dx = 1
arc_params = (q1,q2,dx,R)
points, angles = create_arc_points(*arc_params)
slip_surface_1 = slip_surface("S1",points,angles,arc_params)

q1,q2 = [10,0],[40,-15]
R = vector_between(q1,q2).magnitude
dx = 1
arc_params = (q1,q2,dx,R)
points, angles = create_arc_points(*arc_params)
slip_surface_2 = slip_surface("S2",points,angles,arc_params)

system.slip_surfaces = [slip_surface_1,slip_surface_2]
system.slip_surfaces_plot(file_name="slip_surface_plot")

'''
For rainfall simulations, pore pressure as state variable
is necessary additon since the hydraulic model used 
converts rainfall precipitation to excess pore pressure
in the slope. We do this by setting system.pore_pressure = True.
Pore pressure can also be added in simulations with
surface or volume force.
'''
system.pore_pressure = True

'''
Defining the groundwater table is required once pore pressure
is included. Two coordinates at the left and right boundaries, 
the line connecting these two points is our model of the
groundwater table. They need not be a defined points.
'''
system.groundwater_table = [p4.coordinate,p7.coordinate]

'''
The duration of the rainfall we will simulate is one day
and the instantaneous rainfall (in units of mm) 
will be at every 30 mins. The rainfall distribution is modeled 
with a Dirichlet distribution so that the cumulative 1-day rainfall 
can be set to 100.

The hydraulic model needs two parameters porosity and
dissipation rate which we set as 0.15 and 3-day respectively.
'''
day = 24*60*60 #1-day
dt = 30*60 #30 mins
duration = day
N = int(duration/dt)
system.steps = [i*dt for i in range(N)]
precipitation = np.random.dirichlet(np.ones(N),size=1)[0]
system.precipitation = [P*100 for P in precipitation]
system.hydraulic_parameters["porosity"] = 0.15
system.hydraulic_parameters["dissipation_time"] = 3*day
system.evolve()

'''
In the following, the unit of time is changed to hours ("hr"). Other option
is "min" for minutes and "day" for days, the default is "s" for seconds.
'''
markers_plot(system.markers_dataframe,system.analysis_path,time_unit="hr")
columns_plot(system.columns_dataframe,system.analysis_path,time_unit="hr")
factor_of_safety_plot(system.slip_surfaces_dataframe,system.analysis_path,time_unit="hr")
slip_surfaces_stresses_plot(system.slip_surfaces_dataframe,system.analysis_path,time_unit="hr")

'''
Precipitation is plotted. When gwt=True, the plot includes the 
fluctiuation of groundwater table depth.'''
precipitation_plot(system.precipitation_dataframe,system.analysis_path,gwt=True,time_unit="hr")
