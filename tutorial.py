'''
This tutorial documents simulation of a simple slope 
of a single material acted upon by a surface load.
We use standard units.
'''

from geolaman.entities import *
from geolaman.simulation import *
from geolaman.analysis import *
import numpy as np

'''
First, we define the points that would make the slope with the 
geolaman.entities.point class 
Note: points must be unique in label and in coordinates
'''

p0 = point("p0",[0,0],0.5)
p1 = point("p1",[20,0],0.5)
p2 = point("p2",[40,-15],2.)
p3 = point("p3",[50,-15],5.)
p4 = point("p4",[50,-20],5.)
p5 = point("p5",[0,-20],5.)

'''
Second, we define the boundaries of the slope that we will need later on
as we will see. We do this with the geolaman.entities.boundary class.
'''

left_boundary = boundary("left_boundary",[p5,p0],zero_displacement_at="x")
bottom_boundary = boundary("bottom_boundary",[p4,p5],zero_displacement_at="y")
right_boundary = boundary("right_boundary",[p3,p4],zero_displacement_at="x")
surface = boundary("surface",[p1,p2,p3])
load_surface = boundary("load_surface",[p0,p1])

'''
Third, we define the region that makes up the slope. We do this with
the geolaman.entities.region class. Here, we define the material parameters.
'''

material = region(label="material",
        density = 19.383e3/9.8,
        Young_modulus = 2e7,
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
system.regions = [material]
system.boundaries = [left_boundary,bottom_boundary,right_boundary,surface,load_surface]
system.create_mesh()
