import numpy as np
from dolfinx import geometry
from pandas import DataFrame, concat
import os

class point:
    def __init__(self,label,coordinate,mesh_size):
        '''
        Attributes:
        label: string: label of point
        coordinate: list of int/float: coordinate [x,y] of point
        mesh_size: int/float: length set between the point as
                               mesh node with other neighboring mesh nodes
        '''
        self.label = label
        self.coordinate = coordinate 
        self.mesh_size = mesh_size 

class boundary:
    def __init__(self,label,points,zero_displacement_at=False):
        '''
        Attributes:
        label: string: label of boundary
        points: list of geolaman.entities.point: points making up the boundary
        mark: int: mark of boundary - for mesh processing
        zero_displacement_at: string: indicates displacement boundary condition;
                                      "x": zero displacement at x direction
                                      "y": zero displacement at y direction
                                      "xy": zero displacement at both directions
        '''
        self.label = label
        self.points = points
        self.mark = None
        self.zero_displacement_at = zero_displacement_at

class region:
    def __init__(self,label,density,Young_modulus,Poisson_ratio,
                 cohesion,friction_angle,conductivity,porosity,boundary_points):
        '''
        Attributes:
        label: string: label of region
        density: int/float: material density
        Young_modulus: int/float: material Young modulus
        Poisson_ratio: int/float: material Poisson ratio
        cohesion: int/float: material cohesion
        friction_angle: int/float: material friction angle in radians
        conductivity: int/float: material hydraulic conductivity (assumed istropic)
        porosity: float: material porosity
        boundary_points: list of geolaman.entities.point: boundary points of the region
        mark: int : mark of region - for mesh processing
        '''

        self.label = label
        self.density = density
        self.Young_modulus = Young_modulus                              #elastic model
        self.Poisson_ratio = Poisson_ratio                              #elastic model
        self.cohesion = cohesion                                        #Mohr-Coulomb model
        self.friction_angle = friction_angle                            #Mohr-Coulomb model
        self.conductivity = conductivity                                #u-p model
        self.porosity = porosity                                        #u-p model
        self.boundary_points = boundary_points
        self.mark = None

class marker:
    def __init__(self,label,point_):
        '''
        Attributes:
        label: string: label of marker
        coordinate: list of int/float: coordinate [x,y] where marker is
                                    initially at
        displacement_data: dict: stores displacement data with time/steps 
                                as dict.keys() and displacement as dict.values()
        '''

        self.label = label
        if isinstance(point_,point):
            point_ = point_.coordinate
        self.coordinate = np.zeros(3)
        self.coordinate[:2] = point_
        self.cells_proc = []
        self.displacement_data = {}

    def create_dataframe(self):
        times = list(self.displacement_data.keys())
        disp = np.transpose(list(self.displacement_data.values()))
        df = DataFrame.from_dict({"t":times,"dx":disp[0],"dy":disp[1]})
        self.dataframe = df

class column:
    def __init__(self,label,point_,length,segment_length):
        '''
        Attributes:
        label: string: label of marker
        point: list of int/float: coordinate [x,y] where topmost 
                                node is initially at
        length: int/float: length of column
        segment_length: int/float: length of segments
        '''

        self.label = label
        if isinstance(point_,point):
            point_ = point_.coordinate
        self.point = point_
        self.length = length
        self.segment_length = segment_length
        self.nodes = None
        self.cells_proc = []
        self.displacement_data = {}
        self.create_column()

    def create_column(self):
        number_of_nodes = int(self.length/self.segment_length)
        self.nodes = np.zeros((3,number_of_nodes))
        self.nodes[0] = [self.point[0]]*number_of_nodes
        self.nodes[1] = np.linspace(self.point[1],self.point[1]-self.length,number_of_nodes)
        self.nodes = self.nodes.T

    def create_dataframe(self):
        df = DataFrame(columns=["t","node","dx","dy"])
        N = len(self.nodes)
        nodes = list(range(N))
        for t in self.displacement_data.keys():
            tx = [t]*N
            disp_vals = self.displacement_data[t].T
            data_dict = {"t":tx,"node":nodes,"dx":disp_vals[0],"dy":disp_vals[1]}
            df_ = DataFrame.from_dict(data_dict)
            df = concat([df,df_],ignore_index=True)
        self.dataframe = df

class slip_surface:
    def __init__(self,label,points,angles,arc_params):
        '''
        label: string: label of the slip_surface
        points: list of points coordinates
        angles: list of angles tangential to the points
        arc_params: parameters of arc = (a,b,width,radius)
        '''
        self.label = label
        self.points = np.zeros((3,len(points)))
        self.points[0] = np.transpose(points)[0]
        self.points[1] = np.transpose(points)[1]
        self.points = self.points.T
        self.angles = angles
        self.arc_params = arc_params
        self.dx = arc_params[-1] #width
        self.cells_proc = []

def create_slip_surfaces(arcs,surfaces,radius_factor=1,x_only=False):
    '''
    arcs: list of arc_params = (a,b,width,radius*)
    surfaces: list of geolaman.boundary upper surfaces
    x_only: boolean: if True a & b are only x-coordinates
                     and no radius in arc_params, if False
                     a & b are (x,y)-coordinates
    '''
    #print("Creating slip surfaces ...")
    slip_surfaces = []

    surface_points = []
    for surface in surfaces:
        surface_points += [tuple(p.coordinate) for p in surface.points]
    surface_points = list(set(surface_points))
    surface_points.sort()
    surface_points_x = np.transpose(surface_points)[0]

    count = 1
    for arc in arcs:
        label = "S"+str(count)
        if x_only:
            xl,xr = arc[0],arc[1]
            idx = find_closest_to_value(xl,surface_points_x)
            yl = get_colinear_y(xl,surface_points[idx[0]],surface_points[idx[1]]) if type(idx) is tuple else surface_points[idx][1]
            idx = find_closest_to_value(xr,surface_points_x)
            yr = get_colinear_y(xr,surface_points[idx[0]],surface_points[idx[1]]) if type(idx) is tuple else surface_points[idx][1]
            ql,qr = (xl,yl),(xr,yr)

            radius_0 = radius_factor*vector_between(ql,qr).magnitude
            radius = radius_0.copy()
            check_points = True #check if all points are below surface
            while check_points:
                has_point_above = False
                points, angles = create_arc_points(ql,qr,radius,arc[2])
                for p in points:
                    idx = find_closest_to_value(p[0],surface_points_x)
                    y = get_colinear_y(p[0],surface_points[idx[0]],surface_points[idx[1]]) if type(idx) is tuple else surface_points[idx][1]
                    if p[1] > y:
                        has_point_above = True #above surface
                        break
                if has_point_above:
                    radius = radius-0.1*radius_0
                else:
                    check_points = False

        else:
            ql,qr = arc[0],arc[1]
            radius = arc[3]
            points, angles = create_arc_points(ql,qr,radius,arc[2])

        arc_params = (ql,qr,arc[2],radius)
        slip_surfaces.append(slip_surface(label,points,angles,arc_params))
        count += 1

    return slip_surfaces

def get_colinear_y(x,a,b):
    return a[1]+((b[1]-a[1])/(b[0]-a[0]))*(x-a[0])

def find_closest_to_value(value,array_to_compare):
    '''
    return: index (indices as tuple) of element(s) in array_to_compare closest to value
    '''
    difference = list(array_to_compare-[value]*len(array_to_compare))
    try:
        match_idx = difference.index(0)
    except ValueError:
        #get elements closest to the value at both sides
        difference_pos = [d for d in difference if d > 0]
        match_idx_right = difference.index(min(difference_pos))
        difference_neg = [d for d in difference if d < 0]
        match_idx_left = difference.index(max(difference_neg))
        match_idx = (match_idx_left,match_idx_right)
    return match_idx

class vector_between:
    '''
    Creates a vector from point a to point b
    '''
    def __init__(self,a,b):
        '''
        vector: list of int/float: [x-component,y-component]
        magnitude: float: magnitude of vector
        '''
        if isinstance(a,point):
            a = a.coordinate
        if isinstance(b,point):
            b = b.coordinate
        self.vector = [b[0]-a[0],b[1]-a[1]]
        self.magnitude = np.sqrt((b[0]-a[0])**2+(b[1]-a[0])**2)

def create_arc_points(a,b,dx,r):
    '''
    Creates the points of a circular arc from a to b with radius r subdivided
    by (vertical) slices of width dx. The points are midpoints of the slices.
    '''

    if isinstance(a,point):
        a = a.coordinate
    if isinstance(b,point):
        b = b.coordinate
    q = np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2) #distance between a & b
    mp = [(a[0]+b[0])/2,(a[1]+b[1])/2] #midpoint of a & b
    cp = [0,0]
    result = None
    if q > 2*r:
        result = "No arc segment can be made with these points and radius"
    else:
        if q == 2*r:
            cp=mp
        else:
            #assumes a is left of b
            f = np.sqrt(r**2-(q/2)**2)
            cp[0] = mp[0] + f*(a[1]-b[1])/q
            cp[1] = mp[1] + f*(b[0]-a[0])/q
        points = []
        angles = []
        xs = np.linspace(a[0],b[0],int(abs((b[0]-a[0])/dx)))
        ys = [cp[1] - np.sqrt(r**2 - (x-cp[0])**2) for x in xs]
        xs_mid = [x+dx/2 for x in xs[:-1]]
        for i in range(len(xs)-1):
            l = xs[i+1]-xs[i]
            h = ys[i+1]-ys[i]
            y = (ys[i]+ys[i+1])/2
            angle = np.arctan2(h,l)
            points.append([xs_mid[i],y])
            angles.append(angle)
        result = points,angles
    return result

def print_points(points,ends_only=False):
    if ends_only:
        print(points[0].label,points[0].coordinate)
        print(points[-1].label,points[-1].coordinate)
    else:
        for p in points:
            print(p.label,p.coordinate)
    return

def has_duplicates(points):
    coords = [p.coordinate for p in points]
    duplicates_removed = list(set(map(tuple,coords)))
    has_duplicates = False
    if len(duplicates_removed) < len(coords):
        has_duplicates = True
    print("Has duplicated points? ",has_duplicates)
