import numpy as np
import itertools 
from .gen_dome import gen_dome
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.ray import Ray3D
import functools
from honeybee.aperture import Aperture
from honeybee.room import Room
from ladybug_geometry.geometry3d.face import Face3D






def gentestpts(x_lower, x_upper, y_lower, y_upper, z_height, x_size, y_size):


    # This function alwasy return list
    n_x = round((x_upper - x_lower) / x_size)
    x = np.linspace(x_lower, x_upper, n_x + 1)
    x_testpts = x[:-1] + np.diff(x)/2
    
    n_y = round((y_upper - y_lower) / y_size)
    y = np.linspace(y_lower, y_upper, n_y + 1)
    y_testpts = y[:-1] + np.diff(y)/2
    xy = list(itertools.product(x_testpts, y_testpts))
    
    if z_height is not None:
    

        if isinstance(z_height,(list, tuple, np.ndarray)):
            xyz = []
            for height in z_height:
                z = np.ones((n_x * n_y,  1)) * height
                temp_ = np.append(xy, z, axis =1  ).tolist()
                converted_xyz = [Point3D(*point_i) for point_i in temp_]
                xyz.append(converted_xyz)
        return xyz
        
    else:
        return np.array(xy)


def calcVF(testPts, faces):  # Calculate viewfactor,,, testPts must be numpy array of LB Point3D objects
    dome_vec = gen_dome()
    vecs = [Vector3D(*single_vector) for single_vector in dome_vec.T]
    def calcVF_ind(faces, testPt):   # VF for individual point

        faces_count  = [[] for _ in range(len(faces))]  # generate empty nested list
        for vec in vecs:   # determine the intersection in all direction
            ray = Ray3D(testPt, vec)
            for face_i, face in enumerate(faces):
                intersec = face.intersect_line_ray(ray)
                if intersec:
                    faces_count[face_i].append(intersec)
                    # break
        
        return [len(item)/len(vecs) for item in faces_count]   # View factor = fraction of the rays intercepted by that surface 
  # A signle array contain view factors for all test points and faces

    VFs = [[calcVF_ind(faces, testpt_ind) for testpt_ind in first_list] for first_list in testPts]
    assert ~(np.array(VFs).sum(axis = 2) > 1.01).any()   # Raise flag if any of them is larger than 1.01
    assert ~(np.array(VFs).sum(axis = 2) < 0.99).any()   #  Raise flag if any of them is less than 0.99

        # VFs.append(vfunc(testPts, face))
    
    return VFs
    

    
def detSolar():
    pass
    

 
def genRoom(zonename = 'Test', width = 8, depth = 6, height = 2.7, WWR = 50, numGlz = 2, ori = 'south', sizeX = 1, sizeY = 1):
    
    if ori == 'south' or ori == 'north':
        w = width; d = depth
    elif ori == 'east' or ori == 'west':
        w = depth; d = width
    else:
        raise("Orientation is not understandable")
        
    room = Room.from_box(zonename, w, d, height,0, Point3D(0, 0, 0))  # name, width, depth, height, , orientation_angle, origin
    assert list(room.faces[0].normal) == [0.0, 0.0, -1.0]
    assert list(room.faces[1].normal) == [0.0, 1.0,  0.0]
    assert list(room.faces[2].normal) == [1.0, 0.0,  0.0]
    assert list(room.faces[3].normal) == [0.0, -1.0,  0.0]
    assert list(room.faces[4].normal) == [-1.0, 0.0,  0.0]
    assert list(room.faces[5].normal) == [0.0, 0.0,  1.0]
    
    assert 'Bottom' in room.faces[0].name
    assert 'Front' in room.faces[1].name
    assert 'Right' in room.faces[2].name
    assert 'Back' in room.faces[3].name
    assert 'Left' in room.faces[4].name
    assert 'Top' in room.faces[5].name

    # Change the name to something more understandable
    inddict = {0:"floor", 1:"north", 2:"east", 3:"south", 4:"west", 5:"ceiling"}
    
    for i, face in enumerate(room.faces):
        if inddict[i] == ori:
            face.name = inddict[i] + "_" + 'exterior'    
            exterior_id = i
        else:
            face.name = inddict[i] + "_" + 'interior'
        

    
    
    glz_id = {'north':1, 'south': 3, 'east': 2, 'west': 4}
    
    def addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i):
        if ori == 'north':
            glz_pts = (Point3D(hor_lower, depth, ver_lower), Point3D(hor_lower, depth, ver_upper), 
                        Point3D(hor_upper,depth, ver_upper), Point3D(hor_upper, depth, ver_lower))
            glz_face = Face3D(glz_pts)  ## Init without face
            glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)

        elif ori == 'south':
            glz_pts = (Point3D(hor_lower, 0, ver_lower), Point3D(hor_lower, 0, ver_upper), 
                        Point3D(hor_upper,0, ver_upper), Point3D(hor_upper, 0, ver_lower))
            glz_face = Face3D(glz_pts)  ## Init without face
            glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)

        elif ori == 'east':
            glz_pts = (Point3D(depth, hor_lower, ver_lower), Point3D(depth, hor_lower, ver_upper), 
                        Point3D(depth, hor_upper, ver_upper), Point3D(depth, hor_upper, ver_lower))
            glz_face = Face3D(glz_pts)  ## Init without face
            glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)

        elif ori == 'west': 
            glz_pts = (Point3D(0, hor_lower, ver_lower), Point3D(0, hor_lower, ver_upper), 
                        Point3D(0, hor_upper, ver_upper), Point3D(0, hor_upper, ver_lower))
            glz_face = Face3D(glz_pts)  ## Init without face
            glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)
        
        room.faces[glz_id[ori]].add_aperture(glz_ape)

            
    for glz_i in range(numGlz):
            hor_lower = glz_i * width / numGlz + width / numGlz / 2  - (width / numGlz) * np.sqrt(WWR / 100) / 2
            hor_upper = glz_i * width / numGlz + width / numGlz / 2  + (width / numGlz) * np.sqrt(WWR / 100) / 2
            ver_lower = height / 2 - height * np.sqrt(WWR / 100) / 2
            ver_upper = height / 2 + height * np.sqrt(WWR / 100) / 2
            
            assert (0 < hor_lower < hor_upper < width or 0 < ver_lower < ver_upper < height), "check WWR"
            addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i)
    
    
    testPts = gentestpts(0, w, 0, d, np.linspace(0.01,1.8,3), 1, 1)
    
    opaqueFaces = [room.faces[i].punched_geometry for i in range(6)]
    glzFaces = [item.geometry for item in room.faces[glz_id[ori]].apertures]
    
    VFs = calcVF(testPts, opaqueFaces + glzFaces ) 

    return {'room_obj':room, 'opaqueFaces_geo': opaqueFaces, 'glzFaces_geo':glzFaces, 'exteriorid':exterior_id,
            'inddict':inddict, 'testPts': testPts, 'VF': np.array(VFs)}

def genIDFandRun(zonename, ori, **kwargs):
    pass


    