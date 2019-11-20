from .utils import *
import warnings
import logging
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.ray import Ray3D
import functools
from honeybee.aperture import Aperture
from honeybee.room import Room
from ladybug_geometry.geometry3d.face import Face3D
import itertools

import numpy as np

class ModelInit(object):
    __slots__ = ('_zone_name','_orientation', '_zone_width', '_zone_depth', '_zone_height',
                 '_U_factor', '_SHGC' , '_WWR', '_wea_dir', '_stand','_room', '_viewfactor',
                 '_testPts','__faceidbyori', '_room', '__testptsheight',
                 '_working_dir', '_observers','__xupper', '__yupper')

    def __init__(self, zone_name = None, orientation = None,zone_width = None, zone_depth = None,
                 zone_height = None,  U_factor = None, SHGC = None, WWR = None,
                 stand = True, wea_dir = None, working_dir = None):
        self.zone_name = zone_name
        self.orientation = orientation
        self.zone_width = zone_width
        self.zone_depth = zone_depth
        self.zone_height = zone_height
        self.U_factor = U_factor
        self.SHGC = SHGC
        self.WWR = WWR
        self.stand = stand
        self.wea_dir = wea_dir
        self.working_dir = working_dir
        self.__faceidbyori = {'north':1, 'south': 3, 'east': 2, 'west': 4}
        self._observers = None

    @property
    def zone_name(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_name

    @zone_name.setter
    def zone_name(self, value):
        self._zone_name = value


    @property
    def orientation(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value
        if value is not None:
            if value == 'south' or value == 'north':
                self.__xupper = self.zone_width
                self.__yupper = self.zone_depth
            elif value == 'east' or value == 'west':
                self.__xupper = self.zone_depth
                self.__yupper  = self.zone_width
            else:
                raise ("Orientation is not understandable")

    @property
    def zone_width(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_width

    @zone_width.setter
    def zone_width(self, value):
        self._zone_width = value

    @property
    def zone_depth(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_depth

    @zone_depth.setter
    def zone_depth(self, value):
        self._zone_depth = value

    @property
    def zone_height(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_height

    @zone_height.setter
    def zone_height(self, value):
        self._zone_height = value

    @property
    def wea_dir(self):
        return self._wea_dir

    @wea_dir.setter
    def wea_dir(self,value):
        self._wea_dir = value

    @property
    def U_factor(self):
        return self._U_factor

    @U_factor.setter
    def U_factor(self, value):
        self._U_factor = value

    @property
    def SHGC(self):
        return self._SHGC

    @SHGC.setter
    def SHGC(self, value):
        self._SHGC = value

    @property
    def stand(self):
        return self._stand

    @stand.setter
    def stand(self, value):
        self._stand = value
        if value:
            self.__testptsheight = np.linspace(0.1, 1.8, 3)
        else:
            self.__testptsheight = np.linspace(0.1, 1.3, 3)

    @property
    def WWR(self):
        return self._WWR


    @WWR.setter
    def WWR(self, value):
        try:
            if value < 1:
                warnings.warn("WWR ranges from 0 to 100. The WWR given is quite small")
        except:
            pass
        self._WWR = value

    @property
    def working_dir(self):
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    @property
    def room(self):
        return self._room
        # assert self._zone_name is not None
        # assert self._zone_width is not None
        # assert self._zone_depth is not None
        # assert self._zone_height is not None
        # assert self._WWR is not None
        # assert self._orientation is not None
        # assert self._observers is not None, 'No observer attached'
        # checker = self._observers(self._zone_width, self._zone_depth, self._zone_height)
        # if checker:
        #     warnings.warn("Regenerate Room object")
        #     self._room = self.__genRoom(self._zone_name, self._zone_width , self._zone_depth , self._zone_height ,
        #                    self._WWR , self._orientation)
        # return self._room


    @property
    def opaque_faces_geometry(self):
        return [self.room.faces[i].punched_geometry for i in range(6)]

    @property
    def glazing_faces_geometry(self):
        return [item.geometry for item in self.room.faces[self.__faceidbyori[self.orientation]].apertures]

    @property
    def viewfactor(self):
        return self._viewfactor

    @property
    def testPts(self):
        return self._testPts

    @staticmethod
    def __genRoom(zonename, width , depth , height , WWR , ori ,numGlz = 2,):
        if ori == 'south' or ori == 'north':
            w = width
            d = depth
        elif ori == 'east' or ori == 'west':
            w = depth
            d = width
        else:
            raise ("Orientation is not understandable")

        room = Room.from_box(zonename, w, d, height, 0,
                             Point3D(0, 0, 0))  # name, width, depth, height, , orientation_angle, origin
        assert list(room.faces[0].normal) == [0.0, 0.0, -1.0]
        assert list(room.faces[1].normal) == [0.0, 1.0, 0.0]
        assert list(room.faces[2].normal) == [1.0, 0.0, 0.0]
        assert list(room.faces[3].normal) == [0.0, -1.0, 0.0]
        assert list(room.faces[4].normal) == [-1.0, 0.0, 0.0]
        assert list(room.faces[5].normal) == [0.0, 0.0, 1.0]

        assert 'Bottom' in room.faces[0].name
        assert 'Front' in room.faces[1].name
        assert 'Right' in room.faces[2].name
        assert 'Back' in room.faces[3].name
        assert 'Left' in room.faces[4].name
        assert 'Top' in room.faces[5].name

        # Change the name to something more understandable
        inddict = {0: "floor", 1: "north", 2: "east", 3: "south", 4: "west", 5: "ceiling"}

        for i, face in enumerate(room.faces):
            if inddict[i] == ori:
                face.name = inddict[i] + "_" + 'exterior'
                exterior_id = i
            else:
                face.name = inddict[i] + "_" + 'interior'

        glz_id = {'north': 1, 'south': 3, 'east': 2, 'west': 4}

        def addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i):
            if ori == 'north':
                glz_pts = (Point3D(hor_lower, depth, ver_lower), Point3D(hor_lower, depth, ver_upper),
                           Point3D(hor_upper, depth, ver_upper), Point3D(hor_upper, depth, ver_lower))
                glz_face = Face3D(glz_pts)  ## Init without face
                glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)

            elif ori == 'south':
                glz_pts = (Point3D(hor_lower, 0, ver_lower), Point3D(hor_lower, 0, ver_upper),
                           Point3D(hor_upper, 0, ver_upper), Point3D(hor_upper, 0, ver_lower))
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
            hor_lower = glz_i * width / numGlz + width / numGlz / 2 - (width / numGlz) * np.sqrt(WWR / 100) / 2
            hor_upper = glz_i * width / numGlz + width / numGlz / 2 + (width / numGlz) * np.sqrt(WWR / 100) / 2
            ver_lower = height / 2 - height * np.sqrt(WWR / 100) / 2
            ver_upper = height / 2 + height * np.sqrt(WWR / 100) / 2

            assert (0 < hor_lower < hor_upper < width or 0 < ver_lower < ver_upper < height), "check WWR"
            addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i)
        return room

    def update(self):
        assert self._observers is not None, 'No observer attached'
        recalc_vf, regen_ep,regen_testpts = self._observers(self._zone_name, self._zone_width,
                                                            self._zone_depth, self._zone_height, self._WWR,
                                                            self._orientation)


        if regen_ep:
            logging.info("(Re)calculate ep")
            self._room = self.__genRoom(self.zone_name, self.zone_width, self.zone_depth, self.zone_height,
                                        self.WWR, self.orientation)
        else:
            logging.info("No need to update room object")

        if regen_testpts:
            logging.info("(Re)calculating test points")
            self._testPts = self.__gentestpts(0, self.__xupper, 0, self.__yupper, self.__testptsheight, 1, 1)
        else:
            logging.info("No need to update test points")

        if recalc_vf:
            logging.info("(Re)calculating view factor")
            self._viewfactor = self.__calcVF(self.testPts, self.opaque_faces_geometry + self.glazing_faces_geometry )
        else:
            logging.info("No need to update view factor")



        if not recalc_vf and not regen_ep and not regen_testpts:
            logging.info("No updates are needed")

    @staticmethod
    def __calcVF(testPts, faces):  # Calculate viewfactor,,, testPts must be numpy array of LB Point3D objects
        dome_vec = gen_dome()
        vecs = [Vector3D(*single_vector) for single_vector in dome_vec.T]

        def calcVF_ind(faces, testPt):  # VF for individual point

            faces_count = [[] for _ in range(len(faces))]  # generate empty nested list
            for vec in vecs:  # determine the intersection in all direction
                ray = Ray3D(testPt, vec)
                for face_i, face in enumerate(faces):
                    intersec = face.intersect_line_ray(ray)
                    if intersec:
                        faces_count[face_i].append(intersec)
                        # break

            return [len(item) / len(vecs) for item in
                    faces_count]  # View factor = fraction of the rays intercepted by that surface

        # A signle array contain view factors for all test points and faces

        VFs = [[calcVF_ind(faces, testpt_ind) for testpt_ind in first_list] for first_list in testPts]
        assert not (np.array(VFs).sum(axis=2) > 1.01).any()  # Raise flag if any of them is larger than 1.01
        assert not (np.array(VFs).sum(axis=2) < 0.99).any()  # Raise flag if any of them is less than 0.99

        # VFs.append(vfunc(testPts, face))

        return VFs

    @staticmethod
    def __gentestpts(x_lower, x_upper, y_lower, y_upper, z_height, x_size, y_size):
        # This function alwasy return list
        n_x = round((x_upper - x_lower) / x_size)
        x = np.linspace(x_lower, x_upper, n_x + 1)
        x_testpts = x[:-1] + np.diff(x) / 2

        n_y = round((y_upper - y_lower) / y_size)
        y = np.linspace(y_lower, y_upper, n_y + 1)
        y_testpts = y[:-1] + np.diff(y) / 2
        xy = list(itertools.product(x_testpts, y_testpts))

        if z_height is not None:

            if isinstance(z_height, (list, tuple, np.ndarray)):
                xyz = []
                for height in z_height:
                    z = np.ones((n_x * n_y, 1)) * height
                    temp_ = np.append(xy, z, axis=1).tolist()
                    converted_xyz = [Point3D(*point_i) for point_i in temp_]
                    xyz.append(converted_xyz)
                return xyz

            else:
                return np.array(xy)

    def bind_to(self, callback):
        logging.info('BOUND TO OBSERVER')
        self._observers = callback
    #         print(len(self._observers))

class observer(object):
    def __init__(self, data):
        self.name = None
        self.width = None
        self.depth = None
        self.height = None
        self.WWR = None
        self.orientation = None
        self.data = data
        self.data.bind_to(self.check_update)

    def check_update(self, name, width, depth, height, WWR, orientation):  # If self.width and self.
        recalc_vf = False
        regen_ep = False
        regen_testpts = False
        if name != self.name:
            regen_ep = True
        if WWR != self.WWR:
            recalc_vf = True
            regen_ep = True
        if (width != self.width) or (depth != self.depth) or (height != self.height) or (orientation != self.orientation):
            regen_ep = True
            recalc_vf = True
            regen_testpts = True
        # Update observer after checking the update
        self.name = name
        self.width = width
        self.depth = depth
        self.height = height
        self.WWR = WWR
        self.orientation = orientation
        return recalc_vf, regen_ep,regen_testpts



