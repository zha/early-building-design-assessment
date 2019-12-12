# from .utils import *
import warnings
import logging
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybug_geometry.geometry3d.ray import Ray3D
import functools
from honeybee.aperture import Aperture
from honeybee.room import Room as Room_energy
from ladybug_geometry.geometry3d.face import Face3D
from honeybeeradiance.hbfensurface import HBFenSurface
from honeybeeradiance.vectormath.euclid import Vector3, Point3
from honeybeeradiance.radiance.material.glass import Glass
from honeybeeradiance.room import Room as Room_rad
from ladybug.wea import Wea
from ladybug.sunpath import Sunpath
from ladybug_geometry.geometry3d.line import LineSegment3D
from ladybug_geometry.intersection2d import closest_point2d_on_line2d
from ladybug_geometry.intersection3d import closest_point3d_on_plane
import math
import pandas as pd
import seaborn as sns


import itertools

import numpy as np
from pathlib import Path
import os



# TODO: for update method, add assert to make sure that all the inputs are in
class ModelInit(object):
    __slots__ = ('_zone_name','_orientation', '_zone_width', '_zone_depth', '_zone_height',
                 '_U_factor', '_SHGC' , '_WWR', '_wea_dir', '_stand','_room', '_viewfactor',
                 '_testPts','__faceid', '_room', '__testptsheight', '_interior_wall_faces',
                 '_exterior_wall_face', '_floor_face', '_ceiling_face','__faceid_reversed',
                 '__faceid_rad_reversed', '__faceid_rad','_room_rad','_weather', '_sun_up_hoys',
                 '_sun_up_altitude','testPts_shape', '_angle_factors', '_dist_to_window',
                 '_testPts2D', '_fsvv',
                 '_working_dir', '_observers','__xupper', '__yupper', )

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
        self._weather = None
        self._observers = None
        self._sun_up_hoys = None
        self.__faceid = {0: "floor", 1: "north", 2: "east", 3: "south", 4: "west", 5: "ceiling"}
        self.__faceid_reversed = {v: k for k, v in self.__faceid.items()}
        self.__faceid_rad = {0: "south", 1: "east", 2:"north", 3:"west"}  # TODO: ADD ASSERT TO CHECK RADAINCE
        self.__faceid_rad_reversed = {v: k for k, v in self.__faceid_rad.items()}
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
    def room_rad(self):
        return self._room_rad

    @property
    def opaque_faces_geometry(self):  # this is useful for viewfactor calculation
        return [self.room.faces[i].punched_geometry for i in range(6)]

    @property
    def glazing_faces_geometry(self):  # this is useful for energyplus simulation
        return [item.geometry for item in self.room.faces[self.__faceid_reversed[self.orientation]].apertures]

    @property # this is useful for energyplus simulation
    def interior_wall_faces(self):
       return self._interior_wall_faces

    @property
    def exterior_wall_face(self): # this is useful for energyplus simulation
        return self._exterior_wall_face
    @property
    def floor_face(self): # this is useful for energyplus simulation
        return self._floor_face
    @property
    def ceiling_face(self): # this is useful for energyplus simulation
        return self._ceiling_face

    @property
    def apertures(self): # this is useful for energyplus simulation
        return self.exterior_wall_face.apertures

    @property
    def facenames(self):
        return [item.name for item in list(self.room.faces) + list(self.apertures)]
    @property
    def glzfacenames(self):
        return [item.name for item in self.apertures]

    @property
    def angle_factors(self):
        return self._angle_factors
    @property
    def distance_to_window(self):
        return self._dist_to_window
    @property
    def glzheight(self):
        ver_lower = self.zone_height / 2 - self.zone_height * np.sqrt(self.WWR / 100) / 2
        ver_upper = self.zone_height / 2 + self.zone_height * np.sqrt(self.WWR / 100) / 2
        return ver_upper - ver_lower


    @property
    def viewfactor(self):
        names = [item.name for item in list(self.room.faces) + list(self.apertures)]
        values = np.moveaxis(np.array(self._viewfactor), -1, 0).tolist()

        vf_dict  = dict(zip(names, values))
        return vf_dict

    @property
    def testPts(self):
        return self._testPts
    @property
    def testPts2D(self):
        try: return self._testPts2D
        except:
            self._testPts2D = np.array(self.testPts[0])[:,:-1]
            return self._testPts2D
    @property
    def testPts_x(self):
        return self.testPts2D[:,0]
    @property
    def testPts_y(self):
        return self.testPts2D[:, 1]

    @property
    def weather(self):
        if self._weather is None:
            self._weather = Wea.from_epw_file(self.wea_dir)
        return self._weather
    @property
    def location(self):
        return self.weather.location
    @property
    def sunpath(self):
        return Sunpath.from_location(self.location)
    @property
    def sun_up_hoys(self):
        if self._sun_up_hoys is None:
            suns = tuple(self.sunpath.calculate_sun_from_hoy(hoy) for hoy in range(8760))
            result_list = [[s.hoy, s.altitude] for s in suns if s.altitude > 0]
            self._sun_up_hoys = np.array(result_list).T[0].tolist()
            self._sun_up_altitude = np.array(result_list).T[1].tolist()

        return self._sun_up_hoys
    @property
    def sun_up_altitude(self):
        try: return self._sun_up_altitude
        except:
            _ = self.sun_up_hoys
            return self._sun_up_altitude


    @property
    def dist_to_window(self):
        return self._dist_to_window

    @property
    def fsvv(self):
        try: return self._fsvv
        except:
            dist_to_window_all_pts = np.array([self.dist_to_window[0]] * self.testPts_shape[0])
            self._fsvv = np.degrees(np.arctan(2 / (2 * dist_to_window_all_pts))) * np.degrees(np.arctan(2 / (2 * dist_to_window_all_pts))) / 90 / 180
            return self._fsvv

    def __genRoom(self,numGlz = 2,):

        self._room = Room_energy.from_box(self.zone_name, self.__xupper, self.__yupper, self.zone_height, 0,
                             Point3D(0, 0, 0))  # name, width, depth, height, , orientation_angle, origin

        self._room_rad = Room_rad(origin=(0, 0, 0), width=self.__xupper , depth=self.__yupper, height=self.zone_height,
                                    rotation_angle = 0)


        assert list(self._room.faces[0].normal) == [0.0, 0.0, -1.0]
        assert list(self._room.faces[1].normal) == [0.0, 1.0, 0.0]
        assert list(self._room.faces[2].normal) == [1.0, 0.0, 0.0]
        assert list(self._room.faces[3].normal) == [0.0, -1.0, 0.0]
        assert list(self._room.faces[4].normal) == [-1.0, 0.0, 0.0]
        assert list(self._room.faces[5].normal) == [0.0, 0.0, 1.0]

        assert 'Bottom' in self._room.faces[0].name
        assert 'Front' in self._room.faces[1].name
        assert 'Right' in self._room.faces[2].name
        assert 'Back' in self._room.faces[3].name
        assert 'Left' in self._room.faces[4].name
        assert 'Top' in self._room.faces[5].name

        # Change the name to something more understandable

        self._interior_wall_faces = []
        for i, face in enumerate(self._room.faces):
            if self.__faceid[i] == self.orientation:
                face.name = self.__faceid[i] + "_" + 'exterior'
                self._exterior_wall_face = face
            else:
                face.name = self.__faceid[i] + "_" + 'interior'
                if self.__faceid[i] == 'floor':
                    self._floor_face = face
                elif self.__faceid[i] == 'ceiling':
                    self._ceiling_face = face
                else:
                    self._interior_wall_faces.append(face)

        glz_id = {'north': 1, 'south': 3, 'east': 2, 'west': 4}

        def addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i):
            if self.orientation == 'north':
                ## EnergyPlus
                glz_pts = (Point3D(hor_lower, self.zone_depth, ver_lower), Point3D(hor_lower, self.zone_depth, ver_upper),
                           Point3D(hor_upper, self.zone_depth, ver_upper), Point3D(hor_upper, self.zone_depth, ver_lower))
                ## Radiance
                glz_pts_rad = (Point3(hor_lower, self.zone_depth, ver_lower), Point3(hor_lower, self.zone_depth, ver_upper),
                               Point3(hor_upper, self.zone_depth, ver_upper), Point3(hor_upper, self.zone_depth, ver_lower))

            elif self.orientation == 'south':
                ## EnergyPlus
                glz_pts = (Point3D(hor_lower, 0, ver_lower), Point3D(hor_lower, 0, ver_upper),
                           Point3D(hor_upper, 0, ver_upper), Point3D(hor_upper, 0, ver_lower))
                ## Radiance
                glz_pts_rad = (Point3(hor_lower, 0, ver_lower), Point3(hor_lower, 0, ver_upper),
                               Point3(hor_upper, 0, ver_upper), Point3(hor_upper, 0, ver_lower))


            elif self.orientation == 'east':
                ## EnergyPlus
                glz_pts = (Point3D(self.zone_depth, hor_lower, ver_lower), Point3D(self.zone_depth, hor_lower, ver_upper),
                           Point3D(self.zone_depth, hor_upper, ver_upper), Point3D(self.zone_depth, hor_upper, ver_lower))
                ## Radiance
                glz_pts_rad = (Point3(self.zone_depth, hor_lower, ver_lower), Point3(self.zone_depth, hor_lower, ver_upper),
                               Point3(self.zone_depth, hor_upper, ver_upper), Point3(self.zone_depth, hor_upper, ver_lower))

            elif self.orientation == 'west':
                ## EnergyPlus
                glz_pts = (Point3D(0, hor_lower, ver_lower), Point3D(0, hor_lower, ver_upper),
                           Point3D(0, hor_upper, ver_upper), Point3D(0, hor_upper, ver_lower))
                ## Radiance
                glz_pts_rad = (Point3(0, hor_lower, ver_lower), Point3(0, hor_lower, ver_upper),
                               Point3(0, hor_upper, ver_upper), Point3(0, hor_upper, ver_lower))
            ## EnergyPlus
            glz_face = Face3D(glz_pts)  ## Init without face
            glz_ape = Aperture('glz_{}'.format(glz_i), glz_face)
            self._room.faces[glz_id[self.orientation]].add_aperture(glz_ape)
            ## Radiance
            glz_face_rad = HBFenSurface('glz_{}'.format(glz_i), glz_pts_rad)
            # glass_mat_rad = Glass.by_single_trans_value('Tvis_{}'.format(glz_i), self.SHGC)  # TODO: NEED TO CHECK THIS ASSUMPTION
            # glz_face_rad.radiance_material = glass_mat_rad
            self._room_rad.walls[self.__faceid_rad_reversed[self.orientation]].add_fenestration_surface(glz_face_rad)


        for glz_i in range(numGlz):
            hor_lower = glz_i * self.zone_width / numGlz + self.zone_width / numGlz / 2 - (self.zone_width / numGlz) * np.sqrt(self.WWR / 100) / 2
            hor_upper = glz_i * self.zone_width / numGlz + self.zone_width / numGlz / 2 + (self.zone_width / numGlz) * np.sqrt(self.WWR / 100) / 2
            ver_lower = self.zone_height / 2 - self.zone_height * np.sqrt(self.WWR / 100) / 2
            ver_upper = self.zone_height / 2 + self.zone_height * np.sqrt(self.WWR / 100) / 2

            assert (0 < hor_lower < hor_upper < self.zone_width or 0 < ver_lower < ver_upper < self.zone_height), "check WWR"
            addGlz(hor_lower, hor_upper, ver_lower, ver_upper, glz_i)

    def update(self):
        assert self._observers is not None, 'No observer attached'
        recalc_vf, regen_ep,regen_testpts = self._observers(self._zone_name, self._zone_width,
                                                            self._zone_depth, self._zone_height, self._WWR,
                                                            self._orientation)


        if regen_ep:
            logging.info("(Re)calculate ep")
            self.__genRoom()
        else:
            logging.info("No need to update room object")

        if regen_testpts:
            logging.info("(Re)calculating test points")
            self._testPts = self.__gentestpts(0, self.__xupper, 0, self.__yupper, self.__testptsheight, 1, 1)
            self.testPts_shape = [np.array(self._testPts).shape[0], np.array(self._testPts).shape[1]]
        else:
            logging.info("No need to update test points")

        if recalc_vf: # in this step, calculate both view factors and angle factor
            logging.info("(Re)calculating view factor")
            self._viewfactor = self.__calcVF(self.testPts, self.opaque_faces_geometry + self.glazing_faces_geometry )
            self._angle_factors, self._dist_to_window = self.__calcAngleFactor(self.testPts[0], self.glazing_faces_geometry)
        else:
            logging.info("No need to update view factor")



        if not recalc_vf and not regen_ep and not regen_testpts:
            logging.info("No updates are needed")

    @staticmethod
    def __calcVF(testPts, faces):  # Calculate viewfactor,,, testPts must be numpy array of LB Point3D objects
        # dome_vec = gen_dome()
        # vecs = [Vector3D(*single_vector) for single_vector in dome_vec.T]
        parent_path = Path(__file__).parent
        vec_path = os.path.join(parent_path, 'dat', 'ray_values.csv')
        vecs_array = np.genfromtxt(vec_path, delimiter=',')
        vecs = [Vector3D(*single_vector) for single_vector in vecs_array.T]



        def calcVF_ind(faces, testPt):  # VF for individual point

            faces_count = [[] for _ in range(len(faces))]  # generate empty nested list
            for vec in vecs:  # determine the intersection in all direction
                ray = Ray3D(testPt, vec)
                for face_i, face in enumerate(faces):
                    intersec = face.intersect_line_ray(ray)
                    if intersec:
                        faces_count[face_i].append(intersec)
                        # break
            return [len(item) / len(vecs) for i ,item in
                    enumerate(faces_count)]  # View factor = fraction of the rays intercepted by that surface
        # A signle array contain view factors for all test points and faces

        VFs = [[calcVF_ind(faces, testpt_ind) for testpt_ind in first_list] for first_list in testPts]
        assert not (np.array(VFs).sum(axis=2) > 1.01).any()  # Raise flag if any of them is larger than 1.01
        assert not (np.array(VFs).sum(axis=2) < 0.99).any() # Raise flag if any of them is less than 0.99

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




    @staticmethod
    def __calcAngleFactor(testpts, glz_surfaces):  # angle factor is used for the draft calculation

        def valdiate_closest_point(pre_determined_test_point, glz_face):  # this function is used to validate and determine
                                                                        # if the closest point is in the glazing surface,
                                                                        # if not, a new cloest point will be reported

            twod_point = glz_face._plane.xyz_to_xy(pre_determined_test_point)
            ifin = glz_face.polygon2d.is_point_inside_bound_rect(twod_point)
            if ifin:
                the_closest_point = pre_determined_test_point

            elif not ifin:
                the_closest_point = None
                distance = np.inf
                for _s in glz_face.polygon2d.segments:
                    close_pt = closest_point2d_on_line2d(twod_point, _s)
                    if twod_point.distance_to_point(close_pt) < distance:
                        distance = twod_point.distance_to_point(close_pt)
                        the_closest_point = glz_face._plane.xy_to_xyz(close_pt)
                    else:
                        pass
            #         print(glz_face._plane.xy_to_xyz(the_closest_point))

            return the_closest_point

        anglefactors = []
        distance_to_glazing = []
        for glz_srf in glz_surfaces:
            ind_srf_angle_factor = []
            ind_srf_dist = []
            for testpt in testpts:
                closest = closest_point3d_on_plane(testpt, glz_srf.plane)
                closest = valdiate_closest_point(closest, glz_srf) # correct the cloest test point in case it is not on the galzing

                srfVec = Vector3D((closest - testpt).x, (closest - testpt).y, 0)
                angle2Srf = math.degrees(glz_srf.plane.n.angle(srfVec))
                if angle2Srf > 90:
                    angle2Srf = 180 - angle2Srf
                angFactor = (90 - abs(angle2Srf)) / 90
                ind_srf_angle_factor.append(angFactor)
                dist = glz_srf.plane.distance_to_point(testpt)
                ind_srf_dist.append(dist)
            anglefactors.append(ind_srf_angle_factor)
            distance_to_glazing.append(ind_srf_dist)
        return anglefactors, distance_to_glazing

    def __manual_verify_view_factor__(self, name_of_surface):  # This method is used to verify view factor information
        assert name_of_surface in self.viewfactor.keys(), self.viewfactor.keys()
        final_values = []
        for height_i in range(3):
            df = pd.DataFrame(
                {'vf': self.viewfactor[name_of_surface][height_i], 'x': self.testPts_x, 'y': self.testPts_y})

            df1 = pd.pivot_table(data=df, values='vf', columns='x', index='y').iloc[::-1]
            sns.heatmap(df1)
            final_values.append(df1)

        return final_values
    def __manual_verify_fsvv__(self):
        df = pd.DataFrame(
            {'fsvv': self.fsvv[0], 'x': self.testPts_x, 'y': self.testPts_y})

        df1 = pd.pivot_table(data=df, values='fsvv', columns='x', index='y').iloc[::-1]
        sns.heatmap(df1)
        return df1

    def __manual_verify_angle_facotr__(self):
        final_results = []
        for item in self.angle_factors:
            df = pd.DataFrame(
                {'angle_factor': item, 'x': self.testPts_x, 'y': self.testPts_y})

            df1 = pd.pivot_table(data=df, values='angle_factor', columns='x', index='y').iloc[::-1]
            sns.heatmap(df1)
            final_results.append(df1)
        return final_results

    def __manual_verify_dist_to_window__(self):
        df = pd.DataFrame({'dist_to_window': self.dist_to_window, 'x':self.testPts_x, 'y':self.testPts_y})
        df1 = pd.pivot_table(data=df, values='dist_to_window', columns='x', index='y').iloc[::-1]
        sns.heatmap(df1)
        return df1
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



