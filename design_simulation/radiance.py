
from honeybeeradiance.radiance.material.glass import Glass
import numpy as np
from honeybeeradiance.radiance.analysisgrid import AnalysisGrid
from honeybeeradiance.radiance.sky.skymatrix import SkyMatrix
from honeybeeradiance.radiance.recipe.annual.gridbased import GridBased

import os

from .radiancewrite import write
class RadianceModel(object):
    __slots__ = ('_room_rad', '_model', '_result')

    def __init__(self, model):
        self.model = model


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,value):
        self._model = value

    @property
    def room(self):

        # add in window properties before returing

        glass_mat = Glass.by_single_trans_value('GLASS_MAT_WITH_SHGC_{}'.format(self.model.SHGC), self.model.SHGC)  # todo: this assumption nedds to be checked


        for glz_face in self.model.room_rad.walls[self.model._ModelInit__faceid_rad_reversed[self.model.orientation]].children_surfaces:
            glz_face.radiance_material = glass_mat


        return self.model.room_rad

    @property
    def rp(self):
        sky = SkyMatrix(self.model.weather, hoys=self.model.sun_up_hoys)

        flat_testPts = np.array(self.model.testPts).reshape(-1, 3)
        analysis_grid = AnalysisGrid.from_points_and_vectors(flat_testPts)
        rp = GridBased(sky_mtx=sky, analysis_grids=(analysis_grid,), simulation_type=1,
                       hb_objects=(self.room,), reuse_daylight_mtx=False)  # ,radiance_parameters =  RfluxmtxParameters(0))

        return rp



    @property
    def result(self):
        batch_file = write(self.rp, target_folder=self.model.working_dir, project_name=self.model.zone_name)




class RadianceResult:
    def __init__(self):
        pass
