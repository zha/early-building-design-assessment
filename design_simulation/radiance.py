
from honeybeeradiance.radiance.material.glass import Glass
import numpy as np
from honeybeeradiance.radiance.analysisgrid import AnalysisGrid
from honeybeeradiance.radiance.sky.skymatrix import SkyMatrix
from honeybeeradiance.radiance.recipe.annual.gridbased import GridBased

import os
import logging

from .radiancewrite import write
class RadianceModel(object):
    __slots__ = ('_room_rad', '_model', '_result', '_rp')

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
        try: return self._rp
        except:
            sky = SkyMatrix(self.model.weather, hoys=self.model.sun_up_hoys)

            flat_testPts = np.array(self.model.testPts).reshape(-1, 3)
            analysis_grid = AnalysisGrid.from_points_and_vectors(flat_testPts)
            self._rp = GridBased(sky_mtx=sky, analysis_grids=(analysis_grid,), simulation_type=1,
                           hb_objects=(self.room,), reuse_daylight_mtx=False)  # ,radiance_parameters =  RfluxmtxParameters(0))

            return self._rp



    @property
    def result(self):
        try: return self._result
        except:
            logging.info("No pre-exisitng results found, now running Radaince")
            batch_file = write(self.rp, target_folder=self.model.working_dir, project_name=self.model.zone_name)
            logging.info("Radiance batch file is located in {}".format(batch_file))
            logging.info('Now running Radiance')
            self.rp.run(batch_file, debug=False)
            logging.info('Radiance is now completed')
            self._result = RadianceResult(self.model.working_dir, self.rp)
            return self._result




class RadianceResult:
    __slots__ = ('_rp', '_work_dir', '_scene_daylit')
    def __init__(self, work_dir, rp):
        self._work_dir = work_dir
        self._rp = rp

    @property
    def scene_daylit(self):
        try: return self._scene_daylit
        except:
            with open(os.path.join(self._work_dir, self._rp._skyfiles.sky_mtx_total), 'rb') as f:
                sky_mtx_total = f.readlines()
            with open(self._rp.result_files[0], 'rb') as f:
                dc_ttoal = f.readlines()

            self._scene_daylit = 0
        return 0

