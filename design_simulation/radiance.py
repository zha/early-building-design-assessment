
from honeybeeradiance.radiance.material.glass import Glass
import numpy as np
from honeybeeradiance.radiance.analysisgrid import AnalysisGrid
from honeybeeradiance.radiance.sky.skymatrix import SkyMatrix
from honeybeeradiance.radiance.recipe.annual.gridbased import GridBased

import os
import logging
from pathlib import Path


from .radiancewrite import write
from multiprocessing import Pool

from multiprocessing import Process,freeze_support
import multiprocessing

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
            batch_file, project_folder = write(self.rp, target_folder=self.model.working_dir, project_name=self.model.zone_name)
            logging.info("Radiance batch file is located in {}".format(batch_file))
            logging.info('Now running Radiance')
            self.rp.run(batch_file, debug=False)
            logging.info('Radiance is now completed')
            self._result = RadianceResult(project_folder, self.rp)
            return self._result
import os
def matmul(mat1, mat2):
    print(os.getpid())
    return np.matmul(mat1, mat2).tolist()

def matop(mtx_dir, dc_dir, mode, return_val):
    print(mode)
    print(mtx_dir)
    print(dc_dir)

    if mode ==1 or mode ==2:
        with open(mtx_dir, 'r') as f:
            sky_mtx = f.readlines()

            # read in dc file
        with open(dc_dir, 'r') as f:
            dc = f.readlines()


        dc_str = ' '.join(dc[11:]).replace("\t", " ").replace("\n", " ")
        sky_mtx_str = ' '.join(sky_mtx[8:]).replace("\t", " ").replace("\n", " ")
        dc_parsed = np.fromstring(dc_str, sep=" ")
        sky_mtx_parsed = np.fromstring(sky_mtx_str, sep=" ")
        dc_parsed = dc_parsed.reshape(-1, 146, 3)
        sky_mtx_parsed = sky_mtx_parsed.reshape(146, 4447, 3)
                  # Matrix operation
        p = Pool(3)
        final = p.starmap(matmul, zip(dc_parsed.transpose(2, 0, 1), sky_mtx_parsed.transpose(2, 0, 1)))
        p.close()
        p.join()


    elif mode == 3:
        with open(dc_dir, 'r') as f:
            dc_sun = f.readlines()
        with open(mtx_dir, 'rb') as f:
            sun_mtx = np.load(f)

        dc_sun_str = ' '.join(dc_sun[10:]).replace("\t", " ").replace("\n", " ")
        dc_sun_parsed = np.fromstring(dc_sun_str, sep=" ")
        dc_sun_parsed = dc_sun_parsed.reshape(-1, 4447, 3)

        sun_mtx = np.array([np.diag(sun_mtx[:, i]) for i in range(3)])

        p = Pool()
        final = p.starmap(matmul, zip(dc_sun_parsed.transpose(2, 0, 1), sun_mtx))
        p.close()
        p.join()

    final = (np.array(final).transpose(1,2,0) * [47.4, 119.9, 11.6]).sum(axis = 2)
    # final = np.array(final)

    return_val[mode - 1] = final.tolist()

    # final = []
    #
    # for mat1, mat2 in zip(dc_total_parsed.transpose(2, 0, 1), sky_mtx_total_parsed.transpose(2, 0, 1)):
    #     final.append(np.matmul(mat1, mat2).tolist())






class RadianceResult:
    __slots__ = ('_rp', '_project_dir', '_scene_daylit','_scene_black_daylit', '_results',
                 '_scene_sun', '_total', '_direct', '_diffuse')
    def __init__(self, project_dir, rp):
        self._project_dir = project_dir
        self._rp = rp

    # @property
    # def scene_daylit(self):
    #     try: return self._scene_daylit
    #     except:
    #         logging.info("scene_daylit")
    #         with open(os.path.join(self._project_dir, self._rp._skyfiles.sky_mtx_total), 'r') as f:
    #             sky_mtx_total = f.readlines()
    #         with open(self._rp.result_files[0], 'r') as f:
    #             dc_total = f.readlines()
    #
    #         dc_total_str = ' '.join(dc_total[11:]).replace("\t", " ").replace("\n", " ")
    #         sky_mtx_total_str = ' '.join(sky_mtx_total[8:]).replace("\t", " ").replace("\n", " ")
    #         dc_total_parsed = np.fromstring(dc_total_str, sep=" ")
    #         sky_mtx_total_parsed = np.fromstring(sky_mtx_total_str, sep=" ")
    #         dc_total_parsed = dc_total_parsed.reshape(-1, 146, 3)
    #         sky_mtx_total_parsed = sky_mtx_total_parsed.reshape(146, 4447, 3)
    #                   # Matrix operation
    #         # p = Pool(3)
    #         # final = p.starmap(matmul, zip(dc_total_parsed.transpose(2, 0, 1), sky_mtx_total_parsed.transpose(2, 0, 1)))
    #         # p.close()
    #         # p.join()
    #
    #         final = []
    #
    #         for mat1, mat2 in zip(dc_total_parsed.transpose(2, 0, 1), sky_mtx_total_parsed.transpose(2, 0, 1)):
    #             final.append(np.matmul(mat1, mat2).tolist())
    #
    #
    #
    #         self._scene_daylit = np.array(final)
    #         return self._scene_daylit
    #
    #
    # @property
    # def scene_black_daylit(self):
    #     try: return self._scene_black_daylit
    #     except:
    #         logging.info("scene_black_daylit")
    #         with open(os.path.join(self._project_dir, self._rp._skyfiles.sky_mtx_direct), 'r') as f:
    #             sky_mtx_direct = f.readlines()
    #         with open(self._rp.result_files[1], 'r') as f:
    #             dc_direct = f.readlines()
    #
    #         dc_direct_str = ' '.join(dc_direct[11:]).replace("\t", " ").replace("\n", " ")
    #         sky_mtx_direct_str = ' '.join(sky_mtx_direct[8:]).replace("\t", " ").replace("\n", " ")
    #         dc_direct_parsed = np.fromstring(dc_direct_str, sep=" ")
    #         sky_mtx_direct_parsed = np.fromstring(sky_mtx_direct_str, sep=" ")
    #         dc_direct_parsed = dc_direct_parsed.reshape(-1, 146, 3)
    #         sky_mtx_direct_parsed = sky_mtx_direct_parsed.reshape(146, 4447, 3)
    #         ## Matrix operation
    #
    #
    #         # p = Pool()
    #         # final = p.starmap(matmul, zip(dc_direct_parsed.transpose(2, 0, 1), sky_mtx_direct_parsed.transpose(2, 0, 1)))
    #         # p.close()
    #         # p.join()
    #
    #         final = []
    #         for mat1, mat2 in zip(dc_direct_parsed.transpose(2, 0, 1), sky_mtx_direct_parsed.transpose(2, 0, 1)):
    #             final.append(np.matmul(mat1, mat2).tolist())
    #
    #         self._scene_black_daylit = np.array(final)
    #
    #         return self._scene_black_daylit
    #
    # @property
    # def scene_sun(self):
    #     try: return self._scene_sun
    #     except:
    #         logging.info("scene_sun")
    #         with open(self._rp.result_files[2], 'r') as f:
    #             dc_sun = f.readlines()
    #         dc_sun_str = ' '.join(dc_sun[10:]).replace("\t", " ").replace("\n", " ")
    #         dc_sun_parsed = np.fromstring(dc_sun_str, sep=" ")
    #         dc_sun_parsed = dc_sun_parsed.reshape(-1, 4447, 3)
    #
    #         parent_path = Path(__file__).parent
    #
    #         with open(os.path.join(parent_path,'dat', 'sunmtx.npy'), 'rb') as f:
    #             sun_mtx = np.load(f)
    #
    #
    #
    #         sun_mtx = np.array([np.diag(sun_mtx[:, i]) for i in range(3)])
    #         # p = Pool()
    #         # final = p.starmap(matmul, zip(dc_sun_parsed.transpose(2, 0, 1), sun_mtx))
    #         # p.close()
    #         # p.join()
    #
    #         final = []
    #         for mat1, mat2 in zip(dc_sun_parsed.transpose(2, 0, 1), sun_mtx):
    #             final.append(np.matmul(mat1, mat2).tolist())
    #
    #         self._scene_sun = np.array(final)
    #         return self._scene_sun

    @property
    def results(self):
        try: return self._results
        except:
            logging.info("Now calcualte the final Radiance result")
            # scene_total = self.scene_daylit
            # scene_direct = self.scene_black_daylit
            # scene_sun = self.scene_sun
            # p = Pool(processes= 10)
            # p.map(testone, [self.scene_daylit, self.scene_black_daylit, self.scene_sun])
            # p.close()
            # p.join()
            # freeze_support()
            #

            total_mtx_dir = os.path.join(self._project_dir, self._rp._skyfiles.sky_mtx_total)
            total_dc_dir = self._rp.result_files[0]

            direct_mtx_dir = os.path.join(self._project_dir, self._rp._skyfiles.sky_mtx_direct)
            direct_dc_dir =self._rp.result_files[1]

            parent_path = Path(__file__).parent
            sun_mtx_dir  =os.path.join(parent_path,'dat', 'sunmtx.npy')
            sun_dc_dir = self._rp.result_files[2]

            manager = multiprocessing.Manager()
            result_list = manager.list([None, None, None])
            p1 = Process(target=matop, args=(total_mtx_dir, total_dc_dir, 1,result_list))
            p2 = Process(target=matop, args=(direct_mtx_dir, direct_dc_dir, 2,result_list))
            p3 = Process(target=matop, args=(sun_mtx_dir, sun_dc_dir, 3, result_list))


            p1.start()
            p2.start()
            p3.start()

            p1.join()
            p2.join()
            p3.join()
            # pool = Pool(processes=3)
            #
            # [pool.apply_async(testone, args=(x,)) for x in [self.scene_daylit, self.scene_black_daylit]]
            result_list = list(result_list)
            self._total = result_list[0] - result_list[1] + result_list[2]
            self._direct = result_list[2]
            self._diffuse = self._total - self._direct
            self._results = [self._total, self._direct, self._diffuse]

            return self._results
            # self._final_result = scene_total + scene_direct
            # return self._final_result
