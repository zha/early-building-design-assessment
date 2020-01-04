
from honeybeeradiance.radiance.material.glass import Glass
import numpy as np
from honeybeeradiance.radiance.analysisgrid import AnalysisGrid
from honeybeeradiance.radiance.sky.skymatrix import SkyMatrix
from honeybeeradiance.radiance.recipe.annual.gridbased import GridBased
from honeybeeradiance.radiance.parameters.rfluxmtx import RfluxmtxParameters

import os
import logging
from pathlib import Path


from .radiancewrite import write
from multiprocessing import Pool

from multiprocessing import Process,freeze_support
import multiprocessing

import time

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

        glass_mat = Glass.by_single_trans_value('GLASS_MAT_WITH_TSOL_{}'.format(self.model.Tsol), self.model.Tsol)


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
            # j = {"gridbased_parameters": "-aa 0 -ab 8 -ad 4096 -dc 1 -st 0 -lw 0 -as 1024 -ar 0 -lr 16 -dt 0 -dr 6 -ds 0.02 -dp 0"}
            # params = RfluxmtxParameters.from_json(j)
            self._rp = GridBased(sky_mtx=sky, analysis_grids=(analysis_grid,), simulation_type=1,
                           hb_objects=(self.room,), reuse_daylight_mtx=False,)# radiance_parameters = params)  # ,radiance_parameters =  RfluxmtxParameters(0))

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
def matmul(mat1, mat2, i = None, result_list = None):
    print(os.getpid())
    result = np.matmul(mat1, mat2).tolist()
    if (i is not None ) & (result_list is not None):
        result_list[i] = result
    else:
        pass
    return result

def matop(mtx_dir, dc_dir, mode, return_val):
    # print(mode)
    # print(mtx_dir)
    # print(dc_dir)

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
        # start_time  = time.time()
        #
        # p = Pool(3)
        # final = p.starmap(matmul, zip(dc_parsed.transpose(2, 0, 1), sky_mtx_parsed.transpose(2, 0, 1)))
        # p.close()
        # p.join()
        # print(start_time - time.time())


        # start_time = time.time()
        dc_mtx = dc_parsed.transpose(2, 0, 1)
        sky_mtx = sky_mtx_parsed.transpose(2, 0, 1)
        print(dc_mtx.shape)
        final = []
        final.append(matmul(dc_mtx[0], sky_mtx[0] ))
        final.append(matmul(dc_mtx[1], sky_mtx[1]))
        final.append(matmul(dc_mtx[2], sky_mtx[2]))
        # print(start_time - time.time())



    elif mode == 3:
        with open(dc_dir, 'r') as f:
            dc_sun = f.readlines()
        with open(mtx_dir, 'rb') as f:
            sun_mtx = np.load(f)

        dc_sun_str = ' '.join(dc_sun[10:]).replace("\t", " ").replace("\n", " ")
        dc_sun_parsed = np.fromstring(dc_sun_str, sep=" ")
        dc_sun_parsed = dc_sun_parsed.reshape(-1, 4447, 3)

        sun_mtx = np.array([np.diag(sun_mtx[:, i]) for i in range(3)])

        dc_mtx = dc_sun_parsed.transpose(2, 0, 1)

        # final = []
        # final.append(matmul(dc_mtx[0], sun_mtx[0]))
        # final.append(matmul(dc_mtx[1], sun_mtx[1]))
        # final.append(matmul(dc_mtx[2], sun_mtx[2]))

        manager = multiprocessing.Manager()
        result_list = manager.list([None, None, None])
        p1 = Process(target=matmul, args=(dc_mtx[0], sun_mtx[0],0, result_list))
        p2 = Process(target=matmul, args=(dc_mtx[1], sun_mtx[1], 1, result_list))
        p3 = Process(target=matmul, args=(dc_mtx[2], sun_mtx[2], 2, result_list))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

        final = list(result_list)
        # p = Pool()
        # final = p.starmap(matmul, zip(dc_sun_parsed.transpose(2, 0, 1), sun_mtx))
        # p.close()
        # p.join()






    final = (np.array(final).transpose(1,2,0) * [47.4, 119.9, 11.6]).sum(axis = 2)
    # final = np.array(final)
    if return_val is None:
        pass
    else:
        return_val[mode - 1] = final.tolist()


    # return final

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

    @property
    def total(self):
        try: return self._total
        except:
            _ = self.results
            return self._total
    @property
    def direct(self):
        try: return self._direct
        except:
            _ = self.results
            return self._direct
    @property
    def diffuse(self):
        try: return self._diffuse
        except:
            _  =self.results
            return self._diffuse

    @property
    def results(self):
        try: return self._results
        except:
            logging.info("Now calculate the final Radiance result")
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


            time_start = time.time()
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
            print(time.time()-time_start)


            # time_start = time.time()
            # p = Pool()
            # p.starmap(matop, zip([total_mtx_dir,direct_mtx_dir,sun_mtx_dir],
            #                              [total_dc_dir,direct_dc_dir,sun_dc_dir],
            #                              [1,2,3], [None, None, None]))
            # p.close()
            # p.join()
            # print(time.time()-time_start)


            # pool = Pool(processes=3)
            #
            # [pool.apply_async(testone, args=(x,)) for x in [self.scene_daylit, self.scene_black_daylit]]
            result_list = (np.array(result_list) / 179 ).tolist()
            diffuse = np.array(result_list[0]) - np.array(result_list[1])
            diffuse[diffuse < 0] = 0
            self._diffuse = diffuse
            self._direct = result_list[2]
            self._total = self._direct + self._diffuse
            # self._diffuse = (np.array(self._total) - np.array(self._direct)).tolist()
            self._results = [self._total, self._direct, self._diffuse]

            return self._results
            # self._final_result = scene_total + scene_direct
            # return self._final_result
