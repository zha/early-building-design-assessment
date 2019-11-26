import numpy as np
from ladybug_comfort.solarcal import body_solar_flux_from_horiz_parts,erf_from_body_solar_flux,mrt_delta_from_erf
from multiprocessing import Pool
import time
import logging
from multiprocessing import Process,freeze_support

class ComfortModel:
    def __init__(self, initmodel, energymodel, radiancemodel):
        self._initmodel = initmodel
        self._energymodel = energymodel
        self._radiancemodel = radiancemodel

    @property
    def initmodel(self):   # grab the initial model from
        return self._initmodel

    @property
    def energymodel(self):
        return self._energymodel

    @property
    def radiancemodel(self):
        return self._radiancemodel

    @property
    def LW_MRT(self):
        try: return self._LW_MRT
        except:
            vf_array = np.array(list(self.initmodel.viewfactor.values()))
            vf_array = vf_array.reshape(vf_array.shape[0], -1).T
            surfacetemps = np.array(list(self.energymodel.result.surfacetemps.values()))
            self._LW_MRT = np.matmul(vf_array, (surfacetemps + 273.15) ** 4) ** 0.25 - 273.15
            return self._LW_MRT

    @property
    def delta_MRT(self):

        # return type
        try: return self._delta_mrt
        except:

            direct = np.array(self.radiancemodel.result.direct)
            origional_dims = direct.shape
            diffuse = np.array(self.radiancemodel.result.diffuse)
            sun_up_altitude = np.repeat(self.initmodel.sun_up_altitude, repeats=direct.shape[0]).reshape(-1, direct.shape[0]).T
            direct = direct.reshape(-1).tolist()
            diffuse = diffuse.reshape(-1).tolist()
            sun_up_altitude = sun_up_altitude.reshape(-1).tolist()

            start_time = time.time()

            p1 = Pool(4)
            bodyrad = p1.starmap(body_solar_flux_from_horiz_parts, zip(diffuse, direct, sun_up_altitude))
            p1.close()
            p1.join()

            p2 = Pool(4)
            erf = p2.map(erf_from_body_solar_flux, list(bodyrad))
            p2.close()
            p2.join()

            p3 = Pool(4)
            dmrt = p3.map(erf_from_body_solar_flux, list(erf))
            p3.close()
            p3.join()


            logging.info(time.time() - start_time)
            self._delta_mrt = np.array(list(dmrt)).reshape(*origional_dims)

            return self._delta_mrt

    @property
    def totalMRT(self):
        try: return self._totalMRT
        except:

            dmrt = self.delta_MRT
            orimrt = self.LW_MRT
            sun_up_hoys = np.array(self.initmodel.sun_up_hoys).astype(int).tolist()
            dmrt8760 = np.empty((orimrt.shape[0], orimrt.shape[1]))
            dmrt8760[:, sun_up_hoys] = dmrt
            self._totalMRT = orimrt + dmrt8760
            return self._totalMRT



    @property
    def PMV(self):
        try: return self._PMV
        except:
            pass
    # def runall(self):
    #     p1 = Process(target=get_energy_and_radiance_at_the_same_time, args=(self.energymodel, self.initmodel))
    #     p2 = Process(target=get_energy_and_radiance_at_the_same_time, args=(self.radiancemodel, self.initmodel))
    #     p1.start()
    #     p2.start()
    #     p1.join()
    #     p2.join()


# def get_energy_and_radiance_at_the_same_time(model1, model2):
#     model = model1(model2)
#     model.result
#     return model

