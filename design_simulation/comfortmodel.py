import numpy as np
from ladybug_comfort.solarcal import body_solar_flux_from_horiz_parts,erf_from_body_solar_flux,mrt_delta_from_erf
from multiprocessing import Pool
import time
import logging
from .energy import EnergyModel
from .radiance import RadianceModel
from multiprocessing import Process,freeze_support
import pandas as pd

class ComfortModel:
    def __init__(self, initmodel, ):
        self.initmodel = initmodel
        # self._energymodel = energymodel
        # self._radiancemodel = radiancemodel

    @property
    def initmodel(self):   # grab the initial model from
        return self._initmodel

    @initmodel.setter
    def initmodel(self,value):
        self._initmodel = value
        self._energymodel = EnergyModel(value)
        self._radiancemodel = RadianceModel(value)


    @property
    def energymodel(self):
        return self._energymodel

    @property
    def radiancemodel(self):
        return self._radiancemodel


    @property
    def season_array(self):
        """
        0 -- shoulder
        1 -- winter
        2 -- summer
        :return:
        """

        try: return self._season_array
        except:
            year_range = pd.date_range(start="01/01/2017 00:00", end="12/31/2017 23:00", freq='H')
            summer_range = pd.date_range(start="06/20/2017 00:00", end="09/22/2017 23:00", freq='H')
            winter_range = pd.date_range(start="12/21/2017 00:00", end="12/31/2017 23:00", freq='H').append(
                            pd.date_range(start="01/01/2017 00:00", end="03/19/2017 23:00", freq='H'))
            self._season_array = year_range.isin(winter_range).astype(int) * 1 + year_range.isin(summer_range).astype(int) * 2
            return self._season_array
    @property
    def clo(self):
        try: return self.initmodel.clo
        except:
            clo = (self.season_array == 2) * 0.5 +\
                  (self.season_array == 1) * 1.0 + \
                  (self.season_array == 0) * 0.75
            return clo


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
    def airtemp_mapped(self):
        try: return self._airtemp_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._airtemp_mapped = np.repeat(self.energymodel.result.air_temperature, repeats= testpts_shape[0] * testpts_shape[1]).reshape(8760, -1).T.tolist()
            return self._airtemp_mapped

    @property
    def rh_mapped(self):
        try: return self._rh_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._rh_mapped = np.repeat(self.energymodel.result.relative_humidity,
                                             repeats=testpts_shape[0] * testpts_shape[1]).reshape(8760, -1).T.tolist()
            return self._rh_mapped

    @property
    def clo_mapped(self):
        try:
            return self._clo_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._clo_mapped = np.repeat(self.clo,
                                             repeats=testpts_shape[0] * testpts_shape[1]).reshape(8760, -1).T.tolist()
            return self._clo_mapped
    @property
    def airspeed_mapped(self): # this is only a temporaty implemeation
        testpts_shape = self.initmodel.testPts_shape
        return  np.empty(testpts_shape[0] * testpts_shape[1], 8760).fill(0.1)

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

# def load_radaince_results(model, mode):
#     if mode == 1:
#         epmodel = EnergyModel(model)
#         epmodel.result
#     elif model == 2:
#         pass



# def get_energy_and_radiance_at_the_same_time(model1, model2):
#     model = model1(model2)
#     model.result
#     return model

