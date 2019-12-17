import numpy as np
from ladybug_comfort.solarcal import body_solar_flux_from_horiz_parts, erf_from_body_solar_flux, mrt_delta_from_erf, \
    indoor_sky_heat_exch, body_solar_flux_from_parts
from multiprocessing import Pool
import time
import logging
from .energy import EnergyModel
from .radiance import RadianceModel
from multiprocessing import Process, freeze_support
import pandas as pd
from ladybug_comfort.pmv import fanger_pmv
import multiprocessing
import math
from ladybug_comfort.pmv import ppd_from_pmv
from ladybug.dt import DateTime
from functools import partial


class ComfortModel:
    def __init__(self, initmodel, ):
        self.initmodel = initmodel
        # self._energymodel = energymodel
        # self._radiancemodel = radiancemodel

    @property
    def initmodel(self):  # grab the initial model from
        return self._initmodel

    @initmodel.setter
    def initmodel(self, value):
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

        try:
            return self._season_array
        except:
            year_range = pd.date_range(start="01/01/2017 00:00", end="12/31/2017 23:00", freq='H')
            summer_range = pd.date_range(start="06/20/2017 00:00", end="09/22/2017 23:00", freq='H')
            winter_range = pd.date_range(start="12/21/2017 00:00", end="12/31/2017 23:00", freq='H').append(
                pd.date_range(start="01/01/2017 00:00", end="03/19/2017 23:00", freq='H'))
            self._season_array = year_range.isin(winter_range).astype(int) * 1 + year_range.isin(summer_range).astype(
                int) * 2
            return self._season_array

    @property
    def clo(self):
        """
        return:
            numpy array with shape (8760,)
        """

        try:
            return self.initmodel.clo
        except:
            clo = (self.season_array == 2) * 0.5 + \
                  (self.season_array == 1) * 1.0 + \
                  (self.season_array == 0) * 0.75
            return clo

    @property
    def LW_MRT(self):
        """
        return:
            numpy array with shape (# of test points, 8760)

        """
        try:
            return self._LW_MRT
        except:
            vf_array = np.array(list(self.initmodel.viewfactor.values()))
            assert (vf_array.shape[0] == 8) & (vf_array.shape[1:] == tuple(self.initmodel.testPts_shape))
            vf_array = vf_array.reshape(vf_array.shape[0],
                                        -1).T  # this is to make into shape (#_of_testpts, #_of_walls)
            surfacetemps = np.array(list(self.energymodel.result.surfacetemps.values()))  # shape (#_of_walls, 8760)
            self._LW_MRT = np.matmul(vf_array, (surfacetemps + 273.15) ** 4) ** 0.25 - 273.15
            return self._LW_MRT

    def pd_mapped_2D(self, var_name, hour_i=None, month=None, day=None, hod=None, height_i=0):  # height should be index
        """
        args:
            var_name : the name of the variable that need to be mapped. Curretnly supported
                        Supported types:
        """
        if hour_i is not None:
            assert hour_i in range(8760)
            hoy = hour_i
        elif all([month is not None, day is not None, hod is not None]):
            assert (month in range(12)) and (day in range(31)) and (hod in range(24))
            hoy = DateTime(month=month, day=day, hour=hod).int_hoy
        else:
            raise Exception("Something inputs to this function is not right. Check your inputs ")

        xy = self.initmodel.testPts2D.T
        origional_data = self.__getattribute__(var_name)
        origional_data_shape = origional_data.shape
        if origional_data_shape[0] == self.initmodel.testPts_shape[0] * self.initmodel.testPts_shape[
            1]:  ##  Then this data contains all testpoints
            reshaped = origional_data.reshape(self.initmodel.testPts_shape[0], self.initmodel.testPts_shape[1], 8760)
            mapped_data = self.__generate_pivot_table_2D(xy, reshaped[height_i, :, hoy])
        elif origional_data_shape[0] == self.initmodel.testPts_shape[1]:  ## This data only contain the 2d test points
            mapped_data = self.__generate_pivot_table_2D(xy, origional_data[:, hoy])
        else:
            raise Exception("Shape of the input . Need to debug the code manually")

        return mapped_data

        # reshaped = self.LW_MRT.reshape(3,-1, 8760)
        # partialfunc = partial(self.generate_pivot_table_2D, self.initmodel.testPts2D.T)
        # return np.apply_along_axis(partialfunc, axis = 1 , arr= reshaped)

    @staticmethod
    def __generate_pivot_table_2D(xy, data):
        df = pd.DataFrame({"X": xy[0], 'Y': xy[1], 'values': data})
        return pd.pivot_table(df, values='values', index=['Y'], columns=['X'])

    @property
    def direct_all_hoys(self):
        try:
            return self._direct_all_hoys
        except:
            sun_up_array = np.array(self.radiancemodel.result.direct)
            sun_up_hoys = np.array(self.initmodel.sun_up_hoys).astype(int).tolist()
            direct_all_hoys = np.empty((sun_up_array.shape[0], 8760))
            direct_all_hoys[:, sun_up_hoys] = sun_up_array
            self._direct_all_hoys = direct_all_hoys
            return self._direct_all_hoys

    @property
    def diffuse_all_hoys(self):
        try:
            return self._diffuse_all_hoys
        except:
            sun_up_array = np.array(self.radiancemodel.result.diffuse)
            sun_up_hoys = np.array(self.initmodel.sun_up_hoys).astype(int).tolist()
            diffuse_all_hoys = np.empty((sun_up_array.shape[0], 8760))
            diffuse_all_hoys[:, sun_up_hoys] = sun_up_array
            self._diffuse_all_hoys = diffuse_all_hoys
            return self._diffuse_all_hoys

    @property
    def delta_MRT(self):
        """
        return:
            numpy array with shape (# of test points * # of heights, 8760)

        """
        try:
            return self._delta_mrt
        except:

            direct = np.array(self.radiancemodel.result.direct)
            origional_dims = direct.shape
            diffuse = np.array(self.radiancemodel.result.diffuse)
            sun_up_altitude = np.repeat(self.initmodel.sun_up_altitude, repeats=direct.shape[0]).reshape(-1,
                                                                                                         direct.shape[
                                                                                                             0]).T
            assert sun_up_altitude.shape == direct.shape == diffuse.shape
            direct = direct.reshape(-1).tolist()
            diffuse = diffuse.reshape(-1).tolist()
            sun_up_altitude = sun_up_altitude.reshape(-1).tolist()
            assert self.initmodel.fsvv.shape == tuple(self.initmodel.testPts_shape)
            fsvv = np.array([self.initmodel.fsvv.reshape(-1).tolist()] * len(self.initmodel.sun_up_hoys)).T
            assert fsvv.shape == np.array(self.radiancemodel.result.diffuse).shape == np.array(
                self.radiancemodel.result.direct).shape
            fsvv = fsvv.reshape(-1).tolist()

            # start_time = time.time()
            #
            # p1 = Pool(4)
            # bodyrad = p1.starmap(body_solar_flux_from_horiz_parts, zip(diffuse, direct, sun_up_altitude))
            # p1.close()
            #
            # p2 = Pool(4)
            # erf = p2.map(erf_from_body_solar_flux, list(bodyrad))
            # p2.close()
            #
            # p3 = Pool(4)
            # dmrt = p3.map(mrt_delta_from_erf, list(erf))
            # p3.close()
            #
            #
            # p1.join()
            # p2.join()
            # p3.join()
            # logging.info('it takes' + str(time.time() - start_time)+ 'to calucate dMRT')
            #
            # # method 2
            # start_time = time.time()
            # bodyrad = list(map(body_solar_flux_from_horiz_parts, diffuse, direct, sun_up_altitude))
            # erf = list(map(erf_from_body_solar_flux, bodyrad))
            # dmrt = list(map(mrt_delta_from_erf, erf))
            # logging.info(time.time()- start_time)
            # method 3
            start_time = time.time()
            bodyrad = [body_solar_flux_from_parts(diff_horiz_solar=diffuse, dir_normal_solar=direct,
                                                  altitude=sun_up_altitude, sky_exposure=sky_exposure,
                                                  )
                       for diffuse, direct, sun_up_altitude, sky_exposure in
                       zip(diffuse, direct, sun_up_altitude, fsvv)]
            erf = list(map(erf_from_body_solar_flux, bodyrad))
            dmrt = list(map(mrt_delta_from_erf, erf))
            logging.info(time.time() - start_time)
            # method 4
            # split_input = np.array_split([diffuse, direct, sun_up_altitude], 4, axis =1)
            # start_time = time.time()
            # manager = multiprocessing.Manager()
            # result_list = manager.list([None, None, None, None])
            # p1 = Process(target=calc_dmrt, args=(0, result_list, *split_input[0]))
            # p2 = Process(target=calc_dmrt, args=(1, result_list, *split_input[1]))
            # p3 = Process(target=calc_dmrt, args=(2, result_list, *split_input[2]))
            # p4 = Process(target=calc_dmrt, args=(3, result_list, *split_input[3]))
            # p1.start()
            # p2.start()
            # p3.start()
            # p4.start()
            #
            # p1.join()
            # p2.join()
            # p3.join()
            # p4.join()
            # result_list = list(result_list)
            # logging.info(time.time()- start_time)

            delta_mrt_sun_up_hoys = np.array(list(dmrt)).reshape(*origional_dims)
            sun_up_hoys = np.array(self.initmodel.sun_up_hoys).astype(int).tolist()
            dmrt8760 = np.empty((delta_mrt_sun_up_hoys.shape[0], 8760))
            dmrt8760[:, sun_up_hoys] = delta_mrt_sun_up_hoys
            self._delta_mrt = dmrt8760

            return self._delta_mrt

    @property
    def totalMRT(self):  # This is for all test points and for all heights
        """
        return:
            numpy array with shape (# of test points * # of heights, 8760)

        """
        try:
            return self._totalMRT
        except:
            dmrt = self.delta_MRT
            orimrt = self.LW_MRT
            self._totalMRT = orimrt + dmrt
            return self._totalMRT

    # @property
    # def fsvv_mapped(self):
    #     np.array([self.initmodel.fsvv.tolist()] * 8760).transpose()
    #     return

    @property
    def airtemp_mapped(self):
        """
        return:
            numpy array with shape (# number of test pts with one height, 8760)

        """
        try:
            return self._airtemp_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._airtemp_mapped = np.repeat(self.energymodel.result.air_temperature,
                                             repeats=testpts_shape[1]).reshape(8760, -1).T
            return self._airtemp_mapped

    @property
    def rh_mapped(self):
        """
        return:
            numpy array with shape (# number of test pts with one height, 8760)

        """
        try:
            return self._rh_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._rh_mapped = np.repeat(self.energymodel.result.relative_humidity,
                                        repeats=testpts_shape[1]).reshape(8760, -1).T
            return self._rh_mapped

    @property
    def clo_mapped(self):

        """clothing level
        return:
            numpy array with shape (# number of test pts with one height, 8760)

        """
        try:
            return self._clo_mapped
        except:
            testpts_shape = self.initmodel.testPts_shape
            self._clo_mapped = np.repeat(self.clo,
                                         repeats=testpts_shape[1]).reshape(8760, -1).T
            return self._clo_mapped

    @property
    def airspeed_mapped(self):  # this is only a temporary implementation
        """airspeed
        return:
            numpy array with shape (# number of test pts with one height, 8760)

        """
        testpts_shape = self.initmodel.testPts_shape
        airspeed_array = np.empty((testpts_shape[1], 8760))
        airspeed_array.fill(0.1)
        return airspeed_array

    @property
    def met_mapped(self):
        """airspeed
        return:
            numpy array with shape (# number of test pts with one height, 8760)

        """
        testpts_shape = self.initmodel.testPts_shape
        met_array = np.empty((testpts_shape[1], 8760))
        met_array.fill(1)
        return met_array

    @property
    def firstPMV(self):
        """NO NEED TO RUN DIRECTLY,  RUN unadjustedPMV right below INSTEAD"""

        try:
            return self._PMV, self._heat_loss
        except:
            airtemp = self.airtemp_mapped  # .reshape(-1)
            mrt = self.totalMRT.reshape(self.initmodel.testPts_shape[0], self.initmodel.testPts_shape[1], -1).mean(
                axis=0)  # .reshape(-1)
            rh = self.rh_mapped  # .reshape(-1)
            clo = self.clo_mapped  # .reshape(-1)
            airspeed = self.airspeed_mapped  # .reshape(-1)
            met = self.met_mapped  # .reshape(-1)

            assert airtemp.shape == mrt.shape == rh.shape == clo.shape == airspeed.shape == met.shape
            origional_array_shape = airtemp.shape
            # Now flatten the arrays
            airtemp = airtemp.reshape(-1)
            mrt = mrt.reshape(-1)
            rh = rh.reshape(-1)
            clo = clo.reshape(-1)
            airspeed = airspeed.reshape(-1)
            met = met.reshape(-1)

            logging.info('Calculating PMV and PPD, weeeeee...')
            allarray = np.array([airtemp, mrt, airspeed, rh, met, clo])
            split_input = np.array_split(allarray, 4, axis=1)  # chop up the list to four pieces
            # start_time = time.time()
            #
            # proc_fanger_calc(*split_input[0])
            # proc_fanger_calc(*split_input[1])
            # proc_fanger_calc(*split_input[2])
            # proc_fanger_calc(*split_input[3])
            # logging.info(time.time()- start_time)
            # start_time = time.time()
            # pmv = np.vectorize(fanger_pmv)
            # self.result_1 = pmv(*allarray)
            # logging.info(time.time() - start_time)

            start_time = time.time()
            manager = multiprocessing.Manager()
            result_list = manager.list([None, None, None, None])
            p1 = Process(target=proc_fanger_calc, args=(0, result_list, *split_input[0]))
            p2 = Process(target=proc_fanger_calc, args=(1, result_list, *split_input[1]))
            p3 = Process(target=proc_fanger_calc, args=(2, result_list, *split_input[2]))
            p4 = Process(target=proc_fanger_calc, args=(3, result_list, *split_input[3]))
            p1.start()
            p2.start()
            p3.start()
            p4.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            result_list = list(result_list)
            logging.info(time.time() - start_time)
            self._PMV = np.concatenate((result_list[0][0], result_list[1][0],
                                        result_list[2][0], result_list[3][0])).reshape(*origional_array_shape)
            # self._PPD = np.concatenate((result_list[0][1], result_list[1][1],
            #                             result_list[2][1], result_list[3][1])).reshape(*origional_array_shape)
            self._heat_loss = np.concatenate((result_list[0][1], result_list[1][1],
                                              result_list[2][1], result_list[3][1])).reshape(*origional_array_shape)
            return self._PMV, self._heat_loss
            # self.result_2 = result_list
            # p1 = Pool(4)
            # p1.starmap(fanger_pmv, zip(airtemp, mrt, airspeed, rh, met, clo))
            # p1.close()
            # p1.join()
            # logging.info(time.time()- start_time)

    @property
    def unadjustedPMV(self):
        """
        return:
            numpy array with shape (# of test points; one height, 8760)

        """
        try:
            return self._PMV
        except:
            _ = self.firstPMV  # Run the above fcuntion if the above reuslt is not already exits
            return self._PMV

    @property
    def downdraft_speed_and_temperature(self):
        """
        return:
            draft_speed: numpy array with shape (# of test points; one height, 8760)
            draft_temperature: (# of test points; one height, 8760)
        """
        try:
            return self._draft_speed, self._draft_temp
        except:
            # collect all of the necessary information before caluclation
            airTemp = np.squeeze(self.energymodel.result.air_temperature)  # temporal
            winsrfsTemp = np.array(list(self.energymodel.result.glztemps.values())).T  # temporal

            spreadFac = .97  # spatial
            windowHgt = self.initmodel.glzheight  # spatial

            dists = self.initmodel.distance_to_window
            angFacs = self.initmodel.angle_factors
            num_of_glazing = np.array(dists).shape[0]
            num_of_pts = np.array(dists).shape[1]

            def calcVelTemp_i(airTemp, winSrfTempFinal):

                ptVelLists = []
                ptTemplists = []
                for srfCount in range(num_of_glazing):

                    ptVelLists.append([])
                    ptTemplists.append([])
                    for ptCount in range(num_of_pts):

                        # Compute the temperature difference.
                        glassAirDelta = airTemp - winSrfTempFinal[srfCount]
                        if glassAirDelta > 0:

                            dist = dists[srfCount][ptCount]
                            angFac = angFacs[srfCount][ptCount]

                            if dist < 0.4:
                                windSpd = self.velMaxClose(glassAirDelta, windowHgt)
                            elif dist < 2:
                                windSpd = self.velMaxMid(dist, glassAirDelta, windowHgt)
                            else:
                                windSpd = self.velMaxFar(glassAirDelta, windowHgt)
                            floorAirTemp = self.calcFloorAirTemp(airTemp, dist, glassAirDelta)

                            ptVelLists[srfCount].append((windSpd * ((angFac / (1 / spreadFac)) + (1 - spreadFac))))
                            ptTemplists[srfCount].append(
                                airTemp - ((airTemp - floorAirTemp) * ((angFac / (1 / spreadFac)) + (1 - spreadFac))))
                        else:
                            ptVelLists[srfCount].append(0)
                            ptTemplists[srfCount].append(airTemp)

                # Finally, take the max of the velocity and minimum temperature if there are multiple window surfaces
                ptVelLists = np.amax(ptVelLists, axis=0)
                ptTemplists = np.amin(ptTemplists, axis=0)
                return ptVelLists, ptTemplists

            start_time = time.time()

            final_list = [calcVelTemp_i(inputs[0], inputs[1]) for inputs in zip(airTemp, winsrfsTemp)]
            self._draft_speed = np.array(final_list)[:, 0, :].T
            self._draft_temp = np.array(final_list)[:, 1, :].T

            print(time.time() - start_time)
            return self._draft_speed, self._draft_temp

    def __manual_validate_downdraft_speed_and_temperature__(self):
        assert len(self.initmodel.glazing_faces_geometry) == 2, 'currrently it only works for 2 glazing surface'
        spreadFac = .97  # spatial
        windowHgt = self.initmodel.glzheight  # spatial
        start_time = time.time()
        airtemp = np.array(
            [np.squeeze(self.energymodel.result.air_temperature).tolist()] * self.initmodel.testPts_shape[1])
        winsrfsTemp = np.array([[srftemp] * self.initmodel.testPts_shape[1] for srftemp in
                                list(self.energymodel.result.glztemps.values())])
        dists = np.moveaxis([self.initmodel.distance_to_window] * 8760, 0, -1)  # shape = (#_of_glz, #_of_testpts, 8760)
        angFacs = np.moveaxis([self.initmodel.angle_factors] * 8760, 0, -1)
        print(time.time() - start_time)

        def calcVelTemp(airtemp_i, srftemp_i, dist_i, angfac_i):
            glassAirDelta = airtemp_i - srftemp_i
            if glassAirDelta > 0:
                if dist_i < 0.4:
                    windSpd = self.velMaxClose(glassAirDelta, windowHgt)
                elif dist_i < 2:
                    windSpd = self.velMaxMid(dist_i, glassAirDelta, windowHgt)
                else:
                    windSpd = self.velMaxFar(glassAirDelta, windowHgt)
                floorAirTemp = self.calcFloorAirTemp(airtemp_i, dist_i, glassAirDelta)
                vel = windSpd * ((angfac_i / (1 / spreadFac)) + (1 - spreadFac))
                temp = airtemp_i - ((airtemp_i - floorAirTemp) * ((angfac_i / (1 / spreadFac)) + (1 - spreadFac)))
                return vel, temp
            else:
                return 0, airtemp_i

        results_all = []
        for glz_i in range(2):
            results_all.append(
                list(map(calcVelTemp, airtemp.reshape(-1), winsrfsTemp[glz_i].reshape(-1), dists[glz_i].reshape(-1),
                         angFacs[glz_i].reshape(-1))))
        # results_all should have shape of (#_of_glz, totla, 2 (parameters))
        max_vel = np.array(results_all)[:, :, 0].max(axis=0)
        min_temp = np.array(results_all)[:, :, 1].min(axis=0)
        max_vel = max_vel.reshape(-1, 8760)
        min_temp = min_temp.reshape(-1, 8760)

        assert (max_vel == self.downdraft_speed_and_temperature[0]).all()
        assert (min_temp == self.downdraft_speed_and_temperature[1]).all()
        return True

    @property
    def draft_speed(self):
        """
        return:
            numpy array with shape of (# of test points; one height, 8760)

        """
        try:
            return self._draft_speed
        except:
            _ = self.downdraft_speed_and_temperature
            return self._draft_speed

    @property
    def draft_temp(self):
        """
        return:
            numpy array with shape of (# of test points; one height, 8760)

        """
        try:
            return self._draft_temp
        except:
            _ = self.downdraft_speed_and_temperature
            return self._draft_temp

    @property
    def draft_adjusted_PMV(self):
        """
         return:
             numpy array with shape of (# of test points; one height, 8760)

         """
        try:
            return self._draft_adjusted_PMV
        except:
            assert self.draft_speed.shape == self.draft_temp.shape == self.unadjustedPMV.shape
            origional_shape = self.unadjustedPMV.shape
            draft_speed = self.draft_speed.reshape(-1)
            draft_temp = self.draft_temp.reshape(-1)
            unadjustedPMV = self.unadjustedPMV.reshape(-1)

            start_time = time.time()
            adjusted = list(map(pmv_draft_adjustment, draft_speed, unadjustedPMV, draft_temp))
            logging.info('It takes ' + str(time.time() - start_time) +
                         ' to calculate the adjusted PMV')
            self._draft_adjusted_PMV = np.array(adjusted).reshape(*origional_shape)
            return self._draft_adjusted_PMV

    @property
    def draft_adjusted_PPD(self):
        """
         return:
             numpy array with shape of (# of test points; one height, 8760)

         """
        try:
            return self._draft_adjusted_PPD
        except:
            draft_adjusted_PMV = self.draft_adjusted_PMV
            origional_shape = draft_adjusted_PMV.shape
            draft_adjusted_PMV = draft_adjusted_PMV.reshape(-1)

            draft_adjusted_PPD = np.array(list(map(ppd_from_pmv, draft_adjusted_PMV))).reshape(*origional_shape)
            self._draft_adjusted_PPD = draft_adjusted_PPD
            return self._draft_adjusted_PPD

            # assert self.draft_speed.shape == self.draft_temp.shape == self.unadjustedPMV.shape
            # self._draft_adjusted_PMV =0

    @staticmethod
    def calcFloorAirTemp(airTemp, dist, deltaT):
        return airTemp - ((0.3 - (0.034 * dist)) * deltaT)

    @staticmethod
    def velMaxClose(deltaT, windowHgt):
        return 0.083 * (math.sqrt(deltaT * windowHgt))

    @staticmethod
    def velMaxMid(dist, deltaT, windowHgt):
        return 0.143 * ((math.sqrt(deltaT * windowHgt)) / (dist + 1.32))

    @staticmethod
    def velMaxFar(deltaT, windowHgt):
        return 0.043 * (math.sqrt(deltaT * windowHgt))


def pmv_draft_adjustment(v, pmv, temp):
    equation = lambda v, pmv, temp: -0.03686567 * np.sqrt(v) * temp + 0.73404528 * pmv
    init_result = equation(v, pmv, temp)
    if (init_result < pmv) & (v >= 0.1):
        final_value = init_result  # the adjustment will be accepted if the conditions are satisified
    else:
        final_value = pmv  # No adjustment will be taken into account if either of the condition is not datificationed

    return final_value


def proc_fanger_calc(i, result_list, *input_array, ):
    pmv = np.vectorize(fanger_pmv)
    result_list[i] = pmv(*input_array)


def calc_dmrt(i, result_list, *input_array):
    bodyrad = [body_solar_flux_from_horiz_parts(*item) for item in zip(*input_array)]
    erf = [erf_from_body_solar_flux(item) for item in bodyrad]
    dmrt = [mrt_delta_from_erf(item) for item in erf]
    result_list[i] = dmrt
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
# import math
# def fanger_pmv(ta, tr, vel, rh, met, clo, wme=0):
#     """Calculate PMV using only Fanger's original equation.
#     Note that Fanger's original experiments were conducted at
#     low air speeds (<0.1 m/s) and the 2015 ASHRAE-55 thermal comfort
#     standard states that one should use standard effective temperature
#     (SET) to correct for the cooling effect of air speed in cases
#     where such speeds exceed 0.1 m/s.  The pmv() function in this module
#     will apply this SET correction in cases where it is appropriate.
#     Note:
#         [1] Fanger, P.O. (1970). Thermal Comfort: Analysis and applications
#         in environmental engineering. Copenhagen: Danish Technical Press.
#     Args:
#         ta: Air temperature [C]
#         tr: Mean radiant temperature [C]
#         vel: Relative air velocity [m/s]
#         rh: Relative humidity [%]
#         met: Metabolic rate [met]
#         clo: Clothing [clo]
#         wme: External work [met], normally around 0 when seated
#     Returns:
#         pmv: Predicted mean vote (PMV)
#         ppd: Percentage of people dissatisfied (PPD) [%]
#         heat_loss: A dictionary with the 6 heat loss terms of the PMV model.
#             The dictionary items are as follows:
#                 'cond': heat loss by conduction [W]
#                 'sweat': heat loss by sweating [W]
#                 'res_l': heat loss by latent respiration [W]
#                 'res_s' heat loss by dry respiration [W]
#                 'rad': heat loss by radiation [W]
#                 'conv' heat loss by convection [W]
#     """
#
#     pa = rh * 10. * math.exp(16.6536 - 4030.183 / (ta + 235.))
#
#     icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
#     m = met * 58.15  # metabolic rate in W/M2
#     w = wme * 58.15  # external work in W/M2
#     mw = m - w  # internal heat production in the human body
#     if icl <= 0.078:
#         fcl = 1 + (1.29 * icl)
#     else:
#         fcl = 1.05 + (0.645 * icl)
#
#     # heat transf. coeff. by forced convection
#     hcf = 12.1 * math.sqrt(vel)
#     taa = ta + 273.
#     tra = tr + 273.
#     tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
#
#     p1 = icl * fcl
#     p2 = p1 * 3.96
#     p3 = p1 * 100.
#     p4 = p1 * taa
#     p5 = 308.7 - 0.028 * mw + (p2 * ((tra / 100.) ** 4))
#     xn = tcla / 100.
#     xf = tcla / 50.
#     eps = 0.00015
#
#     n = 0
#     while abs(xn - xf) > eps:
#         xf = (xf + xn) / 2.
#         hcn = 2.38 * (abs(100.0 * xf - taa) ** 0.25)
#         if hcf > hcn:
#             hc = hcf
#         else:
#             hc = hcn
#         xn = (p5 + p4 * hc - p2 * (xf ** 4)) / (100. + p3 * hc)
#         n += 1
#         if n > 150:
#             print('Max iterations exceeded')
#             return 1
#
#     tcl = 100. * xn - 273.
#
#     # heat loss conduction through skin
#     hl1 = 3.05 * 0.001 * (5733. - (6.99 * mw) - pa)
#     # heat loss by sweating
#     if mw > 58.15:
#         hl2 = 0.42 * (mw - 58.15)
#     else:
#         hl2 = 0
#     # latent respiration heat loss
#     hl3 = 1.7 * 0.00001 * m * (5867. - pa)
#     # dry respiration heat loss
#     hl4 = 0.0014 * m * (34. - ta)
#     # heat loss by radiation
#     hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100., 4))
#     # heat loss by convection
#     hl6 = fcl * hc * (tcl - ta)
#
#     ts = 0.303 * math.exp(-0.036 * m) + 0.028
#     pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
#
#     # collect heat loss terms.
#     heat_loss = {
#         'cond': hl1,
#         'sweat': hl2,
#         'res_l': hl3,
#         'res_s': hl4,
#         'rad': hl5,
#         'conv': hl6}
#
#     return pmv, heat_loss
