from .modelinit import ModelInit
from honeybee_energy.load.people import People
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.infiltration import Infiltration
from honeybee_energy.load.ventilation import Ventilation
from honeybee_energy.load.equipment import ElectricEquipment
import honeybee_energy.lib.scheduletypelimits as schedule_types
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.simulationparameter import SimulationParameter
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_energy.simulation.control import SimulationControl
from honeybee_energy.material.opaque import EnergyMaterial
from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.idealair import IdealAirSystem
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from ladybug.designday import DDY
from ladybug.futil import write_to_file_by_name
from honeybee.model import Model
from honeybee_energy.boundarycondition import Adiabatic
from honeybee.boundarycondition import Outdoors
from honeybee_energy.run import run_idf
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np


# TODO: CEHCK INPUT1
class EnergyModel:
    __slots__ = ('_room', '_model', '_result', '_idf')
    def __init__(self, model):
        self.model = model

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        assert isinstance(value, ModelInit)
        self._model = value

    @property
    def room(self):
        return self.model.room

    @property
    def idf(self):
        try: return self._idf
        except:
            concrete125 = EnergyMaterial('Concrete125', 0.125, 0.93, 2300, 653, 'Rough', 0.88, 0.6, 0.7)
            concrete350 = EnergyMaterial('Concrete350', 0.35, 0.93, 2300, 653, 'Rough', 0.88, 0.6, 0.7)
            spandrel_mat = EnergyMaterial('Spadrel_mat', 0.205, 0.064976228, 82.5, 1000, 'Rough', 0.9, 0.7, 0.7)

            concrete_fc = OpaqueConstruction('concrete_fc', [concrete125])
            concrete_w = OpaqueConstruction('concrete_w', [concrete350])
            spandrel_construct = OpaqueConstruction('spandrel_construction', [spandrel_mat])


            self.model.ceiling_face.properties.energy.construction = concrete_fc
            self.model.ceiling_face.boundary_condition = Adiabatic()
            self.model.floor_face.properties.energy.construction = concrete_fc
            self.model.floor_face.boundary_condition = Adiabatic()

            self.model.exterior_wall_face.properties.energy.construction = spandrel_construct
            self.model.exterior_wall_face.boundary_condition = Outdoors()

            for interior_face in self.model.interior_wall_faces:
                interior_face.properties.energy.construction = concrete_w
                interior_face.boundary_condition = Adiabatic()



            glass = EnergyWindowMaterialSimpleGlazSys('simple_glz', self.model.U_factor, self.model.SHGC, 0.72)

            double_low_e = WindowConstruction('Double Low-E Window', [glass])

            for aperture in self.model.apertures:
                aperture.properties.energy.construction = double_low_e

            parent_path = Path(__file__).parent

            occ_sch_dir = os.path.join(parent_path, 'dat', 'occ.idf')
            occ_sch = ScheduleRuleset.extract_all_from_idf_file(occ_sch_dir)

            act_sch_dir = os.path.join(parent_path, 'dat', 'act.idf')
            act_sch = ScheduleRuleset.extract_all_from_idf_file(act_sch_dir)

            ltg_sch_dir = os.path.join(parent_path, 'dat', 'ltg.idf')
            ltg_sch = ScheduleRuleset.extract_all_from_idf_file(ltg_sch_dir)

            eqp_sch_dir = os.path.join(parent_path, 'dat', 'eqp.idf')
            eqp_sch = ScheduleRuleset.extract_all_from_idf_file(eqp_sch_dir)

            clg_avail_dir = os.path.join(parent_path, 'dat', 'CLG_availiability.idf')
            clg_avail = ScheduleRuleset.extract_all_from_idf_file(clg_avail_dir)

            htg_avail_dir = os.path.join(parent_path, 'dat', 'HTG_availiability.idf')
            htg_avail = ScheduleRuleset.extract_all_from_idf_file(htg_avail_dir)

            always_on = ScheduleRuleset.from_constant_value('Always on', 1, schedule_type_limit=schedule_types.on_off)

            people = People('People Obj', 0.04, occ_sch[0], activity_schedule=act_sch[
                0])  # name, people_per_area, occupancy_schedule, activity_schedule=None, radiant_fraction=0.3, latent_fraction='autocalculate'
            self.room.properties.energy.people = people

            lighting = Lighting('Lighting Obj', 5, ltg_sch[0])
            self.room.properties.energy.lighting = lighting

            infiltration = Infiltration('Infiltration obj', 0.00025, always_on)
            self.room.properties.energy.infiltration = infiltration

            ventilation = Ventilation('Ventilation obj', 0, 0.000729166, 0, 0, always_on)
            self.room.properties.energy.ventilation = ventilation

            equip = ElectricEquipment('Equipment obj', 5, eqp_sch[0], radiant_fraction=0.6)
            self.room.properties.energy.electric_equipment = equip

            self.room.properties.energy.hvac = IdealAirSystem(heating_limit=6096, cooling_limit=4570,
                                                         economizer_type="NoEconomizer",
                                                         heating_availability_schedule=htg_avail[0],
                                                         cooling_availability_schedule=clg_avail[0])

            heat_setpt = ScheduleRuleset.from_constant_value(
                'Heat_stp', 22, schedule_types.temperature)
            cool_setpt = ScheduleRuleset.from_constant_value(
                'Cool_stp', 24, schedule_types.temperature)

            setpoint = Setpoint('Setpoints', heat_setpt, cool_setpt)

            self.room.properties.energy.setpoint = setpoint

            model = Model(self.model.zone_name + 'model', [self.room])

            sim_control = SimulationControl(do_zone_sizing=True, do_system_sizing=False, do_plant_sizing=False,
                                            run_for_sizing_periods=False, run_for_run_periods=True)

            sim_par = SimulationParameter(simulation_control=sim_control)
            sim_par.output.add_zone_energy_use()
            sim_par.output.add_hvac_energy_use()

            sim_par.output.add_gains_and_losses()
            sim_par.output.add_comfort_metrics()
            sim_par.output.add_stratification_variables()
            sim_par.output.add_surface_temperature()
            sim_par.output.add_surface_energy_flow()
            sim_par.output.add_glazing_solar()
            sim_par.output.add_energy_balance_variables()
            sim_par.output.add_comfort_map_variables()

            sim_par.output.add_output('System Node Temperature')
            sim_par.output.add_output("Zone Ideal Loads Supply Air Sensible Heating Energy")
            sim_par.output.add_output("Zone Ideal Loads Supply Air Sensible Cooling Energy")

            sim_par.output.add_output("Zone Infiltration Sensible Heat Loss Energy")
            sim_par.output.add_output("Zone Infiltration Sensible Heat Gain Energy")
            sim_par.output.add_output("Zone People Sensible Heating Energy")

            # Get the input design days
            # ddy_file = 'e:/CAN_ON_Toronto.716240_CWEC.ddy'

            wea_dir = self.model.wea_dir

            wea_dirname = os.path.dirname(wea_dir)
            wea_filename = os.path.splitext(os.path.basename(wea_dir))[0]

            ddy_file = os.path.join(wea_dirname, wea_filename + '.ddy')

            ddy_obj = DDY.from_ddy_file(ddy_file)
            ddy_strs = [ddy.ep_style_string for ddy in ddy_obj.design_days if
                        '99.6%' in ddy.name or '.4%' in ddy.name]

            sim_par_str = sim_par.to_idf()
            model_str = model.to.idf(model)

            add_str = """
                SurfaceConvectionAlgorithm:Inside,
                    AdaptiveConvectionAlgorithm;  !- Algorithm
                SurfaceConvectionAlgorithm:Outside,
                    AdaptiveConvectionAlgorithm;  !- Algorithm
    
                ZoneCapacitanceMultiplier:ResearchSpecial,
                ResearchSpecial,          !- Name
                {},                        !- Zone or ZoneList Name
                5,                       !- Temperature Capacity Multiplier
                1,                       !- Humidity Capacity Multiplier
                1,                       !- Carbon Dioxide Capacity Multiplier
                1;                       !- Generic Contaminant Capacity Multiplier""".format(self.model.zone_name)
            idf_str = '\n\n'.join((sim_par_str, '\n\n'.join(ddy_strs), model_str,
                                   add_str))  # add_str   #https://github.com/ladybug-tools/honeybee-energy/blob/master/honeybee_energy/writer.py
            self._idf = idf_str
            return self._idf

    @property
    def result(self):
        try: return self._result
        except:
            logging.info("No pre-exisitng result found, now running EP")
            write_to_file_by_name(os.path.join(self.model.working_dir, self.model.zone_name), 'in.idf', self.idf, True)
            run_idf(os.path.join(self.model.working_dir, self.model.zone_name, 'in.idf'), self.model.wea_dir, r"C:\EnergyPlusV9-0-1")

            self._result = EnergyResult(os.path.join(self.model.working_dir, self.model.zone_name, 'eplusout.csv'), self.model)

            return self._result

class EnergyResult:
    # This object parses the csv files
    # __slots__ = ('_csv_dir', '_results', '_df')
    def __init__(self, csv_dir, model):
        self._csv_dir = csv_dir
        self._model= model

    @property
    def model(self):
        return self._model

    @property
    def csv_dir(self):
        return self._csv_dir

    @property
    def df(self):
        try: return self._df
        except:
            self._df = pd.read_csv(self._csv_dir)
            return self._df

    @property
    def surfacetemps(self):
        try: return self._surfacetemps
        except:
            self._surfacetemps = {}
            for facename in self.model.facenames:
                index = self.df.columns.str.lower().str.contains(facename.lower()) &\
                        self.df.columns.str.lower().str.contains('Surface Inside Face Temperature'.lower())  # searching for commands in the order of the face name
                assert len(np.where(index)[0]) == 1  #check if only one hit
                self._surfacetemps.update({facename: self.df[self.df.columns[index]].T.values.tolist()[0]})
            return self._surfacetemps

    @property
    def glztemps(self):
        try: return self._glztemps
        except:
            self._glztemps = {name: self.surfacetemps[name] for name in self.model.glzfacenames}
            return self._glztemps

    @property
    def air_temperature(self):
        try: return self._air_temperature
        except:
            index = self.df.columns.str.lower().str.contains('Zone Mean Air Temperature'.lower())
            assert len(np.where(index)) == 1
            self._air_temperature = self.df[self._df.columns[index]].values.tolist()
            return self._air_temperature
    @property
    def relative_humidity(self):
        try: return self._relative_humidity
        except:
            index = self.df.columns.str.lower().str.contains('Zone Air Relative Humidity'.lower())
            assert len(np.where(index)) == 1
            self._relative_humidity = self.df[self._df.columns[index]].values.tolist()
            return self._relative_humidity

            #             assert len(np.where(index)) == 1
            #             self.__dict__[key] = self._df[self._df.columns[index]]
    @property
    def supply_air_sensible_heating(self):
        try: return self._supply_air_sensible_heating
        except:
            index = self.df.columns.str.lower().str.contains('Zone Ideal Loads Supply Air Sensible Heating Energy'.lower())
            assert len(np.where(index)) == 1
            self._supply_air_sensible_heating = self.df[self._df.columns[index]].values.tolist()
            return self._supply_air_sensible_heating

    @property
    def supply_air_total_heating(self):
        try: return self._supply_air_total_heating
        except:
            index = self.df.columns.str.lower().str.contains('Zone Ideal Loads Supply Air Total Heating Energy'.lower())
            assert len(np.where(index)) == 1
            self._supply_air_total_heating = self.df[self._df.columns[index]].values.tolist()
            return self._supply_air_total_heating


    @property
    def supply_air_sensible_cooling(self):
        try: return self._supply_air_sensible_cooling
        except:
            index = self.df.columns.str.lower().str.contains('Zone Ideal Loads Supply Air Sensible Cooling Energy'.lower())
            assert len(np.where(index)) == 1
            self._supply_air_sensible_cooling = self.df[self._df.columns[index]].values.tolist()
            return self._supply_air_sensible_cooling

    @property
    def supply_air_total_cooling(self):
        try: return self._supply_air_total_cooling
        except:
            index = self.df.columns.str.lower().str.contains('Zone Ideal Loads Supply Air Total Cooling Energy'.lower())
            assert len(np.where(index)) == 1
            self._supply_air_total_cooling = self.df[self._df.columns[index]].values.tolist()
            return self._supply_air_total_cooling


    # def __getattr__(self, item):
    #     try: return self.__dict__[item]
    #     except:
    #         self.loadresults()
    #         return self.__dict__[item]

    # def loadresults(self):
    #     logging.info("loading result from csv")
    #     param_dict = {'heating':'Zone Ideal Loads Supply Air Total Heating Energy',
    #                   'cooling':'Zone Ideal Loads Supply Air Total Cooling Energy',
    #                   'ceiling_temp':['CEILING','Surface Inside Face Temperature' ],
    #                   'floor_temp': ['FLOOR', 'Surface Inside Face Temperature' ],
    #                   'west_wall_temp': ['WEST', 'Surface Inside Face Temperature'],
    #                   'east_wall_temp': ['EAST', 'Surface Inside Face Temperature'],
    #                   'north_wall_temp': ['NORTH', 'Surface Inside Face Temperature'],
    #                   'south_wall_temp': ['SOUTH', 'Surface Inside Face Temperature'],
    #                   'glazing1_temp': ['GLZ_1', 'Surface Inside Face Temperature'],
    #                   'glazing2_temp': ['GLZ_2', 'Surface Inside Face Temperature']}
    #     self._df = pd.read_csv(self._csv_dir)
    #
    #     for key, item in param_dict.items():
    #         if isinstance(item, str):
    #
    #             index = self._df.columns.str.contains(item)
    #             assert len(np.where(index)) == 1
    #             self.__dict__[key] = self._df[self._df.columns[index]]
    #         elif isinstance(item, list):
    #             index = self._df.columns.str.contains(item[0]) & self._df.columns.str.contains(item[1])
    #             assert len(np.where(index)) == 1
    #             self.__dict__[key] = self._df[self._df.columns[index]]

            #
            # cooling_index = self._df.columns.str.contains('Zone Ideal Loads Supply Air Total Cooling Energy')
            # assert len(np.where(cooling_index)) == 1
            # zone_mean_air = self._df.columns.str.contains('Zone Mean Air Temperature')
            # assert len(np.where(zone_mean_air)) == 1


