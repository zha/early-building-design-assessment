import sys
sys.path.insert(0, r'E:\LB\honeybee-core')
sys.path.insert(0, r'E:\LB\honeybee-energy')


from util import genRoom, gentestpts, calcVF
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

def genIDFandrun(zonename, orientation, width, depth, WWR, U_factor, SHGC,  workingdir):

    stuff  = genRoom(zonename, ori = orientation, width =  width , depth = depth, WWR = WWR)

    room = stuff['room_obj']

    concrete125 = EnergyMaterial('Concrete125', 0.125, 0.93, 2300, 653, 'Rough', 0.88, 0.6, 0.7)
    concrete350 = EnergyMaterial('Concrete350', 0.35, 0.93, 2300, 653, 'Rough', 0.88, 0.6, 0.7)
    spandrel_mat = EnergyMaterial('Spadrel_mat', 0.205,  0.064976228, 82.5, 1000 , 'Rough', 0.9, 0.7, 0.7)

    concrete_fc = OpaqueConstruction('concrete_fc', [concrete125])
    concrete_w = OpaqueConstruction('concrete_w', [concrete350])
    spandrel_construct  = OpaqueConstruction('spandrel_construction', [spandrel_mat])


    for single_face in room.faces:
        if 'floor' in single_face.name or 'ceiling' in single_face.name:
            single_face.properties.energy.construction = concrete_fc
            single_face.boundary_condition = Adiabatic()
        elif 'exterior' in single_face.name:
            single_face.properties.energy.construction = spandrel_construct
            single_face.boundary_condition = Outdoors()
        else:
            single_face.properties.energy.construction = concrete_w
            single_face.boundary_condition = Adiabatic()


    glass = EnergyWindowMaterialSimpleGlazSys('simple_glz', U_factor, SHGC,0.72)

    double_low_e = WindowConstruction( 'Double Low-E Window', [glass])

    for aperture in room.faces[stuff['exteriorid']].apertures:
        aperture.properties.energy.construction = double_low_e

    occ_sch_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\occ.idf'
    occ_sch = ScheduleRuleset.extract_all_from_idf_file(occ_sch_dir)
    act_sch_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\act.idf'
    act_sch = ScheduleRuleset.extract_all_from_idf_file(act_sch_dir)
    ltg_sch_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\ltg.idf'
    ltg_sch = ScheduleRuleset.extract_all_from_idf_file(ltg_sch_dir)
    eqp_sch_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\eqp.idf'
    eqp_sch = ScheduleRuleset.extract_all_from_idf_file(eqp_sch_dir)
    clg_avail_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\CLG_availiability.idf'
    clg_avail = ScheduleRuleset.extract_all_from_idf_file(clg_avail_dir)
    htg_avail_dir = r'C:\Users\Administrator\Google Drive\ladybug_pypi_test\idf\HTG_availiability.idf'
    htg_avail = ScheduleRuleset.extract_all_from_idf_file(htg_avail_dir)



    always_on = ScheduleRuleset.from_constant_value('Always on', 1, schedule_type_limit = schedule_types.on_off)


    people = People('People Obj', 0.04, occ_sch[0],activity_schedule =act_sch[0] )  # name, people_per_area, occupancy_schedule, activity_schedule=None, radiant_fraction=0.3, latent_fraction='autocalculate'
    room.properties.energy.people = people

    lighting = Lighting('Lighting Obj', 5, ltg_sch[0])
    room.properties.energy.lighting = lighting


    infiltration = Infiltration('Infiltration obj', 0.00025, always_on)
    room.properties.energy.infiltration = infiltration


    ventilation = Ventilation('Ventilation obj', 0, 0.000729166, 0, 0, always_on)
    room.properties.energy.ventilation = ventilation


    equip = ElectricEquipment('Equipment obj', 5, eqp_sch[0], radiant_fraction = 0.6)
    room.properties.energy.electric_equipment = equip



    room.properties.energy.hvac = IdealAirSystem(heating_limit =6096, cooling_limit =  4570,economizer_type = "NoEconomizer",
                                                heating_availability_schedule =htg_avail[0], cooling_availability_schedule =clg_avail[0] )


    heat_setpt = ScheduleRuleset.from_constant_value(
        'Heat_stp', 22, schedule_types.temperature)
    cool_setpt = ScheduleRuleset.from_constant_value(
        'Cool_stp', 24, schedule_types.temperature)

    setpoint = Setpoint('Setpoints', heat_setpt, cool_setpt)

    room.properties.energy.setpoint = setpoint


    model = Model(zonename + 'model', [room])



    sim_control = SimulationControl(do_zone_sizing = True, do_system_sizing = False, do_plant_sizing = False, run_for_sizing_periods = False, run_for_run_periods = True)

    sim_par = SimulationParameter(simulation_control = sim_control)
    sim_par.output.add_zone_energy_use()
    sim_par.output.add_gains_and_losses()
    sim_par.output.add_comfort_metrics()

    sim_par.output.add_output('System Node Temperature')
    sim_par.output.add_output("Zone Ideal Loads Supply Air Sensible Heating Energy")
    sim_par.output.add_output("Zone Infiltration Sensible Heat Loss Energy")
    sim_par.output.add_output("Zone Infiltration Sensible Heat Gain Energy")
    sim_par.output.add_output("Zone People Sensible Heating Energy")

    # Get the input design days
    ddy_file = 'e:/CAN_ON_Toronto.716240_CWEC.ddy'
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
    1;                       !- Generic Contaminant Capacity Multiplier""".format(zonename)
    idf_str = '\n\n'.join((sim_par_str, '\n\n'.join(ddy_strs), model_str, add_str))   #add_str   #https://github.com/ladybug-tools/honeybee-energy/blob/master/honeybee_energy/writer.py

    write_to_file_by_name(os.path.join(workingdir, zonename), 'in.idf', idf_str, True)

    if True:
        print(os.path.join(workingdir, str(zonename), 'in.idf'))
        run_idf(os.path.join(workingdir, str(zonename), 'in.idf'),  r"e:/CAN_ON_Toronto.716240_CWEC.epw", r"C:\EnergyPlusV9-0-1")
