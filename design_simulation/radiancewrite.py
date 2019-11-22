
from honeybeeradiance.futil import preparedir, copy_files_to_folder
from honeybeeradiance.radiance.command.rfluxmtx import Rfluxmtx
from honeybeeradiance.radiance.command.dctimestep import Dctimestep
from honeybeeradiance.radiance.command.rmtxop import Rmtxop, RmtxopMatrix
from honeybeeradiance.radiance.command.gendaymtx import Gendaymtx
from honeybeeradiance.radiance.sky.sunmatrix import SunMatrix
from honeybeeradiance.radiance.sky.analemma import AnalemmaReversed as Analemma
from honeybeeradiance.radiance.command.oconv import Oconv
from honeybeeradiance.radiance.command.rpict import Rpict
from honeybeeradiance.radiance.command.rcontrib import Rcontrib
from honeybeeradiance.radiance.command.vwrays import Vwrays
from honeybeeradiance.radiance.parameters.rpict import RpictParameters
from honeybeeradiance.radiance.recipe.recipeutil import glz_srf_to_window_group
from honeybeeradiance.radiance.recipe.parameters import get_radiance_parameters_grid_based, \
    get_radiance_parameters_image_based
from honeybeeradiance.radiance.recipe.recipedcutil import *

from honeybeeradiance.radiance.recipe.recipeutil import write_extra_files
from honeybeeradiance.radiance.recipe.recipedcutil import write_rad_files_daylight_coeff
from honeybeeradiance.radiance.recipe.recipedcutil import get_commands_scene_daylight_coeff
from honeybeeradiance.radiance.recipe.recipedcutil import get_commands_w_groups_daylight_coeff
from honeybeeradiance.radiance.recipe._gridbasedbase import GenericGridBased
from honeybeeradiance.radiance.recipe.parameters import get_radiance_parameters_grid_based
from honeybeeradiance.radiance.sky.skymatrix import SkyMatrix
from honeybeeradiance.futil import write_to_file
from honeybeeradiance.radiance.analysisgrid import AnalysisGrid
from honeybeeradiance.radiance.parameters.rfluxmtx import RfluxmtxParameters
from honeybeeradiance.hbsurface import HBSurface
from honeybeeradiance.radiance.recipe._gridbasedbase import GenericGridBased

from honeybeeradiance.radiance.recipe.recipedcutil import skymtx_to_gendaymtx
from honeybeeradiance.radiance.recipe.recipedcutil import sky_receiver
from honeybeeradiance.radiance.recipe.recipedcutil import coeff_matrix_commands,sun_coeff_matrix_commands

import os
from collections import namedtuple


def write(obj, target_folder, project_name='untitled', header=True,
          transpose=False):
    """Write analysis files to target folder.

    Args:
        target_folder: Path to parent folder. Files will be created under
            target_folder/gridbased. use self.sub_folder to change subfolder name.
        project_name: Name of this project as a string.
        header: A boolean to include the header lines in commands.bat. header
            includes PATH and cd toFolder
    Returns:
        Full path to command.bat
    """
    # 0.prepare target folder
    # create main folder target_folder/project_name
    obj._commands = []
    obj._result_files = []
    project_folder = GenericGridBased.write_content(obj,
                                                    target_folder, project_name, False,
                                                    subfolders=['tmp', 'result/matrix']
                                                    )
    # write geometry and material files
    opqfiles, glzfiles, wgsfiles = write_rad_files_daylight_coeff(
        project_folder + '/scene', project_name, obj.opaque_rad_file,
        obj.glazing_rad_file, obj.window_groups_rad_files
    )
    # additional radiance files added to the recipe as scene
    extrafiles = write_extra_files(obj.scene, project_folder + '/scene', True)

    # 0.write points
    points_file = obj.write_analysis_grids(project_folder, project_name)

    # 2.write batch file
    if header:
        obj._commands.append(obj.header(project_folder))

    # # 2.1.Create sky matrix.
    # # 2.2. Create sun matrix
    skycommands, skyfiles = get_commands_sky_simplified(project_folder, obj.sky_matrix,
                                                        reuse=True)
    obj._skyfiles = skyfiles
    obj._commands.extend(skycommands)

    # for each window group - calculate total, direct and direct-analemma results
    # calculate the contribution of glazing if any with all window groups blacked
    inputfiles = opqfiles, glzfiles, wgsfiles, extrafiles
    commands, results = get_commands_scene_daylight_coeff(
        project_name, obj.sky_matrix.sky_density, project_folder, skyfiles,
        inputfiles, points_file, obj.total_point_count, obj.radiance_parameters,
        obj.reuse_daylight_mtx, obj.total_runs_count, transpose=transpose)

    obj._result_files.extend(
        os.path.join(project_folder, str(result)) for result in results
    )

    obj._add_commands(skycommands, commands)
    if obj.window_groups:
        # calculate the contribution for all window groups
        commands, results = get_commands_w_groups_daylight_coeff(
            project_name, obj.sky_matrix.sky_density, project_folder,
            obj.window_groups, skyfiles, inputfiles, points_file,
            obj.total_point_count, obj.radiance_parameters,
            obj.reuse_daylight_mtx, obj.total_runs_count, transpose=transpose)

        obj._add_commands(skycommands, commands)
        obj._result_files.extend(
            os.path.join(project_folder, str(result)) for result in results
        )

    # # 2.5 write batch file
    batch_file = os.path.join(project_folder, 'commands.bat')

    # add echo to commands and write them to file
    write_to_file(batch_file, '\n'.join(obj.preproc_commands()))

    return batch_file



def get_commands_scene_daylight_coeff(
        project_name, sky_density, project_folder, skyfiles, inputfiles,
        points_file, total_point_count, rfluxmtx_parameters, reuse_daylight_mtx=False,
        total_count=1, radiation_only=False, transpose=False, simplified=False):
    """Get commands for the static windows in the scene.
    Use get_commands_w_groups_daylight_coeff to get the commands for the rest of the
    scene.
    Args:
        project_name: A string to generate uniqe file names for this project.
        sky_density: Sky density for this study.
        project_folder: Path to project_folder.
        skyfiles: Collection of path to sky files. The order must be (sky_mtx_total,
            sky_mtx_direct, analemma, sunlist, analemmaMtx). You can use get_commands_sky
            function to generate this list.
        inputfiles: Input files for this study. The order must be (opqfiles, glzfiles,
            wgsfiles, extrafiles). Each files object is a namedtuple which includes
            filepath to radiance files under fp and filepath to backed out files under
            fpblk.
        points_file: Path to points_file.
        total_point_count: Number of total points inside points_file.
        rfluxmtx_parameters: An instance of rfluxmtx_parameters for daylight matrix.
        reuse_daylight_mtx: A boolean not to include the commands for daylight matrix
            calculation if they already exist inside the folder.
    """
    # unpack inputs
    opqfiles, glzfiles, wgsfiles, extrafiles = inputfiles

    if len(wgsfiles) > 0:
        # material is the first file
        blkmaterial = [wgsfiles[0].fpblk[0]]
        # collect files for blacked geometries for all window groups
        wgsblacked = [f.fpblk[1] for c, f in enumerate(wgsfiles)]
    else:
        # there is no window group, return an empty tuple
        blkmaterial = ()
        wgsblacked = ()

    window_group = glz_srf_to_window_group()
    window_groupfiles = glzfiles.fp

    commands, results = _get_commands_daylight_coeff(
        project_name, sky_density, project_folder, window_group, skyfiles,
        inputfiles, points_file, total_point_count, blkmaterial, wgsblacked,
        rfluxmtx_parameters, 0, window_groupfiles, reuse_daylight_mtx, (1, total_count),
        radiation_only=radiation_only, transpose=transpose, simplified=simplified)

    return commands, results



def _get_commands_daylight_coeff(
        project_name, sky_density, project_folder, window_group, skyfiles, inputfiles,
        points_file, total_point_count, blkmaterial, wgsblacked, rfluxmtx_parameters,
        window_group_count=0, window_groupfiles=None, reuse_daylight_mtx=False,
        counter=None, radiation_only=False, transpose=False, simplified=False):
    """Get commands for the daylight coefficient recipe.
    This function is used by get_commands_scene_daylight_coeff and
    get_commands_w_groups_daylight_coeff. You usually don't want to use this function
    directly.
    """
    commands = []
    result_files = []
    # unpack inputs
    opqfiles, glzfiles, wgsfiles, extrafiles = inputfiles
    if radiation_only:
        if simplified:
            sky_mtxDiff = skyfiles[0]
        else:
            sky_mtxDiff, analemma, sunlist, analemmaMtx = skyfiles
    else:
        sky_mtx_total, sky_mtx_direct, analemma, sunlist, analemmaMtx = skyfiles

    for scount, state in enumerate(window_group.states):
        # 2.3.Generate daylight coefficients using rfluxmtx
        # collect list of radiance files in the scene for both total and direct
        if counter:
            p = ((counter[0] + scount - 1.0) / counter[1]) * 100
            c = int(p / 10)
            commands.append(
                ':: Done with {} of {} ^|{}{}^| ({:.2f}%%)'.format(
                    counter[0] + scount - 1, counter[1], '#' * c,
                    '-' * (10 - c), float(p)
                )
            )
        commands.append('::')
        commands.append(
            ':: start of the calculation for {}, {}. State {} of {}'.format(
                window_group.name, state.name, scount + 1, window_group.state_count
            )
        )
        commands.append('::')

        if scount != 0 or not window_groupfiles:
            # in case window group is not already provided
            window_groupfiles = (wgsfiles[window_group_count].fp[scount],)

        rflux_scene = (
            f for fl in
            (window_groupfiles, opqfiles.fp, extrafiles.fp,
             blkmaterial, wgsblacked)
            for f in fl)

        rflux_scene_blacked = (
            f for fl in
            (window_groupfiles, opqfiles.fpblk, extrafiles.fpblk,
             blkmaterial, wgsblacked)
            for f in fl)

        d_matrix = 'result/matrix/normal_{}..{}..{}.dc'.format(
            project_name, window_group.name, state.name)

        d_matrix_direct = 'result/matrix/black_{}..{}..{}.dc'.format(
            project_name, window_group.name, state.name)

        sun_matrix = 'result/matrix/sun_{}..{}..{}.dc'.format(
            project_name, window_group.name, state.name)

        if not os.path.isfile(os.path.join(project_folder, d_matrix)) \
                or not reuse_daylight_mtx:
            rad_files = tuple(os.path.relpath(f, project_folder) for f in rflux_scene)
            sender = '-'
            receiver = sky_receiver(
                os.path.join(project_folder, 'sky/rfluxSky.rad'), sky_density
            )

            commands.append(':: :: 1. calculating daylight matrices')
            commands.append('::')

            commands.append(':: :: [1/3] scene daylight matrix')
            commands.append(
                ':: :: rfluxmtx - [sky] [points] [wgroup] [blacked wgroups] [scene]'
                ' ^> [dc.mtx]'
            )
            commands.append('::')

            # sampling_rays_count = 1 based on @sariths studies
            rflux = coeff_matrix_commands(
                d_matrix, os.path.relpath(receiver, project_folder), rad_files, sender,
                os.path.relpath(points_file, project_folder), total_point_count,
                rfluxmtx_parameters
            )
            commands.append(rflux.to_rad_string())

            if not simplified:
                rad_files_blacked = tuple(os.path.relpath(f, project_folder)
                                          for f in rflux_scene_blacked)

                commands.append(':: :: [2/3] black scene daylight matrix')
                commands.append(
                    ':: :: rfluxmtx - [sky] [points] [wgroup] [blacked wgroups] '
                    '[blacked scene] ^> [black dc.mtx]'
                )
                commands.append('::')

                original_value = int(rfluxmtx_parameters.ambient_bounces)
                rfluxmtx_parameters.ambient_bounces = 1
                rflux_direct = coeff_matrix_commands(
                    d_matrix_direct, os.path.relpath(receiver, project_folder),
                    rad_files_blacked, sender,
                    os.path.relpath(points_file, project_folder),
                    total_point_count, rfluxmtx_parameters
                )
                commands.append(rflux_direct.to_rad_string())
                rfluxmtx_parameters.ambient_bounces = original_value

                commands.append(':: :: [3/3] black scene analemma daylight matrix')
                commands.append(
                    ':: :: rcontrib - [sun_matrix] [points] [wgroup] [blacked wgroups] '
                    '[blacked scene] ^> [analemma dc.mtx]'
                )
                commands.append('::')
                sun_commands = sun_coeff_matrix_commands(
                    sun_matrix, os.path.relpath(points_file, project_folder),
                    rad_files_blacked, os.path.relpath(analemma, project_folder),
                    sunlist, rfluxmtx_parameters.irradiance_calc
                )

                commands.extend(cmd.to_rad_string() for cmd in sun_commands)
        else:
            commands.append(':: :: 1. reusing daylight matrices')
            commands.append('::')

        commands.append(
            ':: end of calculation for {}, {}'.format(window_group.name, state.name))
        commands.append('::')
        commands.append('::')
        result_files.append(os.path.join(project_folder, d_matrix))
        result_files.append(os.path.join(project_folder, d_matrix_direct))
        result_files.append(os.path.join(project_folder, sun_matrix))

    #         if not simplified:
    #             result_files.append(
    #                 os.path.join(project_folder, str(fmtx.output_file))
    #             )
    #         else:
    #             result_files.append(
    #                 os.path.join(project_folder, str(finalmtx.output_file))
    #             )

    return commands, result_files


def get_commands_sky_simplified(project_folder, sky_matrix, reuse=True):
    """Get list of commands to generate the skies.
    1. total sky matrix
    2. direct only sky matrix
    3. sun matrix (aka analemma)
    This methdo genrates sun matrix under project_folder/sky and return the commands
    to generate skies number 1 and 2.
    Returns a namedtuple for (output_files, commands)
    output_files in a namedtuple itself (sky_mtx_total, sky_mtx_direct, analemma,
        sunlist, analemmaMtx).
    """
    OutputFiles = namedtuple('OutputFiles',
                             'sky_mtx_total sky_mtx_direct analemma sunlist analemmaMtx')

    SkyCommands = namedtuple('SkyCommands', 'output_files commands')

    commands = []

    # # 2.1.Create sky matrix.
    sky_matrix.mode = 0
    sky_mtx_total = 'sky/{}.smx'.format(sky_matrix.name)
    sky_matrix.mode = 1
    sky_mtx_direct = 'sky/{}.smx'.format(sky_matrix.name)
    sky_matrix.mode = 0

    # add commands for total and direct sky matrix.
    if hasattr(sky_matrix, 'isSkyMatrix'):
        for m in range(2):
            sky_matrix.mode = m
            gdm = skymtx_to_gendaymtx(sky_matrix, project_folder)
            if gdm:
                note = ':: {} sky matrix'.format('direct' if m else 'total')
                commands.extend((note, gdm))
        sky_matrix.mode = 0
    else:
        # sky vector
        raise TypeError('You must use a SkyMatrix to generate the sky.')

    # # 2.2. Create sun matrix
    #     sm = SunMatrix.from_wea(sky_matrix.wea, sky_matrix.north, sky_matrix.hoys,
    #                             sky_matrix.sky_type)

    #     analemma_mtx = sm.execute(os.path.join(project_folder, 'sky'))
    #     print(analemma_mtx)
    ann = Analemma.from_wea(sky_matrix.wea, sky_matrix.hoys, sky_matrix.north)
    ann.execute(os.path.join(project_folder, 'sky'))
    sunlist = os.path.join('.', 'sky', ann.sunlist_file)
    analemma = os.path.join(project_folder + '/sky', ann.analemma_file)

    of = OutputFiles(sky_mtx_total, sky_mtx_direct, analemma, sunlist, project_folder)

    return SkyCommands(commands, of)