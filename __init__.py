"""
ASH - A MULTISCALE MODELLING PROGRAM

"""
# Python libraries
import os
import shutil
import numpy as np
import copy
import subprocess as sp
import glob
import sys
import inspect
import time
import atexit

###############
# ASH modules
###############
import ash

# Adding modules,interfaces directories to sys.path
ashpath = os.path.dirname(ash.__file__)
# sys.path.insert(1, ashpath+'/modules')
# sys.path.insert(1, ashpath+'/interfaces')
# sys.path.insert(1, ashpath+'/functions')

from ash.functions.functions_general import blankline, BC, listdiff, print_time_rel, print_time_rel_and_tot, pygrep, \
    printdebug, read_intlist_from_file, frange, writelisttofile, load_julia_interface, read_datafile, write_datafile

# Fragment class and coordinate functions
import ash.modules.module_coords
from ash.modules.module_coords import get_molecules_from_trajectory, eldict_covrad, write_pdbfile, Fragment, read_xyzfile, \
    write_xyzfile, make_cluster_from_box, read_ambercoordinates, read_gromacsfile
from ash.modules.module_coords import remove_atoms_from_system_CHARMM, add_atoms_to_system_CHARMM, getwaterconstraintslist,\
    QMregionfragexpand, read_xyzfiles, Reaction

# Singlepoint
import ash.modules.module_singlepoint
from ash.modules.module_singlepoint import Singlepoint, newSinglepoint, ZeroTheory, Singlepoint_fragments,\
    Singlepoint_theories, Singlepoint_fragments_and_theories, Singlepoint_reaction

# Constants
import ash.constants

# functions related to electronic structure
import ash.functions.functions_elstructure
from ash.functions.functions_elstructure import read_cube, write_cube_diff

# QMcode interfaces
from ash.interfaces.interface_ORCA import ORCATheory, counterpoise_calculation_ORCA, ORCA_External_Optimizer, run_orca_plot, \
        run_orca_mapspc, make_molden_file_ORCA, grab_coordinates_from_ORCA_output
import ash.interfaces.interface_ORCA

from ash.interfaces.interface_MLMM import MLMMTheory
from ash.interfaces.interface_TorchANI import TorchANITheory

# MM: external and internal
from ash.interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
    MDtraj_imagetraj, solvate_small_molecule, MDAnalysis_transform, OpenMM_box_relaxation, write_nonbonded_FF_for_ligand
from ash.modules.module_MM import NonBondedTheory, UFFdict, UFF_modH_dict, LJCoulpy, coulombcharge, LennardJones, \
    LJCoulombv2, LJCoulomb, MMforcefield_read

# QM/MM
from ash.modules.module_QMMM import QMMMTheory, actregiondefine

# Initialize settings
import ash.settings_ash

# Print header
import ash.ash_header
ash_header.print_header()

# Exit command (footer)
if ash.settings_ash.settings_dict["print_exit_footer"] is True:
    atexit.register(ash_header.print_footer)
    if ash.settings_ash.settings_dict["print_full_timings"] is True:
        atexit.register(ash_header.print_timings)
