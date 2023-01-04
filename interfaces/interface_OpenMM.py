from operator import truediv
import os
from sys import stdout
import time
import traceback
import io
import copy
import numpy as np
import mdtraj

#import ash
import ash.constants

ashpath = os.path.dirname(ash.__file__)
from ash.functions.functions_general import ashexit, BC, print_time_rel, listdiff, printdebug, print_line_with_mainheader, \
    print_line_with_subheader1, print_line_with_subheader2, isint, writelisttofile
from ash.functions.functions_elstructure import DDEC_calc, DDEC_to_LJparameters
from ash.modules.module_coords import Fragment, write_pdbfile, distance_between_atoms, list_of_masses, write_xyzfile, \
    change_origin_to_centroid, get_centroid, check_charge_mult
from ash.modules.module_MM import UFF_modH_dict, MMforcefield_read
from ash.interfaces.interface_ORCA import ORCATheory, grabatomcharges_ORCA, chargemodel_select
from ash.modules.module_singlepoint import Singlepoint


class OpenMMTheory:
    def __init__(self, printlevel=2, platform='CPU', numcores=None, topoforce=False, forcefield=None, topology=None,
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None,
                 cluster_fragment=None, ASH_FF_file=None, PBCvectors=None,
                 xmlfiles=None, pdbfile=None, use_parmed=False,
                 xmlsystemfile=None,
                 do_energy_decomposition=False,
                 periodic=False, charmm_periodic_cell_dimensions=None, customnonbondedforce=False,
                 periodic_nonbonded_cutoff=12.0, dispersion_correction=True,
                 switching_function_distance=10.0,
                 ewalderrortolerance=5e-4, PMEparameters=None,
                 delete_QM1_MM1_bonded=False, applyconstraints_in_run=False,
                 constraints=None, restraints=None, frozen_atoms=None, fragment=None, dummysystem=False,
                 autoconstraints='HBonds', hydrogenmass=1.5, rigidwater=True, changed_masses=None):

        print_line_with_mainheader("OpenMM Theory")

        module_init_time = time.time()

        #Indicate that this is a MMtheory
        self.theorytype="MM"

        # OPEN MM load
        try:
            import openmm
            import openmm.app
            import openmm.unit
            print("Imported OpenMM library version:", openmm.__version__)
        except ImportError:
            raise ImportError(
                "OpenMMTheory requires installing the OpenMM library. Try: conda install -c conda-forge openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")
        print_time_rel(module_init_time, modulename="import openMM")
        timeA = time.time()
        # OpenMM variables
        self.openmm = openmm
        self.openmm_app = openmm.app
        self.simulationclass = openmm.app.simulation.Simulation

        self.unit = openmm.unit
        self.Vec3 = openmm.Vec3

        # print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        print_line_with_subheader1("Defining OpenMM object")
        # Printlevel
        self.printlevel = printlevel
        print("self.printlevel:", self.printlevel)
        # Initialize system
        self.system = None
        
        #Degrees of freedom of system (accounts for frozen atoms and constraints)
        #Will be set by update_simulation
        self.dof=None

        # Load Parmed if requested
        if use_parmed is True:
            print("Using Parmed to read topologyfiles")
            try:
                import parmed
            except ImportError:
                print("Problem importing parmed Python library")
                print("Make sure parmed is present in your Python.")
                print("Parmed can be installed using pip: pip install parmed")
                ashexit(code=9)

        # Autoconstraints when creating MM system: Default: None,  Options: Hbonds, AllBonds, HAng
        if autoconstraints == 'HBonds':
            print("HBonds option: X-H bond lengths will automatically be constrained")
            self.autoconstraints = self.openmm.app.HBonds
        elif autoconstraints == 'AllBonds':
            print("AllBonds option: All bond lengths will automatically be constrained")
            self.autoconstraints = self.openmm.app.AllBonds
        elif autoconstraints == 'HAngles':
            print("HAngles option: All bond lengths and H-X-H and H-O-X angles will automatically be constrained")
            self.autoconstraints = self.openmm.app.HAngles
        elif autoconstraints is None or autoconstraints == 'None':
            print("No automatic constraints")
            self.autoconstraints = None
        else:
            print("Unknown autoconstraints option")
            ashexit()
        print("AutoConstraint setting:", self.autoconstraints)
        
        # User constraints, restraints and frozen atoms
        self.user_frozen_atoms = []
        self.user_constraints = []
        self.user_restraints = []
        
        # Rigidwater constraints are on by default. Can be turned off
        self.rigidwater = rigidwater
        print("Rigidwater constraints:", self.rigidwater)
        # Modify hydrogenmass or not
        if hydrogenmass is not None:
            self.hydrogenmass = hydrogenmass * self.unit.amu
        else:
            self.hydrogenmass = None
        print("Hydrogenmass option:", self.hydrogenmass)

        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        # See modify_bonded_forces
        # TODO: Move option to module_QMMM instead
        self.delete_QM1_MM1_bonded = delete_QM1_MM1_bonded
        # Platform (CPU, CUDA, OpenCL) and Parallelization
        self.platform_choice = platform
        # CPU: Control either by provided numcores keyword, or by setting env variable: $OPENMM_CPU_THREADS in shell
        # before running.
        self.properties = {}
        if self.platform_choice == 'CPU':
            print("Using platform: CPU")
            if numcores is not None:
                print("Numcores variable provided to OpenMM object. Will use {} cores with OpenMM".format(numcores))
                self.properties["Threads"] = str(numcores)
                print(BC.WARNING,"Warning: Linux may ignore this user-setting and go with OPENMM_CPU_THREADS variable instead if set.",BC.END)
                print("If OPENMM_CPU_THREADS was not set in jobscript, physical cores will probably be used.")
                print("To be safe: check the running process on the node",BC.END)
            else:
                print("No numcores variable provided to OpenMM object")
                print("Checking if OPENMM_CPU_THREADS shell variable is present")
                try:
                    print("OpenMM will use {} threads according to environment variable: OPENMM_CPU_THREADS".format(
                        os.environ["OPENMM_CPU_THREADS"]))
                except KeyError:
                    print(
                        "OPENMM_CPU_THREADS environment variable not set.\nOpenMM will choose number of physical cores "
                        "present.")
        else:
            print("Using platform:", self.platform_choice)

        # Whether to do energy decomposition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_decomposition = do_energy_decomposition

        # Initializing
        self.coords = []
        self.charges = []
        self.Periodic = periodic
        self.ewalderrortolerance = ewalderrortolerance

        # Whether to apply constraints or not when calculating MM energy via run method (does not apply to OpenMM MD)
        # NOTE: Should be False in general. Only True for special cases
        self.applyconstraints_in_run = applyconstraints_in_run

        # Switching function distance in Angstrom
        self.switching_function_distance = switching_function_distance

        # Residue names,ids,segments,atomtypes of all atoms of system.
        # Grabbed below from PSF-file. Information used to write PDB-file
        self.resnames = []
        self.resids = []
        self.segmentnames = []
        self.atomtypes = []
        self.atomnames = []
        self.mm_elements = []

        # Positions. Generally not used but can be if e.g. grofile has been read in.
        # Purpose: set virtual sites etc.
        self.positions = None

        
        

        self.Forcefield = None
        # What type of forcefield files to read. Reads in different way.
        # print("Now reading forcefield files")
        print_line_with_subheader1("Setting up force fields.")
        print(
            "Note: OpenMM will fail in this step if parameters are missing in topology and\n"
            "      parameter files (e.g. nonbonded entries).\n")

        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            print("Reading CHARMM files.")
            self.psffile = psffile
            if use_parmed is True:
                print("Using Parmed.")
                self.psf = parmed.charmm.CharmmPsfFile(psffile)
                self.params = parmed.charmm.CharmmParameterSet(charmmtopfile, charmmprmfile)
                # Grab resnames from psf-object. Different for parmed object
                # Note: OpenMM uses 0-indexing
                self.resnames = [self.psf.atoms[i].residue.name for i in range(0, len(self.psf.atoms))]
                self.resids = [self.psf.atoms[i].residue.idx for i in range(0, len(self.psf.atoms))]
                self.segmentnames = [self.psf.atoms[i].residue.segid for i in range(0, len(self.psf.atoms))]
                self.atomtypes = [i.type for i in self.psf.atoms]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atoms[i].name for i in range(0, len(self.psf.atoms))]

                #TODO: Elements are unset here. Parmed parses things differently
                #NOTE: we could deduce element from atomname or mass 
                #self.mm_elements = [self.psf.atoms[i].element for i in range(0, len(self.psf.atoms))]
                #self.mm_elements = [i.element.symbol for i in self.psf.topology.atoms()]
            else:
                # Load CHARMM PSF files via native routine.
                self.psf = openmm.app.CharmmPsfFile(psffile)
                self.params = openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
                # Grab resnames from psf-object
                self.resnames = [self.psf.atom_list[i].residue.resname for i in range(0, len(self.psf.atom_list))]
                self.resids = [self.psf.atom_list[i].residue.idx for i in range(0, len(self.psf.atom_list))]
                self.segmentnames = [self.psf.atom_list[i].system for i in range(0, len(self.psf.atom_list))]
                self.atomtypes = [self.psf.atom_list[i].attype for i in range(0, len(self.psf.atom_list))]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atom_list[i].name for i in range(0, len(self.psf.atom_list))]
                self.mm_elements = [i.element.symbol for i in self.psf.topology.atoms()]

            self.topology = self.psf.topology
            self.forcefield = self.psf
            self.topfile = psffile

        elif GROMACSfiles is True:
            print("Reading Gromacs files.")
            # Reading grofile, not for coordinates but for periodic vectors
            if use_parmed is True:
                print("Using Parmed.")
                print("GROMACS top dir:", gromacstopdir)
                parmed.gromacs.GROMACS_TOPDIR = gromacstopdir
                print("Reading GROMACS GRO file:", grofile)
                gmx_gro = parmed.gromacs.GromacsGroFile.parse(grofile)

                print("Reading GROMACS topology file:", gromacstopfile)
                gmx_top = parmed.gromacs.GromacsTopologyFile(gromacstopfile)
                self.topfile = gromacstopfile

                # Getting PBC parameters
                gmx_top.box = gmx_gro.box
                gmx_top.positions = gmx_gro.positions
                self.positions = gmx_top.positions

                self.topology = gmx_top.topology
                self.forcefield = gmx_top

            else:
                print("Using built-in OpenMM routines to read GROMACS topology.")
                print("WARNING: may fail if virtual sites present (e.g. TIP4P residues).")
                print("Use 'parmed=True'  to avoid")
                gro = openmm.app.GromacsGroFile(grofile)
                self.grotop = openmm.app.GromacsTopFile(gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                                        includeDir=gromacstopdir)

                self.topology = self.grotop.topology
                self.forcefield = self.grotop

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??

            # Create an OpenMM system by calling createSystem on grotop
            # self.system = self.grotop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
            #                                    nonbondedCutoff=1 * simtk.openmm.unit.nanometer)

        elif Amberfiles is True:
            print("Reading Amber files.")
            print("WARNING: Only new-style Amber7 prmtop-file will work.")
            print("WARNING: Will take periodic boundary conditions from prmtop file.")
            if use_parmed is True:
                print("Using Parmed to read Amber files.")
                self.prmtop = parmed.load_file(amberprmtopfile)
            else:
                print("Using built-in OpenMM routines to read Amber files.")
                # Note: Only new-style Amber7 prmtop files work
                self.prmtop = openmm.app.AmberPrmtopFile(amberprmtopfile)
            self.topology = self.prmtop.topology
            self.forcefield = self.prmtop
            self.topfile = amberprmtopfile

            #List of resids, resnames and mm_elements. Used by actregiondefine
            self.resids = [i.residue.index for i in self.prmtop.topology.atoms()]
            self.resnames = [i.residue.name for i in self.prmtop.topology.atoms()]
            self.mm_elements = [i.element.symbol for i in self.prmtop.topology.atoms()]
            #NOTE: OpenMM does not grab Amber atomtypes for some reason. Feature request
            #TODO: Grab more topology information
            # TODO: Define segmentnames, atomtypes, atomnames??


        elif topoforce is True:
            print("Using forcefield info from topology and forcefield keyword.")
            self.topology = topology
            self.forcefield = forcefield

        elif ASH_FF_file is not None:
            print("Reading ASH cluster fragment file and ASH Forcefield file.")

            # Converting ASH FF file to OpenMM XML file
            MM_forcefield = MMforcefield_read(ASH_FF_file)

            atomtypes_res = []
            atomnames_res = []
            elements_res = []
            atomcharges_res = []
            sigmas_res = []
            epsilons_res = []
            residue_types = []
            masses_res = []

            for resid, residuetype in enumerate(MM_forcefield['residues']):
                residue_types.append("RS" + str(resid))
                atypelist = MM_forcefield[residuetype + "_atomtypes"]
                # atypelist needs to be more unique due to different charges
                atomtypes_res.append(["R" + residuetype[-1] + str(j) for j, i in enumerate(atypelist)])
                elements_res.append(MM_forcefield[residuetype + "_elements"])
                atomcharges_res.append(MM_forcefield[residuetype + "_charges"])
                # Atomnames, have to be unique and 4 letters, adding number
                atomnames_res.append(["R" + residuetype[-1] + str(j) for j, i in enumerate(atypelist)])
                sigmas_res.append([MM_forcefield[atomtype].LJparameters[0] / 10 for atomtype in
                                   MM_forcefield[residuetype + "_atomtypes"]])
                epsilons_res.append([MM_forcefield[atomtype].LJparameters[1] * 4.184 for atomtype in
                                     MM_forcefield[residuetype + "_atomtypes"]])
                masses_res.append(list_of_masses(elements_res[-1]))

            xmlfile = write_xmlfile_nonbonded(resnames=residue_types, atomnames_per_res=atomnames_res,
                                              atomtypes_per_res=atomtypes_res,
                                              elements_per_res=elements_res, masses_per_res=masses_res,
                                              charges_per_res=atomcharges_res, sigmas_per_res=sigmas_res,
                                              epsilons_per_res=epsilons_res,
                                              filename="cluster_system.xml", coulomb14scale=1.0, lj14scale=1.0)
            # Creating lists for PDB-file
            # requires ffragmenttype_labels to be present in fragment.
            # NOTE: Hence will only work for molcrys-prepared files for now
            atomnames_full = []
            jindex = 0
            resid_index = 1
            residlabels = []
            residue_types_full = []
            for i, fragtypelabel in enumerate(cluster_fragment.fragmenttype_labels):
                atomnames_full.append(atomnames_res[fragtypelabel][jindex])
                residlabels.append(resid_index)
                jindex += 1
                residue_types_full.append("RS" + str(fragtypelabel))
                if jindex == len(atomnames_res[fragtypelabel]):
                    jindex = 0
                    resid_index += 1

            # Creating PDB-file, only for topology (not coordinates)
            write_pdbfile(cluster_fragment, outputname="cluster", resnames=residue_types_full, atomnames=atomnames_full,
                          residlabels=residlabels)
            pdb = openmm.app.PDBFile("cluster.pdb")
            self.topology = pdb.topology
            self.topfile = "cluster.pdb"

            self.forcefield = openmm.app.ForceField(xmlfile)

        # Load XMLfile for whole system
        elif xmlsystemfile is not None:
            print("Reading system XML file:", xmlsystemfile)
            xmlsystemfileobj = open(xmlsystemfile).read()
            # Deserialize the XML text to create a System object.
            print("Now defining OpenMM system using information in file")
            print("Warning: file may contain hardcoded constraints that can not be overridden.")
            self.system = openmm.XmlSerializer.deserializeSystem(xmlsystemfileobj)
            #self.forcefield = system_temp.forcefield
            #NOTE: Big drawback of xmlsystemfile is that constraints have been hardcoded and can
            #NOTE: we could remove all present constraints using: self.remove_all_constraints()
            #NOTE: However, not sure how easy to enforce Hatom, rigidwater etc. constraints again without remaking system object
            #NOTE: Maybe define system object using XmlSerializer, somehow create forcefield object from it.
            #NOTE: Then recreate system below. Not sure if possible

            #TODO: set further properties of system here, e.g. PME parameters
            #otherwise system is not completely set

            # We still need topology from somewhere to using pdbfile
            print("Reading topology from PDBfile:", pdbfile)
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            self.topfile = pdbfile
        # Simple OpenMM system without any forcefield defined. Requires ASH fragment
        # Used for OpenMM_MD with QM Hamiltonian
        elif dummysystem is True:
            #Create list of atomnames, used in PDB topology and XML file
            atomnames_full=[j+str(i) for i,j in enumerate(fragment.elems)]
            #Write PDB-file frag.pdb with dummy atomnames
            write_pdbfile(fragment, outputname="frag", atomnames=atomnames_full)
            #Load PDB-file and create topology
            pdb = openmm.app.PDBFile("frag.pdb")
            self.topology = pdb.topology
            self.topfile = "frag.pdb"

            #Create dummy XML file
            xmlfile = write_xmlfile_nonbonded(filename="dummy.xml", resnames=["DUM"], atomnames_per_res=[atomnames_full], atomtypes_per_res=[fragment.elems],
                                            elements_per_res=[fragment.elems], masses_per_res=[fragment.masses],
                                            charges_per_res=[[0.0]*fragment.numatoms],
                                            sigmas_per_res=[[0.0]*fragment.numatoms], epsilons_per_res=[[0.0]*fragment.numatoms], skip_nb=True)
            #Create dummy forcefield
            self.forcefield = self.openmm_app.ForceField(xmlfile)


        # Read topology from PDB-file and XML-forcefield files to define forcefield
        else:
            print("Reading OpenMM XML forcefield files and PDB file")
            print("xmlfiles:", str(xmlfiles).strip("[]"))
            # This would be regular OpenMM Forcefield definition requiring XML file
            # Topology from PDBfile annoyingly enough
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            self.topfile = pdbfile
            # Todo: support multiple xml file here
            # forcefield = simtk.openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            self.forcefield = openmm.app.ForceField(*xmlfiles)

            #Defining some things. resids is used by actregiondefine
            self.resids = [i.residue.index for i in self.topology.atoms()]



        # NOW CREATE SYSTEM UNLESS already created (xmlsystemfile)
        if self.system is None:
            # Periodic or non-periodic ystem
            if self.Periodic is True:
                print_line_with_subheader1("Setting up periodicity.")

                print("Nonbonded cutoff is {} Angstrom.".format(periodic_nonbonded_cutoff))
                # Parameters here are based on OpenMM DHFR example

                if CHARMMfiles is True:
                    print("Using CHARMM files.")

                    if charmm_periodic_cell_dimensions is None:
                        print(
                            "Error: When using CHARMMfiles and 'Periodic=True', 'charmm_periodic_cell_dimensions' "
                            "keyword needs to be supplied.")
                        print(
                            "Example: charmm_periodic_cell_dimensions= [200, 200, 200, 90, 90, 90]  in Angstrom and "
                            "degrees")
                        ashexit()
                    self.charmm_periodic_cell_dimensions = charmm_periodic_cell_dimensions
                    print("Periodic cell dimensions:", charmm_periodic_cell_dimensions)
                    self.a = charmm_periodic_cell_dimensions[0] * self.unit.angstroms
                    self.b = charmm_periodic_cell_dimensions[1] * self.unit.angstroms
                    self.c = charmm_periodic_cell_dimensions[2] * self.unit.angstroms
                    if use_parmed is True:
                        self.forcefield.box = [self.a, self.b, self.c, charmm_periodic_cell_dimensions[3],
                                               charmm_periodic_cell_dimensions[4], charmm_periodic_cell_dimensions[5]]
                        # print("Set box vectors:", self.forcefield.box)
                        print_line_with_subheader2("Set box vectors:")
                        print("a:", self.a)
                        print("b:", self.b)
                        print("c:", self.c)
                        print("alpha:", charmm_periodic_cell_dimensions[3])
                        print("beta:", charmm_periodic_cell_dimensions[4])
                        print("gamma:", charmm_periodic_cell_dimensions[5])
                    else:
                        self.forcefield.setBox(self.a, self.b, self.c,
                                               alpha=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                        unit=self.unit.degree),
                                               beta=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                       unit=self.unit.degree),
                                               gamma=self.unit.Quantity(value=charmm_periodic_cell_dimensions[3],
                                                                        unit=self.unit.degree))
                        print("Set box vectors:", self.forcefield.box_vectors)


                    # ashexit()
                    self.system = self.forcefield.createSystem(self.params, nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater, ewaldErrorTolerance=self.ewalderrortolerance,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms,
                                                               switchDistance=switching_function_distance * self.unit.angstroms)
                elif GROMACSfiles is True:
                    # NOTE: Gromacs has read PBC info from Gro file already
                    print("Ewald Error tolerance:", self.ewalderrortolerance)
                    # Note: Turned off switchDistance. Not available for GROMACS?
                    #
                    self.system = self.forcefield.createSystem(nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater, ewaldErrorTolerance=self.ewalderrortolerance,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)
                elif Amberfiles is True:
                    # NOTE: Amber-interface has read PBC info from prmtop file already
                    self.system = self.forcefield.createSystem(nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater, ewaldErrorTolerance=self.ewalderrortolerance,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)

                    # print("self.system num con", self.system.getNumConstraints())
                else:
                    print("Setting up periodic system here.")
                    # Modeller and manual xmlfiles
                    self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=openmm.app.PME,
                                                               constraints=self.autoconstraints,
                                                               hydrogenMass=self.hydrogenmass,
                                                               rigidWater=self.rigidwater, ewaldErrorTolerance=self.ewalderrortolerance,
                                                               nonbondedCutoff=periodic_nonbonded_cutoff * self.unit.angstroms)
                    # switchDistance=switching_function_distance*self.unit.angstroms

                # print("self.system dict", self.system.__dict__)

                # TODO: Customnonbonded force option. Currently disabled

                if PBCvectors is not None:
                    # pbcvectors_mod = PBCvectors
                    print("Setting PBC vectors by user request.")
                    print("Assuming list of lists or list of Vec3 objects.")
                    print("Assuming vectors in nanometers.")
                    self.system.setDefaultPeriodicBoxVectors(*PBCvectors)

                a, b, c = self.system.getDefaultPeriodicBoxVectors()
                print_line_with_subheader2("Periodic vectors:")
                print(a)
                print(b)
                print(c)
                # print("Periodic vectors:", self.system.getDefaultPeriodicBoxVectors())
                print("")
                # Force modification here
                # print("OpenMM Forces defined:", self.system.getForces())
                print_line_with_subheader2("OpenMM Forces defined:")
                for force in self.system.getForces():
                    print(force.getName())

                    #NONBONDED FORCE 
                    if isinstance(force, openmm.CustomNonbondedForce):
                        # NOTE: THIS IS CURRENTLY NOT USED
                        pass
                    elif isinstance(force, openmm.NonbondedForce):

                        # Turn Dispersion correction on/off depending on user
                        force.setUseDispersionCorrection(dispersion_correction)

                        # Modify PME Parameters if desired
                        # force.setPMEParameters(1.0/0.34, fftx, ffty, fftz)
                        if PMEparameters is not None:
                            print("Changing PME parameters")
                            force.setPMEParameters(PMEparameters[0], PMEparameters[1], PMEparameters[2],
                                                   PMEparameters[3])
                        # force.setSwitchingDistance(switching_function_distance)
                        # if switching_function is True:
                        #    force.setUseSwitchingFunction(switching_function)
                        #    #Switching distance in nm. To be looked at further
                        #   force.setSwitchingDistance(switching_function_distance)
                        #    print('SwitchingFunction distance: %s' % force.getSwitchingDistance())
                        print_line_with_subheader2("Nonbonded force settings (after all modifications):")
                        print("Periodic cutoff distance: {}".format(force.getCutoffDistance()))
                        print('Use SwitchingFunction: %s' % force.getUseSwitchingFunction())
                        if force.getUseSwitchingFunction() is True:
                            print('SwitchingFunction distance: {}'.format(force.getSwitchingDistance()))
                        print('Use Long-range Dispersion correction: %s' % force.getUseDispersionCorrection())
                        print("PME Parameters:", force.getPMEParameters())
                        print("Ewald error tolerance:", force.getEwaldErrorTolerance())


                print_line_with_subheader2("OpenMM system created.")

            # Non-Periodic
            else:
                print("System is non-periodic.")

                if CHARMMfiles is True:
                    self.system = self.forcefield.createSystem(self.params, nonbondedMethod=openmm.app.NoCutoff,
                                                               constraints=self.autoconstraints,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=1000 * openmm.unit.angstroms,
                                                               hydrogenMass=self.hydrogenmass)
                elif Amberfiles is True:
                    self.system = self.forcefield.createSystem(nonbondedMethod=openmm.app.NoCutoff,
                                                               constraints=self.autoconstraints,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=1000 * openmm.unit.angstroms,
                                                               hydrogenMass=self.hydrogenmass)
                #NOTE: might be unnecessary
                elif dummysystem is True:
                    self. system = self.forcefield.createSystem(self.topology)
                else:
                    self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=openmm.app.NoCutoff,
                                                               constraints=self.autoconstraints,
                                                               rigidWater=self.rigidwater,
                                                               nonbondedCutoff=1000 * openmm.unit.angstroms,
                                                               hydrogenMass=self.hydrogenmass)

                print_line_with_subheader2("OpenMM system created.")
                print("OpenMM Forces defined:", self.system.getForces())
                print("")
                # for i,force in enumerate(self.system.getForces()):
                #    if isinstance(force, openmm.NonbondedForce):
                #        self.getatomcharges()
                #        self.nonbonded_force=force

                # print("original forces: ", forces)
                # Get charges from OpenMM object into self.charges
                # self.getatomcharges(forces['NonbondedForce'])
                # print("self.system.getForces():", self.system.getForces())
                # self.getatomcharges(self.system.getForces()[6])

                # CASE CUSTOMNONBONDED FORCE
                # REPLACING REGULAR NONBONDED FORCE
                if customnonbondedforce is True:
                    print("currently inactive")
                    ashexit()
                    # Create CustomNonbonded force
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            custom_nonbonded_force, custom_bond_force = create_cnb(self.system.getForces()[i])
                    print("1custom_nonbonded_force:", custom_nonbonded_force)
                    print("num exclusions in customnonb:", custom_nonbonded_force.getNumExclusions())
                    print("num 14 exceptions in custom_bond_force:", custom_bond_force.getNumBonds())

                    # TODO: Deal with frozen regions. NOT YET DONE
                    # Frozen-Act interaction
                    # custom_nonbonded_force.addInteractionGroup(self.frozen_atoms,self.active_atoms)
                    # Act-Act interaction
                    # custom_nonbonded_force.addInteractionGroup(self.active_atoms,self.active_atoms)
                    # print("2custom_nonbonded_force:", custom_nonbonded_force)

                    # Pointing self.nonbonded_force to CustomNonBondedForce instead of Nonbonded force
                    self.nonbonded_force = custom_nonbonded_force
                    print("self.nonbonded_force:", self.nonbonded_force)
                    self.custom_bondforce = custom_bond_force

                    # Update system with new forces and delete old force
                    self.system.addForce(self.nonbonded_force)
                    self.system.addForce(self.custom_bondforce)

                    # Remove oldNonbondedForce
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            self.system.removeForce(i)

        # Defining nonbonded force
        for i, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.NonbondedForce):
                # self.getatomcharges()
                self.nonbonded_force = force

        # Set charges in OpenMMobject by taking from Force (used by QM/MM)
        print("Setting charges")
        # self.getatomcharges(self.nonbonded_force)
        self.getatomcharges()

        # Storing numatoms and list of all atoms
        self.numatoms = int(self.system.getNumParticles())
        self.allatoms = list(range(0, self.numatoms))
        print("Number of atoms in OpenMM system:", self.numatoms)

        # Preserve original masses before any mass modifications or frozen atoms (set mass to 0)
        #NOTE: Creates list of Quantity objects (value, unit attributes)
        self.system_masses_original = [self.system.getParticleMass(i) for i in self.allatoms]
        #List of currently used masses. Can be modified by self.modify_masses and self.freeze_atoms
        #NOTE: Regular list of floats
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]


        if constraints or frozen_atoms or restraints:
            print_line_with_subheader1("Adding user constraints, restraints or frozen atoms.")
        # Now adding user-defined system constraints (only bond-constraints supported for now)
        if constraints is not None:
            print("Before adding user constraints, system contains {} constraints".format(self.system.getNumConstraints()))
            print("")

            if len(constraints) < 50:
                print("User-constraints to add:", constraints)
            else:
                print(f"{len(constraints)} user-defined constraints to add.")

            # Cleaning up constraint list. Adding distance if missing
            if 2 in [len(con) for con in constraints]:
                print("Missing distance value for some constraints. Can apply current-geometry distances if ASH\n"
                      "fragment has been provided")
                if fragment is None:
                    print("No ASH fragment provided to OpenMMTheory. Will check if pdbfile is defined and use coordinates from there")
                    if pdbfile is None:
                        print("No PDBfile present either. Either fragment or PDBfile containing \
                            coordinates is required for constraint definition")
                        ashexit()
                    else:
                        fragment=Fragment(pdbfile=pdbfile)
                # Cleaning up constraint list. Adding distance if missing
                constraints = clean_up_constraints_list(fragment=fragment, constraints=constraints)
            self.user_constraints = constraints
            print("")
            self.add_bondconstraints(constraints=constraints)
            print("")
            # print("After adding user constraints, system contains {} constraints".format(self.system.getNumConstraints()))
            print(f"{len(self.user_constraints)} user-defined constraints added.")
        # Now adding user-defined frozen atoms
        if frozen_atoms is not None:
            self.user_frozen_atoms = frozen_atoms
            if len(self.user_frozen_atoms) < 50:
                print("Frozen atoms to add:", str(frozen_atoms).strip("[]"))
            else:
                print(f"{len(self.user_frozen_atoms)} user-defined frozen atoms to add.")
            self.freeze_atoms(frozen_atoms=frozen_atoms)
        
        # Now adding user-defined restraints (only bond-restraints supported for now)
        if restraints is not None:
            # restraints is a list of lists defining bond restraints: constraints = [[atom_i,atom_j, d, k ]]
            # Example: [[700,701, 1.05, 5.0 ]] Unit is Angstrom and kcal/mol * Angstrom^-2
            self.user_restraints = restraints
            if len(self.user_restraints) < 50:
                print("User-restraints to add:", restraints)
            else:
                print(f"{len(self.user_restraints)} user-defined restraints to add.")
            self.add_bondrestraints(restraints=restraints)

        #Now changing masses if requested
        if changed_masses is not None:
            print("Modified masses")
            #changed_masses should be a dict of : atomindex: mass
            self.modify_masses(changed_masses=changed_masses)


        print("\nSystem constraints defined upon system creation:", self.system.getNumConstraints())
        print("Use printlevel =>3 to see list of all constraints")
        if self.printlevel >= 3:
            for i in range(0, self.system.getNumConstraints()):
                print("Defined constraints:", self.system.getConstraintParameters(i))
        print_time_rel(timeA, modulename="system create")
        timeA = time.time()

        # Platform
        # print("Hardware platform:", self.platform_choice)
        self.platform = openmm.Platform.getPlatformByName(self.platform_choice)

        # Create basic simulation (will be overridden by OpenMM_Opt, OpenMM_MD functions) 
        #self.create_simulation()
        self.set_simulation_parameters()
        self.update_simulation()
        # Old:
        # NOTE: If self.system is modified then we have to remake self.simulation
        # self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,self.platform)
        # self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)
        print_time_rel(timeA, modulename="simulation setup")
        # timeA = time.time()
        print_time_rel(module_init_time, modulename="OpenMM object creation")

    # add force that restrains atoms to a fixed point:
    # https://github.com/openmm/openmm/issues/2568

    # To set positions in OpenMMobject (in nm) from np-array (Angstrom)
    def set_positions(self, coords):
        print("Setting coordinates of OpenMM object")
        coords_nm = coords * 0.1  # converting from Angstrom to nm
        pos = [self.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in
               range(len(coords_nm))] * self.unit.nanometer
        self.simulation.context.setPositions(pos)
        print("Coordinates set")

    # Thermaise the OpenMM velocities to a temperature in Kelvin.
    def set_velocities_to_temperature(self, temperature):
        print(f"Thermalizing OpenMM velocities to {temperature} K.")
        self.simulation.context.setVelocitiesToTemperature(temperature * self.unit.kelvin)

    #Add dummy 
    #https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=161&t=10049&p=0&start=0&view=&sid=b844250e55b14682fb21b5f66a4d810f
    #https://github.com/openmm/openmm/issues/2262
    #Helpful for NPT simulations when solute is fixed
    #TODO: Not quiteready. Not sure how to use best
    # Add dummy atom for each solute atom?
    # Or enought to add like a centroid atom and then bind each solute atom via restraint?
    def add_dummy_atom_to_restrain_solute(self,atomindices=None, forceconstant=100):
        print("num particles", self.system.getNumParticles())
        #Adding dummy atom with mass 0
        self.system.addParticle(0)
        print("num particles", self.system.getNumParticles())
        dummyatomindex=self.system.getNumParticles()-1
        print("dummyatomindex:", dummyatomindex)
        #Adding zero-charge and zero-epsilon to Nonbonded force (charge,sigma,epsilon)
        self.nonbonded_force.addParticle(0, 1, 0)
        #Adding dummy-atom to topology
        chain=self.topology.addChain()
        residue=self.topology.addResidue("dummy",chain)
        dummy_element=self.openmm.app.element.Element(0,"Dummyel","Dd",0.0)
        self.topology.addAtom("Dum",dummy_element,residue)

        self.restraint = self.openmm.HarmonicBondForce()
        self.restraint.setUsesPeriodicBoundaryConditions(True)
        self.system.addForce(self.restraint)

        for i in atomindices:
            print("Adding bond")
            self.restraint.addBond(i, dummyatomindex, 0, forceconstant)
        #for force in self.system.getForces():
        #    if isinstance(force,self.openmm.HarmonicBondForce):
        #        print("Adding harmonic bond to dummy atom and atomindex 1")
        #        #Add harmonic bond between first atom in solute
        #        for i in atomindices:
        #            print("Adding bond")
        #            force.addBond(i, dummyatomindex, 0, 20)
    
    #NOTE: we probably can not remove particles actually
    # TOBE DELETED
    def remove_dummy_atom(self):
        #Go through atom labels/names and delete if it has a dummy label ?
        
        #Or remove by index ?

        #1. remove system particle
        
        #2. remove nonbonded force info ?
        #3. remove from topology
        #4. remove system restraint force ?
        self.system.removeForce(-1)




    # This is custom externa force that restrains group of atoms to center of system
    def add_center_force(self, center_coords=None, atomindices=None, forceconstant=1.0):
        print("Inside add_center_force")
        print("center_coords:", center_coords)
        print("atomindices:", atomindices)
        print("forceconstant:", forceconstant)
        #Distinguish periodic and nonperiodic scenarios:
        if self.Periodic is True:
            print("Warning: Add_center_force with PBC is not tested")
            centerforce = self.openmm.CustomExternalForce("k *periodicdistance(x, y, z, x0, y0, z0)")    
        else:
            centerforce = self.openmm.CustomExternalForce("k * (abs(x-x0) + abs(y-y0) + abs(z-z0))")
        centerforce.addGlobalParameter("k",
                                       forceconstant * 4.184 * self.unit.kilojoule / self.unit.angstrom / self.unit.mole)
        centerforce.addPerParticleParameter('x0')
        centerforce.addPerParticleParameter('y0')
        centerforce.addPerParticleParameter('z0')
        # Coordinates of system center
        center_x = center_coords[0] / 10
        center_y = center_coords[1] / 10
        center_z = center_coords[2] / 10
        for i in atomindices:
            # centerforce.addParticle(i, np.array([0.0, 0.0, 0.0]))
            centerforce.addParticle(i, self.Vec3(center_x, center_y, center_z))
        self.system.addForce(centerforce)
        #Updating simulation again in order to update parameters. Making sure not to change integrator etc.
        #self.create_simulation(timestep=self.timestep, integrator=self.integrator, 
        #                       coupling_frequency=self.coupling_frequency, temperature=self.temperature)
        self.update_simulation()
        print("Added center force")
        return centerforce

    def add_custom_external_force(self):
        # customforce=None
        # inspired by https://github.com/CCQC/janus/blob/ba70224cd7872541d279caf0487387104c8253e6/janus/mm_wrapper/openmm_wrapper.py
        customforce = self.openmm.CustomExternalForce("-x*fx -y*fy -z*fz")
        # customforce.addGlobalParameter('shift', 0.0)
        customforce.addPerParticleParameter('fx')
        customforce.addPerParticleParameter('fy')
        customforce.addPerParticleParameter('fz')
        for i in range(self.system.getNumParticles()):
            customforce.addParticle(i, np.array([0.0, 0.0, 0.0]))
        self.system.addForce(customforce)
        # self.externalforce=customforce
        # Necessary:
        #self.create_simulation(timestep=self.timestep, integrator=self.integrator, 
        #                       coupling_frequency=self.coupling_frequency, temperature=self.temperature)
        self.update_simulation()
        # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CustomExternalForce.html

        print("Added force")
        return customforce

    def update_custom_external_force(self, customforce, gradient, conversion_factor=49614.752589207):
        print("Updating custom external force")
        # shiftpar_inkjmol=shiftparameter*2625.4996394799
        # Convert Eh/Bohr gradient to force in kj/mol nm
        # *49614.501681716106452
        #NOTE: default conversion factor (49614.752589207) assumes input gradient in Eh/Bohr and converting to kJ/mol nm
        forces = -gradient * conversion_factor
        for i, f in enumerate(forces):
            customforce.setParticleParameters(i, i, f)
        # print("xx")
        # self.externalforce.X(shiftparameter)
        # NOTE: updateParametersInContext expensive. Avoid somehow???
        # https://github.com/openmm/openmm/issues/1892
        # print("Current value of global par 0:", self.externalforce.getGlobalParameterDefaultValue(0))
        # self.externalforce.setGlobalParameterDefaultValue(0, shiftpar_inkjmol)
        # print("Current value of global par 0:", self.externalforce.getGlobalParameterDefaultValue(0))

        customforce.updateParametersInContext(self.simulation.context)

    # Write XML-file for full system
    def saveXML(self, xmlfile="system_full.xml"):
        serialized_system = self.openmm.XmlSerializer.serialize(self.system)
        with open(xmlfile, 'w') as f:
            f.write(serialized_system)
        print("Wrote system XML file:", xmlfile)

    # Function to add bond constraints to system before MD
    def add_bondconstraints(self, constraints=None):
        # for i in range(0,self.system.getNumConstraints()):
        #    print("Constraint:", i)
        #    print(self.system.getConstraintParameters(i))
        # prevconstraints=[self.system.getConstraintParameters(i) for i in range(0,self.system.getNumConstraints())]
        # print("prevconstraints:", prevconstraints)

        for i, j, d in constraints:
            print("Adding bond constraint between atoms {} and {}. Distance value: {:.4f} Å".format(i, j, d))
            self.system.addConstraint(i, j, d * self.unit.angstroms)

    #Remove all defined constraints in system
    def remove_all_constraints(self):
        todelete=[]
        # Looping over all defined system constraints
        for i in range(0, self.system.getNumConstraints()):
            todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)
    #Remove specific constraints
    def remove_constraints(self, constraints):
        todelete = []
        # Looping over all defined system constraints
        for i in range(0, self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            for usercon in constraints:
                if all(elem in usercon for elem in [con[0], con[1]]):
                    todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)
    #Remove constraints for selected atoms. For example: QM atoms in QM/MM MD
    def remove_constraints_for_atoms(self, atoms):
        print("Removing constraints in OpenMM object for atoms:", atoms)
        todelete = []
        # Looping over all defined system constraints
        for i in range(0, self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            #print("con:", con)
            if con[0] in atoms or con[1] in atoms:
                todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Function to add restraints to system before MD
    def add_bondrestraints(self, restraints=None):
        new_restraints = self.openmm.HarmonicBondForce()
        for i, j, d, k in restraints:
            print(
                "Adding bond restraint between atoms {} and {}. Distance value: {} Å. Force constant: {} kcal/mol*Å^-2".format(
                    i, j, d, k))
            new_restraints.addBond(i, j, d * self.unit.angstroms,
                                   k * self.unit.kilocalories_per_mole / self.unit.angstroms ** 2)
        self.system.addForce(new_restraints)

    # TODO: Angleconstraints and Dihedral restraints

    # Function to freeze atoms during OpenMM MD simulation. Sets masses to zero. Does not modify potential
    # energy-function.
    def freeze_atoms(self, frozen_atoms=None):
        print("Freezing {} atoms by setting particles masses to zero.".format(len(frozen_atoms)))

        # Modify particle masses in system object. For freezing atoms
        for i in frozen_atoms:
            self.system.setParticleMass(i, 0 * self.unit.daltons)
        
        #Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    #Changed masses according to user input dictionary
    def modify_masses(self, changed_masses=None):
        print("Modify masses according: ", changed_masses)
        # Preserve original masses
        #self.system_masses = [self.system.getParticleMass(i) for i in self.allatoms]
        # Modify particle masses in system object.
        for am in changed_masses:
            self.system.setParticleMass(am, changed_masses[am] * self.unit.daltons)

        #Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    def unfreeze_atoms(self):
        # Looping over system_masses if frozen, otherwise empty list
        for atom, mass in zip(self.allatoms, self.system_masses_original):
            self.system.setParticleMass(atom, mass)

        #Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    # Currently unused
    def set_active_and_frozen_regions(self, active_atoms=None, frozen_atoms=None):
        # FROZEN AND ACTIVE ATOMS
        self.allatoms = list(range(0, self.numatoms))
        if active_atoms is None and frozen_atoms is None:
            print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
            self.frozen_atoms = []
        elif active_atoms is not None and frozen_atoms is None:
            self.active_atoms = active_atoms
            self.frozen_atoms = listdiff(self.allatoms, self.active_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms), len(self.frozen_atoms)))
            # listdiff
        elif frozen_atoms is not None and active_atoms is None:
            self.frozen_atoms = frozen_atoms
            self.active_atoms = listdiff(self.allatoms, self.frozen_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms), len(self.frozen_atoms)))
        else:
            print("active_atoms and frozen_atoms can not be both defined")
            ashexit()

    # This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    # Give list of atom indices for which we will remove all pairs
    # Todo: Way too slow to do for big list of e.g. frozen atoms but works well for qmatoms list size
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    # https://github.com/openmm/openmm/issues/1696
    def addexceptions(self, atomlist):
        timeA = time.time()
        import itertools
        print("Add exceptions/exclusions. Removing i-j interactions for list:", len(atomlist), "atoms")

        # Has duplicates
        # [self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        # https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        # [self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]
        numexceptions = 0
        numexclusions = 0
        printdebug("self.system.getForces() ", self.system.getForces())
        # print("self.nonbonded_force:", self.nonbonded_force)

        for force in self.system.getForces():
            printdebug("force:", force)
            if isinstance(force, self.openmm.NonbondedForce):
                print("Case Nonbondedforce. Adding Exception for ij pair.")
                for i in atomlist:
                    for j in atomlist:
                        printdebug("i,j : {} and {} ".format(i, j))
                        force.addException(i, j, 0, 0, 0, replace=True)

                        # NOTE: Case where there is also a CustomNonbonded force present (GROMACS interface).
                        # Then we have to add exclusion there too to avoid this issue: https://github.com/choderalab/perses/issues/357
                        # Basically both nonbonded forces have to have same exclusions (or exception where chargepro=0, eps=0)
                        # TODO: This leads to : Exception: CustomNonbondedForce: Multiple exclusions are specified for particles
                        # Basically we have to inspect what is actually present in CustomNonbondedForce
                        # for force in self.system.getForces():
                        #    if isinstance(force, self.openmm.CustomNonbondedForce):
                        #        force.addExclusion(i,j)

                        numexceptions += 1
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("Case CustomNonbondedforce. Adding Exclusion for kl pair.")
                # NOTE: This step is unfortunately a bit slow (43 seconds for 28 atomlist in 71K system)
                # Only applies to system with CustomNonbondedForce (e.g. GROMACS setup)
                # TODO: look into speeding up
                # Get list of all present exclusions first
                all_exclusions = [force.getExclusionParticles(exclindex) for exclindex in range(0,force.getNumExclusions()) ]
                # Function 
                def check_if_exclusion_present(all_exclusions,pair):
                    for exclusion in all_exclusions:
                        if set(exclusion) == set(pair):
                            return True
                    return False
                for k in atomlist:
                    for l in atomlist:
                        if check_if_exclusion_present(all_exclusions,(k,l)) is False:
                            all_exclusions.append([k,l])
                            force.addExclusion(k, l)
                            numexclusions += 1
        print("Number of exceptions (Nonbondedforce) added:", numexceptions)
        print("Number of exclusions (CustomNonbondedforce) added:", numexclusions)
        printdebug("self.system.getForces() ", self.system.getForces())
        # Seems like updateParametersInContext does not reliably work here so we have to remake the simulation instead
        # Might be bug (https://github.com/openmm/openmm/issues/2709). Revisit
        # self.nonbonded_force.updateParametersInContext(self.simulation.context)
        #self.create_simulation()
        self.update_simulation()

        print_time_rel(timeA, modulename="add exception")

    # Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    # Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    # qmatoms list provided for generality of MM objects. Not used here for now

    def set_simulation_parameters(self, timestep=0.001, coupling_frequency=1, temperature=300, integrator='VerletIntegrator'):
        self.timestep=timestep
        self.coupling_frequency=coupling_frequency
        self.temperature=temperature
        self.integrator_name=integrator
    # Create/update simulation from scratch or after system has been modified (force modification or even deletion)
    #def create_simulation(self, timestep=0.001, integrator='VerletIntegrator', coupling_frequency=1,
    #                      temperature=300):
    def update_simulation(self):
        #Keeping variables
        timeA = time.time()
        print_line_with_subheader1("Creating/updating OpenMM simulation object")
        print("Integrator name:", self.integrator_name)
        print("Timestep:", self.timestep)
        print("Temperature:", self.temperature)
        print("Coupling frequency:", self.coupling_frequency)
        print("Properties:", self.properties)
        print("Topology:", self.topology)
        printdebug("self.system.getForces() ", self.system.getForces())
        #NOTE: Integrator definition has to be here (instead of set_simulation_parameters) as it has to be recreated for each updated simulation
        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        if self.integrator_name == 'VerletIntegrator':
            self.integrator = self.openmm.VerletIntegrator(self.timestep * self.unit.picoseconds)
        elif self.integrator_name == 'VariableVerletIntegrator':
            self.integrator = self.openmm.VariableVerletIntegrator(self.timestep * self.unit.picoseconds)
        elif self.integrator_name == 'LangevinIntegrator':
            self.integrator = self.openmm.LangevinIntegrator(self.temperature * self.unit.kelvin,
                                                             self.coupling_frequency / self.unit.picosecond,
                                                             self.timestep * self.unit.picoseconds)
        elif self.integrator_name == 'LangevinMiddleIntegrator':
            # openmm recommended with 4 fs timestep, Hbonds 1/ps friction
            self.integrator = self.openmm.LangevinMiddleIntegrator(self.temperature * self.unit.kelvin,
                                                                   self.coupling_frequency / self.unit.picosecond,
                                                                   self.timestep * self.unit.picoseconds)
        elif self.integrator_name == 'NoseHooverIntegrator':
            self.integrator = self.openmm.NoseHooverIntegrator(self.temperature * self.unit.kelvin,
                                                               self.coupling_frequency / self.unit.picosecond,
                                                               self.timestep * self.unit.picoseconds)
        # NOTE: Problem with Brownian, disabling
        # elif integrator == 'BrownianIntegrator':
        #    self.integrator = self.openmm.BrownianIntegrator(temperature*self.unit.kelvin, coupling_frequency/self.unit.picosecond, timestep*self.unit.picoseconds)
        elif self.integrator_name == 'VariableLangevinIntegrator':
            self.integrator = self.openmm.VariableLangevinIntegrator(self.temperature * self.unit.kelvin,
                                                                     self.coupling_frequency / self.unit.picosecond,
                                                                     self.timestep * self.unit.picoseconds)
        else:
            print(BC.FAIL,
                  "Unknown integrator.\n Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator, "
                  "LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VariableLangevinIntegrator ",
                  BC.END)
            ashexit()

        self.simulation = self.simulationclass(self.topology, self.system, self.integrator, self.platform,
                                               self.properties)
        #Now calling function to compute the actual degrees of freedom.
        #NOTE: Better place for this? Just needs to be called once, after constraints and frozen atoms are done.
        self.compute_DOF()
        print_time_rel(timeA, modulename="creating/updating simulation")

    # Functions for energy decompositions
    def forcegroupify(self):
        self.forcegroups = {}
        print("inside forcegroupify")
        print("self.system.getForces()", self.system.getForces())
        print("Number of forces:\n", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i
        # print("self.forcegroups :", self.forcegroups)
        # ashexit()

    def getEnergyDecomposition(self, context):
        # Call and set force groups
        self.forcegroupify()
        energies = {}
        # print("self.forcegroups:", self.forcegroups)
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2 ** i).getPotentialEnergy()
        return energies

    def printEnergyDecomposition(self):
        timeA = time.time()
        # Energy composition
        # TODO: Calling this is expensive (seconds)as the energy has to be recalculated.
        # Only do for cases: a) single-point b) First energy-step in optimization and last energy-step
        # OpenMM energy components
        openmm_energy = dict()
        energycomp = self.getEnergyDecomposition(self.simulation.context)
        # print("energycomp: ", energycomp)
        # print("self.forcegroups:", self.forcegroups)
        # print("len energycomp", len(energycomp))
        # print("openmm_energy: ", openmm_energy)
        print("")
        bondterm_set = False
        extrafcount = 0
        # This currently assumes CHARMM36 components, More to be added
        for comp in energycomp.items():
            # print("comp: ", comp)
            if 'HarmonicBondForce' in str(type(comp[0])):
                # Not sure if this works in general.
                if bondterm_set is False:
                    openmm_energy['Bond'] = comp[1]
                    bondterm_set = True
                else:
                    openmm_energy['Urey-Bradley'] = comp[1]
            elif 'HarmonicAngleForce' in str(type(comp[0])):
                openmm_energy['Angle'] = comp[1]
            elif 'PeriodicTorsionForce' in str(type(comp[0])):
                # print("Here")
                openmm_energy['Dihedrals'] = comp[1]
            elif 'CustomTorsionForce' in str(type(comp[0])):
                openmm_energy['Impropers'] = comp[1]
            elif 'CMAPTorsionForce' in str(type(comp[0])):
                openmm_energy['CMAP'] = comp[1]
            elif 'NonbondedForce' in str(type(comp[0])):
                openmm_energy['Nonbonded'] = comp[1]
            elif 'CMMotionRemover' in str(type(comp[0])):
                openmm_energy['CMM'] = comp[1]
            elif 'CustomBondForce' in str(type(comp[0])):
                openmm_energy['14-LJ'] = comp[1]
            else:
                extrafcount += 1
                openmm_energy['Otherforce' + str(extrafcount)] = comp[1]

        print_time_rel(timeA, modulename="energy decomposition")
        # timeA = time.time()

        # The force terms to print in the ordered table.
        # Deprecated. Better to print everything.
        # Missing terms in force_terms will be printed separately
        # if self.Forcefield == 'CHARMM':
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded', '14-LJ']
        # else:
        #    #Modify...
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded']

        # Sum all force-terms
        sumofallcomponents = 0.0
        for val in openmm_energy.values():
            sumofallcomponents += val._value

        # Print energy table
        print('%-20s | %-15s | %-15s' % ('Component', 'kJ/mol', 'kcal/mol'))
        print('-' * 56)
        # TODO: Figure out better sorting of terms
        for name in sorted(openmm_energy):
            print('%-20s | %15.2f | %15.2f' % (name, openmm_energy[name] / self.unit.kilojoules_per_mole,
                                               openmm_energy[name] / self.unit.kilocalorie_per_mole))
        print('-' * 56)
        print('%-20s | %15.2f | %15.2f' % ('Sumcomponents', sumofallcomponents, sumofallcomponents / 4.184))
        print("")
        print('%-20s | %15.2f | %15.2f' % ('Total', self.energy * ash.constants.hartokj, self.energy * ash.constants.harkcal))

        print("")
        print("")
        # Adding sum to table
        openmm_energy['Sum'] = sumofallcomponents
        self.energy_components = openmm_energy

    def compute_DOF(self):
        # Compute the number of degrees of freedom.
        dof = 0
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i) > 0*self.unit.dalton:
                dof += 3
        for i in range(self.system.getNumConstraints()):
            p1, p2, distance = self.system.getConstraintParameters(i)
            if self.system.getParticleMass(p1) > 0*self.unit.dalton or self.system.getParticleMass(p2) > 0*self.unit.dalton:
                dof -= 1
        if any(type(self.system.getForce(i)) == self.openmm.CMMotionRemover for i in range(self.system.getNumForces())):
            dof -= 3
        self.dof=dof

    #NOTE: Adding charge/mult here temporarily to  be consistent with QM_theories. Not used
    def run(self, current_coords=None, elems=None, Grad=False, fragment=None, qmatoms=None, label=None, charge=None, mult=None):
        module_init_time = time.time()
        timeA = time.time()
        # timeA = time.time()
        if self.printlevel > 1:
            print_line_with_subheader1("Running Single-point OpenMM Interface")
        # If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply
        # coords). Then try if fragment object was supplied.
        # Otherwise internal coords if they exist
        if current_coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    if self.printlevel > 1:
                        print("Using internal coordinates (from OpenMM object).")
                    current_coords = self.coords
                else:
                    print("Found no coordinates!")
                    ashexit()
            else:
                current_coords = fragment.coords

        #IMPORTANT: Checking whether constraints have been defined in OpenMM object
        # Defined OpenMM constraints will not work within a Single-point run scheme
        # In fact forces will be all wrong. Thus checking before continuing
        # Constraints and frozen atoms have to instead by enforced by geomeTRICOptimizer, non-OpenMM dynamics module etc.
        defined_constraints=self.system.getNumConstraints()
        if self.printlevel > 1:
            print("Number of OpenMM system constraints defined:", defined_constraints)

        if self.autoconstraints != None or self.rigidwater==True:
            print(BC.FAIL,"OpenMM autoconstraints (HBonds,AllBonds,HAngles) in OpemmTheory are not compatible with OpenMMTheory.run()", BC.END)
            print(BC.WARNING,"Please redefine OpenMMTheory object: autoconstraints=None, rigidwater=False", BC.END)
            ashexit()
            
        if self.user_frozen_atoms or self.user_constraints or self.user_restraints:
            print("User-defined frozen atoms/constraints/restraints in OpemmTheory are not compatible with OpenMMTheory.run()")
            print("Constraints must instead be defined inside the program that called OpenMMtheory.run(), e.g. geomeTRICOptimizer.")
            ashexit()
        if defined_constraints != 0:
            print(BC.FAIL,"OpenMM constraints not zero. Exiting.",BC.END)
            ashexit()

        print_time_rel(timeA, modulename="OpenMMTheory.run: constraints checking", currprintlevel=self.printlevel, currthreshold=1)
        # Making sure coords is np array and not list-of-lists
        current_coords = np.array(current_coords)
        factor = -49614.752589207
        if self.printlevel > 1: print("Updating coordinates.")
        timeA = time.time()

        # NOTE: THIS IS STILL RATHER SLOW
        current_coords_nm = current_coords * 0.1  # converting from Angstrom to nm
        pos = [self.Vec3(current_coords_nm[i, 0], current_coords_nm[i, 1], current_coords_nm[i, 2]) for i in
               range(len(current_coords_nm))] * self.unit.nanometer
        # pos = [self.Vec3(*v) for v in current_coords_nm] * self.unit.nanometer #slower
        print_time_rel(timeA, modulename="Creating pos array", currprintlevel=self.printlevel, currthreshold=1)
        timeA = time.time()
        # THIS IS THE SLOWEST PART. Probably nothing to be done
        self.simulation.context.setPositions(pos)

        print_time_rel(timeA, modulename="Updating MM positions", currprintlevel=self.printlevel, currthreshold=1)
        timeA = time.time()
        # While these distance constraints should not matter, applying them makes the energy function agree with
        # previous benchmarking for bonded and nonbonded
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/
        # Using 1e-6 hardcoded value since how used in paper
        # NOTE: Weirdly, applyconstraints is True result in constraints for TIP3P disappearing
        if self.applyconstraints_in_run is True:
            if self.printlevel > 1: print("Applying constraints before calculating MM energy.")
            self.simulation.context.applyConstraints(1e-6)
            print_time_rel(timeA, modulename="context: apply constraints", currprintlevel=self.printlevel, currthreshold=1)
            timeA = time.time()

        if self.printlevel > 1:
            print("Calling OpenMM getState.")
        if Grad is True:
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / ash.constants.hartokj
            self.gradient = np.array(state.getForces(asNumpy=True) / factor)
        else:
            state = self.simulation.context.getState(getEnergy=True, getForces=False)
            self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / ash.constants.hartokj
        
        print_time_rel(timeA, modulename="OpenMM getState", currprintlevel=self.printlevel, currthreshold=1)

        if self.printlevel > 1:
            print("OpenMM Energy:", self.energy, "Eh")
            print("OpenMM Energy:", self.energy * ash.constants.harkcal, "kcal/mol")

        # Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_decomposition is True:
            self.printEnergyDecomposition()
        if self.printlevel > 1:
            print_line_with_subheader2("Ending OpenMM interface")
        print_time_rel(module_init_time, modulename="OpenMM run", moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy

    # Get list of charges from chosen force object (usually original nonbonded force object)
    def getatomcharges_old(self, force):
        chargelist = []
        for i in range(force.getNumParticles()):
            charge = force.getParticleParameters(i)[0]
            if isinstance(charge, self.unit.Quantity):
                charge = charge / self.unit.elementary_charge
                chargelist.append(charge)
        self.charges = chargelist
        return chargelist

    def getatomcharges(self):
        chargelist = []
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge = force.getParticleParameters(i)[0]
                    if isinstance(charge, self.unit.Quantity):
                        charge = charge / self.unit.elementary_charge
                        chargelist.append(charge)
                self.charges = chargelist
        return chargelist

    # Delete selected exceptions. Only for Coulomb.
    # Used to delete Coulomb interactions involving QM-QM and QM-MM atoms
    def delete_exceptions(self, atomlist):
        timeA = time.time()
        print("Deleting Coulombexceptions for atomlist:", atomlist)
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                for exc in range(force.getNumExceptions()):
                    # print(force.getExceptionParameters(exc))
                    # force.getExceptionParameters(exc)
                    p1, p2, chargeprod, sigmaij, epsilonij = force.getExceptionParameters(exc)
                    if p1 in atomlist or p2 in atomlist:
                        # print("p1: {} and p2: {}".format(p1,p2))
                        # print("chargeprod:", chargeprod)
                        # print("sigmaij:", sigmaij)
                        # print("epsilonij:", epsilonij)
                        chargeprod._value = 0.0
                        force.setExceptionParameters(exc, p1, p2, chargeprod, sigmaij, epsilonij)
                        # print("New:", force.getExceptionParameters(exc))
        #self.create_simulation()
        self.update_simulation()
        print_time_rel(timeA, modulename="delete_exceptions")

    # Function to
    def zero_nonbondedforce(self, atomlist, zeroCoulomb=True, zeroLJ=True):
        timeA = time.time()
        print("Zero-ing nonbondedforce")

        def charge_sigma_epsilon(charge, sigma, epsilon):
            if zeroCoulomb is True:
                newcharge = charge
                newcharge._value = 0.0

            else:
                newcharge = charge
            if zeroLJ is True:
                newsigma = sigma
                newsigma._value = 0.0
                newepsilon = epsilon
                newepsilon._value = 0.0
            else:
                newsigma = sigma
                newepsilon = epsilon
            return [newcharge, newsigma, newepsilon]

        # Zero all nonbonding interactions for atomlist
        for force in self.system.getForces():
            if isinstance(force, self.openmm.NonbondedForce):
                # Setting single particle parameters
                for atomindex in atomlist:
                    oldcharge, oldsigma, oldepsilon = force.getParticleParameters(atomindex)
                    newpars = charge_sigma_epsilon(oldcharge, oldsigma, oldepsilon)
                    print(newpars)
                    force.setParticleParameters(atomindex, newpars[0], newpars[1], newpars[2])
                print("force.getNumExceptions() ", force.getNumExceptions())
                print("force.getNumExceptionParameterOffsets() ", force.getNumExceptionParameterOffsets())
                print("force.getNonbondedMethod():", force.getNonbondedMethod())
                print("force.getNumGlobalParameters() ", force.getNumGlobalParameters())
                # Now doing exceptions
                for exc in range(force.getNumExceptions()):
                    print(force.getExceptionParameters(exc))
                    force.getExceptionParameters(exc)
                    p1, p2, chargeprod, sigmaij, epsilonij = force.getExceptionParameters(exc)
                    # chargeprod._value=0.0
                    # sigmaij._value=0.0
                    # epsilonij._value=0.0
                    newpars2 = charge_sigma_epsilon(chargeprod, sigmaij, epsilonij)
                    force.setExceptionParameters(exc, p1, p2, newpars2[0], newpars2[1], newpars2[2])
                    # print("New:", force.getExceptionParameters(exc))
                # force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("customnonbondedforce not implemented")
                ashexit()
        #self.create_simulation()
        self.update_simulation()
        print_time_rel(timeA, modulename="zero_nonbondedforce")
        # self.create_simulation()

    # Updating charges in OpenMM object. Used to set QM charges to 0 for example
    # Taking list of atom-indices and list of charges (usually zero) and setting new charge
    # Note: Exceptions also needs to be dealt with (see delete_exceptions)
    def update_charges(self, atomlist, atomcharges):
        timeA = time.time()
        print("Updating charges in OpenMM object.")
        assert len(atomlist) == len(atomcharges)
        # newcharges = []
        # print("atomlist:", atomlist)
        for atomindex, newcharge in zip(atomlist, atomcharges):
            # Updating big chargelist of OpenMM object.
            # TODO: Is this actually used?
            self.charges[atomindex] = newcharge
            # print("atomindex: ", atomindex)
            # print("newcharge: ",newcharge)
            oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            # Different depending on type of NonbondedForce
            if isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [newcharge, sigma, epsilon])
                # bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(i)
                # print("bla1,bla2,bla3", bla1,bla2,bla3)
            elif isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, newcharge, sigma, epsilon)
                # bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(atomindex)
                # print("bla1,bla2,bla3", bla1,bla2,bla3)

        # Instead of recreating simulation we can just update like this:
        print("Updating simulation object for modified Nonbonded force.")
        printdebug("self.nonbonded_force:", self.nonbonded_force)
        # Making sure that there still is a nonbonded force present in system (in case deleted)
        for i, force in enumerate(self.system.getForces()):
            printdebug("i is {} and force is {}".format(i, force))
            if isinstance(force, self.openmm.NonbondedForce):
                printdebug("here")
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
            if isinstance(force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.updateParametersInContext(self.simulation.context)
        #self.create_simulation()
        self.update_simulation()
        printdebug("done here")
        print_time_rel(timeA, modulename="update_charges")

    def modify_bonded_forces(self, atomlist):
        timeA = time.time()
        print("Modifying bonded forces.")
        print("")
        # This is typically used by QM/MM object to set bonded forces to zero for qmatoms (atomlist)
        # Mimicking: https://github.com/openmm/openmm/issues/2792

        numharmbondterms_removed = 0
        numharmangleterms_removed = 0
        numpertorsionterms_removed = 0
        numcustomtorsionterms_removed = 0
        numcmaptorsionterms_removed = 0
        # numcmmotionterms_removed = 0
        numcustombondterms_removed = 0

        for force in self.system.getForces():
            if isinstance(force, self.openmm.HarmonicBondForce):
                printdebug("HarmonicBonded force")
                printdebug("There are {} HarmonicBond terms defined.".format(force.getNumBonds()))
                printdebug("")
                # REVISIT: Neglecting QM-QM and sQM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    # print("i:", i)
                    p1, p2, length, k = force.getBondParameters(i)
                    # print("p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                    # or: delete QM-QM and QM-MM
                    # and: delete QM-QM

                    if self.delete_QM1_MM1_bonded is True:
                        exclude = (p1 in atomlist or p2 in atomlist)
                    else:
                        exclude = (p1 in atomlist and p2 in atomlist)
                    # print("exclude:", exclude)
                    if exclude is True:
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} length: {} k: {}".format(p1, p2, length, k))
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed += 1
                        p1, p2, length, k = force.getBondParameters(i)
                        printdebug("After p1: {} p2: {} length: {} k: {}".format(p1, p2, length, k))
                        printdebug("")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.HarmonicAngleForce):
                printdebug("HarmonicAngle force")
                printdebug("There are {} HarmonicAngle terms defined.".format(force.getNumAngles()))
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    # Are angle-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3]]
                    # Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    # Originally set to 2
                    if presence.count(True) >= 2:
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} angle: {} k: {}".format(p1, p2, p3, angle, k))
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed += 1
                        p1, p2, p3, angle, k = force.getAngleParameters(i)
                        printdebug("After p1: {} p2: {} p3: {} angle: {} k: {}".format(p1, p2, p3, angle, k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.PeriodicTorsionForce):
                printdebug("PeriodicTorsionForce force")
                printdebug("There are {} PeriodicTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    # print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                    # Originally set to 3
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug(
                            "Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1, p2, p3, p4,
                                                                                                        periodicity,
                                                                                                        phase, k))
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed += 1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        printdebug(
                            "After p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1, p2, p3, p4,
                                                                                                       periodicity,
                                                                                                       phase, k))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomTorsionForce):
                printdebug("CustomTorsionForce force")
                printdebug("There are {} CustomTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    # print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    # print("pars:", pars)
                    if presence.count(True) >= 3:
                        printdebug("Found torsion in QM-region")
                        printdebug("presence.count(True):", presence.count(True))
                        printdebug("exclude True")
                        printdebug("atomlist:", atomlist)
                        printdebug("i:", i)
                        printdebug("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1, p2, p3, p4, pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0, 0.0))
                        numcustomtorsionterms_removed += 1
                        p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        printdebug("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1, p2, p3, p4, pars))
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CMAPTorsionForce):
                printdebug("CMAPTorsionForce force")
                printdebug("There are {} CMAP terms defined.".format(force.getNumTorsions()))
                printdebug("There are {} CMAP maps defined".format(force.getNumMaps()))
                # print("Assuming no CMAP terms in QM-region. Continuing")
                # Note (RB). CMAP is between pairs of backbone dihedrals.
                # Not sure if we can delete the terms:
                # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CMAPTorsionForce.html
                #  
                # print("Map num 0", force.getMapParameters(0))
                # print("Map num 1", force.getMapParameters(1))
                # print("Map num 2", force.getMapParameters(2))
                for i in range(force.getNumTorsions()):
                    jj, p1, p2, p3, p4, v1, v2, v3, v4 = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4, v1, v2, v3, v4]]
                    # NOTE: Not sure how to use count properly here when dealing with torsion atoms in QM-region
                    if presence.count(True) >= 4:
                        printdebug(
                            "jj: {} p1: {} p2: {} p3: {} p4: {}      v1: {} v2: {} v3: {} v4: {}".format(jj, p1, p2, p3,
                                                                                                         p4, v1, v2, v3,
                                                                                                         v4))
                        printdebug("presence:", presence)
                        printdebug("Found CMAP torsion partner in QM-region")
                        printdebug("Not deleting. To be revisited...")
                        # print("presence.count(True):", presence.count(True))
                        # print("exclude True")
                        # print("atomlist:", atomlist)
                        # print("i:", i)
                        # print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        # force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        # numcustomtorsionterms_removed+=1
                        # p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        # print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                # force.updateParametersInContext(self.simulation.context)

            elif isinstance(force, self.openmm.CustomBondForce):
                printdebug("CustomBondForce")
                printdebug("There are {} force terms defined.".format(force.getNumBonds()))
                # Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, vars = force.getBondParameters(i)
                    #print("p1: {} p2: {}".format(p1,p2))
                    #print("vars:", vars)
                    exclude = (p1 in atomlist and p2 in atomlist)
                    #print("exclude:", exclude)
                    #print("-----")
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before")
                        #print("p1: {} p2: {}")
                        #force.setBondParameters(i, p1, p2, [0.0, 0.0, 0.0])
                        #NOTE: list of parameters now set to 0.0 for any number of parameters
                        force.setBondParameters(i, p1, p2, [0.0 for i in vars])
                        numcustombondterms_removed += 1
                        p1, p2, vars = force.getBondParameters(i)
                        #print("After:")
                        #print("p1: {} p2: {}")
                        #print("vars:", vars)
                        # ashexit()
                force.updateParametersInContext(self.simulation.context)

            elif isinstance(force, self.openmm.CMMotionRemover):
                pass
                # print("CMMotionRemover ")
                # print("nothing to be done")
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                pass
                # print("CustomNonbondedForce force")
                # print("nothing to be done")
            elif isinstance(force, self.openmm.NonbondedForce):
                pass
                # print("NonbondedForce force")
                # print("nothing to be done")
            else:
                pass
                # print("Other force: ", force)
                # print("nothing to be done")

        print("")
        print("Number of bonded terms removed:", )
        print("Harmonic Bond terms:", numharmbondterms_removed)
        print("Harmonic Angle terms:", numharmangleterms_removed)
        print("Periodic Torsion terms:", numpertorsionterms_removed)
        print("Custom Torsion terms:", numcustomtorsionterms_removed)
        print("CMAP Torsion terms:", numcmaptorsionterms_removed)
        print("CustomBond terms", numcustombondterms_removed)
        print("")
        #self.create_simulation()
        self.update_simulation()
        print_time_rel(timeA, modulename="modify_bonded_forces")


# For frozen systems we use Customforce in order to specify interaction groups
# if len(self.frozen_atoms) > 0:

# Two possible ways.
# https://github.com/openmm/openmm/issues/2698
# 1. Use CustomNonbondedForce  with interaction groups. Could be slow
# 2. CustomNonbondedForce but with scaling


# https://ahy3nz.github.io/posts/2019/30/openmm2/
# http://www.maccallumlab.org/news/2015/1/23/testing

# Comes close to NonbondedForce results (after exclusions) but still not correct
# The issue is most likely that the 1-4 LJ interactions should not be excluded but rather scaled.
# See https://github.com/openmm/openmm/issues/1200
# https://github.com/openmm/openmm/issues/1696
# How to do:
# 1. Keep nonbonded force for only those interactions and maybe also electrostatics?
# Mimic this??: https://github.com/openmm/openmm/blob/master/devtools/forcefield-scripts/processCharmmForceField.py
# Or do it via Parmed? Better supported for future??
# 2. Go through the 1-4 interactions and not exclude but scale somehow manually. But maybe we can't do that in
# CustomNonbonded Force?
# Presumably not but maybe can add a special force object just for 1-4 interactions. We
def create_cnb(original_nbforce):
    """Creates a CustomNonbondedForce object that mimics the original nonbonded force
    and also a Custombondforce to handle 14 exceptions
    """
    # Next, create a CustomNonbondedForce with LJ and Coulomb terms
    ONE_4PI_EPS0 = 138.935456
    # ONE_4PI_EPS0=1.0
    # TODO: Not sure whether sqrt should be present or not in epsilon???
    energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
    # sqrt ??
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    energy_expression += "chargeprod = charge1*charge2;"
    custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
    custom_nonbonded_force.addPerParticleParameter('charge')
    custom_nonbonded_force.addPerParticleParameter('sigma')
    custom_nonbonded_force.addPerParticleParameter('epsilon')
    # Configure force
    custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    # custom_nonbonded_force.setCutoffDistance(9999999999)
    custom_nonbonded_force.setUseLongRangeCorrection(False)
    # custom_nonbonded_force.setUseSwitchingFunction(True)
    # custom_nonbonded_force.setSwitchingDistance(99999)
    print('Adding particles to custom force.')
    for index in range(self.system.getNumParticles()):
        [charge, sigma, epsilon] = original_nbforce.getParticleParameters(index)
        custom_nonbonded_force.addParticle([charge, sigma, epsilon])
    # For CustomNonbondedForce we need (unlike NonbondedForce) to create exclusions that correspond to the automatic
    # exceptions in NonbondedForce
    # These are interactions that are skipped for bonded atoms
    numexceptions = original_nbforce.getNumExceptions()
    print("numexceptions in original_nbforce: ", numexceptions)

    # Turn exceptions from NonbondedForce into exclusions in CustombondedForce
    # except 1-4 which are not zeroed but are scaled. These are added to Custombondforce
    exceptions_14 = []
    numexclusions = 0
    for i in range(0, numexceptions):
        # print("i:", i)
        # Get exception parameters (indices)
        p1, p2, charge, sigma, epsilon = original_nbforce.getExceptionParameters(i)
        # print("p1,p2,charge,sigma,epsilon:", p1,p2,charge,sigma,epsilon)
        # If 0.0 then these are CHARMM 1-2 and 1-3 interactions set to zero
        if charge._value == 0.0 and epsilon._value == 0.0:
            # print("Charge and epsilons are 0.0. Add proper exclusion")
            # Set corresponding exclusion in customnonbforce
            custom_nonbonded_force.addExclusion(p1, p2)
            numexclusions += 1
        else:
            # print("This is not an exclusion but a scaled interaction as it is is non-zero. Need to keep")
            exceptions_14.append([p1, p2, charge, sigma, epsilon])
            # [798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632, unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]

    print("len exceptions_14", len(exceptions_14))
    # print("exceptions_14:", exceptions_14)
    print("numexclusions:", numexclusions)

    # Creating custombondforce to handle these special exceptions
    # Now defining pair parameters
    # https://github.com/openmm/openmm/issues/2698
    energy_expression = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    custom_bond_force = self.openmm.CustomBondForce(energy_expression)
    custom_bond_force.addPerBondParameter('chargeprod')
    custom_bond_force.addPerBondParameter('sigma')
    custom_bond_force.addPerBondParameter('epsilon')

    for exception in exceptions_14:
        idx = exception[0]
        jdx = exception[1]
        c = exception[2]
        sig = exception[3]
        eps = exception[4]
        custom_bond_force.addBond(idx, jdx, [c, sig, eps])

    print('Number of defined 14 bonds in custom_bond_force:', custom_bond_force.getNumBonds())

    return custom_nonbonded_force, custom_bond_force


# TODO: Look into: https://github.com/ParmEd/ParmEd/blob/7e411fd03c7db6977e450c2461e065004adab471/parmed/structure.py#L2554

# myCustomNBForce= simtk.openmm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
# myCustomNBForce.setNonbondedMethod(simtk.openmm.app.NoCutoff)
# myCustomNBForce.setCutoffDistance(1000*simtk.openmm.unit.angstroms)
# Frozen-Act interaction
# myCustomNBForce.addInteractionGroup(self.frozen_atoms,self.active_atoms)
# Act-Act interaction
# myCustomNBForce.addInteractionGroup(self.active_atoms,self.active_atoms)


# Clean up list of lists of constraint definition. Add distance if missing
def clean_up_constraints_list(fragment=None, constraints=None):
    print("Checking defined constraints.")
    newconstraints = []
    for con in constraints:
        if len(con) == 3:
            newconstraints.append(con)
        elif len(con) == 2:
            distance = distance_between_atoms(fragment=fragment, atom1=con[0], atom2=con[1])
            print("Adding missing distance definition between atoms {} and {}: {:.4f}".format(con[0], con[1], distance))
            newcon = [con[0], con[1], distance]
            newconstraints.append(newcon)
    return newconstraints


def OpenMM_Opt(fragment=None, theory=None, maxiter=1000, tolerance=1, enforcePeriodicBox=True):
    module_init_time = time.time()
    print_line_with_mainheader("OpenMM Optimization")

    if fragment is None:
        print("No fragment object. Exiting.")
        ashexit()

    # Distinguish between OpenMM theory or QM/MM theory
    if isinstance(theory, OpenMMTheory):
        openmmobject = theory
    else:
        print("Only OpenMMTheory allowed in OpenMM_Opt. Exiting.")
        ashexit()

    print("Number of atoms:", fragment.numatoms)
    print("Max iterations:", maxiter)
    print("Energy tolerance:", tolerance)

    print("OpenMM autoconstraints:", openmmobject.autoconstraints)
    print("OpenMM hydrogenmass:", openmmobject.hydrogenmass)
    print("OpenMM rigidwater constraints:", openmmobject.rigidwater)

    if openmmobject.user_constraints:
        print("User constraints:", openmmobject.user_constraints)
    else:
        print("User constraints: None")

    if openmmobject.user_restraints:
        print("User restraints:", openmmobject.user_restraints)
    else:
        print("User restraints: None")
    print("Number of frozen atoms:", len(openmmobject.user_frozen_atoms))
    if 0 < len(openmmobject.user_frozen_atoms) < 50:
        print("Frozen atoms", openmmobject.user_frozen_atoms)
    print("")

    if openmmobject.autoconstraints is None:
        print(f"{BC.WARNING}WARNING: Autoconstraints have not been set in OpenMMTheory object definition.{BC.END}")
        print(f"{BC.WARNING}This means that by default no bonds are constrained in the optimization.{BC.END}")
        print("Will continue...")
    if openmmobject.rigidwater is True and len(openmmobject.user_frozen_atoms) != 0 or (
            openmmobject.autoconstraints is not None and len(openmmobject.user_frozen_atoms) != 0):
        print(
            f"{BC.WARNING}WARNING: Frozen_atoms options selected but there are general constraints defined in{BC.END} "
            f"{BC.WARNING}the OpenMM object (either rigidwater=True or autoconstraints is not None)\n{BC.END}"
            f"{BC.WARNING}OpenMM will crash if constraints and frozen atoms involve the same atoms{BC.END}")


    openmmobject.set_simulation_parameters(timestep=0.001, temperature=1, integrator='VerletIntegrator')
    openmmobject.update_simulation()
    print("Simulation created.")

    # Context: settings positions
    print("Now adding coordinates")
    openmmobject.set_positions(fragment.coords)

    print("")
    state = openmmobject.simulation.context.getState(getEnergy=True, getForces=True,
                                                     enforcePeriodicBox=enforcePeriodicBox)
    print("Initial potential energy is: {} Eh".format(
        state.getPotentialEnergy().value_in_unit_system(openmmobject.unit.md_unit_system) / ash.constants.hartokj))
    kjmolnm_to_atomic_factor = -49614.752589207
    forces_init = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_init.flatten()) / len(forces_init.flatten()))
    print("RMS force: {} Eh/Bohr".format(rms_force))
    print("Max force component: {} Eh/Bohr".format(forces_init.max()))
    print("")
    print("Starting minimization.")

    openmmobject.simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance)
    print("Minimization done.")
    print("")
    state = openmmobject.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True,
                                                     enforcePeriodicBox=enforcePeriodicBox)
    print("Potential energy is: {} Eh".format(
        state.getPotentialEnergy().value_in_unit_system(openmmobject.unit.md_unit_system) / ash.constants.hartokj))
    forces_final = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_final.flatten()) / len(forces_final.flatten()))
    print("RMS force: {} Eh/Bohr".format(rms_force))
    print("Max force component: {} Eh/Bohr".format(forces_final.max()))

    # Get coordinates
    newcoords = state.getPositions(asNumpy=True).value_in_unit(openmmobject.unit.angstrom)
    print("")
    print("Updating coordinates in ASH fragment.")
    fragment.coords = newcoords

    with open('frag-minimized.pdb', 'w') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeHeader(openmmobject.topology, f)
    with open('frag-minimized.pdb', 'a') as f:
        openmmobject.openmm.app.pdbfile.PDBFile.writeModel(openmmobject.topology,
                                                           openmmobject.simulation.context.getState(getPositions=True,
                                                                                                    enforcePeriodicBox=enforcePeriodicBox).getPositions(),
                                                           f)

    print('All Done!')
    print_time_rel(module_init_time, modulename="OpenMM_Opt", moduleindex=1)


def OpenMM_Modeller(pdbfile=None, forcefield=None, xmlfile=None, waterxmlfile=None, watermodel=None, pH=7.0,
                    solvent_padding=10.0, solvent_boxdims=None, extraxmlfile=None, residue_variants=None,
                    ionicstrength=0.1, pos_iontype='Na+', neg_iontype='Cl-', use_higher_occupancy=False,
                    platform="CPU"):
    module_init_time = time.time()
    print_line_with_mainheader("OpenMM Modeller")
    try:
        import openmm as openmm
        import openmm.app as openmm_app
        import openmm.unit as openmm_unit
        print("Imported OpenMM library version:", openmm.__version__)

    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: 'conda install -c conda-forge openmm'  \
            Also see http://docs.openmm.org/latest/userguide/application.html")
    try:
        import pdbfixer
    except ImportError:
        print("Problem importing pdbfixer. Install first via conda:")
        print("conda install -c conda-forge pdbfixer")
        ashexit()

    if pdbfile == None:
        print("You must provide a pdbfile= keyword argument")
        ashexit()

    def write_pdbfile_openMM(topology, positions, filename):
        openmm.app.PDBFile.writeFile(topology, positions, file=open(filename, 'w'))
        print("Wrote PDB-file:", filename)

    def print_systemsize():
        print("System size: {} atoms\n".format(len(modeller.getPositions())))

    # https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#template


    if residue_variants == None:
        residue_variants={}


    # Water model. May be overridden by forcefield below
    if watermodel == "tip3p":
        # Possible Problem: this only has water, no ions.
        waterxmlfile = "tip3p.xml"
    elif waterxmlfile is not None:
        # Problem: we need to define watermodel also
        print("Using waterxmlfile:", waterxmlfile)
    # Forcefield options
    if forcefield is not None:
        if forcefield == 'Amber99':
            xmlfile = "amber99sb.xml"
        elif forcefield == 'Amber96':
            xmlfile = "amber96.xml"
        elif forcefield == 'Amber03':
            xmlfile = "amber03.xml"
        elif forcefield == 'Amber10':
            xmlfile = "amber10.xml"
        elif forcefield == 'Amber14':
            xmlfile = "amber14-all.xml"
            # Using specific Amber FB version of TIP3P
            if watermodel == "tip3p":
                waterxmlfile = "amber14/tip3pfb.xml"
        elif forcefield == 'Amber96':
            xmlfile = "amber96.xml"
        elif forcefield == 'CHARMM36':
            xmlfile = "charmm36.xml"
            # Using specific CHARMM36 version of TIP3P
            watermodel="tip3p"
            waterxmlfile = "charmm36/water.xml"
        elif forcefield == 'CHARMM2013':
            xmlfile = "charmm_polar_2013.xml"
        elif forcefield == 'Amoeba2013':
            xmlfile = "amoeba2013.xml"
        elif forcefield == 'Amoeba2009':
            xmlfile = "amoeba2009.xml"
    elif xmlfile is not None:
        print("Using xmlfile:", xmlfile)
    else:
        print("You must provide a forcefield or xmlfile keyword!")
        ashexit()

    print("PDBfile:", pdbfile)
    print("Forcefield:", forcefield)
    print("XMfile:", xmlfile)
    print("Water model:", watermodel)
    print("Xmlfile:", waterxmlfile)
    print("pH:", pH)

    print("User-provided dictionary of residue_variants:", residue_variants)
    # Define a forcefield
    if extraxmlfile is None:
        forcefield = openmm_app.forcefield.ForceField(xmlfile, waterxmlfile)
    else:
        print("Using extra XML file:", extraxmlfile)
        #Checking if file exists first
        if os.path.isfile(extraxmlfile) is not True:
            print(BC.FAIL,"File {} can not be found. Exiting.".format(extraxmlfile),BC.END)
            ashexit()
        forcefield = openmm_app.forcefield.ForceField(xmlfile, waterxmlfile, extraxmlfile)


    print("\nNow checking PDB-file for alternate locations, i.e. multiple occupancies:\n")

    
    #Check PDB-file whether it contains alternate locations of residue atoms (multiple occupations)
    #Default behaviour: 
    # - if no multiple occupancies return input PDBfile and go on
    # - if multiple occupancies, print list of residues and tell user to fix them. Exiting
    # - if use_higher_occupancy is set to True, user higher occupancy location, write new PDB_file and use
    pdbfile=find_alternate_locations_residues(pdbfile, use_higher_occupancy=use_higher_occupancy)

    print("Using PDB-file", pdbfile)

    # Fix basic mistakes in PDB by PDBFixer
    # This will e.g. fix bad terminii
    print("\nRunning PDBFixer")
    fixer = pdbfixer.PDBFixer(pdbfile)
    fixer.findMissingResidues()
    print("Found missing residues:", fixer.missingResidues)
    fixer.findNonstandardResidues()
    print("Found non-standard residues:", fixer.nonstandardResidues)
    # fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    print("Found missing atoms:", fixer.missingAtoms)
    print("Found missing terminals:", fixer.missingTerminals)
    fixer.addMissingAtoms()
    print("Added missing atoms.")
    #exit()

    openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open('system_afterfixes.pdb', 'w'))
    print("PDBFixer done.")
    print(BC.WARNING,"Warning: PDBFixer can create unreasonable orientations of residues if residues are missing or multiple occupancies are present.\n \
    You should inspect the created PDB-file to be sure.",BC.END)
    print("Wrote PDBfile: system_afterfixes.pdb")

    # Load fixed PDB-file and create Modeller object
    pdb = openmm_app.PDBFile("system_afterfixes.pdb")
    print("\n\nNow loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    modeller_numatoms = modeller.topology.getNumAtoms()
    numresidues = modeller.topology.getNumResidues()
    numchains = modeller.topology.getNumChains()
    modeller_atoms=list(modeller.topology.atoms())
    modeller_bonds=list(modeller.topology.bonds())
    modeller_chains=list(modeller.topology.chains())
    modeller_residues=list(modeller.topology.residues())
    print("Modeller topology has {} residues.".format(numresidues))
    print("Modeller topology has {} chains.".format(numchains))
    print("Modeller topology has {} atoms.".format(modeller_numatoms))
    print("Chains:", modeller_chains)
    #Getting residues for each chain
    for chain_x in modeller_chains:
        print("This is chain {}, it has {} residues and they are: {}\n".format(chain_x.index,len(chain_x._residues),chain_x._residues))
    print("\n")

    #PRINTING big table of residues
    print("User defined residue variants per chain:")
    for rv_key,rv_vals in residue_variants.items():
        print("Chain {} : {}".format(rv_key,rv_vals))
    print("\nMODELLER TOPOLOGY - RESIDUES TABLE\n")
    print("  {:<12}{:<13}{:<13}{:<13}{:<13}       {}".format("ASH-resid","Resname","Chain-index", "Chain-name", "ResID-in-chain","User-modification"))
    print("-"*100)
    current_chainindex=0
    #Also using loop to get residue_states list that we pass on to modeller.addHydrogens
    residue_states=[]
    for each_residue in modeller_residues:
        #Division line between chains
        if each_residue.chain.index != current_chainindex:
            print("--"*30)
        resid=each_residue.index
        resid_in_chain=int(each_residue.id)
        resname=each_residue.name
        chain=each_residue.chain
        current_chainindex=each_residue.chain.index
        if chain.id in residue_variants:
            if resid_in_chain in residue_variants[chain.id]:
                residue_states.append(residue_variants[chain.id][resid_in_chain])
                FLAGLABEL="-- This residue will be changed to: {} --".format(residue_variants[chain.id][resid_in_chain])
            else:
                residue_states.append(None) #Note: we add None since we don't want to influence addHydrogens 
                FLAGLABEL=""
        else:
            residue_states.append(None)  #Note: we add None since we don't want to influence addHydrogens
            FLAGLABEL=""

        print("  {:<12}{:<13}{:<13}{:<13}{:<13}       {}".format(resid,resname,chain.index,chain.id, resid_in_chain,FLAGLABEL))

    openmm_app.PDBFile.writeFile(modeller.topology, modeller.positions, open('system_afterfixes2.pdb', 'w'))


    #NOTE: to be deleted
    if len(residue_states) != numresidues:
        print("residue_states != numresidues. Something went wrong")
        ashexit()

    # Adding hydrogens feeding in residue_states
    # This is were missing residue/atom errors will come
    print("")
    print("Adding hydrogens for pH:", pH)
    #print("Providing full list of residue_states", residue_states)
    print("Warning: OpenMM Modeller will fail in this step if residue information is missing")
    try:
        modeller.addHydrogens(forcefield, pH=pH, variants=residue_states)
    except ValueError as errormessage:
        print(BC.FAIL,"\nError: OpenMM modeller.addHydrogens signalled a ValueError",BC.END)
        print("This is a common error and suggests a problem in PDB-file or missing residue information in the forcefield.")
        print("Non-standard inorganic/organic residues require providing an additional XML-file via extraxmlfile= option")
        print("Note that C-terminii require the dangling O-atom to be named OXT ")
        print("Read the ASH documentation or the OpenMM documentation on dealing with this problem.")
        print("\nFull error message from OpenMM:")
        print(errormessage)
        print()
        ashexit()

    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_afterH.pdb")
    print_systemsize()

    # Adding Solvent
    print("Adding solvent, watermodel:", watermodel)
    if solvent_boxdims is not None:
        print("Solvent boxdimension provided: {} Å".format(solvent_boxdims))
        modeller.addSolvent(forcefield, neutralize=False, boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1],
                                                            solvent_boxdims[2]) * openmm_unit.angstrom)
    else:
        print("Using solvent padding (solvent_padding=X keyword): {} Å".format(solvent_padding))
        modeller.addSolvent(forcefield, neutralize=False, padding=solvent_padding * openmm_unit.angstrom, model=watermodel)
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize()

    # Ions
    print("Adding ionic strength: {} M, using ions: {} and {}".format(ionicstrength, pos_iontype, neg_iontype))
    modeller.addSolvent(forcefield, neutralize=True, positiveIon=pos_iontype, negativeIon=neg_iontype, 
        ionicStrength=ionicstrength * openmm_unit.molar)
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_afterions.pdb")
    write_pdbfile_openMM(modeller.topology, modeller.positions, "finalsystem.pdb")
    print_systemsize()

    # Create ASH fragment and write to disk
    fragment = Fragment(pdbfile="system_afterions.pdb")
    fragment.print_system(filename="finalsystem.ygg")
    fragment.write_xyzfile(xyzfilename="finalsystem.xyz")

    print("\nOpenMM_Modeller used the following XML-files to define system:")
    print("General forcefield XML file:", xmlfile)
    print("Solvent forcefield XML file:", waterxmlfile)
    print("Extra forcefield XML file:", extraxmlfile)

    #Creating new OpenMM object from forcefield so that we can write out system XMLfile
    print("Creating OpenMMTheory object")
    openmmobject =OpenMMTheory(platform=platform, forcefield=forcefield, topoforce=True,
                        topology=modeller.topology, pdbfile=None, periodic=True,
                        autoconstraints='HBonds', rigidwater=True)
    #Write out System XMLfile
    #TODO: Disable ?
    systemxmlfile="system_full.xml"

    serialized_system = openmm.XmlSerializer.serialize(openmmobject.system)
    with open(systemxmlfile, 'w') as f:
        f.write(serialized_system)
    
    print("\n\nFiles written to disk:")
    print("system_afteratlocfixes.pdb")
    print("system_afterfixes.pdb")
    print("system_afterfixes2.pdb")
    print("system_afterH.pdb")
    print("system_aftersolvent.pdb")
    print("system_afterions.pdb and finalsystem.pdb (same)")
    print("\nFinal files:")
    print("finalsystem.pdb  (PDB file)")
    print("finalsystem.ygg  (ASH fragment file)")
    print("finalsystem.xyz   (XYZ coordinate file)")
    print("{}   (System XML file)".format(systemxmlfile))
    print(BC.OKGREEN,"\n\n OpenMM_Modeller done! System has been fully set up!\n",BC.END)
    print(BC.WARNING,"Strongly recommended: Check finalsystem.pdb carefully for correctness!", BC.END)
    print("\nTo use this system setup to define a future OpenMMTheory object you can either do:\n")

    print(BC.OKMAGENTA,"1. Define using separate forcefield XML files:",BC.END)
    if extraxmlfile is None:
        print("omm = OpenMMTheory(xmlfiles=[\"{}\", \"{}\"], pdbfile=\"finalsystem.pdb\", periodic=True)".format(xmlfile,waterxmlfile),BC.END)
    else:
        print("omm = OpenMMTheory(xmlfiles=[\"{}\", \"{}\", \"{}\"], pdbfile=\"finalsystem.pdb\", periodic=True)".format(xmlfile,waterxmlfile,extraxmlfile),BC.END)

    print(BC.OKMAGENTA,"2. Use full system XML-file (USUALLY NOT RECOMMENDED ):\n",BC.END, \
        "omm = OpenMMTheory(xmlsystemfile=\"system_full.xml\", pdbfile=\"finalsystem.pdb\", periodic=True)\n",BC.END)
    print_time_rel(module_init_time, modulename="OpenMM_Modeller", moduleindex=1)
    
    #Return openmmobject. Could be used directly
    return openmmobject, fragment


def MDtraj_import_():
    print("Importing mdtraj (https://www.mdtraj.org)")
    try:
        import mdtraj
    except ImportError:
        print("Problem importing mdtraj. Try: 'pip install mdtraj' or 'conda install -c conda-forge mdtraj'")
        ashexit()
    return mdtraj


# anchor_molecules. Use if automatic guess fails
def MDtraj_imagetraj(trajectory, pdbtopology, format='DCD', unitcell_lengths=None, unitcell_angles=None,
                     solute_anchor=None):
    #Trajectory basename
    traj_basename = os.path.splitext(trajectory)[0]
    #PDB-file basename
    pdb_basename = os.path.splitext(pdbtopology)[0]
    
    #Import mdtraj library
    mdtraj = MDtraj_import_()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)

    #Also load the pdbfile as a trajectory-snapshot (in addition to being topology)
    pdbsnap = mdtraj.load(pdbtopology, top=pdbtopology)
    pdbsnap_imaged = pdbsnap.image_molecules()

    numframes = len(traj._time)
    print("Found {} frames in trajectory.".format(numframes))
    print("PBC information in trajectory:")
    print("Unitcell lengths:", traj.unitcell_lengths[0])
    print("Unitcell angles", traj.unitcell_angles[0])
    # If PBC information is missing from traj file (OpenMM: Charmmfiles, Amberfiles option etc) then provide this info
    if unitcell_lengths is not None:
        print("unitcell_lengths info provided by user.")
        unitcell_lengths_nm = [i / 10 for i in unitcell_lengths]
        traj.unitcell_lengths = np.array(unitcell_lengths_nm * numframes).reshape(numframes, 3)
        traj.unitcell_angles = np.array(unitcell_angles * numframes).reshape(numframes, 3)
    # else:
    #    print("Missing PBC info. This can be provided by unitcell_lengths and unitcell_angles keywords")

    # Manual anchor if needed
    # NOTE: not sure how well this works but it's something
    if solute_anchor is True:
        anchors = [set(traj.topology.residue(0).atoms)]
        print("anchors:", anchors)
        # Re-imaging trajectory
        imaged = traj.image_molecules(anchor_molecules=anchors)
    else:
        imaged = traj.image_molecules()

    # Save trajectory in format
    if format == 'DCD':
        imaged.save(traj_basename + '_imaged.dcd')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.dcd')
    elif format == 'PDB':
        imaged.save(traj_basename + '_imaged.pdb')
        print("Saved reimaged trajectory:", traj_basename + '_imaged.pdb')
    else:
        print("Unknown trajectory format.")

    #Save PDB-snapshot
    pdbsnap_imaged.save(pdb_basename + '_imaged.pdb')
    print("Saved reimaged PDB-file:", pdb_basename + '_imaged.pdb')
    #Return last frame as coords or ASH fragment ?
    #Last frame coordinates as Angstrom
    lastframe=imaged[-1]._xyz[-1]*10

    return lastframe


def MDAnalysis_transform(topfile, trajfile, solute_indices=None, trajoutputformat='PDB', trajname="MDAnalysis_traj"):
    # Load traj
    print("MDAnalysis interface: transform")

    try:
        import MDAnalysis as mda
        import MDAnalysis.transformations as trans
    except ImportError:
        print("Problem importing MDAnalysis library.")
        print("Install via: 'pip install mdtraj'")
        ashexit()

    print("Loading trajecory using MDAnalysis")
    print("Topology file:", topfile)
    print("Trajectory file:", trajfile)
    print("Solute_indices:", solute_indices)
    print("Trajectory output format:", trajoutputformat)
    print("Will unwrap solute and center in box.")
    print("Will then wrap full system.")

    # Load trajectory
    u = mda.Universe(topfile, trajfile, in_memory=True)
    print(u.trajectory.ts, u.trajectory.time)

    # Grab solute
    numatoms = len(u.atoms)
    solutenum = len(solute_indices)
    solute = u.atoms[:solutenum]
    # solvent = u.atoms[solutenum:numatoms]
    fullsystem = u.atoms[:numatoms]
    elems_list = list(fullsystem.types)
    # Guess bonds. Could also read in vdW radii. Could also read in connectivity from ASH if this fails
    solute.guess_bonds()
    # Unwrap solute, center solute and wraps full system (or solvent)
    workflow = (trans.unwrap(solute),
                trans.center_in_box(solute, center='mass'),
                trans.wrap(fullsystem, compound='residues'))

    u.trajectory.add_transformations(*workflow)
    if trajoutputformat == 'PDB':
        fullsystem.write(trajname + ".pdb", frames='all')

    # TODO: Distinguish between transforming whole trajectory vs. single geometry
    # Maybe just read in single-frame trajectory so that things are general
    # Returning last frame. To be used in ASH workflow
    lastframe = u.trajectory[-1]

    return elems_list, lastframe._pos


# Assumes all atoms present (including hydrogens)
def solvate_small_molecule(fragment=None, charge=None, mult=None, watermodel=None, solvent_boxdims=[70.0, 70.0, 70.0],
                           nonbonded_pars="CM5_UFF", orcatheory=None, numcores=1):
    # , ionicstrength=0.1, iontype='K+'
    print_line_with_mainheader("SmallMolecule Solvator")
    try:
        import openmm as openmm
        import openmm.app as openmm_app
        import openmm.unit as openmm_unit
        from openmm import XmlSerializer
        print("Imported OpenMM library version:", openmm.__version__)

    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: conda install -c conda-forge openmm  \
            Also see http://docs.openmm.org/latest/userguide/application.html")

    def write_pdbfile_openMM(topology, positions, filename):
        openmm.app.PDBFile.writeFile(topology, positions, file=open(filename, 'w'))
        print("Wrote PDB-file:", filename)

    def print_systemsize():
        print("System size: {} atoms\n".format(len(modeller.getPositions())))

    # Defining simple atomnames and atomtypes to be used for solute
    atomnames = [el + "Y" + str(i) for i, el in enumerate(fragment.elems)]
    atomtypes = [el + "X" + str(i) for i, el in enumerate(fragment.elems)]

    # Take input ASH fragment and write a basic PDB file via ASH
    write_pdbfile(fragment, outputname="smallmol", dummyname='LIG', atomnames=atomnames)

    # Load PDB-file and create Modeller object
    pdb = openmm_app.PDBFile("smallmol.pdb")
    print("Loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    numresidues = modeller.topology.getNumResidues()
    print("Modeller topology has {} residues.".format(numresidues))

    # Forcefield

    # TODO: generalize to other solvents.
    # Create local ASH library of XML files
    if watermodel == "tip3p":
        print("Using watermodel=TIP3P . Using parameters in:", ashpath + "/databases/forcefields")
        forcefieldpath = ashpath + "/databases/forcefields"
        waterxmlfile = forcefieldpath + "/tip3p_water_ions.xml"
        coulomb14scale = 1.0
        lj14scale = 1.0
    elif watermodel == "charmm_tip3p":
        coulomb14scale = 1.0
        lj14scale = 1.0
        # NOTE: Problem combining this and solute XML file.
        print("Using watermodel: CHARMM-TIP3P (has ion parameters also)")
        # This is the modified CHARMM-TIP3P (LJ parameters on H at least, maybe bonded parameters defined also)
        # Advantage: also contains ion parameters
        waterxmlfile = "charmm36/water.xml"
    else:
        print("Unknown watermodel.")
        ashexit()

    # Define nonbonded paramers
    if nonbonded_pars == "CM5_UFF":
        print("Using CM5 atomcharges and UFF-LJ parameters.")
        atompropdict = basic_atom_charges_ORCA(fragment=fragment, charge=charge, mult=mult,
                                               orcatheory=orcatheory, chargemodel="CM5", numcores=numcores)
        charges = atompropdict['charges']
        # Basic UFF LJ parameters
        # Converting r0 parameters from Ang to nm and to sigma
        sigmas = [UFF_modH_dict[el][0] * 0.1 / (2 ** (1 / 6)) for el in fragment.elems]
        # Convering epsilon from kcal/mol to kJ/mol
        epsilons = [UFF_modH_dict[el][1] * 4.184 for el in fragment.elems]
    elif nonbonded_pars == "DDEC3" or nonbonded_pars == "DDEC6":
        print("Using {} atomcharges and DDEC-derived parameters.".format(nonbonded_pars))
        atompropdict = basic_atom_charges_ORCA(fragment=fragment, charge=charge, mult=mult,
                                               orcatheory=orcatheory, chargemodel=nonbonded_pars, numcores=numcores)
        charges = atompropdict['charges']
        r0 = atompropdict['r0s']
        eps = atompropdict['epsilons']
        sigmas = [s * 0.1 / (2 ** (1 / 6)) for s in r0]
        epsilons = [e * 4.184 for e in eps]
    elif nonbonded_pars == "xtb_UFF":
        print("Using xTB charges and UFF-LJ parameters.")
        charges = basic_atomcharges_xTB(fragment=fragment, charge=charge, mult=mult, xtbmethod='GFN2')
        # Basic UFF LJ parameters
        # Converting r0 parameters from Ang to nm and to sigma
        sigmas = [UFF_modH_dict[el][0] * 0.1 / (2 ** (1 / 6)) for el in fragment.elems]
        # Convering epsilon from kcal/mol to kJ/mol
        epsilons = [UFF_modH_dict[el][1] * 4.184 for el in fragment.elems]
    else:
        print("Unknown nonbonded_pars option.")
        ashexit()

    print("sigmas:", sigmas)
    print("epsilons:", epsilons)

    # Creating XML-file for solute

    xmlfile = write_xmlfile_nonbonded(resnames=["LIG"], atomnames_per_res=[atomnames], atomtypes_per_res=[atomtypes],
                                      elements_per_res=[fragment.elems], masses_per_res=[fragment.masses],
                                      charges_per_res=[charges],
                                      sigmas_per_res=[sigmas], epsilons_per_res=[epsilons], filename="solute.xml",
                                      coulomb14scale=coulomb14scale, lj14scale=lj14scale)

    print("Creating forcefield using XML-files:", xmlfile, waterxmlfile)
    forcefield = openmm_app.forcefield.ForceField(*[xmlfile, waterxmlfile])

    # , waterxmlfile
    # if extraxmlfile == None:
    #    print("here")
    #    forcefield=openmm_app.forcefield.ForceField(xmlfile, waterxmlfile)
    # else:
    #    print("Using extra XML file:", extraxmlfile)
    #    forcefield=openmm_app.forcefield.ForceField(xmlfile, waterxmlfile, extraxmlfile)

    # Solvent+Ions
    print("Adding solvent, watermodel:", watermodel)
    # NOTE: modeller.addsolvent will automatically add ions to neutralize any excess charge
    # TODO: Replace with something simpler
    if solvent_boxdims is not None:
        print("Solvent boxdimension provided: {} Å".format(solvent_boxdims))
        modeller.addSolvent(forcefield, boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1],
                                                            solvent_boxdims[2]) * openmm_unit.angstrom)

    # Write out solvated system coordinates
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize()
    # Create ASH fragment and write to disk
    newfragment = Fragment(pdbfile="system_aftersolvent.pdb")
    newfragment.print_system(filename="newfragment.ygg")
    newfragment.write_xyzfile(xyzfilename="newfragment.xyz")

    # Return forcefield object,  topology object and ASH fragment
    return forcefield, modeller.topology, newfragment


# Simple XML-writing function. Will only write nonbonded parameters
def write_xmlfile_nonbonded(resnames=None, atomnames_per_res=None, atomtypes_per_res=None, elements_per_res=None,
                            masses_per_res=None, charges_per_res=None, sigmas_per_res=None,
                            epsilons_per_res=None, filename="system.xml", coulomb14scale=0.833333, 
                            lj14scale=0.5, skip_nb=False, charmm=False):
    print("Inside write_xml file")
    # resnames=["MOL1", "MOL2"]
    # atomnames_per_res=[["CM1","CM2","HX1","HX2"],["OT1","HT1","HT2"]]
    # atomtypes_per_res=[["CM","CM","H","H"],["OT","HT","HT"]]
    # sigmas_per_res=[[1.2,1.2,1.3,1.3],[1.25,1.17,1.17]]
    # epsilons_per_res=[[0.2,0.2,0.3,0.3],[0.25,0.17,0.17]]
    # etc.
    # Always list of lists now

    assert len(resnames) == len(atomnames_per_res) == len(atomtypes_per_res)
    # Get list of all unique atomtypes, elements, masses
    # all_atomtypes=list(set([item for sublist in atomtypes_per_res for item in sublist]))
    # all_elements=list(set([item for sublist in elements_per_res for item in sublist]))
    # all_masses=list(set([item for sublist in masses_per_res for item in sublist]))

    # Create list of all AtomTypelines (unique)
    atomtypelines = []
    for resname, atomtypelist, elemlist, masslist in zip(resnames, atomtypes_per_res, elements_per_res, masses_per_res):
        for atype, elem, mass in zip(atomtypelist, elemlist, masslist):
            atomtypeline = "<Type name=\"{}\" class=\"{}\" element=\"{}\" mass=\"{}\"/>\n".format(atype, atype, elem,
                                                                                                  str(mass))
            if atomtypeline not in atomtypelines:
                atomtypelines.append(atomtypeline)
    # Create list of all nonbonded lines (unique)
    nonbondedlines = []
    LJforcelines = []
    for resname, atomtypelist, chargelist, sigmalist, epsilonlist in zip(resnames, atomtypes_per_res, charges_per_res,
                                                                         sigmas_per_res, epsilons_per_res):
        print("atomtypelist:", atomtypelist)
        print("chargelist.", chargelist)
        print("sigmalist", sigmalist)
        for atype, charge, sigma, epsilon in zip(atomtypelist, chargelist, sigmalist, epsilonlist):
            if charmm == True:
                #LJ parameters zero here
                nonbondedline = "<Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, charge,0.0, 0.0)
                #Here we set LJ parameters
                ljline = "<Atom type=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, sigma, epsilon) 
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)
                if ljline not in LJforcelines:
                    LJforcelines.append(ljline)
            else:
                nonbondedline = "<Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, charge,
                                                                                                        sigma, epsilon)
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)

    with open(filename, 'w') as xmlfile:
        xmlfile.write("<ForceField>\n")
        xmlfile.write("<AtomTypes>\n")
        for atomtypeline in atomtypelines:
            xmlfile.write(atomtypeline)
        xmlfile.write("</AtomTypes>\n")
        xmlfile.write("<Residues>\n")
        for resname, atomnamelist, atomtypelist in zip(resnames, atomnames_per_res, atomtypes_per_res):
            xmlfile.write("<Residue name=\"{}\">\n".format(resname))
            for i, (atomname, atomtype) in enumerate(zip(atomnamelist, atomtypelist)):
                xmlfile.write("<Atom name=\"{}\" type=\"{}\"/>\n".format(atomname, atomtype))
            # All other atoms
            xmlfile.write("</Residue>\n")
        xmlfile.write("</Residues>\n")
        if skip_nb is False:

            if charmm == True:
                #Writing both Nonbnded force block and also LennardJonesForce block
                xmlfile.write("<NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n".format(coulomb14scale, lj14scale))
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
                xmlfile.write("<LennardJonesForce lj14scale=\"{}\">\n".format(lj14scale))
                for ljline in LJforcelines:
                    xmlfile.write(ljline)
                xmlfile.write("</LennardJonesForce>\n")
            else:
                #Only NonbondedForce block
                xmlfile.write("<NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n".format(coulomb14scale, lj14scale))
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
        xmlfile.write("</ForceField>\n")
    print("Wrote XML-file:", filename)
    return filename


# TODO: Move elsewhere?
def basic_atomcharges_xTB(fragment=None, charge=None, mult=None, xtbmethod='GFN2'):
    print("Now calculating atom charges for fragment.")
    print("Using default xTB charges.")
    calc = xTBTheory(runmode='inputfile',xtbmethod=xtbmethod)

    Singlepoint(theory=calc, fragment=fragment, charge=charge, mult=mult)
    atomcharges = grabatomcharges_xTB()
    print("atomcharges:", atomcharges)
    print("fragment elems:", fragment.elems)
    return atomcharges


# TODO: Move elsewhere?
def basic_atom_charges_ORCA(fragment=None, charge=None, mult=None, orcatheory=None, chargemodel=None, numcores=1):
    atompropdict = {}
    print("Will calculate charges using ORCA.")

    # Define default ORCA object if notprovided
    if orcatheory is None:
        print("orcatheory not provided. Will do r2SCAN/def2-SVP single-point calculation")
        orcasimpleinput = "! r2SCAN def2-SVP tightscf "
        orcablocks = "%scf maxiter 300 end"
        orcatheory = ORCATheory(orcasimpleinput=orcasimpleinput,
                                orcablocks=orcablocks, numcores=numcores)
    if chargemodel == 'CM5':
        orcatheory.extraline = chargemodel_select(chargemodel)
    # Run ORCA calculation
    Singlepoint(theory=orcatheory, fragment=fragment, charge=charge, mult=mult)
    if 'DDEC' not in chargemodel:
        atomcharges = grabatomcharges_ORCA(chargemodel, orcatheory.filename + '.out')
        atompropdict['charges'] = atomcharges
    else:
        atomcharges, molmoms, voldict = DDEC_calc(elems=fragment.elems, theory=orcatheory,
                                                  gbwfile=orcatheory.filename + '.gbw', numcores=numcores,
                                                  DDECmodel='DDEC3', calcdir='DDEC', molecule_charge=charge,
                                                  molecule_spinmult=mult)
        atompropdict['charges'] = atomcharges
        r0list, epsilonlist = DDEC_to_LJparameters(fragment.elems, molmoms, voldict)
        print("r0list:", r0list)
        print("epsilonlist:", epsilonlist)
        atompropdict['r0s'] = r0list
        atompropdict['epsilons'] = epsilonlist

    print("atomcharges:", atomcharges)
    print("fragment elems:", fragment.elems)
    return atompropdict


def read_NPT_statefile(npt_output):
    import csv
    from collections import defaultdict
    # Read in CSV file of last NPT simulation and store in lists
    columns = defaultdict(list)

    with open(npt_output, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)

    # Extract step number, volume and density and cast as floats
    steps = np.array(columns['#"Step"'])
    volume = np.array(columns["Box Volume (nm^3)"]).astype(float)
    density = np.array(columns["Density (g/mL)"]).astype(float)

    resultdict = {"steps": steps, "volume": volume, "density": density}
    return resultdict

###########################
# CLASS-BASED OpenMM_MD
###########################


# Wrapper function for OpenMM_MDclass
def OpenMM_MD(fragment=None, theory=None, timestep=0.004, simulation_steps=None, simulation_time=None,
              traj_frequency=1000, temperature=300, integrator='LangevinMiddleIntegrator',
              barostat=None, pressure=1, trajectory_file_option='DCD', trajfilename='trajectory',
              coupling_frequency=1, charge=None, mult=None,
              anderson_thermostat=False,
              enforcePeriodicBox=True, dummyatomrestraint=False, center_on_atoms=None, solute_indices=None,
              datafilename=None, dummy_MM=False, plumed_object=None, add_center_force=False,
              center_force_atoms=None, centerforce_constant=1.0, barostat_frequency=25, specialbox=False):
    print_line_with_mainheader("OpenMM MD wrapper function")
    md = OpenMM_MDclass(fragment=fragment, theory=theory, charge=charge, mult=mult, timestep=timestep,
                        traj_frequency=traj_frequency, temperature=temperature, integrator=integrator,
                        barostat=barostat, pressure=pressure, trajectory_file_option=trajectory_file_option,
                        coupling_frequency=coupling_frequency, anderson_thermostat=anderson_thermostat,
                        enforcePeriodicBox=enforcePeriodicBox, dummyatomrestraint=dummyatomrestraint, center_on_atoms=center_on_atoms, solute_indices=solute_indices,
                        datafilename=datafilename, dummy_MM=dummy_MM,
                        plumed_object=plumed_object, add_center_force=add_center_force,trajfilename=trajfilename,
                        center_force_atoms=center_force_atoms, centerforce_constant=centerforce_constant,
                        barostat_frequency=barostat_frequency, specialbox=specialbox)
    if simulation_steps is not None:
        md.run(simulation_steps=simulation_steps)
    elif simulation_time is not None:
        md.run(simulation_time=simulation_time)
    else:
        print("Either simulation_steps or simulation_time need to be defined (not both).")
        ashexit()


class OpenMM_MDclass:
    def __init__(self, fragment=None, theory=None, charge=None, mult=None, timestep=0.004,
                 traj_frequency=1000, temperature=300, integrator='LangevinMiddleIntegrator',
                 barostat=None, pressure=1, trajectory_file_option='DCD', trajfilename='trajectory',
                 coupling_frequency=1,
                 anderson_thermostat=False,
                 enforcePeriodicBox=True, dummyatomrestraint=False, center_on_atoms=None, solute_indices=None,
                 datafilename=None, dummy_MM=False, plumed_object=None, add_center_force=False,
                 center_force_atoms=None, centerforce_constant=1.0,
                 barostat_frequency=25, specialbox=False):
        module_init_time = time.time()

        print_line_with_mainheader("OpenMM Molecular Dynamics Initialization")

        if fragment is None:
            print("No fragment object. Exiting.")
            ashexit()
        else:
            self.fragment = fragment

        #Check charge/mult
        self.charge, self.mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "OpenMM_MD", theory=theory)

        #External QM option off by default
        self.externalqm=False

        # Distinguish between OpenMM theory QM/MM theory or QM theory
        self.dummy_MM=dummy_MM

        #Case: OpenMMTheory
        if isinstance(theory, OpenMMTheory):
            self.openmmobject = theory
            self.QM_MM_object = None
        #Case: QM/MM theory with OpenMM mm_theory
        elif isinstance(theory, ash.QMMMTheory):
            self.QM_MM_object = theory
            self.openmmobject = theory.mm_theory
            if barostat is not None:
                print("QM/MM MD currently only works in NVT ensemble.")
                ashexit()

        #Case: OpenMM with external QM
        else:
            #NOTE: Recognize QM theories here ??
            print("Unrecognized theory.")
            print("Will assume to be QM theory and will continue")
            print("QM-program forces will be added as a custom external force to OpenMM")
            self.externalqm=True

            #Creating dummy OpenMMTheory (basic topology, particle masses, no forces except CMMRemoval)
            self.openmmobject = OpenMMTheory(fragment=fragment, dummysystem=True) #NOTE: might add more options here
            self.QM_MM_object = None
            self.qmtheory=theory
        
        # Assigning some basic variables
        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.coupling_frequency = coupling_frequency
        self.timestep = timestep
        self.traj_frequency = int(traj_frequency)
        self.plumed_object = plumed_object
        self.barostat_frequency = barostat_frequency
        #PERIODIC or not
        if self.openmmobject.Periodic is True:
            #Generally we want True but for now allowing user to modify (default=True)
            self.enforcePeriodicBox=enforcePeriodicBox
        else:
            print("System is non-periodic. Setting enforcePeriodicBox to False")
            #Non-periodic. Setting enforcePeriodicBox to False (otherwise nonsense)
            self.enforcePeriodicBox=False
        
        print_line_with_subheader2("MD system parameters")
        print("Temperature: {} K".format(self.temperature))
        print("OpenMM autoconstraints:", self.openmmobject.autoconstraints)
        print("OpenMM hydrogenmass:",
               self.openmmobject.hydrogenmass)  # Note 1.5 amu mass is recommended for LangevinMiddle with 4fs timestep
        print("OpenMM rigidwater constraints:", self.openmmobject.rigidwater)
        print("User Constraints:", self.openmmobject.user_constraints)
        print("User Restraints:", self.openmmobject.user_restraints)
        print("Number of atoms:", self.fragment.numatoms)
        print("Number of frozen atoms:", len(self.openmmobject.user_frozen_atoms))
        if len(self.openmmobject.user_frozen_atoms) < 50:
             print("Frozen atoms", self.openmmobject.user_frozen_atoms)
        print("Integrator:", self.integrator)
        print("Timestep: {} ps".format(self.timestep))
        print("Anderon Thermostat:", anderson_thermostat)
        print("coupling_frequency: {} ps^-1 (for Nosé-Hoover and Langevin integrators)".format(self.coupling_frequency))
        print("Barostat:", barostat)

        print("")
        print("Will write trajectory in format:", trajectory_file_option)
        print("Trajectory write frequency:", self.traj_frequency)
        print("enforcePeriodicBox:", self.enforcePeriodicBox)
        print("")
        #specialbox for QM/MM
        self.specialbox=specialbox

        if self.openmmobject.autoconstraints is None:
            print(f"""{BC.WARNING}
                WARNING: Autoconstraints have not been set in OpenMMTheory object definition. This means that by 
                         default no bonds are constrained in the MD simulation. This usually requires a small 
                         timestep: 0.5 fs or so.
                         autoconstraints='HBonds' is recommended for 2 fs timesteps with LangevinIntegrator and 4fs with LangevinMiddleIntegrator).
                         autoconstraints='AllBonds' or autoconstraints='HAngles' allows even larger timesteps to be used.
                         See : https://github.com/openmm/openmm/pull/2754 and https://github.com/openmm/openmm/issues/2520 
                         for recommended simulation settings in OpenMM.
                         {BC.END}""")
            print("Will continue...")
        if self.openmmobject.rigidwater is True and len(self.openmmobject.user_frozen_atoms) != 0 or (
                self.openmmobject.autoconstraints is not None and len(self.openmmobject.user_frozen_atoms) != 0):
            print(
                f"{BC.WARNING}WARNING: Frozen_atoms options selected but there are general constraints defined in{BC.END} "
                f"{BC.WARNING}the OpenMM object (either rigidwater=True or autoconstraints is not None){BC.END}"
                f"{BC.WARNING}\nOpenMM will crash if constraints and frozen atoms involve the same atoms{BC.END}")
        print("")

        print("Defining atom positions from fragment")
        #Note: using self.positions as we may add dummy atoms (e.g. dummyatomrestraint below) 
        self.positions = self.fragment.coords
        
        #Dummy-atom restraint to deal with NPT simulations that contain constraints/restraints/frozen_atoms
        self.dummyatomrestraint=dummyatomrestraint
        if self.dummyatomrestraint is True:
            if solute_indices == None:
                print("Dummyatomrestraint requires solute_indices to be set")
                ashexit()
            print(BC.WARNING,"Warning: Using dummyatomrestraints. This means that we will add a dummy atom to topology and OpenMM coordinates")
            print("We do not add the dummy atom to ASH-fragment")
            print("Affects visualization of trajectory (make sure to use PDB-file that contains the dummy-atom, printed in the end)",BC.END)
            #Should be centroid of solute or something rather
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            dummypos=get_centroid(solute_coords)
            print("Dummy atom will be added to position:", dummypos)
            #Adding dummy-atom coordinates to self.positions
            self.positions = np.append(self.positions, [dummypos], axis=0)
            print("len self.pos", len(self.positions))
            print("len self.fragment.coords", len(self.fragment.coords))

            #Restraining solute atoms to dummy-atom
            self.openmmobject.add_dummy_atom_to_restrain_solute(atomindices=solute_indices)




        #TRANSLATE solute: #https://github.com/openmm/openmm/issues/1854
        # Translate solute to geometric center on origin
        #centroid = np.mean(positions[solute, :] / positions.unit, axis=0) * positions.unit
        #positions -= centroid            
        if center_on_atoms != None:
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            changed_origin_coords = change_origin_to_centroid(self.fragment.coords, subsetcoords=solute_coords)
            print("changed_origin_coords", changed_origin_coords)

        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        # Set up system with chosen barostat, thermostat, integrator
        if barostat is not None:
            print("Attempting to add barostat.")
            if "MonteCarloBarostat" not in forceclassnames:
                print("Adding barostat.")
                montecarlobarostat=self.openmmobject.openmm.MonteCarloBarostat(self.pressure * self.openmmobject.openmm.unit.bar,
                                                                self.temperature * self.openmmobject.openmm.unit.kelvin)
                #Setting barostat frequency to chosen value or default (25)
                montecarlobarostat.setFrequency(self.barostat_frequency)
                self.openmmobject.system.addForce(montecarlobarostat)
            else:
                print("Barostat already present. Skipping.")
            # print("after barostat added")

            self.integrator = "LangevinMiddleIntegrator"
            print("Barostat requires using integrator:", integrator)
            #self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature,
            #                                    integrator=self.integrator,
            #                                    coupling_frequency=self.coupling_frequency)
            self.openmmobject.set_simulation_parameters(timestep=self.timestep, temperature=self.temperature, 
                                                        integrator=self.integrator, coupling_frequency=self.coupling_frequency)
        elif anderson_thermostat is True:
            print("Anderson thermostat is on.")
            if "AndersenThermostat" not in forceclassnames:
                self.openmmobject.system.addForce(
                    self.openmmobject.openmm.AndersenThermostat(self.temperature * self.openmmobject.openmm.unit.kelvin,
                                                                1 / self.openmmobject.openmm.unit.picosecond))
            self.integrator = "VerletIntegrator"
            print("Now using integrator:", integrator)
            #self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature,
            #                                    integrator=self.integrator,
            #                                    coupling_frequency=coupling_frequency)
            self.openmmobject.set_simulation_parameters(timestep=self.timestep, temperature=self.temperature, 
                                                        integrator=self.integrator, coupling_frequency=self.coupling_frequency)
        else:
            # Deleting barostat and Andersen thermostat if present from previous sims
            for i, forcename in enumerate(forceclassnames):
                if forcename == "MonteCarloBarostat" or forcename == "AndersenThermostat":
                    print("Removing old force:", forcename)
                    self.openmmobject.system.removeForce(i)

            # Regular thermostat or integrator without barostat
            # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
            # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
            #self.openmmobject.create_simulation(timestep=self.timestep, temperature=self.temperature,
            #                                    integrator=self.integrator,
            #                                    coupling_frequency=self.coupling_frequency)
            self.openmmobject.set_simulation_parameters(timestep=self.timestep, temperature=self.temperature, 
                                                        integrator=self.integrator, coupling_frequency=self.coupling_frequency)
        
        self.openmmobject.update_simulation()
        print("Simulation updated.")
        #if self.openmmobject.Periodic is True:
        #    print("PME parameters in context", self.openmmobject.nonbonded_force.getPMEParametersInContext(self.openmmobject.simulation.context))
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        print("OpenMM System forces present:", forceclassnames)
        if self.openmmobject.Periodic is True:
            print("Checking Initial PBC vectors.")
            self.state = self.openmmobject.simulation.context.getState()
            a, b, c = self.state.getPeriodicBoxVectors()
            print(f"A: ", a)
            print(f"B: ", b)
            print(f"C: ", c)
        else:
            print("System is not periodic")


        # THIS DOES NOT APPLY TO QM/MM. MOVE ELSEWHERE??
        #TODO: See if this can be made to work for simulations with step-by-step
        if trajectory_file_option == 'PDB':
            self.openmmobject.simulation.reporters.append(
                self.openmmobject.openmm.app.PDBReporter(trajfilename+'.pdb', self.traj_frequency,
                                                         enforcePeriodicBox=self.enforcePeriodicBox))
        elif trajectory_file_option == 'DCD':
            # NOTE: Disabling for now
            # with open('initial_MDfrag_step1.pdb', 'w') as f: openmmobject.openmm.app.pdbfile.PDBFile
            # .writeModel(openmmobject.topology, openmmobject.simulation.context.getState(getPositions=True,
            # enforcePeriodicBox=enforcePeriodicBox).getPositions(), f)
            # print("Wrote PDB")
            self.openmmobject.simulation.reporters.append(
                self.openmmobject.openmm.app.DCDReporter(trajfilename+'.dcd', self.traj_frequency,
                                                         enforcePeriodicBox=self.enforcePeriodicBox))
        elif trajectory_file_option == 'NetCDFReporter':
            print("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import_()
            self.openmmobject.simulation.reporters.append(
                mdtraj.reporters.NetCDFReporter(trajfilename+'.nc', self.traj_frequency))
        elif trajectory_file_option == 'HDF5Reporter':
            print("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import_()
            self.openmmobject.simulation.reporters.append(
                mdtraj.reporters.HDF5Reporter(trajfilename+'.lh5', self.traj_frequency,
                                              enforcePeriodicBox=self.enforcePeriodicBox))

        if barostat is not None:
            volume = density = True
        else:
            volume = density = False

        # If statedatareporter filename set:
        self.datafilename=datafilename
        if self.datafilename is not None:
            #Remove old file
            try:
                os.remove(self.datafilename)
            except FileNotFoundError:
                pass

            #Now doing open file object in append mode instead of just filename.
            #Just filename does not play nice when running simulation step by step
            #Future OpenMM update may do this automatically?
            self.dataoutputoption = open(self.datafilename,'a')
            print("Will write data to file:", self.datafilename)
        # otherwise stdout:
        else:
            self.dataoutputoption = stdout
        self.statedatareporter=self.openmmobject.openmm.app.StateDataReporter(self.dataoutputoption, self.traj_frequency, step=True, time=True,
                                                           potentialEnergy=True, kineticEnergy=True, volume=volume,
                                                           density=density, temperature=True, separator=' ')
        self.openmmobject.simulation.reporters.append(self.statedatareporter)

        # NOTE: Better to use OpenMM-plumed interface instead??
        if plumed_object is not None:
            print("Plumed active")
            # Create new OpenMM custom external force
            print("Creating new OpenMM custom external force for Plumed.")
            self.plumedcustomforce = self.openmmobject.add_custom_external_force()

        #QM MD
        if self.externalqm is True:
            print("Creating new OpenMM custom external force for external QM theory.")
            self.qmcustomforce = self.openmmobject.add_custom_external_force()
        # QM/MM MD
        if self.QM_MM_object is not None:
            print("QM_MM_object provided. Switching to QM/MM loop.")
            #print("QM/MM requires enforcePeriodicBox to be False.")
            #True means we end up with solute in corner of box (wrong for nonPBC QM code)
            #NOTE: but OK for proteins?
            #self.enforcePeriodicBox = True
            # enforcePeriodicBox or not
            print("self.enforcePeriodicBox:", self.enforcePeriodicBox)

            # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
            # (it would calculate OpenMM energy twice) so turning on in case forgotten
            if self.QM_MM_object.openmm_externalforce is False:
                print("QM/MM object was not set to have 'openmm_externalforce=True'.")
                print("Turning on externalforce option.")
                self.QM_MM_object.openmm_externalforce = True
                self.QM_MM_object.openmm_externalforceobject = self.QM_MM_object.mm_theory.add_custom_external_force()
                print("1openmm obj integrator:", self.openmmobject.integrator)
            # TODO:
            # Should we set parallelization of QM theory here also in case forgotten?

            centercoordinates = False
            # CENTER COORDINATES HERE on SOLUTE HERE ??
            # TODO: Deprecated I think
            if centercoordinates is True:
                # Solute atoms assumed to be QM-region
                self.fragment.write_xyzfile(xyzfilename="fragment-before-centering.xyz")
                soluteatoms = self.QM_MM_object.qmatoms
                solutecoords = self.fragment.get_coords_for_atoms(soluteatoms)[0]
                print("Changing origin to centroid.")
                self.fragment.coords = change_origin_to_centroid(fragment.coords, subsetcoords=solutecoords)
                self.fragment.write_xyzfile(xyzfilename="fragment-after-centering.xyz")

            # Now adding center force acting on solute
            if add_center_force is True:
                # print("add_center_force is True")
                print("Forceconstant is: {} kcal/mol/Ang^2".format(centerforce_constant))
                if center_force_atoms is None:
                    print("center_force_atoms unset. Using QM/MM atoms:", self.QM_MM_object.qmatoms)
                    center_force_atoms = self.QM_MM_object.qmatoms
                # Get geometric center of system (Angstrom)
                center = self.fragment.get_coordinate_center()
                print("center:", center)

                self.openmmobject.add_center_force(center_coords=center, atomindices=center_force_atoms,
                                                   forceconstant=centerforce_constant)


            # After adding QM/MM force, possible Plumed force, possible center force
            # Let's list all OpenMM object system forces
            print("2openmm obj integrator:", self.openmmobject.integrator)
            print("OpenMM Forces defined:", self.openmmobject.system.getForces())
            print("Now starting QM/MM MD simulation.")
            print("openmm obj integrator:", self.openmmobject.integrator)
            # Does step by step

            print_time_rel(module_init_time, modulename="OpenMM_MD setup", moduleindex=1)

    # Simulation loop
    def run(self, simulation_steps=None, simulation_time=None):
        module_init_time = time.time()
        print_line_with_mainheader("OpenMM Molecular Dynamics Run")
        if simulation_steps is None and simulation_time is None:
            print("Either simulation_steps or simulation_time needs to be set.")
            ashexit()
        if simulation_time is not None:
            simulation_steps = int(simulation_time / self.timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * self.timestep

        print_line_with_subheader2("MD run parameters")
        print("Simulation time: {} ps".format(simulation_time))
        print("Simulation steps: {}".format(simulation_steps))
        print("Timestep: {} ps".format(self.timestep))
        print("Set temperature: {} K".format(self.temperature))
        print("OpenMM integrator:", self.openmmobject.integrator_name)
        print()
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        print("OpenMM System forces present before run:", forceclassnames)
        # Delete old traj
        try:
            os.remove("OpenMMMD_traj.xyz")
        # Crashes when permissions not present or file is folder. Should never occur.
        except FileNotFoundError:
            pass

        #Make sure file associated with StateDataReporter is open
        if self.datafilename is not None:
            self.dataoutputoption = open(self.datafilename,'a')

        # Setting coordinates of OpenMM object from current fragment.coords
        self.openmmobject.set_positions(self.positions)
        print()
        # Thermalize the system.
        self.openmmobject.set_velocities_to_temperature(self.temperature)
        # Run simulation
        # kjmolnm_to_atomic_factor = -49614.752589207

        if self.QM_MM_object is not None:
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                print("Step:", step)

                # Get the current simulation state from the context. Use unwrapped
                # coordinates, since we will wrap and recenter molecules with MDTraj.
                current_state = self.openmmobject.simulation.context.getState(
                    getPositions=True,
                    enforcePeriodicBox=False,
                    getEnergy=True
                )

                # Get the current coordinates.
                coords = current_state.getPositions()

                # Write a DCD frame for the current state. Note that we use the box
                # vectors from the topology, so this won't work for NPT simulations.
                with open("state.dcd", "wb") as file:
                    dcd_file = self.openmmobject.openmm_app.DCDFile(
                        file,
                        self.openmmobject.topology,
                        dt=1
                    )
                    dcd_file.writeModel(coords)

                # Load the state with MDTraj. Here we use the topology file
                # used to create the OpenMM system.
                traj = mdtraj.load_dcd("state.dcd", top=self.openmmobject.topfile)

                # Image the molecules so that the solute is centered.
                traj.image_molecules(inplace=True)

                # Get back a NumPy arrray of the current positions, in Angstrom.
                current_coords = traj.xyz[0].astype("float64") * 10

                print_time_rel(checkpoint, modulename="get OpenMM state", moduleindex=2)
                checkpoint = time.time()

                #TODO: Translate box coordinates so that they are centered on solute
                #Do manually or use mdtraj, mdanalysis or something??
                if self.specialbox is True:
                    print("not ready")
                    ashexit()
                    solute_coords = np.take(current_coords, solute_indices, axis=0)
                    changed_origin_coords = change_origin_to_centroid(self.fragment.coords, subsetcoords=solute_coords)
                    current_coords = center_coordinates(current_coords,)

                
                #Printing step-info or write-trajectory at regular intervals
                if step % self.traj_frequency == 0:
                    # Manual step info option
                    print_current_step_info(step,current_state,self.openmmobject)
                    print_time_rel(checkpoint, modulename="print_current_step_info", moduleindex=2)
                    checkpoint = time.time()
                    # Manual trajectory option (reporters do not work for manual dynamics steps)
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", printlevel=1, writemode='a')
                    print_time_rel(checkpoint, modulename="OpenMM_MD writetraj", moduleindex=2)
                    checkpoint = time.time()
                

                checkpoint = time.time()
                print_time_rel(checkpoint, modulename="get current_coords", moduleindex=2)
                # Run QM/MM step to get full system QM+PC gradient.
                # Updates OpenMM object with QM-PC forces
                self.QM_MM_object.run(current_coords=current_coords, elems=self.fragment.elems, Grad=True,
                                      exit_after_customexternalforce_update=True, charge=self.charge, mult=self.mult, step=step)
                print_time_rel(checkpoint, modulename="QM/MM run", moduleindex=2)
                # NOTE: Think about energy correction (currently skipped above)
                # Now take OpenMM step (E+G + displacement etc.)
                checkpoint = time.time()
                self.openmmobject.simulation.step(1)
                print_time_rel(checkpoint, modulename="openmmobject sim step", moduleindex=2)
                print_time_rel(checkpoint_begin_step, modulename="Total sim step", moduleindex=2)
                
                # NOTE: Better to use OpenMM-plumed interface instead??
                # After MM step, grab coordinates and forces
                if self.plumed_object is not None:
                    print("Plumed active. Untested. Hopefully works.")
                    #Necessary to call again
                    current_state_forces=self.openmmobject.simulation.context.getState(getForces=True, enforcePeriodicBox=self.enforcePeriodicBox,)
                    #current_coords = np.array(
                    #    self.openmmobject.simulation.context.getState(getPositions=True).getPositions(
                    #        asNumpy=True))  # in nm
                    current_coords = np.array(current_state.getPositions(asNumpy=True)) #in nm
                    current_forces = np.array(current_state_forces.getForces(asNumpy=True)) # in kJ/mol /nm
                    #np.array(
                    #    self.openmmobject.simulation.context.getState(getForces=True).getForces(
                    #        asNumpy=True))  # in kJ/mol /nm
                    # Plumed object needs to be configured for OpenMM
                    energy, newforces = self.plumed_object.run(coords=current_coords, forces=current_forces,
                                                               step=step)
                    self.openmmobject.update_custom_external_force(self.plumedcustomforce, newforces)

        #External QM for OpenMMtheory
        #Used to run QM dynamics with OpenMM
        elif self.externalqm is True:
            print("External QM with OpenMM option")
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                print("Step:", step)
                #Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state=self.openmmobject.simulation.context.getState(getPositions=True, enforcePeriodicBox=self.enforcePeriodicBox, getEnergy=True)
                print_time_rel(checkpoint, modulename="get OpenMM state", moduleindex=2)
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True))*10
                print_time_rel(checkpoint, modulename="get current coords", moduleindex=2)
                checkpoint = time.time()

                #Printing step-info or write-trajectory at regular intervals
                if step % self.traj_frequency == 0:
                    # Manual step info option
                    print_current_step_info(step,current_state,self.openmmobject)
                    print_time_rel(checkpoint, modulename="print_current_step_info", moduleindex=2)
                    checkpoint = time.time()
                    # Manual trajectory option (reporters do not work for manual dynamics steps)
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", printlevel=1, writemode='a')
                    print_time_rel(checkpoint, modulename="OpenMM_MD writetraj", moduleindex=2)
                    checkpoint = time.time()

                # Run QM step to get full system QM gradient.
                # Updates OpenMM object with QM forces
                energy,gradient=self.qmtheory.run(current_coords=current_coords, elems=self.fragment.elems, Grad=True, charge=self.charge, mult=self.mult)
                print("energy:", energy)
                print_time_rel(checkpoint, modulename="QM run", moduleindex=2)


                self.openmmobject.update_custom_external_force(self.qmcustomforce,gradient)

                #Calculate energy associated with external force so that we can subtract it later
                #TODO: take this and QM energy and add to print_current_step_info
                extforce_energy=3*np.mean(sum(gradient*current_coords*1.88972612546))
                print("extforce_energy:", extforce_energy)

                self.openmmobject.simulation.step(1)
                print_time_rel(checkpoint, modulename="OpenMM sim step", moduleindex=2)
                print_time_rel(checkpoint_begin_step, modulename="Total sim step", moduleindex=2)


                # NOTE: Better to use OpenMM-plumed interface instead??
                # After MM step, grab coordinates and forces
                if self.plumed_object is not None:
                    print("Plumed active. Untested. Hopefully works.")
                    #Necessary to call again
                    current_state_forces=self.openmmobject.simulation.context.getState(getForces=True, enforcePeriodicBox=self.enforcePeriodicBox,)
                    #Keep coords as default OpenMM nm and forces ad kJ/mol/nm. Avoid conversion
                    plumed_coords = np.array(current_state.getPositions(asNumpy=True)) #in nm
                    plumed_forces = np.array(current_state_forces.getForces(asNumpy=True)) # in kJ/mol /nm
                    # Plumed object needs to be configured for OpenMM
                    energy, newforces = self.plumed_object.run(coords=plumed_coords, forces=plumed_forces,
                                                               step=step)
                    self.openmmobject.update_custom_external_force(self.plumedcustomforce, newforces, conversion_factor=1.0)




        #TODO: Delete at some point once testing and debugging are over
        elif self.dummy_MM is True:
            print("Dummy MM option")
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                print("Step:", step)
                #Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state=self.openmmobject.simulation.context.getState(getPositions=True, enforcePeriodicBox=self.enforcePeriodicBox, getEnergy=True)
                print_time_rel(checkpoint, modulename="get OpenMM state", moduleindex=2)
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True))*10
                print_time_rel(checkpoint, modulename="get current coords", moduleindex=2)
                checkpoint = time.time()
                #Printing step-info or write-trajectory at regular intervals
                if step % self.traj_frequency == 0:
                    # Manual step info option
                    print_current_step_info(step,current_state,self.openmmobject)
                    print_time_rel(checkpoint, modulename="print_current_step_info", moduleindex=2)
                    checkpoint = time.time()
                    # Manual trajectory option (reporters do not work for manual dynamics steps)
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", printlevel=1, writemode='a')
                    print_time_rel(checkpoint, modulename="OpenMM_MD writetraj", moduleindex=2)
                    checkpoint = time.time()

                self.openmmobject.simulation.step(1)
                print_time_rel(checkpoint, modulename="OpenMM sim step", moduleindex=2)
                print_time_rel(checkpoint_begin_step, modulename="Total sim step", moduleindex=2)
        else:
            print("Regular classical OpenMM MD option chosen.")
            #This is the fastest option as getState is never called in each loop iteration like above
            # Running all steps in one go
            # TODO: If we wanted to support plumed then we would have to do step 1-by-1 instead
            self.openmmobject.simulation.step(simulation_steps)

        print_line_with_subheader2("OpenMM MD simulation finished!")

        
        #Delete dummyatoms if defined
        #NOTE: probably not possible
        #if self.dummyatomrestraint is True:
        #    print("Removing dummy atom from OpenMM topology and system")
        #    self.openmmobject.remove_dummy_atom()

        #Close Statadatareporter file if open
        if self.datafilename != None:
            self.dataoutputoption.close()


        # Close Plumed also if active. Flushes HILLS/COLVAR etc.
        if self.plumed_object is not None:
            self.plumed_object.close()

        # enforcePeriodicBox=True
        self.state = self.openmmobject.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforcePeriodicBox)
        print("Checking PBC vectors:")
        a, b, c = self.state.getPeriodicBoxVectors()
        print(f"A: ", a)
        print(f"B: ", b)
        print(f"C: ", c)

        # Set new PBC vectors since they may have changed
        print("Updating PBC vectors.")
        # Context. Used?
        self.openmmobject.simulation.context.setPeriodicBoxVectors(a, b, c)
        # System. Necessary
        self.openmmobject.system.setDefaultPeriodicBoxVectors(a, b, c)

        # Writing final frame to disk as PDB
        with open('final_MDfrag_laststep.pdb', 'w') as f:
            self.openmmobject.openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
        with open('final_MDfrag_laststep.pdb', 'a') as f:
            self.openmmobject.openmm.app.pdbfile.PDBFile.writeModel(self.openmmobject.topology,
                                                                    self.state.getPositions(asNumpy=True).value_in_unit(
                                                                        self.openmmobject.unit.angstrom), f)
        # Updating ASH fragment
        newcoords = self.state.getPositions(asNumpy=True).value_in_unit(self.openmmobject.unit.angstrom)
        print("Updating coordinates in ASH fragment.")
        self.fragment.coords = newcoords
        #Updating positions array also in case we call run again
        self.positions = newcoords
        

        print_time_rel(module_init_time, modulename="OpenMM_MD run", moduleindex=1)


#############################
#  Multi-step MD protocols  #
#############################

#Note: dummyatomrestraints necessary for NPT simulation when constraining atoms in space
def OpenMM_box_relaxation(fragment=None, theory=None, datafilename="nptsim.csv", numsteps_per_NPT=10000,
                          volume_threshold=1.3, density_threshold=0.0012, temperature=300, timestep=0.004,
                          traj_frequency=100, trajfilename='relaxbox_NPT', trajectory_file_option='DCD', coupling_frequency=1,
                          dummyatomrestraint=False, solute_indices=None, barostat_frequency=25):
    """NPT simulations until volume and density stops changing

    Args:
        fragment ([type], optional): [description]. Defaults to None.
        theory ([type], optional): [description]. Defaults to None.
        datafilename (str, optional): [description]. Defaults to "nptsim.csv".
        numsteps_per_NPT (int, optional): [description]. Defaults to 10000.
        volume_threshold (float, optional): [description]. Defaults to 1.0.
        density_threshold (float, optional): [description]. Defaults to 0.001.
        temperature (int, optional): [description]. Defaults to 300.
        timestep (float, optional): [description]. Defaults to 0.004.
        traj_frequency (int, optional): [description]. Defaults to 100.
        trajectory_file_option (str, optional): [description]. Defaults to 'DCD'.
        coupling_frequency (int, optional): [description]. Defaults to 1.
        barostat_frequency (int, optional): [description]. Defaults to 25 (timesteps).
    """


    print_line_with_mainheader("Periodic Box Size Relaxation")

    if fragment is None or theory is None:
        print("Fragment and theory required.")
        ashexit()

    if numsteps_per_NPT < traj_frequency:
        print("Parameter 'numpsteps_per_NPT' must be greater than 'traj_frequency', otherwise"
              "no data will be written during the relaxation!")
        ashexit()

    print_line_with_subheader2("Relaxation Parameters")
    print("Steps per NPT cycle:", numsteps_per_NPT)
    print(f"Step size: {timestep * 1000} fs")
    print("Density threshold:", density_threshold)
    print("Volume threshold:", volume_threshold)
    print("Intermediate MD trajectory data file:", datafilename)

    if len(theory.user_frozen_atoms) > 0:
        print("Frozen_atoms:", theory.user_frozen_atoms)
        print(BC.WARNING,"OpenMM object has frozen atoms defined. This is known to cause strange issues for NPT simulations.",BC.END)
        print(BC.WARNING,"Check the results carefully!",BC.END)


    # Starting parameters
    steps = 0
    volume_std = 10
    density_std = 1

    md = OpenMM_MDclass(fragment=fragment, theory=theory, timestep=timestep, traj_frequency=traj_frequency,
                        temperature=temperature, integrator="LangevinMiddleIntegrator",
                        coupling_frequency=coupling_frequency, barostat='MonteCarloBarostat', trajfilename=trajfilename,
                        datafilename=datafilename, trajectory_file_option=trajectory_file_option,
                        dummyatomrestraint=dummyatomrestraint, solute_indices=solute_indices,
                        barostat_frequency=barostat_frequency)

    while volume_std >= volume_threshold and density_std >= density_threshold:
        md.run(numsteps_per_NPT)
        steps += numsteps_per_NPT

        # Read reporter file and calculate stdev
        NPTresults = read_NPT_statefile(datafilename)
        volume = NPTresults["volume"][-traj_frequency:]
        density = NPTresults["density"][-traj_frequency:]
        # volume = volume[-traj_frequency:]
        # density = density[-traj_frequency:]
        volume_std = np.std(volume)
        density_std = np.std(density)

        print_line_with_subheader2("Relaxation Status")
        print("Total steps taken:", steps)
        print(f"Total simulation time: {timestep * steps} ps")
        print("Current Volume:", volume[-1])
        print("Current Volume SD:", volume_std)
        print("Current Density", density[-1])
        print("Current Density SD", density_std)
        # print("Steps\tVolume\tVol. SD\tDensity\tDens. SD")
        # print(f"{steps}\t{volume[-1]}\t{volume_std}\t{density[-1]}\t{density_std}")
        # print("{} steps taken. Volume : {} stdev: {}\tDensity: {} stdev: {}".format(steps,
        #                                                                                     volume[-1],
        #                                                                                      volume_std,
        #                                                                                      density[-1],
        #                                                                                      density_std))

    print("Relaxation of periodic box size finished!\n")
    return theory.simulation.context.getState().getPeriodicBoxVectors()


#Kinetic energy from velocities
def calc_kinetic_energy(velocities,dof):
    kin=0.0
    for v in velocities:
        kin+=0.5*np.dot(v,v)
    return 2*kin / (dof*ash.constants.BOLTZ)

#Used in OpenMM_MD when doing simulation step-by-step (e.g. QM/MM MD)
def print_current_step_info(step,state,openmmobject):
    kinetic_energy=state.getKineticEnergy()
    pot_energy=state.getPotentialEnergy()
    temp=(2*kinetic_energy/(openmmobject.dof*openmmobject.unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(openmmobject.unit.kelvin)
    
    print("="*50)
    print("SIMULATION STATUS (STEP {})".format(step))
    print("_"*50)
    print("Time: {}".format(state.getTime()))
    print("Potential energy:", pot_energy)
    print("Kinetic energy:", kinetic_energy )
    print("Temperature: {}".format(temp))
    print("="*50)
    

#CHECKING PDB-FILE FOR multiple occupations.
#Default behaviour: 
# - if no multiple occupancies return input PDBfile and go on
# - if multiple occupancies, print list of residues and tell user to fix them. Exiting
# - if use_higher_occupancy is set to True, user higher occupancy location, write new PDB_file and use

def find_alternate_locations_residues(pdbfile, use_higher_occupancy=False):
    if use_higher_occupancy is True:
        print("Will keep higher occupancy atoms for alternate locations")
    
    #List of ATOM/HETATM lines to grab from PDB-file
    pdb_atomlines=[]
    #Dict of residues with alternate location labels
    bad_resids_dict={}

    #Alternate location dict for atoms found
    altloc_dict={}

    #Looping through PDB-file
    with open(pdbfile) as pfile:
        for line in pfile:
            #Only keep ATOM/HETATM lines
            if line.startswith("ATOM") or line.startswith("HETATM"):
                altloc=line[16]
                #Adding info to dicts and adding marker if alternate location info present for atom
                if altloc != " ":
                    chain=line[21:22]
                    #New dict item with chain as key
                    if chain not in bad_resids_dict:
                        bad_resids_dict[chain] = []
                    resid=int(line[22:26].replace(" ", ""))
                    resname=line[17:20].replace(" ", "")
                    residue=resname+str(resid)
                    atomname=line[12:16].replace(" ","")
                    occupancy=float(line[54:60])
                    #Atomstring contains only the atom-information (not alt-location label)
                    atomstring=chain+"_"+resname+"_"+str(resid)+"_"+atomname
                    #Adding residue to dict
                    if residue not in bad_resids_dict[chain]:
                        bad_resids_dict[chain].append(residue)
                    #Adding atom-info to dict
                    altloc_dict[(atomstring,altloc)]=[altloc,occupancy,line]
                    #Adding atomstring to list as a marker
                    if ["REPLACE_",atomstring] not in pdb_atomlines:
                        pdb_atomlines.append(["REPLACE_",atomstring])
                #Use unmodifed ATOM line
                else:
                    pdb_atomlines.append(line)
    #For debugging
    #for k,v in altloc_dict.items():
    #    print(k, v)
    def find_index_of_sublist_with_max_col(l,index):
        max=0
        result=None
        for i,s in enumerate(l):
            if s[index] > max:
                max=s[index]
                result=i
        return result
    
    #Now going through pdb_atomlines, finding marker and looking up the best occupancy atom from altloc_dict
    finalpdblines=[]
    for pdbline in pdb_atomlines:

        if pdbline[0]== "REPLACE_":
            print("Alternate locations for atom:", pdbline[1])
            options=[]
            #Looping through altloc_dict items
            for i,j in altloc_dict.items():
                #Matching atomstring
                if i[0] == pdbline[1]:
                    options.append([j[0],j[1],j[2]])
            for l in options:
                pdblinestring=''.join(map(str,l[2:]))
                print(pdblinestring)
            #Get max occupancy item
            ind = find_index_of_sublist_with_max_col(options,1)
            fline = options[ind][2][:16] + " " + options[ind][2][16 + 1:]
            #print(f"Choosing line {fline} based on occupancy {options[ind][1]}.")
            print(f"Choosing line with occupancy {options[ind][1]}.")
            print("-"*90)
            if fline not in finalpdblines:
                finalpdblines.append(fline)
        else:
            finalpdblines.append(pdbline)

    if len(bad_resids_dict) > 0:
        print(BC.WARNING,"\nFound residues in PDB-file that have alternate location labels i.e. multiple occupancies:", BC.END)
        for chain,residues in bad_resids_dict.items():
            print(f"\nChain {chain}:")
            for res in residues:
                print(res)
        print(BC.WARNING,"\nThese residues should be manually inspected and fixed in the PDB-file before continuing", BC.END)
        #if alternatelocation_label != None:
        #    print(BC.WARNING,"\nalternatelocation_label option chosen. Will choose form {} and go on.\n".format(alternatelocation_label), BC.END)
        #    writelisttofile(pdb_atomlines, "system_afteratlocfixes.pdb", separator="")
        #    return "system_afteratlocfixes.pdb"
        if use_higher_occupancy is True:
            print(BC.WARNING,"\n Use higher-occupancy location opton was selected, so continuing.", BC.END)
            writelisttofile(finalpdblines, "system_afteratlocfixes.pdb", separator="")
            return "system_afteratlocfixes.pdb"
        else:
            print(BC.WARNING,"You should delete either the labelled A or B location of the residue-atom/atoms and then remove the A/B label from column 17 in the file")
            print("Alternatively, you can choose use_higher_occupancy=True keyword in OpenMM_Modeller and ASH will keep the higher occupied form and go on ", BC.END)
            print("Make sure that there is always an A or B form present.")
            print(BC.FAIL,"Exiting.", BC.END)
            ashexit()
    #Returning original pdbfile if all OK        

    return pdbfile

#Function to get nonbonded model parameters for a metal cluster
#TODO: Add option to symmetrize charges for similar atoms in residue
def write_nonbonded_FF_for_ligand(fragment=None, xyzfile=None, charge=None, mult=None, coulomb14scale=1.0, lj14scale=1.0, 
    charmm=True, charge_model="xTB", theory=None, LJ_model="UFF", resname="LIG"):
    print_line_with_mainheader("OpenMM write_nonbonded_FF_for_ligand")

    if charmm == True:
        print("CHARMM option: True")
        print("Will create XML file so that the Nonbonded Interaction is compatible with CHARMM.\n")

    else:
        print("CHARMM option: False")
        print("Will create XML file in the regular way\n")

    #Coulomb and LJ scaling. Needs to be FF compatible. CHARMM values below

    #Creating ASH fragment
    if fragment != None:
        if fragment.charge == None or fragment.mult == None:
            print("No charge/mult information present in fragment")
            if charge == None or mult == None:
                print("No charge/mult info provided to function write_nonbonded_FF_for_ligand either.")
                print("Exiting")
                ashexit()
            else:
                fragment.charge=charge; fragment.mult=mult

        #Charge
    elif xyzfile != None:
        if os.path.exists(xyzfile) == False:
            print("XYZ-file does not exist. Exiting")
            ashexit()
        if charge == None or mult == None :
            print("XYZ-file option requires charge and mult definition. Exiting.")
            ashexit()
        fragment=Fragment(xyzfile=xyzfile, charge=charge,mult=mult)
    else:
        print("Neither fragment or xyzfile was provided to write_nonbonded_FF_for_ligand")
        ashexit()

    # Defining simple atomnames and atomtypes to be used for ligand
    atomnames = [el + "Y" + str(i) for i, el in enumerate(fragment.elems)]
    atomtypes = [el + "X" + str(i) for i, el in enumerate(fragment.elems)]

    if charge_model == "xTB":
        print("Using xTB charges")
        charges = basic_atomcharges_xTB(fragment=fragment, charge=fragment.charge, mult=fragment.mult, xtbmethod='GFN2')
    elif charge_model == "CM5_ORCA":
        print("CM5_ORCA option chosen")
        if theory == None: print("theory keyword required");ashexit()
        atompropdict = basic_atom_charges_ORCA(fragment=fragment, charge=fragment.charge, mult=fragment.mult,
                                               orcatheory=theory, chargemodel="CM5", numcores=theory.numcores)
        charges = atompropdict['charges']
    else:
        print("Unknown nonbonded_pars option")
        exit()

    if LJ_model == "UFF":
        # Basic UFF LJ parameters
        # Converting r0 parameters from Ang to nm and to sigma
        sigmas = [UFF_modH_dict[el][0] * 0.1 / (2 ** (1 / 6)) for el in fragment.elems]
        # Convering epsilon from kcal/mol to kJ/mol
        epsilons = [UFF_modH_dict[el][1] * 4.184 for el in fragment.elems]
    else:
        print("other LJ model not available")
        ashexit()

    # Creating XML-file for ligand
    xmlfile = write_xmlfile_nonbonded(resnames=[resname], atomnames_per_res=[atomnames], atomtypes_per_res=[atomtypes],
                                        elements_per_res=[fragment.elems], masses_per_res=[fragment.masses],
                                        charges_per_res=[charges],
                                        sigmas_per_res=[sigmas], epsilons_per_res=[epsilons], filename=resname+".xml",
                                        coulomb14scale=coulomb14scale, lj14scale=lj14scale, charmm=charmm)
    return xmlfile
