# Example showing ML/MM molecular dynamics for an alanine-dipeptide system.

from ash import *

numcores = 4

# Define working directory and input files.
input_dir = "./input"
prmfile = f"{input_dir}/adp.parm7"
crdfile = f"{input_dir}/adp.rst7"

# Create a fragment.
frag = Fragment(
    amber_prmtopfile=prmfile,
    amber_inpcrdfile=crdfile,
    charge=0,
    mult=1
)

# Define QM region and active atoms.
qmatoms = [x for x in range(0, 22)]

# Create the ML/MM theory object.
mlmm = MLMMTheory(
    fragment=frag,
    qmatoms=qmatoms
)

# Create the OpenMMTheory object.
openmm = OpenMMTheory(
    Amberfiles=True,
    amberprmtopfile=prmfile,
    periodic=True,
    autoconstraints="HBonds",
    rigidwater=True
)

# Create QM/MM OBJECT by combining QM and MM objects above
qmmmobject = QMMMTheory(
    qm_theory=mlmm,
    mm_theory=openmm,
    printlevel=2,
    fragment=frag,
    embedding="Elstat",
    qmatoms=qmatoms,
    numcores=numcores
)

# Perform molecular dynamics using OpenMM.
OpenMM_MD(
    fragment=frag,
    theory=qmmmobject,
    timestep=0.002,
    simulation_time=1000000,
    traj_frequency=1000,
    temperature=300,
    integrator='LangevinMiddleIntegrator',
    datafilename="state.txt",
    coupling_frequency=1
)
