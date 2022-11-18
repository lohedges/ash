# Example showing geometric optimization of an alanine-dipeptide system
# using ML/MM energies and gradients.

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

# Create ORCA QM object.
orcainput = "! BLYP 6-31G* TightSCF NoFrozenCore KeepDens"
orcablocks = "%MaxCore 1024"
orca = ORCATheory(
    orcasimpleinput=orcainput,
    orcablocks=orcablocks,
    numcores=numcores,
    printlevel=0
)

# Bind the QM object to the ML/MM theory object so that we
# can compute pure QM/MM energies and gradients for comparison.
mlmm._orca = orca

# Create the OpenMMTheory object.
openmm = OpenMMTheory(
    Amberfiles=True,
    amberprmtopfile=prmfile,
    periodic=True,
    hydrogenmass=1,
    autoconstraints="HBonds",
    rigidwater=True
)

# Create QM/MM OBJECT by combining QM and MM objects above
qmmmobject = QMMMTheory(
    qm_theory=mlmm,
    mm_theory=openmm,
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
    simulation_steps=1000000,
    traj_frequency=1,
    temperature=300,
    pressure=1,
    integrator="LangevinMiddleIntegrator",
    barostat="MonteCarloBarostat",
    datafilename="state.txt",
    coupling_frequency=1
)
