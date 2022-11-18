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
    qmatoms=qmatoms,
    in_vacuo_backend="torchani",
    comparison_frequency=1000,
    numcores=numcores
)

# Create the OpenMMTheory object.
openmm = OpenMMTheory(
    Amberfiles=True,
    amberprmtopfile=prmfile,
    periodic=True,
    hydrogenmass=1,
    autoconstraints=None,
    rigidwater=True,
    platform="CUDA"
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
    timestep=0.001,
    simulation_steps=1000000,
    traj_frequency=100,
    temperature=300,
    integrator='LangevinMiddleIntegrator',
    datafilename="state.txt",
    coupling_frequency=1
)
