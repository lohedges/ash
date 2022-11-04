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
actatoms = qmatoms

# Create the ML/MM theory object.
mlmm = MLMMTheory(
    fragment=frag,
    qmatoms=qmatoms,
    in_vacuo_backend="torchani"
)

# Create the OpenMMTheory object.
openmm = OpenMMTheory(
    Amberfiles=True,
    amberprmtopfile=prmfile,
    periodic=True,
    do_energy_decomposition=True,
    autoconstraints=None,
    rigidwater=False
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

# Run geometry optimization using geomeTRIC optimizer and HDLC coordinates.
# Using active region.
geomeTRICOptimizer(
    theory=qmmmobject,
    fragment=frag,
    ActiveRegion=True,
    actatoms=actatoms,
    maxiter=500,
    coordsystem='hdlc',
    charge=frag.charge,
    mult=frag.mult
)
