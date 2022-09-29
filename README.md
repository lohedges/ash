<img src="ash-simple-logo-letterbig.png" alt="drawing" width="300" align="right"/>

# ASH: a computational chemistry environment
ASH is a Python-based computational chemistry and QM/MM environment, primarily for molecular calculations in the gas phase, explicit solution, crystal or protein environment. Can do single-point calculations, geometry optimizations, surface scans, molecular dynamics, numerical frequencies etc. using a MM, QM or QM/MM Hamiltonian.
Interfaces to popular QM codes: ORCA, xTB, Psi4, PySCF, Dalton, CFour, MRCC.

While ASH is ready to be used in computational chemistry research, it is a young project and there will probably be some issues and bugs to be discovered if you start using it.

**In case of problems:**
Please open an issue on Github and I will try to fix any problems as soon as possible.


**Documentation:**

 https://ash.readthedocs.io/en/latest


**Development:**

ASH welcomes any contributions.

Ongoing priorities:
- Improve documentation of code, write docstrings.
- Write unit tests
- Rewrite silly old code.
- Reduce code redundancy.
- Improve program documentation


**Example:**

```sh
from ash import *

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment from multi-line string
HF_frag=Fragment(coordsstring=coords, charge=0, mult=1)

#Alternative: Create fragment from XYZ-file
HF_frag=Fragment(xyzfile="hf.xyz", charge=0, mult=1)

#Define ORCA theory settings strings
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
#Define ORCA theory object
ORCAcalc = ORCATheory(orcasimpleinput=input, orcablocks=blocks)

#Call optimizer with ORCAtheory object and fragment as input
geomeTRICOptimizer(theory=ORCAcalc,fragment=HF_frag)
```

## ML/MM functionality

To use ML/MM via the [interface_MLMM](interfaces/interface_MLMM.py) module, you'll need to install some
additional pacakges into your ASH Conda environment. A quick-and-dirty
environment setup can be performed as follows.

(The following assumes that you are in the ASH root directory. We advise
using a Python 3.9 MiniConda, available [here](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh).)

```sh
conda create -n ash -c conda-forge -c psi4 -c pyscf geometric openmm julia xtb pdbfixer plumed parmed mdanalysis ase scipy matplotlib psi4 pyscf sympy
conda activate ash
./conda_setup_ash.sh
```

(Make sure that the `PATH` and `LD_LIBRARY_PATH` environment variables within
the `set_environment_ash.sh` script are updated to reflect your local
[ORCA](https://www.orcasoftware.de/tutorials_orca/)
installation.)

Now install the additional dependencies required to enable ML/MM functionality:

* `jax`: (CPU only is fine for demonstration purposes.)

```sh
pip install --upgrade "jax[cpu]"
```

* [librascal](https://github.com/lab-cosmo/librascal)

This requires access to C++ compiler and the [Eigen](https://www.google.com/search?client=firefox-b-d&q=eigen)
template library. We recommend installing these into
your ASH environment to ensure compatibility.

```sh
conda install -c conda-forge compilers eigen
git clone https://github.com/lab-cosmo/librascal.git
cd librascal
pip install .
```
