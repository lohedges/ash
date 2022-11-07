######################################################################
# ML/MM: https://github.com/emedio/embedding
#
# Copyright: 2022
#
# Authors: Kirill Zinovjev <kzinovjev@gmail.com>
#          Lester Hedges   <lester.hedges@gmail.com>
#
# ML/MM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# ML/MM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ML/MM. If not, see <http://www.gnu.org/licenses/>.
######################################################################

import os
import numpy as np

import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy.special import erf as jerf

import scipy
import scipy.io

from ase import Atoms
import ase.io.xyz

from rascal.representations import (
    SphericalExpansion,
    SphericalInvariants
)
from rascal.utils import (
    ClebschGordanReal,
    compute_lambda_soap,
    spherical_expansion_reshape
)

from warnings import warn

from ash.functions.functions_general import (
    ashexit,
    BC
)
from  ash.interfaces.interface_ORCA import ORCATheory
from  ash.interfaces.interface_TorchANI import TorchANITheory
import ash.modules.module_coords

ANGSTROM_TO_BOHR = 1.88973
SPECIES = (1, 6, 7, 8, 16)
SIGMA = 1E-3

SPHERICAL_EXPANSION_HYPERS_COMMON = {
    "gaussian_sigma_constant": 0.5,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    'expansion_by_species_method': 'user defined',
    'global_species': SPECIES
}

Z_DICT = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 19}
SPECIES_DICT = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}

class GPRCalculator:
    '''Predicts an atomic property for a molecule with GPR'''

    def __init__(self, ref_values, ref_soap, n_ref, sigma):
        '''
        ref_values: (N_Z, N_REF)
        ref_soap: (N_Z, N_REF, N_SOAP)
        n_ref: (N_Z,)
        sigma: ()
        '''
        self.ref_soap = ref_soap
        Kinv = self.get_Kinv(ref_soap, sigma)
        self.n_ref = n_ref
        self.n_z = len(n_ref)
        self.ref_mean = np.sum(ref_values, axis=1) / n_ref
        ref_shifted = ref_values - self.ref_mean[:, None]
        self.c = (Kinv @ ref_shifted[:, :, None]).squeeze()

    def __call__(self, mol_soap, zid, gradient=False):
        '''
        mol_soap: (N_ATOMS, N_SOAP)
        zid: (N_ATOMS,)
        '''

        result = np.zeros(len(zid))
        for i in range(self.n_z):
            n_ref = self.n_ref[i]
            ref_soap_z = self.ref_soap[i, :n_ref]
            mol_soap_z = mol_soap[zid == i, :, None]
            K_mol_ref2 = (ref_soap_z @ mol_soap_z).squeeze() ** 2
            result[zid == i] = K_mol_ref2 @ self.c[i, :n_ref] + self.ref_mean[i]
        if not gradient:
            return result
        return result, self.get_gradient(mol_soap, zid)

    def get_gradient(self, mol_soap, zid):
        n_at, n_soap = mol_soap.shape
        df_dsoap = np.zeros((n_at, n_soap))
        for i in range(self.n_z):
            n_ref = self.n_ref[i]
            ref_soap_z = self.ref_soap[i, :n_ref]
            mol_soap_z = mol_soap[zid == i, :, None]
            K_mol_ref = (ref_soap_z @ mol_soap_z).squeeze()
            c = self.c[i, :n_ref]
            df_dsoap[zid == i] = (K_mol_ref[:,None,:] * ref_soap_z.T) @ c * 2
        return df_dsoap

    @classmethod
    def get_Kinv(cls, ref_soap, sigma):
        '''
        ref_soap: (N_Z, MAX_N_REF, N_SOAP)
        sigma: ()
        '''
        n = ref_soap.shape[1]
        K = (ref_soap @ ref_soap.swapaxes(1, 2)) ** 2
        return np.linalg.inv(K + sigma ** 2 * np.identity(n))

class SOAPCalculatorSpinv:
    '''Calculates SOAP feature vectors for a given system'''

    def __init__(self, hypers):
        self.spinv = SphericalInvariants(**hypers)

    def __call__(self, z, xyz, gradient=False):
        mol = self.get_mol(z, xyz)
        return self.get_soap(mol, self.spinv, gradient)

    @staticmethod
    def get_mol(z, xyz):
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_range = xyz_max - xyz_min
        return Atoms(z, positions=xyz - xyz_min, cell=xyz_range, pbc=0)

    @staticmethod
    def get_soap(atoms, spinv, gradient=False):
        managers = spinv.transform(atoms)
        soap = managers.get_features(spinv)
        if not gradient:
            return soap
        grad = managers.get_features_gradient(spinv)
        meta = managers.get_gradients_info()
        n_at, n_soap = soap.shape
        dsoap_dxyz = np.zeros((n_at, n_soap, n_at, 3))
        dsoap_dxyz[meta[:,1],:,meta[:,2],:] = grad.reshape((-1, 3, n_soap)).swapaxes(2,1)
        return soap, dsoap_dxyz

# ML/MM theory. Predicts ML/MM energies (and gradients) allowing QM/MM with
# ML/MM embedding. Requires the use of a QM engine to compute in vacuo energies
# and forces, to which those from the ML/MM model are added. For now we use
# ORCA as the QM backend, but this could be generalised to any supported engine.
class MLMMTheory:
    # Class attributes.

    # Get the directory of this module file.
    _module_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the name of the default model file.
    _default_model = os.path.join(_module_dir, "mlmm_adp.mat")

    # ML model parameters. For now we'll hard-code our own model parameters.
    # Could allow the user to specify their own model, but that would require
    # the use of consistent hyper-paramters, naming, etc.

    # Model hyper-parameters.
    _hypers = {
        "interaction_cutoff": 3.,
        "max_radial": 4,
        "max_angular": 4,
        "compute_gradients": True,
        **SPHERICAL_EXPANSION_HYPERS_COMMON,
    }

    # Supported backends for calculation of in-vacuo energies of the QM region.
    _supported_backends = ["orca", "torchani"]

    # Follow ASH style for constructor, where required arguments are keywords,
    # rather than positional arguments.
    def __init__(self, fragment=None, qmatoms=None, model=None, in_vacuo_backend="TorchANI",
                 comparison_frequency=0, printlevel=2, numcores=1):
        """Constructor.

           Parameters
           ----------

           fragment : ash.Fragment
               The ASH fragment object.

           qmatoms : [int]
               Indices of atoms in the QM region.

           model : str
               Path to the ML model parameter file. If None, then a default
               model will be used.

           in_vacuo_backend : str
               The backend used to compute in-vacuo energies of the QM region.
               Options are: "ORCA", "TorchANI"

           comparison_frequency : int
               The frequency at which to compare in-vacuo eneriges and gradients.
               We compute the delta energy and root mean squared difference in
               gradients between TorchANI and a reference QM engine (ORCA).

           printlevel : int
               Verbosity level.

           numcores : int
               The number of CPU cores to use for the QM backend.
        """

        # Validate input.

        if not isinstance(fragment, ash.modules.module_coords.Fragment):
            raise TypeError("'fragment' must be of type 'ash.modules.module_coords.Fragment'.")
        self._fragment = fragment

        if not isinstance(qmatoms, (list, tuple, np.ndarray)):
            raise TypeError("'qmatoms' must be of type 'list', 'tuple', or 'numpy.ndarray'.")
        if not all(isinstance(x, int) for x in qmatoms):
            raise TypeError("'qmatoms' can only contain items of type 'int'.")
        num_atoms = self._fragment.numatoms
        for qmatom in qmatoms:
            if qmatom < 0 or qmatom >= num_atoms:
                raise ValueError(f"'qmatoms' index {qmatom} outside range of "
                                 f"fragment with {num_atoms} atoms.")
        self._qmatoms = qmatoms

        if model is not None:
            if not isinstance(model, str):
                raise TypeError("'model' must be of type 'str'")
            if not os.path.exists(model):
                raise ValueError(f"Unable to locate model file: '{model}'")
            self._model = model
        else:
            self._model = self._default_model

        # Load the model parameters.
        try:
            self._params = scipy.io.loadmat(self._model, squeeze_me=True)
        except:
            raise ValueError(f"Unable to load model parameters from: '{self._model}'")

        if not isinstance(in_vacuo_backend, str):

            raise TypeError("'in_vacuo_backend' must be of type 'str'.")
        # Strip whitespace and convert to lower case.
        in_vacuo_backend = in_vacuo_backend.replace(" ", "").lower()
        if not in_vacuo_backend in self._supported_backends:
            raise ValueError(f"'in_vacuo_backend' ({in_vacuo_backend}) not supported. "
                             f"Valid options are {', '.join(self._supported_backends)}")
        self._backend = in_vacuo_backend

        if type(comparison_frequency) is not int:
            raise TypeError("'comparison_frequency' must be of type 'int'.")
        if comparison_frequency < 0:
            raise ValueError("'comparison_frequency' must be >= 0.")
        self._comparison_frequency = comparison_frequency

        # Only make comparison when the backend is TorchANI.
        if self._backend != "torchani" and self._comparison_frequency > 0:
            self._comparison_frequency = 0
            warn("'comparison_frequency' can only be used with 'TorchANI' backend.")

        if type(numcores) is not int:
            raise TypeError("'numcores' must be of type 'int'.")
        if numcores < 1:
            raise ValueError("'numcores' must be >= 1.")

        if type(printlevel) is not int:
            raise TypeError("'printlevel' must be of type 'int'.")
        self._printlevel = printlevel

        # Try initialising an ORCATheory object for the in-vacuo backend.
        if self._backend == "orca":
            try:
                # ORCA input.
                orcainput = "! BLYP 6-31G* TightSCF NoFrozenCore KeepDens"
                orcablocks = "%MaxCore 1024"

                # Create the ORCA theory object.
                self._backend_theory = ORCATheory(
                        orcasimpleinput=orcainput,
                        orcablocks=orcablocks,
                        numcores=numcores,
                        printlevel=0
                )
            except:
                raise Exception("Unable to create ORCATheory object for QM backend!")

        # Use TorchANI to predict in-vacuo energies for the QM region.
        else:
            # Create the ML/MM theory object.
            self._backend_theory = TorchANITheory(
                fragment=self._fragment,
                qmatoms=self._qmatoms,
                printlevel=self._printlevel
            )

            # Create an ORCA theory object to allow comparioons of in-vacuo
            # energies and gradients.
            if self._comparison_frequency > 0:
                # ORCA input.
                orcainput = "! BLYP 6-31G* TightSCF NoFrozenCore KeepDens"
                orcablocks = "%MaxCore 1024"

                # Create the ORCA theory object.
                self._reference_theory = ORCATheory(
                        orcasimpleinput=orcainput,
                        orcablocks=orcablocks,
                        numcores=numcores,
                        printlevel=0
                )

        # Initialise ML-model attributes.

        self._get_soap = SOAPCalculatorSpinv(self._hypers)
        self._q_core = self._params["q_core"]
        self._a_QEq = self._params["a_QEq"]
        self._a_Thole = self._params["a_Thole"]
        self._k_Z = self._params["k_Z"]
        self._get_s = GPRCalculator(
            self._params["s_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3
        )
        self._get_chi = GPRCalculator(
            self._params["chi_ref"],
            self._params["ref_soap"],
            self._params["n_ref"],
            1e-3
        )
        self._get_E_with_grad = value_and_grad(self._get_E, argnums=(0, 2, 3, 4))

        # Work out element and species indicies for QM atom elements.

        # Initialise lists.
        self._z = []            # Element index.
        self._zid = []          # Species index.

        # Loop over all of the QM atoms.
        for idx in self._qmatoms:
            # Get the element from the fragment.
            elem = self._fragment.elems[idx]

            try:
                self._z.append(Z_DICT[elem])
                self._zid.append(SPECIES_DICT[self._z[-1]])
            except:
                raise ValueError(f"Unsupported element '{elem}'. "
                                 f"We currently support {', '.join(Z_DICT.keys())}.")

        # Convert to NumPy arrays.
        self._z = np.array(self._z)
        self._zid = np.array(self._zid)

    # Match run function of other interface objects.
    def run(self, current_coords=None, charge=None, mult=None,
            current_MM_coords=None, MMcharges=None, qm_elems=None,
            Grad=False, numcores=None, label=None, step=None):
        """Calculate the energy and (optionally) gradients.

           Parameters
           ----------

           current_coords : numpy.ndarray
               The current QM coordinates.

           charge : int
               Charge of the QM region.

           mult : int
               Spin multiplicity of the QM region.

           current_MM_coords : numpy.ndarray
               The MM point-charge coordinates.

           MMcharges : [ float ]
               The MM point-charge charges.

           qm_elems : [ str ]
               A list of elements for the QM region.

           Grad : bool
               Whether to compute gradients.

           numcores : int
               The number of cores to use for the QM backend.

           label : str
               Job identification string.

           step : int
               The current integration step. (If running ML/MM MD.)
        """

        if self._printlevel >= 2:
            print(BC.OKBLUE,BC.BOLD, "------------RUNNING ML-MM INTERFACE-------------", BC.END)

        # Validate the input. Annoyingly, the user could call the run method
        # using input that is inconsistent with that used to instatiate the
        # MLMM object, i.e. coordinates corresponding to a different fragment,
        # or QM region. This is a general issue with ASH, though, and there
        # is no type checking or data validation elsewhere in the code. Here
        # we assume that the data is consistent and simply type check the input.

        if not isinstance(current_coords, np.ndarray):
            raise TypeError("'current_coords' must be of type 'numpy.ndarray'.")
        if current_coords.dtype != np.float64:
            raise TypeError("'current_coords' must have dtype 'float64'.")

        if not isinstance(current_MM_coords, np.ndarray):
            raise TypeError("'current_MM_coords' must be of type 'numpy.ndarray'.")
        if current_MM_coords.dtype != np.float64:
            raise TypeError("'current_MM_coords' must have dtype 'float64'.")

        if not isinstance(MMcharges, list):
            raise TypeError("'MMcharges' must be of type 'list'.")
        if not all(isinstance(x, float) for x in MMcharges):
            raise TypeError("'MMcharges' must be a list of 'float' types.")

        if not isinstance(qm_elems, list):
            raise TypeError("'qm_elems' must be of type 'list'.")
        if not all(isinstance(x, str) for x in qm_elems):
            raise TypeError("'qm_elems' must be a list of 'str' types.")

        if not isinstance(Grad, bool):
            raise TypeError("'Grad' must be of type 'bool'.")

        if label:
            if not isinstance(label, str):
                raise TypeError("'label' must be of type 'str'.")

        if step:
            if type(step) is not int:
                raise TypeError("'step' must be of type 'int'.")

        try:
            charge = int(charge)
        except:
            # Try using the fragment charge instead.
            try:
                charge = int(self._fragment.charge)
            except:
                raise TypeError("'charge' must be of type 'int'.")

        try:
            mult = int(mult)
        except:
            # Try using the fragment spin multiplicity instead.
            try:
                mult = int(mult)
            except:
                raise TypeError("'mult' must be of type 'int'")

        # First try to use the qm_theory backend to compute in vacuo
        # energies and (optionally) gradients.

        try:
            if Grad:
                if self._printlevel >= 2:
                    print(f"Calculating in-vacuo energies and gradients using {self._backend.upper()}.")

                E_vac, grad_vac = self._backend_theory.run(
                        current_coords=current_coords,
                        charge=charge,
                        mult=mult,
                        qm_elems=qm_elems,
                        Grad=True,
                        numcores=numcores,
                        label="MLMM in vacuo QM backend."
                )
            else:
                if self._printlevel >= 2:
                    print(f"Calculating in-vacuo energies using {self._backend.upper()}.")

                E_vac = self._backend_theory.run(
                        current_coords=xyz,
                        charge=self._qm_charge,
                        mult=self._qm_mult,
                        qm_elems=qm_elems,
                        Grad=False,
                        numcores=numcores,
                        label="MLMM in vacuo QM backend."
                )
        except:
            raise RuntimeError("Failed to calculate in vacuo energies using backend!")

        if self._printlevel >= 2:
            if Grad:
                print("Predicting MM energies and gradients.")
            else:
                print("Predicting MM energies.")

        # Convert coordinate units.
        xyz_bohr = current_coords * ANGSTROM_TO_BOHR
        pc_xyz_bohr = current_MM_coords * ANGSTROM_TO_BOHR

        # Convert point-charge list to an NumPy array.
        MMcharges = np.array(MMcharges)

        if Grad:
            mol_soap, dsoap_dxyz = self._get_soap(self._z, current_coords, True)
            dsoap_dxyz_bohr = dsoap_dxyz / ANGSTROM_TO_BOHR
        else:
            mol_soap = self._get_soap(self._z, current_coords)

        if Grad:
            s, ds_dsoap = self._get_s(mol_soap, self._zid, True)
            chi, dchi_dsoap = self._get_chi(mol_soap, self._zid, True)
            ds_dxyz_bohr = self._get_df_dxyz(ds_dsoap, dsoap_dxyz_bohr)
            dchi_dxyz_bohr = self._get_df_dxyz(dchi_dsoap, dsoap_dxyz_bohr)
        else:
            s = self._get_s(mol_soap, self._zid)
            chi = self._get_chi(mol_soap, self._zid)

        if not Grad:
            E = self._get(xyz_bohr, self._zid, s, chi, pc_xyz_bohr, MMcharges)
            return E + E_vac

        E, grads = self._get_E_with_grad(xyz_bohr, self._zid, s, chi, pc_xyz_bohr, MMcharges)
        dE_dxyz_bohr_part, dE_ds, dE_dchi, dE_dpc_xyz_bohr = grads
        dE_dxyz_bohr = (dE_dxyz_bohr_part +
                        dE_ds @ ds_dxyz_bohr.swapaxes(0,1) +
                        dE_dchi @ dchi_dxyz_bohr.swapaxes(0,1))
        dE_dxyz = dE_dxyz_bohr * ANGSTROM_TO_BOHR
        dE_dpc_xyz = dE_dpc_xyz_bohr * ANGSTROM_TO_BOHR

        # Compare energies and gradients to those obtained by QM/MM with ORCA.
        if step % self._comparison_frequency == 0:
            # Clear the file.
            if step == 0:
                with open("ml_vs_qm.txt", "w") as f:
                    pass
            else:
                if self._printlevel >= 2:
                    print("Comparing ML energies and gradients to QM/MM.")

                E_qm_vac, grad_qm_vac = self._reference_theory.run(
                        current_coords=current_coords,
                        charge=charge,
                        mult=mult,
                        qm_elems=qm_elems,
                        Grad=True,
                        numcores=numcores,
                        label="ORCA in-vacuo reference QM theory."
                )

                # Compute the difference between the ML/MM and QM energies.
                delta_E = E_vac - E_qm_vac

                # Work out the RMSD of the gradients, both QM and PC.
                rmsd_grad = np.sqrt(np.mean(grad_qm_vac - grad_vac)**2)

                # Write to file.
                with open("ml_vs_qm.txt", "a") as f:
                    f.write(f"{step} {delta_E} {rmsd_grad}\n")

        return (E + E_vac, np.array(dE_dxyz) + grad_vac, np.array(dE_dpc_xyz))

    def _get_E(self, xyz_bohr, zid, s, chi, pc_xyz_bohr, MMcharges):
        return jnp.sum(self._get_E_components(xyz_bohr, zid, s, chi, pc_xyz_bohr, MMcharges))

    def _get_E_components(self, xyz_bohr, zid, s, chi, pc_xyz_bohr, MMcharges):
        q_core = self._q_core[zid]
        k_Z = self._k_Z[zid]
        r_data = self._get_r_data(xyz_bohr)
        mesh_data = self._get_mesh_data(xyz_bohr, pc_xyz_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, MMcharges, s, q_val, k_Z)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data['T0_mesh'])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data['T0_mesh_slater'])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = jnp.sum(vpot_static @ MMcharges)

        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data['T1_mesh'])
        E_ind = jnp.sum(vpot_ind @ MMcharges) * 0.5
        return jnp.array([E_static, E_ind])

    def _get_q(self, r_data, s, chi):
        A = self._get_A_QEq(r_data, s)
        b = jnp.hstack([-chi,  0])
        return jnp.linalg.solve(A, b)[:-1]

    def _get_A_QEq(self, r_data, s):
        s_gauss = s * self._a_QEq
        s2 = s_gauss ** 2
        s_mat = jnp.sqrt(s2[:, None] + s2[None, :])

        A = self._get_T0_gaussian(r_data['T01'], r_data['r_mat'], s_mat)
        A = A.at[jnp.diag_indices_from(A)].set(1. / (s_gauss * jnp.sqrt(jnp.pi)))

        ones = jnp.ones((len(A), 1))
        return jnp.block([[A, ones], [ones.T, 0.]])

    def _get_mu_ind(self, r_data, mesh_data, q, s, q_val, k_Z):
        A = self._get_A_thole(r_data, s, q_val, k_Z)
        fields = jnp.sum(mesh_data['T1_mesh'] * q[:, None],
                        axis=1).flatten()
        mu_ind = jnp.linalg.solve(A, fields)
        E_ind = mu_ind @ fields * 0.5
        return mu_ind.reshape((-1, 3))

    def _get_A_thole(self, r_data, s, q_val, k_Z):
        N = - q_val
        v = 60 * N * s ** 3
        alpha = jnp.array(v * k_Z)

        alphap = alpha * self._a_Thole
        alphap_mat = alphap[:, None] * alphap[None, :]

        au3 = r_data['r_mat'] ** 3 / jnp.sqrt(alphap_mat)
        au31 = au3.repeat(3, axis=1)
        au32 = au31.repeat(3, axis=0)
        A = - self._get_T2_thole(r_data['T21'], r_data['T22'], au32)
        A = A.at[jnp.diag_indices_from(A)].set(1. / alpha.repeat(3))
        return A

    @staticmethod
    def _get_df_dxyz(df_dsoap, dsoap_dxyz):
        return jnp.einsum('ij,ijkl->ikl', df_dsoap, dsoap_dxyz)

    @staticmethod
    def _get_vpot_q(q, T0):
        return jnp.sum(T0 * q[:, None], axis=0)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        return - jnp.tensordot(T1, mu, ((0, 2), (0, 1)))

    @classmethod
    def _get_r_data(cls, xyz):
        n_atoms = len(xyz)
        t01 = jnp.zeros((n_atoms, n_atoms))
        t11 = jnp.zeros((n_atoms, n_atoms * 3))
        t21 = jnp.zeros((n_atoms * 3, n_atoms * 3))
        t22 = jnp.zeros((n_atoms * 3, n_atoms * 3))

        rr_mat = xyz[:, None, :] - xyz[None, :, :]

        r2_mat = jnp.sum(rr_mat**2, axis=2)
        r_mat = jnp.sqrt(jnp.where(r2_mat > 0., r2_mat, 1.))
        r_mat = r_mat.at[jnp.diag_indices_from(r_mat)].set(0.)

        tmp = jnp.where(r_mat == 0.0, 1.0, r_mat)
        r_inv = jnp.where(r_mat == 0.0, 0.0, 1. / tmp)

        r_inv1 = r_inv.repeat(3, axis=1)
        r_inv2 = r_inv1.repeat(3, axis=0)
        outer = cls._get_outer(rr_mat)
        id2 = jnp.tile(jnp.tile(jnp.eye(3).T, n_atoms).T, n_atoms)

        t01 = r_inv
        t11 = -rr_mat.reshape(n_atoms, n_atoms * 3) * r_inv1 ** 3
        t21 = -id2 * r_inv2 ** 3
        t22 = 3 * outer  * r_inv2 ** 5

        return {'r_mat': r_mat, 'T01': t01, 'T11': t11, 'T21': t21, 'T22': t22}

    @staticmethod
    def _get_outer(a):
        n = len(a)
        idx = jnp.triu_indices(n, 1)

        result = jnp.zeros((n, n, 3, 3))
        result = result.at[idx].set(a[idx][:, :, None] @ a[idx][:, None, :])
        result = result.swapaxes(0,1).at[idx].set(result[idx])

        return result.swapaxes(1, 2).reshape((n * 3, n * 3))

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        rr = xyz_mesh[None, :, :] - xyz[:, None, :]
        r = jnp.linalg.norm(rr, axis=2)

        return {'T0_mesh': 1. / r,
                'T0_mesh_slater': cls._get_T0_slater(r, s[:, None]),
                'T1_mesh': - rr / r[:, :, None] ** 3}

    @staticmethod
    def _get_T0_slater(r, s):
        return (1 - (1 + r / (s * 2)) * jnp.exp(-r / s)) / r

    @staticmethod
    def _get_T0_gaussian(t01, r, s_mat):
        return t01 * jerf(r / (s_mat * jnp.sqrt(2)))

    @staticmethod
    def _get_T1_gaussian(t11, r, s_mat):
        s_invsq2 = 1. / (s_mat * jnp.sqrt(2))
        return t11 * (
            jerf(r * s_invsq2) -
            r * s_invsq2 * 2 / jnp.sqrt(jnp.pi) * jnp.exp(-r * s_invsq2) ** 2
        ).repeat(3, axis=1)

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        return cls._lambda3(au3) * tr21 + cls._lambda5(au3) * tr22

    @staticmethod
    def _lambda3(au3):
        return 1 - jnp.exp(-au3)

    @staticmethod
    def _lambda5(au3):
        return 1 - (1 + au3) * jnp.exp(-au3)
