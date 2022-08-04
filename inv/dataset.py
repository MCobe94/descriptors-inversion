#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Datadict class

@author: matteo
"""
import numpy as np

from ase import Atoms, Atom
from ase.io import read

from inv.descriptors import Bispectrum, Inversion

from dask.distributed import Client
from dask import delayed, compute


class Dataset:
    """
    Dataset object.

    The Dataset object can represent a collection of chemical structures and
    allows for the calculation of atomic descriptors (bispectrum components)
    and their inversion.

    Parameters
    ----------
    data : dict of key-value pairs
        keywords

        all_types : list of int
            Atomic numbers of the species contained in the dataset.

        files_info : dict of key-value pairs
            Information regardin the files used to create the dataset.

        descriptors : ndarray
            3D array containing the descriptors of the local chemical
            environment (n_structures, n_atoms, n_b).

        descriptors_d : ndarray
            5D array containing the derivatives of the descriptors of the local
            chemical environment (n_structures, n_atoms, n_types, 3, n_b).

        index : list of int
            Unique index associated to each structure.

        structures : list of Atoms
            List of Atoms objects from as implemented in the ASE library
            containing the structural information of the chemical structures
            of the Dataset.

        types : list of ndarray
            Atomic numbers of all the atoms contained in each structure of
            the Dataset.
    """

    def __init__(self, data=None):

        if data == None:
            data = {
                "all_types": None,
                "files_info": {
                    "n_files": None,
                    "filenames": [],
                    "start_index": [],
                },
                "descriptors": [],
                "descriptors_d": [],
                "index": [],
                "structures": [],
                "types": [],
            }

        self.data = data

    @classmethod
    def fromFiles(cls, *args, types):
        """
        Creates a Dataset instance from input xyz files.

        Parameters
        ----------
        *args : str
            Files containing atomic structures.
        types : list of int
            List of atomic numbers of all the unique species
            present in the structures passed.

        Returns
        -------
        Dataset
            Dataset instance created from the xyz files given as argument.

        """
        data = {
            "all_types": None,
            "files_info": {
                "n_files": None,
                "filenames": [],
                "start_index": [],
            },
            "descriptors": [],
            "descriptors_d": [],
            "index": [],
            "structures": [],
            "types": [],
        }

        data["all_types"] = [Atom(_).number for _ in sorted(set(types))]
        data["files_info"]["n_files"] = len(args)

        for arg in args:

            data["files_info"]["filenames"].append(arg)
            data["files_info"]["start_index"].append(len(data["structures"]))

            data["structures"] += list(
                map(Atoms, read(arg, index=":", format="extxyz"))
            )

        for i in range(len(data["structures"])):
            data["types"].append(data["structures"][i].get_atomic_numbers())

            ### Setting default cell for non-periodic systems

            if np.any(data["structures"][i].pbc) == False:

                minc = np.amin(data["structures"][i].positions, axis=0)
                maxc = np.amax(data["structures"][i].positions, axis=0)
                v = maxc - minc + 100
                data["structures"][i].cell = np.diag(v)
                data["structures"][i].wrap(
                    pbc=True, center=True, pretty_translation=True
                )

        data["index"] = [i for i in range(len(data["structures"]))]

        return cls(data)

    def __len__(self):
        try:
            return len(self.data["structures"])
        except:
            try:
                return len(self.data["descriptors"])
            except:
                print("Error unable to determine the size of the dataset")
        return None

    def __getitem__(self, index):

        return Dataset(
            {
                "all_types": self.data["all_types"],
                "files_info": None,
                "descriptors": self.data["descriptors"][index]
                if len(self.data["descriptors"]) > 0
                else [],
                "descriptors_d": self.data["descriptors_d"][index]
                if len(self.data["descriptors_d"]) > 0
                else [],
                "index": self.data["index"][index]
                if len(self.data["index"]) > 0
                else [],
                "structures": self.data["structures"][index]
                if len(self.data["structures"]) > 0
                else [],
                "types": self.data["types"][index]
                if len(self.data["types"]) > 0
                else [],
            }
        )

    def computeBispectrum(self, rcutfac, rfac0, twojmax):
        """
        Compute Bispectrum components for each structure in the Dataset.

        Based on lammps implementation see https://docs.lammps.org/compute_sna_atom.html.

        Parameters
        ----------
        rcutfac : float
            Scale factor applied to all cutof radii which are set to 1.0 ang.
        rfac0 : float
            distance to angle conversion (0 < rcutfac < 1).
        twojmax : int
            Angular momentum limit for bispectrum components.

        Returns
        -------
        ndarray
            3D array bispectrum components (n_structures, n_atoms, n_b).
        ndarray
            5D array bispectrum components derivatives
            (n_structures, n_atoms, n_types, 3, n_b).

        """
        b_a = []
        b_d_a = []

        for i in range(len(self)):
            b, b_d = Bispectrum(
                self.data["structures"][i],
                [
                    Atom(_).number
                    for _ in sorted(
                        set([Atom(t).symbol for t in self.data["types"][i]])
                    )
                ],
                rcutfac,
                rfac0,
                twojmax,
            )

            b_a.append(b)
            b_d_a.append(b_d)

        self.data["descriptors"] = np.asarray(b_a)
        self.data["descriptors_d"] = np.asarray(b_d_a)

        return self.data["descriptors"], self.data["descriptors_d"]

    def computeBispectrumDask(self, rcutfac, rfac0, twojmax, n_workers=4):
        """
        Distributed version of computeBispectrum.

        Based on lammps implementation, https://docs.lammps.org/compute_sna_atom.html.
        Makes use of Dask, https://dask.org.

        Parameters
        ----------
        rcutfac : float
            Scale factor applied to all cutof radii which are set to 1.0 ang.
        rfac0 : float
            distance to angle conversion (0 < rcutfac < 1).
        twojmax : int
            Angular momentum limit for bispectrum components.
        n_workers: int, optional
            Number of workers for the Dask client. The default is 4.

        Returns
        -------
        ndarray
            3D array bispectrum components (n_structures, n_atoms, n_b).
        ndarray
            5D array bispectrum components derivatives
            (n_structures, n_atoms, n_types, 3, n_b).

        """

        client = Client(n_workers=n_workers)

        b_out = []

        for i in range(len(self)):
            b = delayed(Bispectrum)(
                self.data["structures"][i],
                [
                    Atom(_).number
                    for _ in sorted(
                        set([Atom(t).symbol for t in self.data["types"][i]])
                    )
                ],
                rcutfac,
                rfac0,
                twojmax,
            )

            b_out.append(b)

        b_out = compute(*b_out)
        client.close()

        self.data["descriptors"] = [_b[0] for _b in b_out]
        self.data["descriptors_d"] = [_b[1] for _b in b_out]

        return self.data["descriptors"], self.data["descriptors_d"]

    def computeInversion(
        self,
        targets,
        rcutfac,
        rfac0,
        twojmax,
        N=100,
        gamma=0.0000008,
        eta=5e-2,
        nu=1e-4,
    ):
        """
        Inverts the descriptors of each structure in the Dataset to resemble
        the descriptors of the target structures.

        Parameters
        ----------
        targets : Dataset
            Target structures.
        rcutfac : float
            Scale factor applied to all cutof radii which are set to 1.0 ang.
        rfac0 : float
            distance to angle conversion (0 < rcutfac < 1).
        twojmax : int
            Angular momentum limit for bispectrum components.
        N : int, optional
            Number of iterations of the invertion algorithm. The default is 100.
        gamma : float, optional
            Learning rate of the gradient descent. The default is 0.0000008.
        eta : float, optional
            Coupling constant of the loss with the noise term. The default is 5e-2.
        nu : float, optional
            Damping constant of the noise term. The default is 1e-4.

        Returns
        -------
        inverted_a : list of Atoms
            List of Atoms objects containing the configurations resulting
            from the inversion process.
        loss_a : ndarray
            2D array Loss value at each step of the inversion process for each
            structure in the Dataset (n_structures, N).

        """

        inverted_a = []
        loss_a = []

        for i in range(len(targets)):
            inverted, loss = Inversion(
                self[0], targets[i], rcutfac, rfac0, twojmax, N, gamma, eta, nu
            )

            inverted_a.append(inverted)
            loss_a.append(loss)

        return inverted_a, np.asarray(loss_a)

    def computeInversionDask(
        self,
        targets,
        rcutfac,
        rfac0,
        twojmax,
        N=100,
        gamma=0.0000008,
        eta=5e-2,
        nu=1e-4,
        n_workers=4,
    ):
        """
        Distributed version of computeIversion.

        Makes use of Dask, https://dask.org.

        Parameters
        ----------
        targets : Dataset
            Target structures.
        rcutfac : float
            Scale factor applied to all cutof radii which are set to 1.0 ang.
        rfac0 : float
            distance to angle conversion (0 < rcutfac < 1).
        twojmax : int
            Angular momentum limit for bispectrum components.
        N : int, optional
            Number of iterations of the invertion algorithm. The default is 100.
        gamma : float, optional
            Learning rate of the gradient descent. The default is 0.0000008.
        eta : float, optional
            Coupling constant of the loss with the noise term. The default is 5e-2.
        nu : float, optional
            Damping constant of the noise term. The default is 1e-4.
        n_workers: int, optional
            Number of workers for the Dask client. The default is 4.

        Returns
        -------
        inverted_a : list of Atoms
            List of Atoms objects containing the configurations resulting
            from the inversion process.
        loss_a : ndarray
            2D array Loss value at each step of the inversion process for each
            structure in the Dataset (n_structures, N).

        """
        client = Client(n_workers=n_workers)

        inversion_a = []

        for i in range(len(targets)):
            inversion = delayed(Inversion)(
                self[0], targets[i], rcutfac, rfac0, twojmax, N, gamma, eta, nu
            )

            inversion_a.append(inversion)

        inversion_a = compute(*inversion_a)
        client.close()

        inverted_a = [_inv[0] for _inv in inversion_a]
        loss_a = [_inv[1] for _inv in inversion_a]

        return inverted_a, loss_a
