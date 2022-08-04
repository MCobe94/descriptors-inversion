#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is dedicated to the computation of the bisectrum 
components (it makes use of LAMMPS)

@author: matteo
"""

from lammps import lammps
from inv.utils.lammps import atoms2lammpscmds
import numpy as np
from lammps import LMP_STYLE_ATOM, LMP_TYPE_ARRAY
from ase.data import atomic_masses


def Bispectrum(structure, types, rcutfac, rfac0, twojmax):
    """
    Compute Bispectrum components.

    Based on lammps implementation see https://docs.lammps.org/compute_sna_atom.html.

    Parameters
    ----------
    structure : Atoms
        Input structure.
    types : list of int
        Atomic numbers of the species contained in the structure
    rcutfac : float
        Scale factor applied to all cutof radii which are set to 1.0 ang.
    rfac0 : float
        distance to angle conversion (0 < rcutfac < 1).
    twojmax : int
        Angular momentum limit for bispectrum components.

    Returns
    -------
    b : ndarray
        2D array bispectrum components (n_atoms, n_b).
    b_d_out : ndarray
        4D array bispectrum components derivatives
        (n_atoms, n_types, 3, n_b).

    """

    header = [
        "units metal",
        "dimension 3",
        "boundary p p p",
        "atom_style atomic",
        "box tilt large",
        "atom_modify map array sort 0 10.0",
    ]

    typeMap = {}
    for i, n in enumerate(types):
        typeMap[n] = [i + 1, atomic_masses[n]]

    structure_cmd = atoms2lammpscmds(structure, typeMap)

    compute_settings = str(rcutfac) + " "
    compute_settings += str(rfac0) + " "
    compute_settings += str(twojmax) + " "
    compute_settings += " ".join([str(1.0) for _ in range(len(set(types)))]) + " "
    compute_settings += " ".join([str(1.0) for _ in range(len(set(types)))])
    compute_settings += " rmin0 0.0"

    b_cmd = "compute b all sna/atom " + compute_settings
    bd_cmd = "compute bd all snad/atom " + compute_settings

    n_b = int(
        (twojmax / 2 + 1) * (twojmax / 2 + 1.5) * (twojmax / 2 + 2) / 3
    )  # Wood, Thopson J., Chem. Phys. 148, 241721 (2018) tab.1

    body = (
        ["pair_style lj/cut 20", "pair_coeff * * 1 1"] + [b_cmd] + [bd_cmd] + ["run 0"]
    )

    cmd_list = header + structure_cmd + body

    lmp = lammps(cmdargs=["-l", "none", "-screen", "none"])
    lmp.commands_list(cmd_list)

    b = lmp.numpy.extract_compute("b", LMP_STYLE_ATOM, LMP_TYPE_ARRAY).copy()
    bd = lmp.numpy.extract_compute("bd", LMP_STYLE_ATOM, LMP_TYPE_ARRAY).copy()

    lmp.close()

    bd_out = np.zeros((len(bd), len(types), 3, n_b))

    for i in range(len(bd)):
        c = 0

        for j in range(len(types)):
            for k in range(3):
                for p in range(n_b):
                    bd_out[i][j][k][p] = bd[i][c]
                    c += 1

    return b, bd_out
