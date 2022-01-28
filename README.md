# Inversion of the chemical environment representations
Implementation of the algorithm for the inversion of the local chemical environments descriptors.

https://arxiv.org/abs/2201.11591

# Dependency

This library makes use of ASE for the manipulation of the chemical structures, LAMMPS for the computation of the bispectrum components and of Dask to run in parallel:

https://www.lammps.org/

https://dask.org/

https://wiki.fysik.dtu.dk/ase/

## Examples

- example.py shows an example of the inversion procedure, it makes use of the configurations in the Dataset directory.
- pbc.py small script to wrap the atoms into the unit cell for visualization purposes.
