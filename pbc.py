#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix to enforce the wrapping inside the unit cell.

@author: matteo
"""

from ase.io import write, read

molecule = read('out_data/benzene_no_noise.xyz', index=':', format='extxyz')[0]
                                           
molecule.wrap(pbc=True, center=True, pretty_translation=True)
write('./out_data/benzene_no_noise.xyz', molecule)                                           
        
             

     
  
