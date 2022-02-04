#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of an inversion

@author: matteo
"""
import numpy as np
import os
import matplotlib.pyplot as plt 
from ase.io import write

from inv import Dataset

if __name__ == '__main__':

    N_config = 1
    N_workers = 1

    name = 'benzene'
        
    types = ['H', 'C']
    
    path = './datasets/'
    
    start = Dataset.fromFiles(path+name+'_start.xyz',
                              types=types)
    
    target = Dataset.fromFiles(path+'deformed.xyz',
                               types=types)
    
    # Test Bispectrum inversion
    
    inverted, loss = start.computeInversionDask(target[0:N_config], 
                                                     rcutfac=3,
                                                     rfac0=0.99,
                                                     twojmax=8,
                                                     N=5000,
                                                     gamma=4.0e-8,
                                                     eta=1e-2, 
                                                     nu=1e-3,
                                                     n_workers=N_workers
                                                     )
    
    try:
        os.system('mkdir ./out_data')
    except:
        pass  
    
    for n in range(len(inverted)):
        if n==0:            
            write('out_data/'+name+'.xyz', inverted[n], format='extxyz')
        else:            
            write('out_data/'+name+'.xyz', inverted[n], format='extxyz', append=True)

    np.save('out_data/'+name+'_loss.npy', np.asarray(loss))
    
