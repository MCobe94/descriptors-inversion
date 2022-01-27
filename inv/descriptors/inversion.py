#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invert bispectrum components

@author: matteo
"""

import numpy as np
from ase import Atom
from inv.descriptors import Bispectrum

def Inversion(start, target, rcutfac, rfac0, twojmax, N, gamma, eta, nu):  
    '''
    Updates the starting configuration in order to reproduce the descriptors of the target structure.

    Parameters
    ----------
    start : Atoms
        Starting configuration        
    target : Atoms
        Target structure only its bispectrum components are considered in the 
        inversion process.
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
    Atoms
        Atoms objects containing the configuration resulting 
        from the inversion process.
    loss : list of float
        Loss value at each step of the inversion process 

    '''
  
    # Compute target descriptors
      
    target_b, target_bd = Bispectrum(target.data['structures'],
                                     [Atom(_).number for _ in sorted(set([Atom(t).symbol for t in target.data['types']]))],  
                                     rcutfac,
                                     rfac0,
                                     twojmax
                                     )
    
    ###
   
    n_atoms = len(start.data['structures'])
    n_b = int((twojmax/2+1)*(twojmax/2+1.5)*(twojmax/2+2)/3)  # Wood, Thopson J., Chem. Phys. 148, 241721 (2018) tab.1   
   
    sum_bt = [np.zeros(n_b) for _ in range(len(target.data['all_types']))]    
    for i in range(n_atoms):
        sum_bt[target.data['all_types'].index(target.data['types'][i])] += target_b[i]
    
    loss = []
    grad_loss = []   
    
    for step in range(N): 
        
        prototype_b, prototype_bd = Bispectrum(start.data['structures'],
                                               [Atom(_).number for _ in sorted(set([Atom(t).symbol for t in start.data['types']]))], 
                                               rcutfac,
                                               rfac0,
                                               twojmax
                                               )        
           
        loss.append(0.0)
        grad_loss = [[0.0, 0.0, 0.0] for _ in range(n_atoms)]
    
        sum_b = [np.zeros(n_b) for _ in range(len(start.data['all_types']))]
        
        for i in range(n_atoms):            
            sum_b[start.data['all_types'].index(start.data['types'][i])] += prototype_b[i]
        
        for t in range(len(start.data['all_types'])):
            loss[step] += np.dot(sum_bt[t]-sum_b[t], sum_bt[t]-sum_b[t])
            
        for i in range(n_atoms):   
            for j in range(len(start.data['all_types'])):
                for k in range(3):        
                    grad_loss[i][k] += -2*np.dot(sum_bt[j]-sum_b[j], prototype_bd[i][j][k])
            
            start.data['structures'].positions[i] = np.array(start.data['structures'].positions[i]) + gamma*np.array(grad_loss[i]) + eta*np.exp(-nu*step)*np.random.uniform()
                
    return start.data['structures'], loss   