#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 03:13:56 2021

@author: patilu
"""
import numpy as np
from ase.atoms import Atoms

def rotate2lammps(atoms):
    A,B,C = atoms.cell
    if np.dot(np.cross(A,B),C) > 0:

        ax = np.linalg.norm(A)
        Ahat = A/ax
        bx = np.dot(B,Ahat)
        by = np.linalg.norm(np.cross(Ahat,B))
        cx = np.dot(C,Ahat)
        cy = (np.dot(B,C) - bx*cx)/by
        cz = np.sqrt((np.linalg.norm(C)*np.linalg.norm(C)) - (cx*cx) - (cy*cy))
        
        cell = np.array([[ax,bx,cx],[0,by,cy],[0,0,cz]])

        atoms.wrap()
        pos = atoms.positions
        
        T = np.array([np.cross(B,C), np.cross(C,A), np.cross(A,B)])/atoms.get_volume()
        T = np.matmul(cell,T)
        
        newpos = np.array([np.matmul(T,p) for p in pos])
        
        return Atoms(positions = newpos,numbers = atoms.numbers,cell = cell.T, pbc = atoms.pbc.tolist())



def atomslattice2lammpsprism(atoms):
    a,b,c,alpha,beta,gamma = atoms.cell.cellpar()
    
    if (alpha == 90.0) and (beta == 90.0) and (gamma == 90.0):
        lx = a
        ly = b
        lz = c
        prism = False
    else:
        
        lx = a
        xy = b*np.cos(np.deg2rad(gamma))
        xz = c*np.cos(np.deg2rad(beta))
        ly = np.sqrt( b*b - xy*xy )
        yz = ( ( b*c*np.cos(np.deg2rad(alpha)) ) - ( xy*xz ) ) / ly
        lz = np.sqrt( (c*c) - (xz*xz) - (yz*yz) )
        prism = True
    
    
    xlo,ylo,zlo = 0,0,0
    xhi,yhi,zhi = lx,ly,lz
    if prism:
        region = np.array([xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz])
    else:
        region = np.array([xlo,xhi,ylo,yhi,zlo,zhi])
    
    return prism,region

    
def getPrismStr(atoms):

    prism,bounds = atomslattice2lammpsprism(atoms)
    if prism:
        return "prism " + " ".join([str(_a) for _a in bounds])
    else:
        return "block " + " ".join([str(_a) for _a in bounds])

def genatomcmdlist(atoms,typeMap,cmdstring1 = None,cmdstring2 = None):

    if cmdstring1 is None:
        cmdstring1 = "create_atoms {} single {} {} {} remap yes"
    if cmdstring2 is None:    
        cmdstring2 = "mass {} {}"
   
    cmdlist1 = [cmdstring1.format(typeMap[atom.number][0],atom.position[0],atom.position[1],atom.position[2]) for atom in atoms]
    cmdlist2 = [cmdstring2.format(d[0],d[1]) for d in typeMap.values()]
    #print(cmdlist1+cmdlist2)
    return cmdlist1 + cmdlist2

def atoms2lammpscmds(atoms,typeMap,create_atoms_cmdstring = None,mass_cmdstring = None):
    atoms.wrap()
    _atoms = rotate2lammps(atoms)
    _atoms.wrap()
    region = "region box " + getPrismStr(_atoms)
    create_box = "create_box {} box".format(len(typeMap))
    create_commands = genatomcmdlist(_atoms,typeMap,create_atoms_cmdstring,mass_cmdstring)
    
    return [region] + [create_box] + create_commands
