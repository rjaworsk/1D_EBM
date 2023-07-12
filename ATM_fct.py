#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:10:34 2023

@author: ricij
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy

from scipy import sparse
from scipy import special

import Functions



def calc_jacobian_atm(mesh, diffusion_coeff, heat_capacity, phi):
    jacobian = np.zeros((mesh.n_latitude, mesh.n_latitude))
    test_temperature = np.zeros(diffusion_coeff.size)

    index = 0
    for j in range(mesh.n_latitude):
            test_temperature[j] = 1.0
            diffusion_op = Functions.calc_diffusion_operator(mesh, diffusion_coeff, test_temperature, phi)
           
            op = (diffusion_op + (-mesh.B_dn-mesh.B_olr) * test_temperature) / heat_capacity
            
            # Convert matrix to vector
            jacobian[:, index] = op

            # Reset test_temperature
            test_temperature[j] = 0.0
            index += 1

    return jacobian

def timestep_euler_forward_atm(T_ATM, t, delta_t, mesh, diffusion_coeff, heat_capacity, T_S, phi):
   
    diffusion_op = Functions.calc_diffusion_operator(mesh, diffusion_coeff, T_ATM, phi)
    T_ATM_new = T_ATM + delta_t/heat_capacity * (diffusion_op + mesh.A_up + mesh.B_up * T_S - mesh.A_dn - mesh.B_dn * T_ATM - mesh.A_olr -mesh.B_olr * T_ATM)
    return T_ATM_new

    
def timestep_euler_backward_atm(jacobian, delta_t,  T_ATM, T_S, t,  mesh, heat_capacity):
    m, n = jacobian.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian = sparse.csc_matrix(jacobian)
    solve = sparse.linalg.factorized(eye - delta_t * jacobian)    

    source_terms = (mesh.A_up - mesh.A_dn - mesh.A_olr  + mesh.B_up * T_S) / heat_capacity 
   # print( delta_t *(mesh.A_up - mesh.A_dn - mesh.A_olr  + mesh.B_up * T_S) / heat_capacity )
    T_ATM_New = solve((T_ATM + delta_t * source_terms))
    return T_ATM_New
    
