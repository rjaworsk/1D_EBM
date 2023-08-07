#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:10:34 2023

@author: ricij
"""


import numpy as np

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


    
def timestep_euler_backward_atm(solve, delta_t,  T_ATM, T_S, t,  mesh, heat_capacity, F_w): 

    source_terms = (mesh.A_up - mesh.A_dn - mesh.A_olr  + mesh.B_up * T_S + F_w) / heat_capacity 
    T_ATM_New = solve((T_ATM + delta_t * source_terms))
    return T_ATM_New
    
