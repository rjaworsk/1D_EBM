#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:11:54 2023

@author: ricij
"""


import numpy as np
from scipy import special

import Functions


def calc_coalbedo(phi, phi_i_n, phi_i_s, mesh): #Calculation of the coalbedo where phi_i_n and phi_i_s are the latitudes wich have ice and are closest to the equator
    a0 = 0.72
    ai = 0.36
    a2 = (a0 - ai)/((np.pi/2)**2)
    equator = int((mesh.n_latitude - 1) / 2)
    
    north_h =  0.5 * ((a0-a2*phi[equator:(len(phi))]**2 + ai) - (a0-a2*phi[equator:len(phi)]**2 - ai) * special.erf((phi[equator:len(phi)]-phi_i_n)/0.04))
    
    south_h = 0.5 * ((a0-a2*phi[0:equator]**2 + ai) - (a0-a2*phi[0:equator]**2 - ai) * special.erf((phi_i_s-phi[0:equator])/0.04))
    
    return np.concatenate(( south_h, north_h))


def calc_albedo_north(phi, phi_i_n): 
    a0 = 0.72
    ai = 0.36
    a2 = (a0 - ai)/((np.pi/2)**2)
   
    
    return    0.5 * ((a0-a2*phi**2 + ai) - (a0-a2*phi**2 - ai) * special.erf((phi-phi_i_n)/0.04))


def BasalFlux(phi):
    def f(phi):
        return -(1.3E16/(2*np.pi*6.37E6**2)) * np.cos(phi)**8 * (1-11*np.sin(phi)**2)
    def f_schlange(phi):
        return (1-3*np.cos(2*phi))/4
    F_bp = 2
    return f(phi) + F_bp * f_schlange(phi)



def calc_jacobian_ocn(mesh, diffusion_coeff, heat_capacity, phi):
    jacobian = np.zeros((mesh.n_latitude, mesh.n_latitude))
    test_temperature = np.zeros(diffusion_coeff.size)

    index = 0
    for j in range(mesh.n_latitude):
            test_temperature[j] = 1.0
           
            diffusion_op = Functions.calc_diffusion_operator(mesh, diffusion_coeff, test_temperature, phi)
            op = diffusion_op/heat_capacity

            jacobian[:, index] = op

            # Reset test_temperature
            test_temperature[j] = 0.0
            index += 1

    return jacobian


def timestep_euler_backward_ocn(solve, delta_t, T_OCN, T_S, T_ATM, t, mesh, heat_capacity, solar_forcing, F_b, H_I):
    
    source_terms = ((solar_forcing - mesh.A_up - mesh.B_up *T_S + mesh.A_dn + mesh.B_dn * T_ATM + F_b) / heat_capacity) * (H_I <=0)

    T_OCN_New = solve(T_OCN + delta_t * source_terms)
    return T_OCN_New 

   