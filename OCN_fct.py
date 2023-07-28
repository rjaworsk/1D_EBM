#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:11:54 2023

@author: ricij
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy

from scipy import sparse
from scipy import special

import Functions


def calc_albedo(phi, phi_i_n, phi_i_s, mesh): #Berechung des Coalbedo, wobei phi der Breitengrad und phi_i der Breitengrad der südlichsten Eisschicht ist
    a0 = 0.72
    ai = 0.36
    a2 = (a0 - ai)/((np.pi/2)**2)
    equator = int((mesh.n_latitude - 1) / 2)
    
    north_h =  0.5 * ((a0-a2*phi[equator:(len(phi))]**2 + ai) - (a0-a2*phi[equator:len(phi)]**2 - ai) * special.erf((phi[equator:len(phi)]-phi_i_n)/0.04))
    
    south_h = 0.5 * ((a0-a2*phi[0:equator]**2 + ai) - (a0-a2*phi[0:equator]**2 - ai) * special.erf((phi_i_s-phi[0:equator])/0.04))
    
    return np.concatenate(( south_h, north_h))


def calc_albedo_n(phi, phi_i_n): #Berechung des Coalbedo, wobei phi der Breitengrad und phi_i der Breitengrad der südlichsten Eisschicht ist
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

def timestep_euler_forward_ocn(T_OCN, t, delta_t, mesh, heat_capacity, solar_forcing, Fb, T_S, T_ATM, H_I):
    # Note that this function modifies the first argument instead of returning the result
    #diffusion_op = Functions.calc_diffusion_operator(mesh, diffusion_coeff, T_OCN, phi)
    
    T_OCN_New = T_OCN + delta_t * ((solar_forcing - mesh.A_up - mesh.B_up * T_S + mesh.A_dn + mesh.B_dn * T_ATM + Fb)/ heat_capacity) * (H_I<= 0)
    return T_OCN_New

def timestep_euler_backward_ocn(solve, delta_t, T_OCN, T_S, T_ATM, t, mesh, heat_capacity, solar_forcing, F_b, H_I):
    
    source_terms = ((solar_forcing - mesh.A_up - mesh.B_up *T_S + mesh.A_dn + mesh.B_dn * T_ATM + F_b) / heat_capacity) * (H_I <=0)

    T_OCN_New = solve(T_OCN + delta_t * source_terms)
    return T_OCN_New 

   