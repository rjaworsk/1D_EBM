#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:51:38 2023

@author: ricij
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy

from scipy import sparse
from scipy import special

import Functions
import ATM_fct
import OCN_fct

class Mesh:
    def __init__(self):
        self.n_latitude  = 361
        self.n_longitude = 128
        self.ndof = self.n_latitude **2
        self.h = np.pi / (self.n_latitude - 1)
        
        self.RE = 6.37E6  #Radius der Erde
        self.Tf = -1.8    # Freezing point of sea water [degC]
        self.ki = 2.0     #Thermal conductivity of sea ice [W m^-1 degC^-1]
        self.Tm = -0.1    # Melting Temperature of sea water
        self.Lf = 10.0    # Latent heat of fusion of sea ice [W yr m^-3]
        
        self.A_up = 380   # Fluxes
        self.B_up = 7.9
        self.A_dn = 335
        self.B_dn = 5.9
        self.A_olr = 241
        self.B_olr = 2.4
        
        
        #self.csc2 = np.array([1 / np.sin(self.h * j) ** 2 for j in range(1, self.n_latitude - 1)])
        #self.cot = np.array([1 / np.tan(self.h * j) for j in range(1, self.n_latitude - 1)])

class P_atm:
    def __init__(self):        
        self.heat_capacity = 0.3 * np.ones(mesh.n_latitude) #Atmosphere column heat capacity [W yr m^-2 degC^-1]
        self.diffusion_coeff = 2E14 #Large-scale atmospheric diffusivity [m^2 yr^-1]
        
class P_ocn:
    def __init__(self):
        self.K_O = 4.4E11   #Large-scale ocean diffusivity [m^2 yr^-1]
        self.c_O = 1.27E-4  #Ocean specific heat capacity [W yr kg^-1 degC^-1]
        self.rhoo = 1025    #Density of sea water [kg m^-3]
        self.Hml_const = 75 #Mixed-layer depth when set constant [m]
        
        
def ice_edge(H_I, phi):    
        if H_I[len(H_I)-1] == 0: 
            index = len(H_I)
            ice_latitude = phi[len(phi)-1]
        else:
            index = 0
            while (H_I[index] <= 0): 
              index = index + 1
            ice_latitude = phi[index]
        return index, ice_latitude
  
def surface_temp(T_ATM, T_OCN, H_I, solar_forcing_ocn, phi, mesh):
    T_S = copy.copy(T_OCN)
    if any(H_I) > 0 : #sea ice exists
    
        phi_i_index, phi_i  = ice_edge(H_I,phi)
        
        T_d = (mesh.ki * mesh.Tf + H_I * (solar_forcing_ocn - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM))/(mesh.ki + mesh.B_up * H_I)
        
        for j in range(phi_i_index,len(phi)):
            T_S[j] = mesh.Tm * (T_d[j] > mesh.Tm) + T_d[j] * (T_d[j] <= mesh.Tm)
    
    return T_S       

def FreezeAndMelt(T_OCN, H_I, Hml, mesh):
    T_OCN_new = copy.copy(T_OCN)
    H_I_new = copy.copy(H_I)
    z = mesh.Lf/(P_ocn.c_O*P_ocn.rhoo*Hml);
    
   
    for j in range(len(T_OCN)):   
       if H_I[j] < 0:
           
           H_I_new[j] = 0
           T_OCN_new[j] = T_OCN[j] - z[j]*H_I[j]
           
           if T_OCN_new[j] < mesh.Tf:
               H_I_new[j] = (mesh.Tf-T_OCN_new[j])/z[j]
               T_OCN_new[j] = mesh.Tf
        
       elif H_I[j] == 0 and T_OCN[j] < mesh.Tf:
           
               H_I_new[j] = (mesh.Tf-T_OCN[j])/z[j]
               T_OCN_new[j] = mesh.Tf
       
           
       elif H_I[j] > 0:
           H_I_new[j] = H_I[j] + (mesh.Tf-T_OCN[j])/z[j]
           T_OCN_new[j] = mesh.Tf
      
           if H_I_new[j] < 0:
               T_OCN_new[j] = mesh.Tf -z[j]*H_I_new[j]
               H_I_new[j] = 0
               
    return T_OCN_new, H_I_new


def timestep_euler_forward(mesh,T_S, T_ATM, Fb, solar_forcing, H_I, t, delta_t):
    # Note that this function modifies the first argument instead of returning the result
    H_I_new = H_I - delta_t * (1/mesh.Lf * (-mesh.A_up - mesh.B_up * T_S + mesh.A_dn + mesh.B_dn * T_ATM + Fb + solar_forcing) * (H_I >0))
    return H_I_new

def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, T_ATM_0, T_OCN_0, T_S_0,P_ocn,
                          diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude, max_iterations=150, rel_error=2e-5, verbose=True):
    # Number of time steps per year
    ntimesteps = 48

    # Step size
    delta_t = 1 / ntimesteps
    
    T_ATM = np.zeros((mesh.n_latitude, ntimesteps))
    T_ATM[:,-1] = T_ATM_0
    T_OCN = np.zeros((mesh.n_latitude, ntimesteps))
    T_OCN[:,-1] = T_OCN_0
    T_S = np.zeros((mesh.n_latitude, ntimesteps))
    T_S[:,-1] = T_S_0
    H_I = np.zeros((mesh.n_latitude, ntimesteps))
    H_I[:,-1] = H_I_0
    
    
    # Area-mean in every time step
    temp_atm = np.zeros(ntimesteps)
    temp_ocn=  np.zeros(ntimesteps)
    temp_s =  np.zeros(ntimesteps)
    

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    
    jacobian_atm = ATM_fct.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi)
    jacobian_ocn = OCN_fct.calc_jacobian_ocn(mesh, diffusion_coeff_ocn, heat_capacity_ocn, phi)

    for i in range(max_iterations):
        print(i)
        
        for t in range(ntimesteps):    
            
            phi_i = ice_edge(H_I[:,t-1], phi)[1]  # neuer Ice_Edge Index 
            
            Fb = OCN_fct.BasalFlux(phi)
            Hml = P_ocn.Hml_const * np.ones(len(phi))
           
            albedo_ocn = OCN_fct.calc_albedo(phi, phi_i)
            solar_forcing_ocn  = Functions.calc_solar_forcing(albedo_ocn, true_longitude)[:,t]

            
           # T_ATM[:,t] =  ATM_fct.timestep_euler_forward_atm(T_ATM[:,t-1], t, delta_t, mesh, diffusion_coeff_atm, P_atm.heat_capacity, T_S[:,t-1], phi)
            T_ATM[:,t] =   ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1 / ntimesteps, T_ATM[:,t-1], T_S[:,t-1], t, mesh, P_atm.heat_capacity)
            
           
           # T_OCN[:,t] = OCN_fct.timestep_euler_forward_ocn(T_OCN[:,t-1], t, delta_t, mesh, diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, Fb, T_S[:,t-1], T_ATM[:,t-1], H_I[:,t-1], phi)
            T_OCN[:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1 / ntimesteps, T_OCN[:,t-1], T_S[:,t-1], T_ATM[:,t-1], t, mesh, heat_capacity_ocn, solar_forcing_ocn, Fb, H_I[:,t-1])
           
            H_I[:,t] = timestep_euler_forward(mesh,T_S[:,t-1], T_ATM[:,t-1], Fb, solar_forcing_ocn, H_I[:,t-1], t, delta_t)
            
            T_OCN[:,t], H_I[:,t] = FreezeAndMelt(T_OCN[:,t], H_I[:,t], Hml, mesh)
          
            T_S[:,t] = surface_temp(T_ATM[:,t], T_OCN[:,t], H_I[:,t], solar_forcing_ocn, phi, mesh)
    
            
            temp_atm[t] = np.mean(T_ATM[:,t])
            temp_ocn[t] = np.mean(T_OCN[:,t])
            temp_s[t] = np.mean(T_S[:,t])
            
       
    
        avg_temperature_atm = np.sum(temp_atm) / ntimesteps
        avg_temperature_ocn = np.sum(temp_ocn) / ntimesteps
        avg_temperature_s = np.sum(temp_s) / ntimesteps
      
      
        print(np.abs(avg_temperature_atm - old_avg_atm))
        if (np.abs(avg_temperature_atm - old_avg_atm) and np.abs(avg_temperature_ocn - old_avg_ocn)  and  np.abs(avg_temperature_s - old_avg_s)) < rel_error:
            # We can assume that the error is sufficiently small now.
            verbose and print("Equilibrium reached!")
            
            break
        
        else:
              old_avg_atm = avg_temperature_atm
              old_avg_ocn = avg_temperature_ocn
              old_avg_s = avg_temperature_s
           
       
         
    return  T_ATM, T_S, T_OCN, H_I
       
# Run code
if __name__ == '__main__':
    file_path_lambda  = 'input/True_Longitude.dat'     
    file_path = 'input/The_World128x65.dat'  
  
    true_longitude = Functions.read_true_longitude(file_path_lambda)
    ntimesteps = len(true_longitude)
    
    mesh = Mesh()
    phi = np.linspace(0,np.pi/2,mesh.n_latitude) # nur noch bis zum Äquator
    phi_i_deg = 75 #belibiger Startwert für den Breitengrad der Eisschicht 
   
    
    P_atm = P_atm() #Parameter für die Atmosphäre
    P_ocn = P_ocn() #Parameter für den Ozean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones(mesh.n_latitude)  # Hml kann man auch variable setzen 
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusionskoeffizient
    
    
    #Inital Conditions
    T_ATM_0 = 0.5 * (-15 + 35 * np.cos(2*phi))
    T_OCN_0 = 0.5 * (28.2 + 31.8 * np.cos(180*phi/phi_i_deg))
    B = 3/((np.pi/2) - phi_i_deg * np.pi/180)
    A = -B *phi_i_deg * np.pi/180
    H_I_0 = A + B * phi
    H_I_0 = H_I_0 * (H_I_0 > 0) #da Eisdicke nicht negativ sein kann
    T_OCN_0 = T_OCN_0 * (H_I_0 <= 0) + mesh.Tf * (H_I_0 > 0)
    
    Functions.plot_annual_temperature(T_OCN_0, mesh.Tf , "Ocean inital temperature") #begrenzen uns hier auf die Nordhalbkugel
    Functions.plot_annual_temperature(T_ATM_0, mesh.Tf , "Atmosphere inital temperature")
    
    
    phi_index, phi_i = ice_edge(H_I_0, phi)
    albedo_ocn = OCN_fct.calc_albedo(phi, phi_i)
    solar_forcing_ocn  = Functions.calc_solar_forcing(albedo_ocn, true_longitude)
    T_S_0 = surface_temp(T_ATM_0, T_OCN_0, H_I_0, solar_forcing_ocn[:,-1], phi, mesh)
    
    Functions.plot_annual_temperature(T_S_0, mesh.Tf , "Surface inital temperature")


    T_ATM, T_S, T_OCN, H_I = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, T_ATM_0, T_OCN_0, T_S_0, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude)


    Functions.plot_annual_temperature(np.mean(T_S, axis = 1), np.mean(T_S), "Surface Temp ")
    Functions.plot_annual_temperature(np.mean(T_ATM, axis=1),np.mean(T_ATM), "Atmosphere Temp")
    Functions.plot_annual_temperature(np.mean(T_OCN, axis = 1), np.mean(T_OCN) , "Ocean Temp")
    Functions.plot_annual_temperature(np.mean(H_I, axis = 1), np.mean(H_I) , "Mean Ice Thickness")
    

    
    
    
    
    

