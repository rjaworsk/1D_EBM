#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:16:38 2023

@author: ricij
"""

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
from numba import njit
import time

from scipy import sparse
from scipy import special

import Functions
import ATM_fct
import OCN_fct

class Mesh():
    def __init__(self, geo_dat):
        self.n_latitude  = 722
        self.n_longitude = 128
        self.ndof = self.n_latitude **2
        self.h = np.pi / (self.n_latitude - 1)
        self.area = Functions.calc_area(self.n_latitude)
        
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

class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""

class Timer:
    
    """ Timer class from https://realpython.com/python-timer/#python-timer-functions """
    
    def __init__(self):

        self._start_time = None


    def start(self):

        """Start a new timer"""

        if self._start_time is not None:

            raise TimerError(f"Timer is running. Use .stop() to stop it")


        self._start_time = time.perf_counter()


    def stop(self, text=""):

        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:

            raise TimerError(f"Timer is not running. Use .start() to start it")


        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None

        print("Elapsed time ("+text+f"): {elapsed_time:0.10e} seconds")

@njit   
def ice_edge(H_I, phi):    
        if H_I[len(H_I)-1] == 0: 
            index_n = len(H_I)-1
            ice_latitude_n = phi[len(phi)-1]
        elif H_I[len(H_I)-1] != 0:
            index_n = int(len(H_I)/2)
            while (H_I[index_n] <= 0): 
              index_n = index_n + 1
            ice_latitude_n = phi[index_n]
        if H_I[0] == 0: 
            index_s = 0
            ice_latitude_s = phi[0]
        elif H_I[0] != 0: 
            index_s = int(len(H_I)/2)
            while (H_I[index_s] <= 0): 
              index_s = index_s - 1
            ice_latitude_s = phi[index_s]
                 
        return index_s, ice_latitude_s, index_n, ice_latitude_n
  
def surface_temp(T_ATM, T_OCN, H_I, solar_forcing_ocn, phi, mesh):
    T_S = copy.copy(T_OCN)
    if any(H_I) > 0 : #sea ice exists
    
        phi_index_s, phi_i_s, phi_index_n, phi_i_n  = ice_edge(H_I,phi)
        
        T_d = (mesh.ki * mesh.Tf + H_I * (solar_forcing_ocn - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM))/(mesh.ki + mesh.B_up * H_I)
       
        for j in range(phi_index_n,len(phi)):
            T_S[j] = mesh.Tm * (T_d[j] > mesh.Tm) + T_d[j] * (T_d[j] <= mesh.Tm)
            
        for j in range(0,phi_index_s+1):
            T_S[j] = mesh.Tm * (T_d[j] > mesh.Tm) + T_d[j] * (T_d[j] <= mesh.Tm)    
    
    return T_S       

def FreezeAndMelt(T_OCN, H_I, Hml, mesh):
    T_OCN_new = copy.copy(T_OCN)
    H_I_new = copy.copy(H_I)
    z = mesh.Lf/(P_ocn.c_O*P_ocn.rhoo*Hml)
   
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
                          diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude,n_timesteps, max_iterations=15, rel_error=2e-5, verbose=True):
    # Number of time steps per year
    
    # Step size
    delta_t = 1 / ntimesteps
    
    #Startwerte
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
    phi_index_s = np.zeros(ntimesteps)
    phi_index_n = np.zeros(ntimesteps)
    

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    
    
    Fb = OCN_fct.BasalFlux(phi)
    Hml = P_ocn.Hml_const * np.ones(len(phi))
    
    # Construct and factorize Jacobian for the atmosphere
    jacobian_atm = ATM_fct.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi)
    m, n = jacobian_atm.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_atm = sparse.csc_matrix(jacobian_atm)
    jacobian_atm = sparse.linalg.factorized(eye - delta_t * jacobian_atm)   
    
    # Construct and factorize Jacobian for the ocean
    jacobian_ocn = OCN_fct.calc_jacobian_ocn(mesh, diffusion_coeff_ocn, heat_capacity_ocn, phi)
    m, n = jacobian_ocn.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_ocn = sparse.csc_matrix(jacobian_ocn)
    jacobian_ocn = sparse.linalg.factorized(eye - delta_t * jacobian_ocn)
    
    # Compute insolation
    insolation = Functions.calc_insolation(phi, true_longitude)
    
    timer = Timer()
    for i in range(max_iterations):
        print(i)
        timer.start()
        for t in range(ntimesteps):    
            #print(t)
            
            phi_index_s[t], phi_i_s, phi_index_n[t], phi_i_n = ice_edge(H_I[:,t-1], phi)  # neuer Ice_Edge Index 
            
            albedo_ocn = OCN_fct.calc_albedo(phi, phi_i_n, phi_i_s, mesh)
            
            solar_forcing_ocn  = insolation[:,t-1] * albedo_ocn
            #solar_forcing_ocn = np.multiply(solar_forcing_paper[:,t-1], albedo_ocn)
   
            T_ATM[:,t] =   ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1 / ntimesteps, T_ATM[:,t-1], T_S[:,t-1], t, mesh, P_atm.heat_capacity)
            
            T_OCN[:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1 / ntimesteps, T_OCN[:,t-1], T_S[:,t-1], T_ATM[:,t-1], t, mesh, heat_capacity_ocn, solar_forcing_ocn, Fb, H_I[:,t-1])
       
          
            H_I[:,t] = timestep_euler_forward(mesh,T_S[:,t-1], T_ATM[:,t-1], Fb, solar_forcing_ocn, H_I[:,t-1], t, delta_t)
            
  
           
            T_OCN[:,t], H_I[:,t] = FreezeAndMelt(T_OCN[:,t], H_I[:,t], Hml, mesh)
            
 
            T_S[:,t] = surface_temp(T_ATM[:,t], T_OCN[:,t], H_I[:,t], solar_forcing_ocn, phi, mesh)
           
            
            temp_atm[t] = np.mean(T_ATM[:,t])
            temp_ocn[t] = np.mean(T_OCN[:,t])
            temp_s[t] = np.mean(T_S[:,t])
            
       
        timer.stop("one year")
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
           
       
         
    return  T_ATM, T_S, T_OCN, H_I, phi_index_s, phi_index_n
       
# Run code
if __name__ == '__main__':
    start = time.time()
    #file_path_lambda  = 'input/True_Longitude.dat'     
    file_path = 'input/The_World128x65.dat'  
    geo_dat_ = Functions.read_geography("input/The_World128x65.dat")
    
    ntimesteps = 48
    dt = 1/ ntimesteps
    #ecc= 0.016740
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc=0.016740, per = 1.783037)
   
    
    mesh = Mesh(geo_dat_)
    phi = np.linspace(-np.pi/2,np.pi/2,mesh.n_latitude) # nur noch bis zum Äquator
    phi_i_deg_n = 75 #belibiger Startwert für den Breitengrad der Eisschicht 
    phi_i_deg_s  = -75
    
    P_atm = P_atm() #Parameter für die Atmosphäre
    P_ocn = P_ocn() #Parameter für den Ozean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones(mesh.n_latitude)  # Hml kann man auch variable setzen 
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusionskoeffizient
    
    
    #Inital Conditions
    T_ATM_0 = 0.5 * (-15 + 35 * np.cos(2*phi))
    T_OCN_0 = 0.5 * (28.2 + 31.8 * np.cos(180*phi/phi_i_deg_n))
    B = 3/((np.pi/2) - phi_i_deg_n * np.pi/180)
    A = -B *phi_i_deg_n * np.pi/180
    H_I_0 = A + B * phi
    H_I_0 = H_I_0 * (H_I_0 > 0) #da Eisdicke nicht negativ sein kann
    H_I_0 = H_I_0[::-1]  + H_I_0
    T_OCN_0 = T_OCN_0 * (H_I_0 <= 0) + mesh.Tf * (H_I_0 > 0)
    
    Functions.plot_annual_temperature(T_OCN_0, mesh.Tf , "Ocean inital temperature") #begrenzen uns hier auf die Nordhalbkugel
    Functions.plot_annual_temperature(T_ATM_0, mesh.Tf , "Atmosphere inital temperature")
    
    
    phi_index_s, phi_i_s, phi_index_n, phi_i_n = ice_edge(H_I_0, phi)
    albedo_ocn = OCN_fct.calc_albedo(phi, phi_i_n, phi_i_s, mesh)
    solar_forcing_ocn  = Functions.calc_solar_forcing(phi,albedo_ocn, true_longitude)
    T_S_0 = surface_temp(T_ATM_0, T_OCN_0, H_I_0, solar_forcing_ocn[:,0], phi, mesh)
    
    Functions.plot_annual_temperature(T_S_0, mesh.Tf , "Surface inital temperature")


    T_ATM, T_S, T_OCN, H_I, phi_index_s, phi_index_n  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, T_ATM_0, T_OCN_0, T_S_0, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude, ntimesteps)

    end = time.time()
    print(end - start)
    
    Functions.plot_annual_temperature(np.mean(T_S, axis = 1), np.mean(T_S), "Surface Temp lat")
    Functions.plot_annual_temperature(np.mean(T_ATM, axis=1),np.mean(T_ATM), "Atmosphere Temp  lat")
    Functions.plot_annual_temperature(np.mean(T_OCN, axis = 1), np.mean(T_OCN) , "Ocean Temp lat")
    Functions.plot_annual_temperature(np.mean(H_I, axis = 1), np.mean(H_I) , "Mean Ice Thickness lat")
    
    
    
   # Functions.plot_over_time(np.mean(T_ATM, axis=0),np.mean(T_ATM), "Atmosphere Temp")
    annual_mean_temperature_total_ATM =[Functions.calc_mean_1D(T_ATM[:,t], mesh.area) for t in range(ntimesteps)]
    average_temperature_total_ATM = np.sum(annual_mean_temperature_total_ATM) / ntimesteps
    Functions.plot_over_time(annual_mean_temperature_total_ATM, average_temperature_total_ATM,  "Atmosphere Temp")
    
    #Functions.plot_over_time(np.mean(T_OCN, axis = 0), np.mean(T_OCN) , "Ocean Temp")
    annual_mean_temperature_total_OCN =[Functions.calc_mean_1D(T_OCN[:,t], mesh.area) for t in range(ntimesteps)]
    average_temperature_total_OCN= np.sum(annual_mean_temperature_total_OCN) / ntimesteps
    Functions.plot_over_time(annual_mean_temperature_total_OCN, average_temperature_total_OCN,  "Ocean Temp")
    
    #Functions.plot_over_time(np.mean(T_S, axis = 0), np.mean(T_S) , "Surface Temp")
    annual_mean_temperature_total_S =[Functions.calc_mean_1D(T_S[:,t], mesh.area) for t in range(ntimesteps)]
    average_temperature_total_S = np.sum(annual_mean_temperature_total_S) / ntimesteps
    Functions.plot_over_time(annual_mean_temperature_total_S, average_temperature_total_S,  "Surface Temp")
    
    
   # H_I2 = H_I[180:361, :]
    H_I2 = H_I[361:722, :]
    H_I2[H_I2 == 0] = np.NAN
    
   # H_I2_t = np.array([H_I2[:,t][np.logical_not(np.isnan(H_I2[:,t]))] for t in range(ntimesteps)])
    
   # annual_mean_H_I2 =[Functions.calc_mean_1D(H_I2_t[:,t], mesh.area) for t in range(ntimesteps)]
    means = np.nanmean(H_I2, axis=0)  #Wie ohne NaN aber trotzdem mit Calc_mean Function?
    Functions.plot_over_time(means, np.mean(means) , "Mean Ice Thickness North")
    
    # H_I2_s = H_I[0:180, :]
    # H_I2_s[H_I2_s == 0] = np.NAN
    # means_s = np.nanmean(H_I2_s, axis=0)  #Wie ohne NaN aber trotzdem mit Calc_mean Function?
    # Functions.plot_over_time(means_s, np.mean(means_s) , "Mean Ice Thickness South")
   
    
    latitude = np.linspace(-90,90,722)
    ice = np.zeros(phi_index_n.size)
    for i in range(phi_index_n.size):
        ice[i] = latitude[int(phi_index_n[i])]
    Functions.plot_over_time(ice, np.mean(ice) , "Ice_Edge Latitude")
    
    # ice_s = np.zeros(phi_index_n.size)
    # for i in range(phi_index_s.size):
    #     ice_s[i] = latitude[int(phi_index_s[i])]
    # Functions.plot_over_time(ice, np.mean(ice) , "Ice_Edge Latitude South")
    
    
    
    #Plotten von Ergebnissen vom paper und hier erstellte Ergebnisse 
   # Paper_results = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/T_S_Paper.txt")
    Paper_results = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/T_S_361.txt")
    Functions.plot_annual_temperature_vgl(np.mean(T_S[361:722,:], axis = 1),Paper_results[:,1])
    
   # Paper_ice_thickness = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/thick.txt")
    Paper_ice_thickness = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_48.txt")
   # Paper_ice_thickness = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_730.txt")
    Functions.plot_annual_temperature_vgl(means, Paper_ice_thickness[:,1])
    
    #Ice_edge_lat_paper = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Paper_ice_edge_latitude.txt")
    Ice_edge_lat_paper = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/ice_edge_48.txt") # hier mit nur 48 Zeitabschnitten
    Functions.plot_annual_temperature_vgl(ice, Ice_edge_lat_paper[:,1])
    
    
    
    
    