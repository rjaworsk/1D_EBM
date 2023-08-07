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
import numpy as np
import time
from scipy import sparse

import Functions
import ATM_fct
import OCN_fct

class Mesh():
    def __init__(self, geo_dat):
        self.n_latitude  = 721 #Equator is at 361
        #self.n_longitude = 128
        self.ndof = self.n_latitude **2
        self.h = np.pi / (self.n_latitude - 1)
        self.area = Functions.calc_area(self.n_latitude)
        
        self.RE = 6.37E6  #Radius of the earth
        self.Tf = -1.8    #Freezing point of sea water [degC]
        self.ki = 2.0     #Thermal conductivity of sea ice [W m^-1 degC^-1]
        self.Tm = -0.1    #Melting Temperature of sea water
        self.Lf = 10.0    #Latent heat of fusion of sea ice [W yr m^-3]
        
        self.A_up = 380   #Fluxes
        self.B_up = 7.9
        self.A_dn = 335
        self.B_dn = 5.9
        self.A_olr = 241
        self.B_olr = 2.4
        

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
  
def ice_edge(H_I, phi):    
    
        if H_I[len(H_I)-1] == 0: #calc ice edge north 
            index_n = len(H_I)-1
            ice_latitude_n = phi[len(phi)-1]
            
        elif H_I[len(H_I)-1] != 0:
            index_n = int(len(H_I)/2) 
            
            while (H_I[index_n] <= 0): 
              index_n = index_n + 1
              
            ice_latitude_n = phi[index_n]
          
        if H_I[0] == 0: #calc ice edge south
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


def timestep_euler_forward_H_I(mesh,T_S, T_ATM, Fb, solar_forcing, H_I, t, delta_t): #calc new ice thickness 
    
    H_I_new = H_I - delta_t * (1/mesh.Lf * (-mesh.A_up - mesh.B_up * T_S + mesh.A_dn + mesh.B_dn * T_ATM + Fb + solar_forcing) * (H_I >0))
    return H_I_new



def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, T_ATM_0, T_OCN_0, T_S_0, P_ocn,
                          diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude,n_timesteps, F_w, Solar_forcing_paper, max_iterations=100, rel_error=1e-7, verbose=True):
    # Step size
    delta_t = 1 / ntimesteps
    
    #Inital Conditions
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
    temp_H_I =  np.zeros(ntimesteps)
    
    phi_index_s = np.zeros(ntimesteps)
    phi_index_n = np.zeros(ntimesteps)
    

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    old_avg_H_I = 0
    
    
    Fb = OCN_fct.BasalFlux(phi) #additional ocean flux
    #Fb = np.zeros(721)
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
    #insolation = Functions.calc_insolation(phi, true_longitude)
    insolation = Solar_forcing_paper 
    
    timer = Timer()
    for i in range(max_iterations):
        print("Iteration: ",i)
        timer.start()
        for t in range(ntimesteps):    
            
            phi_index_s[t], phi_i_s, phi_index_n[t], phi_i_n = ice_edge(H_I[:,t-1], phi)  # new Ice_Edge Index 
            
            coalbedo_ocn = OCN_fct.calc_coalbedo(phi, phi_i_n, phi_i_s, mesh) #new coalbdeo dependant on the ice_edge 
            
            solar_forcing_ocn  = insolation[:,t-1] * coalbedo_ocn
   
            T_ATM[:,t] =   ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1/ntimesteps, T_ATM[:,t-1], T_S[:,t-1], t, mesh, P_atm.heat_capacity, F_w)
            
            T_OCN[:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1 / ntimesteps, T_OCN[:,t-1], T_S[:,t-1], T_ATM[:,t-1], t, mesh, heat_capacity_ocn, solar_forcing_ocn, Fb, H_I[:,t-1])
     
            H_I[:,t] = timestep_euler_forward_H_I(mesh,T_S[:,t-1], T_ATM[:,t-1], Fb, solar_forcing_ocn, H_I[:,t-1], t, delta_t)
            
            T_OCN[:,t], H_I[:,t] = FreezeAndMelt(T_OCN[:,t], H_I[:,t], Hml, mesh)
            
            solar_forcing_ocn_new  = insolation[:,t] * coalbedo_ocn
 
            T_S[:,t] = surface_temp(T_ATM[:,t], T_OCN[:,t], H_I[:,t], solar_forcing_ocn_new, phi, mesh)
           
            
            temp_atm[t] = np.mean(T_ATM[:,t])
            temp_ocn[t] = np.mean(T_OCN[:,t])
            temp_s[t] = np.mean(T_S[:,t])
            temp_H_I[t] = np.mean(H_I[:,t])
            
        
        timer.stop("one year")
        avg_temperature_atm = np.sum(temp_atm) / ntimesteps
        avg_temperature_ocn = np.sum(temp_ocn) / ntimesteps
        avg_temperature_s = np.sum(temp_s) / ntimesteps
        avg_H_I = np.sum(temp_H_I) / ntimesteps
      
        print("Fehler: ",np.abs(avg_H_I - old_avg_H_I))
        if (np.abs(avg_temperature_atm - old_avg_atm) and np.abs(avg_temperature_ocn - old_avg_ocn)  and  np.abs(avg_temperature_s - old_avg_s) and np.abs(avg_H_I - old_avg_H_I)) < rel_error:
            # We can assume that the error is sufficiently small now.
            verbose and print("Equilibrium reached!")
            
            break
        
        else:
              old_avg_atm = avg_temperature_atm
              old_avg_ocn = avg_temperature_ocn
              old_avg_s = avg_temperature_s
              old_avg_H_I = avg_H_I
  
         
    return  T_ATM, T_S, T_OCN, H_I, phi_index_s, phi_index_n
       
# Run code
if __name__ == '__main__':
    start = time.time()   
    file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    geo_dat_ = Functions.read_geography(file_path)
    
    ntimesteps = 730 #take one timestep twice a day
    dt = 1/ ntimesteps
    #ecc= 0.016740 #old ecc
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc=0, per = 1.783037)
   
    co2_ppm = 315.0
    #F_w = Functions.calc_radiative_cooling_co2(co2_ppm) #Influence of global warming
    F_w = 0
    
    mesh = Mesh(geo_dat_)
    phi = np.linspace(-np.pi/2,np.pi/2,mesh.n_latitude) #from 90° south to 90° north
    phi_i_deg_n = 75 #inital value for the latitude of the ice-edge 
    phi_i_deg_s = 75 
    
    P_atm = P_atm() #Parameters for the atmosphere
    P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones(mesh.n_latitude)  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
    
    
    #Solar Forcing used in Paper 
    Solar_forcing_paper_nrd = np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/solar_focring_qrtdeg_halfday.txt"))
    Solar_forcing_paper_sdl =  np.flip( Solar_forcing_paper_nrd , axis = 0)
   
     
    S_temp = np.zeros((721,730))
    Solar_forcing_paper = np.zeros((721,730))  
    # S_temp[0:361,0:365] =   Solar_forcing_paper_sdl[:,365:731] #adjust for different seasons in northern and southern hemisphere
    # S_temp[0:361,365:731] =   Solar_forcing_paper_sdl[:,0:365]
    S_temp[0:361,:] =   Solar_forcing_paper_sdl
    S_temp[360:723,:] =   Solar_forcing_paper_nrd[0:361,:]
    Solar_forcing_paper[:,0:572] = S_temp[:,158:731]
    Solar_forcing_paper[:,572:731] = S_temp[:,0:158]
    #Solar_forcing_paper = S_temp #für original Solar Forcing wie im Paper (Start Januar)

    
    #Inital Conditions
    T_ATM_0 = 0.5 * (-15 + 35 * np.cos(2*phi))
    T_OCN_0 = 0.5 * (28.2 + 31.8 * np.cos(180*phi/phi_i_deg_n))
    B = 3/((np.pi/2) - phi_i_deg_n * np.pi/180)
    A = -B *phi_i_deg_n * np.pi/180
    H_I_0 = A + B * phi
    H_I_0 = H_I_0 * (H_I_0 > 0) #because ice thickness cannot be negative
    H_I_0 = H_I_0[::-1]  + H_I_0
    T_OCN_0 = T_OCN_0 * (H_I_0 <= 0) + mesh.Tf * (H_I_0 > 0)
    
    Functions.plot_annual_temperature(T_OCN_0, mesh.Tf , "Ocean inital temperature") 
    Functions.plot_annual_temperature(T_ATM_0, mesh.Tf , "Atmosphere inital temperature")
    
    
    phi_index_s, phi_i_s, phi_index_n, phi_i_n = ice_edge(H_I_0, phi)
    coalbedo_ocn = OCN_fct.calc_coalbedo(phi, phi_i_n, phi_i_s, mesh)
    #solar_forcing_ocn  = Functions.calc_solar_forcing(phi,coalbedo_ocn, true_longitude) #og
    solar_forcing_ocn  = Solar_forcing_paper[:,0] * coalbedo_ocn #paper
    #T_S_0 = surface_temp(T_ATM_0, T_OCN_0, H_I_0, solar_forcing_ocn[:,0], phi, mesh) #og
    T_S_0 = surface_temp(T_ATM_0, T_OCN_0, H_I_0, solar_forcing_ocn, phi, mesh) #paper
    
    Functions.plot_annual_temperature(T_S_0, mesh.Tf , "Surface inital temperature")
    

    #inital conditions calculated in Matlab (for March) 
    # HI_Paper= np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/HI_stepsize1.txt"))
    # TA_Paper= np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TA_stepsize1.txt"))
    # TML_Paper= np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TML_stepsize1.txt"))
    # TS_Paper= np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TS_stepsize1.txt"))
    
    # T_ATM_0 = np.zeros(721)
    # T_ATM_0[361:722] = TA_Paper[1:361,158]
    # T_ATM_0[0:361] = np.flip(TA_Paper[:,158], axis=0)
    
    # T_OCN_0 = np.zeros(721)
    # T_OCN_0[361:722] = TML_Paper[1:361,158]
    # T_OCN_0[0:361] = np.flip(TML_Paper[:,158], axis=0)
    
    # T_S_0 = np.zeros(721)
    # T_S_0[361:722] = TS_Paper[1:361,158]
    # T_S_0[0:361] = np.flip(TS_Paper[:,158], axis=0)
    
    # H_I_0 = np.zeros(721)
    # H_I_0[361:722] = HI_Paper[1:361,158]
    # H_I_0[0:361] = np.flip(HI_Paper[:,158], axis=0)


    ###### run model #####
    T_ATM, T_S, T_OCN, H_I, phi_index_s, phi_index_n  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, T_ATM_0, T_OCN_0, T_S_0, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing_ocn, phi, true_longitude, ntimesteps, F_w , Solar_forcing_paper)

    end = time.time()
    print(end - start)
    
    #Plot temperature for each latitude
    Functions.plot_annual_temperature(np.mean(T_S, axis = 1), np.mean(T_S), "Surface Temp lat")
    Functions.plot_annual_temperature(np.mean(T_ATM, axis=1),np.mean(T_ATM), "Atmosphere Temp  lat")
    Functions.plot_annual_temperature(np.mean(T_OCN, axis = 1), np.mean(T_OCN) , "Ocean Temp lat")
    Functions.plot_annual_ice_thickness(np.mean(H_I, axis = 1), np.mean(H_I) , "Mean Ice Thickness lat")
    
    
    #Plot annual temperature
    annual_T_ATM_total =[Functions.calc_mean_1D(T_ATM[:,t], mesh.area) for t in range(ntimesteps)]      
    average_T_ATM_total = np.sum(annual_T_ATM_total) / ntimesteps
    Functions.plot_temperature_time(annual_T_ATM_total, average_T_ATM_total, "Atmosphere Temp")
    
    annual_T_OCN_total =[Functions.calc_mean_1D(T_OCN[:,t], mesh.area) for t in range(ntimesteps)]
    average_T_OCN_total= np.sum(annual_T_OCN_total) / ntimesteps
    Functions.plot_temperature_time(annual_T_OCN_total, average_T_OCN_total, "Ocean Temp")
    
    annual_T_S_total =[Functions.calc_mean_1D(T_S[:,t], mesh.area) for t in range(ntimesteps)]
    average_T_S_total = np.sum(annual_T_S_total) / ntimesteps
    Functions.plot_temperature_time(annual_T_S_total, average_T_S_total,  "Surface Temp")
    
    
    #Plot for ice thickness (Average is only calculated where ice is present)
    H_I_nan = H_I
    H_I_nan[H_I_nan == 0] = np.NAN

    H_I_wo_area_weighting_north = np.nanmean(H_I_nan[360:722], axis=0)  
    Functions.plot_ice_thickness_time(H_I_wo_area_weighting_north, np.mean(H_I_wo_area_weighting_north) , "Mean Ice Thickness North without area weighting")
    
    annual_mean_H_I_north =[Functions.calc_mean_1D_north_H_I(H_I_nan[:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_north, np.mean(annual_mean_H_I_north) , "Mean Ice Thickness North")
    
    annual_mean_H_I_south =[Functions.calc_mean_1D_south_H_I(H_I_nan[:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_south, np.mean(annual_mean_H_I_south) , "Mean Ice Thickness South")
   
    #Plot for the ice edge latitude in the northern Hemisphere
    latitude = np.linspace(-90,90,721)
    ice = np.zeros(phi_index_n.size)
    for i in range(phi_index_n.size):
        ice[i] = latitude[int(phi_index_n[i])]
    Functions.plot_ice_edge_time(ice, np.mean(ice) , "Ice_Edge Latitude")
    
    # ice_s = np.zeros(phi_index_n.size)
    # for i in range(phi_index_s.size):
    #     ice_s[i] = latitude[int(phi_index_s[i])]
    # ice = ice * -1    
    # Functions.plot_ice_edge_time(ice, np.mean(ice) , "Ice_Edge Latitude South")
    

    #Plot results of our model and results of the paper
    #Paper_results = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/T_S_Paper.txt") #48 timesteps 
    Paper_results = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/T_S_361.txt") #730 timesteps 
    Functions.plot_annual_T_S_vgl(np.mean(T_S[360:722,:], axis = 1),Paper_results[:,1])
    
    #Paper_ice_thickness = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_48.txt") #48 timesteps
    Paper_ice_thickness_without_area_weighting = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_730.txt") #730 timesteps
    Paper_ice_thickness_with_area_weighting = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_730_w_area.txt") #730 timesteps
    #Paper_ice_thickness_with_area_weighting_73steps = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Ice_thickness_73_w_area.txt") #73 timesteps
    
    Functions.plot_annual_ice_thickness_vgl(annual_mean_H_I_north, Paper_ice_thickness_with_area_weighting[:,1], "Comparison Ice Thickness with area weighting")
    Functions.plot_annual_ice_thickness_vgl(H_I_wo_area_weighting_north, Paper_ice_thickness_without_area_weighting[:,1], "Comparison Ice Thickness without area weighting")
    
    Ice_edge_lat_paper = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/Paper_ice_edge_latitude.txt") #730 timesteps
    #Ice_edge_lat_paper = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/ice_edge_48.txt") # hier mit nur 48 Zeitabschnitten
    Functions.plot_annual_ice_edge_vgl(ice, Ice_edge_lat_paper[:,1])
    
    ###### Comparison with Matlab Output ######
    
    H_I_Paper= np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/HI_30years.txt"))
    H_I_paper_new = np.zeros((721,730))
    H_I_paper_new[360:722,0:572] = H_I_Paper[:,21328:21900] 
    H_I_paper_new[360:722,572:731] = H_I_Paper[:,21170:21328]
    #H_I_paper_new[360:722,:] = H_I_Paper[:,26280:27010] #normal Solar Forcing (starting Jan)
    H_I_paper_new[H_I_paper_new == 0] = np.NAN
    
    H_I_Paper_mean_wo_area = np.nanmean(H_I_paper_new, axis = 0)
    H_I_Paper_mean_w_area =[Functions.calc_mean_1D_north_H_I_test(H_I_paper_new[:,t], mesh.area) for t in range(730)]
        
   # Functions.plot_ice_thickness_time(H_I_Paper_normal_mean, np.mean(H_I_Paper) , "Mean Ice Thickness North")
    Functions.plot_annual_ice_thickness_vgl(H_I_wo_area_weighting_north, H_I_Paper_mean_wo_area,  "Comparison Ice Thickness without area weighting and output Matlab")
    Functions.plot_annual_ice_thickness_vgl(annual_mean_H_I_north, H_I_Paper_mean_w_area, "Comparison Ice Thickness with area weighting and output Matlab")
    
    # Comparison with Ice Edge latitude with only 73 Steps
    j = 0
    ice_10th = np.zeros(73)
    for i in range(730): 
        if i%10 == 0:
            ice_10th[j] = ice[i]
            j +=1
            
    Ice_edge_lat_paper_73 = np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/ice_edge_73.txt") # hier mit nur 48 Zeitabschnitten        
    Functions.plot_annual_ice_edge_vgl(ice_10th, Ice_edge_lat_paper_73[:,1])
    
    #Comparison Temperature Matlab 
    T_S_Matlab = np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TS_30years.txt")) 
    T_ML_Matlab = np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TML_30years.txt")) 
    T_A_Matlab = np.transpose(np.genfromtxt(r"/Users/ricij/Documents/Universität/Master/Masterarbeit/TA_30years.txt")) 
    
    