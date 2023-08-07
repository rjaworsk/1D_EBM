#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:43:09 2023

@author: ricij
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

def read_geography(filepath):
    return np.genfromtxt(filepath, dtype=np.int8)

def robinson_projection(nlatitude, nlongitude):
    def x_fun(lon, lat):
        return lon / np.pi * (0.0379 * lat ** 6 - 0.15 * lat ** 4 - 0.367 * lat ** 2 + 2.666)

    def y_fun(_, lat):
        return 0.96047 * lat - 0.00857 * np.sign(lat) * np.abs(lat) ** 6.41

    # Longitude goes from -pi to pi (not included), latitude from -pi/2 to pi/2.
    # Latitude goes backwards because the data starts in the North, which corresponds to a latitude of pi/2.
    x_lon = np.linspace(-np.pi, np.pi, nlongitude, endpoint=False)
    y_lat = np.linspace(np.pi / 2, -np.pi / 2, nlatitude)

    x = np.array([[x_fun(lon, lat) for lon in x_lon] for lat in y_lat])
    y = np.array([[y_fun(lon, lat) for lon in x_lon] for lat in y_lat])

    return x, y


# Plot data at grid points in Robinson projection. Return the colorbar for customization.
# This will be reused in other milestones.
def plot_robinson_projection(data, title, plot_continents=False, geo_dat=[], **kwargs):
    # Get the coordinates for the Robinson projection.
    nlatitude, nlongitude = data.shape
    x, y = robinson_projection(nlatitude, nlongitude)

    # Start plotting.
    fig, ax = plt.subplots()

    # Create contour plot of geography information against x and y.
    im = ax.contourf(x, y, data, **kwargs)
    if plot_continents:
        ax.contour(x,y,geo_dat,colors='black',linewidths=0.25, linestyles='solid')
    plt.title(title)
    ax.set_aspect("equal")

    # Remove axes and ticks.
    plt.xticks([])
    plt.yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Colorbar with the same height as the plot. Code copied from
    # https://stackoverflow.com/a/18195921
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    return cbar

def read_true_longitude(filepath):
    return np.genfromtxt(filepath, dtype=np.float64)


def calc_radiative_cooling_co2(co2_concentration, co2_concentration_base=315.0,
                               radiative_cooling_base=210.3):
    return radiative_cooling_base - 5.35 * np.log(co2_concentration / co2_concentration_base)

def calc_albedo(geo_dat):
    def legendre(latitude):
        return 0.5 * (3 * np.sin(latitude) ** 2 - 1)

    def albedo(surface_type, latitude):
        if surface_type == 1:
            return 0.3 + 0.12 * legendre(latitude)
        elif surface_type == 2:
            return 0.6
        elif surface_type == 3:
            return 0.75
        elif surface_type == 5:
            return 0.29 + 0.12 * legendre(latitude)
        else:
            raise ValueError(f"Unknown surface type {surface_type}.")

    nlatitude, nlongitude = geo_dat.shape
    y_lat = np.linspace(0, np.pi/2, nlatitude)

    # Map surface type to albedo.
    return np.array([[albedo(geo_dat[i, j], y_lat[i])
                      for j in range(nlongitude)]
                     for i in range(nlatitude)])


def insolation(latitude, true_longitude, solar_constant, eccentricity,
               obliquity, precession_distance):
    # Determine if there is no sunset or no sunrise.
    sin_delta = np.sin(obliquity) * np.sin(true_longitude)
    cos_delta = np.sqrt(1 - sin_delta ** 2)
    tan_delta = sin_delta / cos_delta

    # Note that z can be +-infinity.
    # This is not a problem, as it is only used for the comparison with +-1.
    # We will never enter the `else` case below if z is +-infinity.
    z = -np.tan(latitude) * tan_delta

    if z >= 1:
        # Latitude where there is no sunrise
        return 0.0
    else:
        rho = ((1 - eccentricity * np.cos(true_longitude - precession_distance))
               / (1 - eccentricity ** 2)) ** 2

        if z <= -1:
            # Latitude where there is no sunset
            return solar_constant * rho * np.sin(latitude) * sin_delta
        else:
            h0 = np.arccos(z)
            second_term = h0 * np.sin(latitude) * sin_delta + np.cos(latitude) * cos_delta * np.sin(h0)
            return solar_constant * rho / np.pi * second_term
   
def calc_insolation(y_lat, true_longitudes, solar_constant=1371.685,    
                       eccentricity=0.0, obliquity=0.409253,
                       precession_distance=1.783037):
    nlatitude = y_lat.size
    
    return np.array([[insolation(y_lat[j], true_longitude, solar_constant, eccentricity,
                   obliquity, precession_distance)
                       for true_longitude in true_longitudes]
                     for j in range(nlatitude)])

def calc_solar_forcing(y_lat,albedo, true_longitudes, solar_constant=1371.685,    
                       eccentricity=0.0, obliquity=0.409253,
                       precession_distance=1.783037):
    def solar_forcing(theta, true_longitude, albedo_loc):
        s = insolation(theta, true_longitude, solar_constant, eccentricity,
                       obliquity, precession_distance)
       # a_c = 1 - albedo_loc

        return  s * albedo_loc

    # Latitude values at the grid points
    nlatitude = albedo.size
    #y_lat = np.linspace( 0,np.pi/2 , nlatitude)

    return np.array([[solar_forcing(y_lat[j], true_longitude, albedo[j])
                       for true_longitude in true_longitudes]
                     for j in range(nlatitude)])      



def calc_diffusion_operator(mesh,  D, temperature, phi):
    n_latitude = mesh.n_latitude
    h = mesh.h
    RE = mesh.RE
    return calc_diffusion_operator_inner(n_latitude, h, RE,  D, temperature, phi)
    
@njit
def calc_diffusion_operator_inner(n_latitude, h, RE,  D, temperature, phi):
    
    delta_phi = np.pi/(n_latitude-1)
    op = np.zeros(phi.size)
    op[0] = RE**2 * 2 * np.pi * D[1] * np.sin(h/2) * (temperature[1]- temperature[0] )/h   #0  # Äquator 
    op[-1]= RE**2 * 2 * np.pi  * D[1] * np.sin(h/2) * (temperature[-2]- temperature[-1] )/h #0  #north pole --> zero flux boundary condition

    for j in range(1,phi.size-1):
        op[j] = D[j] * ((temperature[j+1] - 2 * temperature[j] + temperature[j-1])/(delta_phi**2) -  np.tan(phi[j]) * (temperature[j+1] - temperature[j-1])/(2*delta_phi))
    return op
    



def plot_annual_temperature(annual_temperature, average_temperature, title):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature)
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")
    plt.plot(annual_temperature, label="annual temperature")

    plt.xlim((0, ntimesteps - 1))
    labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
    
def plot_annual_ice_thickness(annual_ice_thickness, average_ice_thickness, title):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness)
    plt.plot(average_ice_thickness * np.ones(ntimesteps), label="average ice thickness")
    plt.plot(annual_ice_thickness, label="annual ice thickness")

    plt.xlim((0, ntimesteps - 1))
    labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("Hi")
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()    
    

def calc_mean_1D(data, area):
    nlatitude = data.size
    
    mean_data = 0
    for i in range(0, nlatitude):
            mean_data += area[i] * data[i] 

    return mean_data 


def calc_mean_1D_north_H_I(data, area):
    nlatitude = data.size
    mean_data = 0
    j = 0
    for i in range(360,nlatitude):
        if (np.isnan(data[i]) == False):
            mean_data += area[i] * data[i]  
            j +=1        
    Area_normalized = 1/np.sum(area[nlatitude-j:nlatitude]) #Area normalization is different due to a different array size
    return mean_data * Area_normalized

def calc_mean_1D_north_H_I_test(data, area):
    nlatitude = 721
    mean_data = 0
    j = 0
    for i in range(360,nlatitude):
        if (np.isnan(data[i]) == False):
            mean_data += area[i] * data[i]  
            j +=1        
    Area_normalized = 1/np.sum(area[nlatitude-j:nlatitude]) #Area normalization is different due to a different array size
    return mean_data * Area_normalized

def calc_mean_1D_south_H_I(data, area):
    nlatitude = data.size
    mean_data = 0
    #area = np.flip(area, axis=0)
    j = 0
    for i in range(0,361):
        if (np.isnan(data[i]) == False):
            mean_data += area[i] * data[i]  
            j +=1
    Area_normalized = 1/np.sum(area[nlatitude-j:nlatitude]) #Area normalization is different due to a different array size
    return mean_data * Area_normalized


def calc_area(n_latitude):
    area = np.zeros(n_latitude, dtype=np.float64)
    delta_theta = np.pi / (n_latitude - 1)

    # Poles
    area[0] = area[-1] = 0.5 * (1 - np.cos(0.5 * delta_theta))

    # Inner cells
    for j in range(1, n_latitude - 1):
        area[j] = np.sin(0.5 * delta_theta) * np.sin(delta_theta * j) 

    return area


def calc_lambda(dt = 1.0 / 48,  nt=48, ecc= 0, per = 1.783037):
    eccfac = 1.0 - ecc**2
    rzero  = (2.0*np.pi)/eccfac**1.5
  
    lambda_ = np.zeros(nt)
    
    for n in range(1, nt):  #hier plus 2??
    
      nu = lambda_[n-1] - per
      t1 = dt*(rzero*(1.0 - ecc * np.cos(nu))**2)
      t2 = dt*(rzero*(1.0 - ecc * np.cos(nu+0.5*t1))**2)
      t3 = dt*(rzero*(1.0 - ecc * np.cos(nu+0.5*t2))**2)
      t4 = dt*(rzero*(1.0 - ecc * np.cos(nu + t3))**2)
      lambda_[n] = lambda_[n-1] + (t1 + 2.0*t2 + 2.0*t3 + t4)/6.0

    return lambda_

def plot_annual_T_S_vgl(annual_temperature_og,  annual_temperature_paper):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_og)
    plt.plot(annual_temperature_og, label="temperature (EBM)")
    plt.plot(annual_temperature_paper, label="temperature paper")
   

    plt.xlim((0, ntimesteps - 1))
    #labels = ["Äquator" ,"Nrd. Wendekreis", "Nrd. Polarkreis", "Nordpol"  ]
    labels = ["0°", "18°", "36°" , "54°", "72°", "90°" ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 6), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title("Comparison Surface temperature ")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
def plot_annual_ice_thickness_vgl(annual_ice_thickness,  annual_ice_thickness_paper, titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness)
    plt.plot(annual_ice_thickness, label="Ice thickness (EBM)")
    plt.plot(annual_ice_thickness_paper, label="Ice thickness (paper)")
   

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    #labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("H_i")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()    
    
def plot_annual_ice_edge_vgl(annual_ice_edge, annual_ice_edge_paper):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_edge)
    plt.plot(annual_ice_edge, label="Ice-edge (EBM)")
    plt.plot(annual_ice_edge_paper, label="Ice-edge (paper)")
   

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    #labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("$\phi$i")
    #plt.rc('axes', labelsize = 20) #geht nicht für nur einen Plot?
    plt.grid()
    plt.title("Comparison Ice-edge")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()     
    
    
    
def plot_temperature_time(annual_temperature_total, average_temperature , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_total)
    plt.plot(annual_temperature_total, label="temperature (total)")
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")
    

    plt.tight_layout()
    plt.show()    
    
def plot_ice_edge_time(annual_ice_edge_total, average_ice_edge , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_edge_total)
    plt.plot(annual_ice_edge_total, label="Ice-edge")
    plt.plot(average_ice_edge  * np.ones(ntimesteps), label="average Ice-edge")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("$\phi$i (°N)")
    #plt.rc('axes', labelsize = 20)
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()        
    
def plot_ice_thickness_time(annual_ice_thickness_total, average_ice_thickness, titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness_total)
    plt.plot(annual_ice_thickness_total, label="ice thickness")
    plt.plot(average_ice_thickness * np.ones(ntimesteps), label="average ice thickness")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("Hi")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()        
    
def plot_temperature_latitude(annual_temperature_total, average_temperature , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_total)
    plt.plot(annual_temperature_total, label="temperature (total)")
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")
   

    plt.xlim((0, ntimesteps - 1))
    labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()        


