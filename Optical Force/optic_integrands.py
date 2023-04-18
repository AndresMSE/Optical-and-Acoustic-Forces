from ast import Raise
from locale import normalize
import numpy as np
import math as m
from scipy.special  import jv
from scipy import integrate
'''Useful functions for the integrands'''
def snell(theta_i,n_t,n_i):
    """
    Snell's law function
    theta_i: incident angle
    n_t: transmited refraction index 
    n_i: incident refraction index
    """
    return np.arcsin((n_i/n_t)*np.sin(theta_i))

def param (x,z):
    """
    Calculates the r-vector magnitude
    and the gamma parameter
    """
    r = np.sqrt(x**2+z**2)
    gamma= m.atan2(x,z)
    return r,gamma

def get_theta(theta,phi,x,z,a):
    """
    Get the allowed max_theta according to the 
    arcos arg testing arround the given theta 
    and phi vector densities. This max_theta 
    depends on the particle's radius & position.
    theta: var theta values vector
    phi: var phi values vector
    x: r-position i-component
    z: r-position k-component
    a: particle radius size
    """
    rmag,gamma = param(x,z)    # Get the model position parameters
    new_theta = []    # Max value for theta value
    max_theta = theta[-1]    
    for i in range(len(theta)):
        o1 = np.sin(theta[i])*np.sin(gamma)*np.cos(phi[0])
        o2 = np.cos(theta[i])*np.cos(gamma)
        omega = (o1+o2)**2
        d = np.real(rmag*omega+np.emath.sqrt(a**2 -(rmag**2) +(rmag**2)*omega)) #magnitude of d-vector: d=a+r
        # Given the arcos arg. c1 & c2 implies an alpha angle greater than 0 lower than pi/2
        condition1 = abs(d-a) <= abs(rmag)    # alpha lower than pi/2    
        condition2 = d**2+a**2 >= rmag**2    # alpha greater than 0 
        if (condition1.any() and condition2.any())==True:
            new_theta.append(theta[i])    # Update new_theta    
        else:
            continue    # Keep searching
    if len(new_theta) == 0 :
        theta_found = np.zeros(len(theta))
    else:
        theta_found = np.linspace(new_theta[0],new_theta[-1],len(theta))
    return theta_found
def Bessel(rho,rho_B):
    """
    Definition of the Bessel Intensity Profile as a zeroth-order
    first kind, Bessel Function.
    rho := transversal axis
    rho_b := Bessel radius
    """
    z = 2.4*rho/rho_B
    I = jv(0,z)
    return I
def closed_Bessel(theta,rho_B,f):
    rho = f*np.sin(theta)
    drho = f*np.cos(theta)
    return (Bessel(rho,rho_B)**2)*rho*drho
    
def Gaussian(rho,sigma):
    """
    Definition of the Gaussian Intensity Profile
    """
    z = -rho**2/(2*sigma**2)
    I = np.exp(z)
    return I

def intensity_selec(struct,PL,f,theta,W,theta0):
    if struct == 'Gaussian':
        rho_l = (f*np.sin(theta0))
        sigma = 1/2*W
        norm = PL/( 2*np.pi*(sigma**2)*(1-Gaussian(rho_l,sigma)))
        rho = (f*np.sin(theta))
        dP1 = norm*Gaussian(rho,sigma)
    elif struct == 'Bessel':
        rho = f*np.sin(theta)
        int = integrate.quad(closed_Bessel, 0, theta0, args=(W,f))[0]
        norm = PL/(2*np.pi* int)
        dP1 = norm*Bessel(rho,W)**2
    else:
        Raise('Wrong selected structure, available: "Gaussian" or "Bessel"')
    return dP1



# 3D Integrands 
def F_Integrand(theta,phi,x,z,a,f,W,PL,theta0,n_t,n_i,struct):
    """
    Defines the numerical integrand of the total optical force
    : f_scatt + f_grad.
    theta: var theta @ integrate
    phi: var phi @ integrate
    x: r-position i-component
    z: r-position k-component 
    a: particle radius
    f: beam focal length
    w0: beam waist
    PI: beam power
    theta0: critical angle according to NA
    and r_index of the medium
    n_t: medium r_index
    n_i: particle r_index
    struct: laser intensity profile
    """
    c=3e8    # Light speed
    r = np.array([x,0,z])    # postion vector
    rmag,gamma = param(x,z)    # get position parameters
    o1 = np.sin(theta)*np.sin(gamma)*np.cos(phi)
    o2 = np.cos(theta)*np.cos(gamma)
    omega = (o1+o2)**2
    d = np.real(rmag*omega + np.emath.sqrt(a**2 -rmag**2+(rmag**2)*omega)) #magnitude of d-vector: d=a+r
    if d ==0:
        return np.array([0,0,0])    # Out of the beam contition
    else:
        arg = (d**2+a**2-rmag**2)/(2*a*d)
        if not(0 <= arg <=1):    # Exclude tangent rays 
            return np.array([0,0,0])
        else:
            alfa = np.arccos(arg) #incident angle
            if alfa == 0:
                return np.array([0,0,0])
            else:
                beta = snell(alfa,n_t,n_i) #refracted angle
                R = 1/2* ( (np.tan(alfa-beta)/np.tan(alfa+beta))**2 + (np.sin(alfa-beta)/np.sin(alfa+beta))**2)
                T = 1-R
                zp = np.array([-np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi),-np.cos(theta)]) #z-vector pointing towards the ray direction
                yp_mag = np.linalg.norm(np.cross(zp,np.cross(r,zp)))
                if yp_mag ==0:
                    yp = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta) ])
                else:
                    yp = np.cross(zp,np.cross(r,zp) )/ yp_mag #y-vector transversal to z-p
                Q = 1+ R*np.exp(1j*2*alfa) - (T**2)*( (np.exp(1j*2*(alfa-beta))+ R*np.exp(1j*2*alfa))/ (1+R**2 +2*R*np.cos(2*beta) ) ) #Ashkin efficiency expression
                # sigma = 1/2 * w0 #Gauss ray sigma
                # Pl = PI*(1-Gaussian((f**2)*(np.sin(theta0)**2),sigma))
                # Pl = PI*(1-np.exp((-f**2) * (np.sin(theta0)**2) / (2*sigma**2))) #Local power of the ray given theta0
                # dP1 = Pl*(f**2)*np.sin(theta)*np.cos(theta)
                # dP2 = Gaussian((f**2)*(np.sin(theta0)**2),sigma))
                # dP2 = np.exp((-f**2) * (np.sin(theta)**2) / (2*sigma**2) )
                # dP3 = 2*np.pi*sigma*(1- Gaussian(f**2*(np.sin(theta0)**2),sigma))
                # dP3 = 2*np.pi*sigma *(1- (np.exp( (-f**2)*(np.sin(theta0)**2) / (2*sigma**2))))
                dP1 = intensity_selec(struct,PL,f,theta,W,theta0)
                dP = dP1*(f**2)*np.sin(theta)*np.cos(theta)  #power differential
                dF = (n_i/c)*dP*((np.real(Q)*zp) + (np.imag(Q)*yp)) #vector of dF
                return dF

def F_Integrand_GRAD(theta,phi,x,z,a,f,W,PL,theta0,n_t,n_i,struct):
    """
    Defines the numerical integrand of the gradient optical force
    theta: var theta @ integrate
    phi: var phi @ integrate
    x: r-position i-component
    z: r-position k-component 
    a: particle radius
    f: beam focal length
    w0: beam waist
    PI: beam power
    theta0: critical angle according to NA
    and r_index of the medium
    n_t: medium r_index
    r_i: particle r_index
    """
    c=3e8
    r = np.array([x,0,z])
    rmag,gamma = param(x,z)
    o1 = np.sin(theta)*np.sin(gamma)*np.cos(phi)
    o2 = np.cos(theta)*np.cos(gamma)
    omega = (o1+o2)**2
    d = np.real(rmag*omega + np.emath.sqrt(a**2 -rmag**2+(rmag**2)*omega)) #magnitude of d-vector: d=a+r
    if d ==0:
        return np.array([0,0,0])
    else:
        arg = (d**2+a**2-rmag**2)/(2*a*d)
        if not(0 <= arg <=1):
            return np.array([0,0,0])
        else:
            alfa = np.arccos(arg) #incident angle
            if alfa == 0:
                return np.array([0,0,0])
            else:
                beta = snell(alfa,n_t,n_i) #refracted angle
                R = 1/2* ( (np.tan(alfa-beta)/np.tan(alfa+beta))**2 + (np.sin(alfa-beta)/np.sin(alfa+beta))**2)
                T = 1-R
                zp = np.array([-np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi),-np.cos(theta)]) #z-vector pointing towards the ray direction
                yp_mag = np.linalg.norm(np.cross(zp,np.cross(r,zp)))
                if yp_mag ==0:
                    yp = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta) ])
                else:
                    yp = np.cross(zp,np.cross(r,zp) )/ yp_mag #y-vector transversal to z-p
                Q = 1+ R*np.exp(1j*2*alfa) - (T**2)*( (np.exp(1j*2*(alfa-beta))+ R*np.exp(1j*2*alfa))/ (1+R**2 +2*R*np.cos(2*beta) ) ) #Ashkin efficiency expression
                # sigma = 1/2 * w0 #Gauss ray sigma
                # Pl = PI*(1-np.exp((-f**2) * (np.sin(theta0)**2) / (2*sigma**2))) #Local power of the ray given theta0
                # dP1 = Pl*(f**2)*np.sin(theta)*np.cos(theta)
                # dP2 = np.exp((-f**2) * (np.sin(theta)**2) / (2*sigma**2) )
                # dP3 = 2*np.pi*sigma *(1- (np.exp( (-f**2)*(np.sin(theta0)**2) / (2*sigma**2))))
                dP1 = intensity_selec(struct,PL,f,theta,W,theta0)
                dP = dP1*(f**2)*np.sin(theta)*np.cos(theta)  #power differential
                dF = (n_i/c)*dP*(np.imag(Q)*yp) #vector of dF gradient
                return dF

def F_Integrand_SCATT(theta,phi,x,z,a,f,W,PL,theta0,n_t,n_i,struct):
    """
    Defines the numerical integrand of the scattering optical force
    theta: var theta @ integrate
    phi: var phi @ integrate
    x: r-position i-component
    z: r-position k-component 
    a: particle radius
    f: beam focal length
    w0: beam waist
    PI: beam power
    theta0: critical angle according to NA
    and r_index of the medium
    n_t: medium r_index
    r_i: particle r_index
    """
    c=3e8
    r = np.array([x,0,z])
    rmag,gamma = param(x,z)
    o1 = np.sin(theta)*np.sin(gamma)*np.cos(phi)
    o2 = np.cos(theta)*np.cos(gamma)
    omega = (o1+o2)**2
    d = np.real(rmag*omega + np.emath.sqrt(a**2 -rmag**2+(rmag**2)*omega)) #magnitude of d-vector: d=a+r
    if d==0:
        return np.array([0,0,0])
    else:
        arg = (d**2+a**2-rmag**2)/(2*a*d)
        if not(0 <= arg <=1):
            return np.array([0,0,0])
        else:
            alfa = np.arccos(arg) #incident angle
            if alfa == 0:
                return np.array([0,0,0])
            else:
                beta = snell(alfa,n_t,n_i) #refracted angle
                R = 1/2* ( (np.tan(alfa-beta)/np.tan(alfa+beta))**2 + (np.sin(alfa-beta)/np.sin(alfa+beta))**2)
                T = 1-R
                zp = np.array([-np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi),-np.cos(theta)]) #z-vector pointing towards the ray direction
                yp_mag = np.linalg.norm(np.cross(zp,np.cross(r,zp)))
                if yp_mag ==0:
                    yp = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta) ])
                else:
                    yp = np.cross(zp,np.cross(r,zp) )/ yp_mag #y-vector transversal to z-p
                Q = 1+ R*np.exp(1j*2*alfa) - (T**2)*( (np.exp(1j*2*(alfa-beta))+ R*np.exp(1j*2*alfa))/ (1+R**2 +2*R*np.cos(2*beta) ) ) #Ashkin efficiency expression
                # sigma = 1/2 * w0 #Gauss ray sigma
                # Pl = PI*(1-np.exp((-f**2) * (np.sin(theta0)**2) / (2*sigma**2))) #Local power of the ray given theta0
                # dP1 = Pl*(f**2)*np.sin(theta)*np.cos(theta)
                # dP2 = np.exp((-f**2) * (np.sin(theta)**2) / (2*sigma**2) )
                # dP3 = 2*np.pi*sigma *(1- (np.exp( (-f**2)*(np.sin(theta0)**2) / (2*sigma**2))))
                # dP = dP1*dP2/dP3  #power differential
                dP1 = intensity_selec(struct,PL,f,theta,W,theta0)
                dP = dP1*(f**2)*np.sin(theta)*np.cos(theta)  #power differential
                dF = (n_i/c)*dP*(np.real(Q))*zp #vector of dF scatter
                return dF

