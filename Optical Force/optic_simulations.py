import struct
import numpy as np
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt

from optic_integrands import get_theta
from optic_integrands_components import F_INTEGRAND_X, F_INTEGRAND_Z
from optic_integrands_components import F_INTEGRAND_GRAD_X, F_INTEGRAND_GRAD_Z
from optic_integrands_components import F_INTEGRAND_SCATT_X, F_INTEGRAND_SCATT_Z

# Calculations and plotting
def optic_forces(r_i,r_f,**kwargs):
    """
    Creates a model dictionary where the simulation results,
    positions and parameters are stored. 
    r_i: initial position vector (iterable)
    r_f: final position vector (iterable)
    kwargs: {'a': p. radius, 'NA': numerical aperture,
    'R': lensa radius, 'n_o': objective r_index , 'n_t': particle r_index
    'n_i': medium r_index, 'w0': beam waist, 'PI': beam power,
    'n_points': size of samplings}
    """
    '''Data preparation'''
    #  Model dictionary will store the useful results
    model = {'positions':{},'results':{}}
    # The p dictionary will store and update the parameters according
    # to the user in **kwars
    p = {'a': 1e-6, 'NA': 1.25, 'R': 3.5e-3, 'n_o': 1.51, 'n_t': 1.5,
             'n_i': 1.33, 'W': 8e-3, 'PL': 30e-3, 'n_points': 500, 'structure':'Gaussian'}
    for key, value in kwargs.items():
        p[key] = value
    # Calculate the f-length and the critical theta according
    # to the given parameters
    p['f'] = p['n_o']*p['R']/p['NA']
    p['theta_crit'] = np.arcsin(p['NA']/p['n_o'])
    # Get the positions vectors
    n = p['n_points']
    z = np.linspace(r_i[2],r_f[2],n)
    x = np.linspace(r_i[0],r_f[0],n)
    # Store them in the model dictionary
    model['positions']['x'] = x
    model['positions']['z'] = z
    # Create the 3D-force results array
    F_tot, F_scatt, F_grad = np.zeros((2,n)),np.zeros((2,n)),np.zeros((2,n))
    '''Integration'''
    # thetacrit/k is the resolution to get the critical angle for each position
    k = 10000
    thetacrit = p['theta_crit']
    theta = np.linspace(0,thetacrit,k)   #theta vector
    phi = np.linspace(0,2*np.pi,k)    #phi vector
    # Get the parameters from dictionary
    a, f, W, PL, n_t, n_i, struc = p['a'], p['f'], p['W'], p['PL'], p['n_t'], p['n_i'], p['structure']
    for i in tqdm(range(n)):   # Loop within positions
        dz, dx = z[i], x[i]    # Position steps
        thetacrit_pos = get_theta(theta,phi,dx,dz,a)[-1]    # Allowed theta vector
        # Total force 3D integration
        F_tot[0][i] = integrate.nquad(F_INTEGRAND_X,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
        F_tot[1][i] = integrate.nquad(F_INTEGRAND_Z,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
        # Scattering force 3D integration
        F_scatt[0][i] = integrate.nquad(F_INTEGRAND_SCATT_X,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
        F_scatt[1][i] = integrate.nquad(F_INTEGRAND_SCATT_Z,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
        # Gradient force 3D integration
        F_grad[0][i] = integrate.nquad(F_INTEGRAND_GRAD_X,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
        F_grad[1][i] = integrate.nquad(F_INTEGRAND_GRAD_Z,
                                       [[0,thetacrit_pos],[0,np.pi*2]],
                                       args =(dx,dz,a,f,W,PL,thetacrit,n_t,n_i,struc))[0]
    # Save the integration results to the model
    model['results']['total'] = F_tot
    model['results']['scatt'] = F_scatt
    model['results']['grad'] = F_grad
    # Store the parameters used 
    model['parameters'] = p
    return model

def plot_forces(model, kind, inline,**kwargs):
    """
    Given model dictionary from optic_forces, this functions
    plots the result. Asks the axis where you want to plot
    the results.
    kind: 
        'single' - One axis plotted
        'stacked' - Two axis plotted
    inline:
        True - %matplotlib inline
        False - %matplotlib qt
    """
    # Load kwargs
    parameters = {'figsize': (15,8), 't_color': 'black', 't_ticks': '.-',
    'g_color': 'blue', 'g_ticks': '.-', 's_color': 'red', 't_ticks': '.-',  }
    # Unpack results & positions
    z, x = model['positions']['z'], model['positions']['x']
    f_tot = model['results']['total']
    f_scatt = model['results']['scatt']
    f_grad = model['results']['grad']
    a = model['parameters']['a']
    # Plot 
    label = ['x','z']
    if inline == True:
        get_ipython().run_line_magic('matplotlib', 'inline')
    else:
        get_ipython().run_line_magic('matplotlib', 'qt')
    if kind == 'single':
        fig, ax = plt.subplots(1,1,figsize=(15,8))
        selec = input('Single: moving axis?')
        if selec == 'x':
            axis_ = 0
            position = x*1e6
        elif selec == 'z':
            axis_ = 1
            position = z*1e6
        scatlab = 'F_s' + '_' + label[axis_] 
        gradlab = 'F_g' + '_'  + label[axis_] 
        totlab = 'F_tot' + '_'  + label[axis_]
        ax.plot(-position,-f_tot[axis_]*1e12,'.-',c='black',label = f'{totlab}')
        ax.plot(-position,-f_scatt[axis_]*1e12,'.-',c='red',label = f'{scatlab}')
        ax.plot(-position,-f_grad[axis_]*1e12,'.-',c='blue',label = f'{gradlab}')
        ax.set_title(f'Componente {label[axis_]} de las fuerzas como función de la posición de la partícula en el eje z ',fontsize=15)
        ax.set_xlabel(f'Distancia longitudinal {label[axis_]}'+r'$\mu m$')
        ax.set_ylabel('Fuerza [pN]')
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.yaxis.label.set_size(15)
        ax.xaxis.label.set_size(15)
        ax.grid()
        ax.legend()
        plt.show()
    elif kind == 'stacked':
        fig, axes = plt.subplots(2,1,figsize=(15,20))
        selec = input('Stacked: moving axis?')
        if selec == 'x':
            position = x*1e6
        elif selec == 'z':
            position = z*1e6
        for i in range(2):
            ax = axes[i]
            scatlab = 'F_s' + '_' + label[i] 
            gradlab = 'F_g' + '_'  + label[i] 
            totlab = 'F_tot' + '_'  + label[i]
            ax.plot(position,f_tot[i]*1e12,'.-',c='black',label = f'{totlab}')
            ax.plot(position,f_scatt[i]*1e12,'.-',c='red',label = f'{scatlab}')
            ax.plot(position,f_grad[i]*1e12,'.-',c='blue',label = f'{gradlab}')
            ax.set_title(f'Componente {label[i]} de las fuerzas como función de la posición de la partícula en el eje z ',fontsize=15)
            ax.set_xlabel(f'Distancia longitudinal {selec} '+r'$\mu m$')
            ax.set_ylabel('Fuerza [pN]')
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.yaxis.label.set_size(15)
            ax.xaxis.label.set_size(15)
            ax.grid()
            ax.legend()
        plt.show()