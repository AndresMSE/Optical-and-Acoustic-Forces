from ast import Raise
import itertools
from tarfile import USTAR_FORMAT
from urllib.parse import uses_relative
from constraint import Variable
import numpy as np
from pandas import NA
from pyrsistent import v
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
import multiprocessing
import os
import json
from mycolorpy import colorlist as mcp

from optic_integrands import get_theta, param
from optic_integrands_components import F_INTEGRAND_X, F_INTEGRAND_Z
from optic_integrands_components import F_INTEGRAND_GRAD_X, F_INTEGRAND_GRAD_Z
from optic_integrands_components import F_INTEGRAND_SCATT_X, F_INTEGRAND_SCATT_Z
from optic_simulations import plot_forces

from sklearn.linear_model import LinearRegression

class OpticRTX:
    
    def __init__(self, variable,structure='Gaussian',aberration=False, **kwargs):
        """
        The object instance is created. 
        
        The type of experimental variable and the model parameters
        are selected choosing between particle material and size,
        type of structured light and optical aberration.
        Parameters
        ----------- 
        variable(str) := 'radii', 'NA', 'material','medium', 'waist'
                            variables for experimentation
        structure(str) := 'Gaussian', 'Bessel
                            structure of light
        aberration(boolean) := If True, the model takes
                            account for optical aberration
        kwargs (dict) := {'a': p. radius [m], 'NA': numerical aperture,
                'R': lensa radius [m], 'n_o': objective r_index , 'n_t': particle r_index
                'n_i': medium r_index, 'W': beam waist/bessel radius [m], 'PL': beam power [W],
                'n_points': size of samplings}
        """
        self.variable = variable
        self.structure = structure
        self.aberration = aberration
        self.materials = {'latex':1.5905, 'silica':1.4585, 'glass': 1.5, 'oil':1.496, 'liver': 1.369,'poly':1.59}
        self.materials_density = {'latex': 1050, 'silica': 2000, 'glass': 1040, 'oil':1020, 'liver': 1070, 'poly':1065}    # kg/m3
        self.mediums = {'water': 1.333, 'air': 1, 'glycerol': 1.4729,'PBS':1.3295}
        self.mediums_density = {'water': 1000, 'air': 1.225, 'glycerol': 1260, 'PBS':1062}
        self.param = {'a': 1e-6, 'NA': 1.25, 'R': 3.5e-3, 'n_o': 1.51, 'material': 'glass','medium': 'water',  'lambda': 532, 'W': 8e-3, 'PL': 100e-3, 'n_points': 500}

        # The p dictionary will store and update the parameters according
        # to the user in **kwars
        for key, value in kwargs.items():
            if key in list(self.param.keys()):
                self.param[key] = value    # Update the simulation parameters
                # raise KeyError('Select a valid parameter. The allowed parameters are: a, NA, R, etc.')
            else:
                raise KeyError('Select a valid parameter. The allowed parameters are: a, NA, R, etc.')
        # Get the refraction index given the material and the medium
        
        if self.param['material'] in self.materials:
            mat = self.param['material']
            self.param['n_t'] = self.materials[mat]
        else:
            raise KeyError('Select a valid material. The allowed materials are: latex,sillica, glass, oil or liver')
        if self.param['medium'] in self.mediums:
            med = self.param['medium']
            self.param['n_i'] = self.mediums[med]
        else:
            raise KeyError('Select a valid medium. The allowed mediyms are: water, air, glycerol')
        # Calculate the f-length and the critical theta according
        # to the given parameters
        self.param['f'] = self.param['n_o']*self.param['R']/self.param['NA']
        self.param['theta_crit'] = np.arcsin(self.param['NA']/self.param['n_o'])
        print('RTXModel_structure-{st}_aberration-{ab}'.format(
            st = structure, ab = aberration
        ))
        # print("File __name__ is set to: {}" .format(__name__))
    # Calculations and plotting
    def optic_forces(self,r_i,r_f,**kwargs):
        """
        Creates a model dictionary where the simulation results,
        positions and parameters are stored. 
        Parameters
        -----------
        r_i: initial position vector (iterable)
        r_f: final position vector (iterable)
        kwargs: {'a': p. radius, 'NA': numerical aperture,
        'R': lensa radius, 'n_o': objective r_index , 'n_t': particle r_index
        'n_i': medium r_index, 'w0': beam waist, 'PI': beam power,
        'n_points': size of samplings}
        """
        self.r_i = r_i
        self.r_f = r_f
        '''Data preparation'''
        #  Model dictionary will store the useful results
        model = {'positions':{},'results':{}}
        # The p dictionary will store and update the parameters according
        # to the user in **kwars
        p = self.param

        for key, value in kwargs.items():
            if key in list(p.keys()):
                p[key] = value    # Update the simulation parameters
                # raise KeyError('Select a valid parameter. The allowed parameters are: a, NA, R, etc.')
            else:
                raise KeyError('Select a valid parameter. The allowed parameters are: a, NA, R, etc.')
        # Get the refraction index given the material and the medium
        if p['material'] in self.materials:
        # if self.param['material'] == mat:
            mat = p['material']
            p['n_t'] = self.materials[mat]
        else:
            raise KeyError('Select a valid material. The allowed materials are: latex,sillica, glass, oil or liver')
        if p['medium'] in self.mediums:
            med = p['medium']
            p['n_i'] = self.mediums[med]
        else:
            raise KeyError('Select a valid medium. The allowed mediyms are: water, air, glycerol')
        # Calculate the f-length and the critical theta according
        # to the given parameters
        p['f'] = p['n_o']*p['R']/p['NA']
        p['theta_crit'] = np.arcsin(p['NA']/p['n_o'])
        self.__dict__.update(p)  # Update class attributes with the parameters
        # print(p)
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
        a, f, W, PL, n_t, n_i,struc = p['a'], p['f'], p['W'], p['PL'], p['n_t'], p['n_i'], self.structure
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
        self.model = model
        return model
    def select_model(self, mode):
        
        if mode == 'auto':
            model = self.model
        elif mode == 'parallel':
            key = float(input('Selected variable from {} simulation ? '.format(self.variable)))
            model = self.return_dict[key]
        else:
            raise TypeError('Mode can only be "auto" or "parallel')
        return model

    def plot_forces(self, mode, kind, inline,**kwargs):
        """
        Given model dictionary from optic_forces, this functions
        plots the result. Asks the axis where you want to plot
        the results.
        Parameters
        -----------
        mode(str) := 
            'auto' - The functions retrieves the model dictionary generated by optic_forces()
            'parellel' - A model dictionary generated by simulation() is used, a key needs to be selected
        kind(str) := 
            'single' - One axis plotted
            'stacked' - Two axis plotted
        inline (boolean) :=
            True - %matplotlib inline
            False - %matplotlib qt
        """
        model = self.select_model(mode)
        # Load & update kwargs
        parameters = {'figsize': (15,8), 't_color': 'black', 't_ticks': '.-',
        'g_color': 'blue', 'g_ticks': '.-', 's_color': 'red', 't_ticks': '.-',  }
        for key, value in kwargs.items():
            parameters[key] = value
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
            fig, ax = plt.subplots(1,1,figsize=parameters['figsize'])
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
            ax.set_title(f'Componente {label[axis_]} de las fuerzas como función de la posición de la partícula en el eje {selec}. Haz {self.structure}.',fontsize=15)
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
                ax.set_title(f'Componente {label[i]} de las fuerzas como función de la posición de la partícula en el eje {selec}. Haz {self.structure}.',fontsize=15)
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

    def parallel_integration(self,var_val,z_eq):
        print(os.getpid())
        c = z_eq
        if self.variable == 'radii':
            rad = var_val
        else:
            rad = self.param['a']
        if self.disp_mode == 'times':
            dist = self.times
            axis = self.axis_
            eq = self.offset
            if eq == 'yes':
                if axis == 'x':
                    r_i = [-dist*rad,0,c*rad]
                    r_f = [dist*rad,0,c*rad]
                elif axis == 'z':
                    r_i = [0,0,-dist*rad]
                    r_f = [0,0,dist*rad]
                elif axis == 'xz':
                    r_i = [-dist*rad,0,-dist*rad+c*rad]
                    r_f = [dist*rad,0,dist*rad+c*rad]
            elif eq == 'no':
                if axis == 'x':
                    r_i = [-dist*rad,0,0]
                    r_f = [dist*rad,0,0]
                elif axis == 'z':
                    r_i = [0,0,-dist*rad]
                    r_f = [0,0,dist*rad]
                elif axis == 'xz':
                    r_i = [-dist*rad,0,-dist*rad]
                    r_f = [dist*rad,0,dist*rad]
        elif self.disp_mode == 'manual':
            dist = self.movement_lim
            axis = self.axis_
            eq = self.offset
            if eq == 'yes':
                if axis == 'x':
                    r_i = [-dist,0,c*rad]
                    r_f = [dist,0,c*rad]
                elif axis == 'z':
                    r_i = [0,0,-dist]
                    r_f = [0,0,dist]
                elif axis == 'xz':
                    r_i = [-dist,0,-dist+c]
                    r_f = [dist,0,dist+c]
            elif eq == 'no':
                if axis == 'x':
                    r_i = [-dist,0,0]
                    r_f = [dist,0,0]
                elif axis == 'z':
                    r_i = [0,0,-dist]
                    r_f = [0,0,dist]
                elif axis == 'xz':
                    r_i = [-dist,0,-dist]
                    r_f = [dist,0,dist]
        res = self.sim_selector(r_i, r_f, var_val)
        self.return_dict[var_val] = res
        pass
    
    def usr_query (self):
        """
        Given the selected experimental variable, the user is 
        asked with the experimental parameters.
        """
        if self.simulation_mode == 'input-manual':
            self.input_var_array = input('Enter variable values separated by commma')
        else:
            self.var_in = float(input('Initial {} ?'.format(self.variable)))
            self.var_fi = float(input('Final {} ?'.format(self.variable)))
            self.var_delta = float(input('{} delta ?'.format(self.variable)))
        self.disp_mode = input('Particle movement mode (times/manual)?')
        if self.disp_mode == 'manual':
            self.movement_lim = float(input('Enter min/max value for displacement (um)...'))*1e-6
        else:
            self.times = int(input('How many radii should the particle move?'))
        self.axis_ = input('Where is the particle moving (x, z, xz) ?')
        self.offset = input('Start at z eq. (yes/no) ?')
        if self.offset == 'yes':
            input_aray = input('Enter known equilibrium positions array [um] separated by comma:')
            self.offset_array_eq = np.array([float(i) for  i in input_aray.split(',')])*1e-6
        self.cust = input('Custom name for file (yes/no)?')
        if self.cust == 'yes':
            self.cust_name = input('Type custom filename... ')
        else:
            self.name = input('Additional name for file (yes/no) ?')
            if self.name == 'yes':
                self.ow = input('Additional name?')
                self.ow = '_'+self.ow
            else:
                self.ow = ''
    
    def manual_query (self, **kwargs):
        """
        Given the selected experimental variable, the user
        gives the the experimental parameters.
        input_var_array : = Variable values separated by commma (str)
        movement_lim : = min/max value for displacement (um)
        axis_ : particle moving axis (x, z, xz) 
        offset : = particle offset to the equilibrium plane
        eq_array : = known eq positions separated by comma (str)
        cust : = apply custom name to saved file (yes/no)
        cust_name := custom name 
        name : = add additional name (not used)
        add_name := (not used)

        """
        p = {'input_var_array': '','movement_lim': 10,'axis_': 'z','offset': 'no' ,'eq_array': '','cust': 'no','cust_name': '','name': 'no','add_name':'' }
        for key,value in kwargs.items():
            p[key] = value
        if self.simulation_mode == 'manual':
            self.input_var_array = p['input_var_array']
        else:
            Raise ('Only manual mode supported')
        self.disp_mode = 'manual'
        self.movement_lim = float(p['movement_lim'])*1e-6
        self.axis_ = p['axis_']
        self.offset = p['offset']
        if self.offset == 'yes':
            input_aray = p['eq_array']
            self.offset_array_eq = np.array([float(i) for  i in input_aray.split(',')])*1e-6
        self.cust = p['cust']
        if self.cust == p['cust']:
            self.cust_name = p['cust_name']
        else:
            self.name = p['name']
            if self.name == 'yes':
                self.ow = p['add_name']
                self.ow = '_'+self.ow
            else:
                self.ow = ''

        pass
    
    def sim_generation(self):
        """
        Using the selected experimental parameters, the array of 
        values for the variable are created accordingly
        """
        def eq_positoins_verf():
            if self.offset == 'yes':
                n = len(self.array)
                try:
                    m = len(self.offset_array_eq)
                    if n != m:
                        raise ValueError("The number of eq positions doesn't match the number of simulations")
                    else:
                        pass
                except NameError:
                    self.offset_array_eq = np.zeros(n)
                pass
        if self.variable == 'radii':
            if self.simulation_mode != 'auto':
                self.array = np.array([float(i) for i in self.input_var_array.split(',')])*1e-6
            else:
                self.array = np.arange(self.var_in,self.var_fi,self.var_delta)*1e-6
            eq_positoins_verf()
        elif self.variable == 'NA':
            if self.simulation_mode != 'auto':
                self.array = np.array([float(i) for i in self.input_var_array.split(',')])
            else:
                self.array = np.arange(self.var_in,self.var_fi,self.var_delta)
            eq_positoins_verf()
        elif self.variable == 'material':
            self.array = list(self.materials.keys())
            eq_positoins_verf()
        elif self.variable == 'medium':
            self.array = list(self.mediums.keys())
            eq_positoins_verf()
        elif self.variable == 'waist':
            if self.structure == 'Bessel':
                if self.simulation_mode != 'auto':
                    self.array = np.array([float(i) for i in self.input_var_array.split(',')])*1e-3
                else:
                    self.array = np.arange(self.var_in,self.var_fi,self.var_delta)*1e-3
                eq_positoins_verf()
            elif self.structure == 'Gaussian':
                if self.simulation_mode != 'auto':
                    self.array = (np.array([float(i) for i in self.input_var_array.split(',')])/2)*np.sqrt(2)*1e-3
                else:
                    self.array = (np.arange(self.var_in,self.var_fi,self.var_delta)/2)*np.sqrt(2)*1e-3
                eq_positoins_verf()
        pass
    
    def sim_selector(self,r_i,r_f,var_val):
        """
        Using the selected experimental parameters, the according variable is selected
        """
        if self.variable == 'radii':
            res = self.optic_forces(r_i,r_f,a=var_val)
        elif self.variable == 'NA':
            res = self.optic_forces(r_i,r_f,NA=var_val)
        elif self.variable == 'material':
            res = self.optic_forces(r_i,r_f,material=var_val)
        elif self.variable == 'medium':
            res = self.optic_forces(r_i,r_f,medium=var_val)
        elif self.variable == 'waist':
            res = self.optic_forces(r_i,r_f,W=var_val)
        return res

    def simulation(self,mode='auto',**kwargs):
        """
        Runs a parallelized simulation for the experimental variables selected
        """
        if __name__ == 'rtx_model':
            # File charge
            manager = multiprocessing.Manager()
            self.return_dict = manager.dict()
            process_list = []
            cores = os.cpu_count()
            print('No. cores:',cores)
            print('Creating job instances')
            # self.r = float(input('Particle radius (um) ?'))*1e-6
            self.simulation_mode = mode
            if self.simulation_mode == 'auto' or self.simulation_mode == 'input-manual':
                self.usr_query()
            elif self.simulation_mode == 'manual':
                self.manual_query(**kwargs)
            self.sim_generation()
            print(f'\nSimulation started...') 
            print('Selected parameters...')
            if mode == 'manual':
                print('Manual variables selected...')
            else:
                print('{v} [{v_in}, {v_fi}] \nMoving along: {a}'.format( v = self.variable, v_in = self.var_in, v_fi = self.var_fi, a = self.axis_))
            for i in range(len(self.array)):
                item = self.array[i]
                if self.offset == 'yes':
                    z_eq = self.offset_array_eq[i]
                elif self.offset == 'no':
                    z_eq = 0
                process = multiprocessing.Process(target=self.parallel_integration, args=(item,z_eq,))
                process_list.append(process)
            print('Execute jobs')
            for p in process_list:
                p.start()

            print('Working....')
            for p in process_list:
                p.join()
            print(('Done!'))

            simulation_results = self.return_dict.copy()
            self.simulation_gen = self.return_dict.copy()
            for item, res in simulation_results.items():
                for position, axis in res['positions'].items():
                    res['positions'][position] = axis.tolist()
                for force, values in res['results'].items():
                    res['results'][force] = values.tolist()
            if self.cust == 'yes':
                self.filename = self.cust_name + '.json'
            else:
                if self.mode == 'auto':
                    self.filename = f'{self.variable}_{self.structure}-simulations_m-{self.axis_}_p-({self.var_in},{self.var_fi})_eq-{self.offset}'
                    self.filename = self.filename  +self.ow + '.json'
                else:
                    self.filename = f'{self.variable}_{self.structure}-simulations_m-{self.axis_}_p-({self.input_var_array})_eq-{self.offset}'
                    self.filename = self.filename  +self.ow + '.json' 
            with open(self.filename,'w') as f:
                json.dump(simulation_results,f)
            print(f'Simulation for {self.variable}:', self.return_dict.keys())
            print('Saved as:',self.filename)
        pass
    
    def load_simulation(self, filename):
        """
        If needed this object can load previous simulations via json unpacking
        Parameters
        ----------
        filename(str) := 
                        Filename or sys path to json file storing dictionary of simulations
        """
        with open(filename) as json_file:
            load = json.load(json_file)
        for pos, res in load.items():
            for position, axis in res['positions'].items():
                res['positions'][position] = np.array(axis)
            for force, values in res['results'].items():
                res['results'][force] = np.array(values)
        self.simulation_load = load
        pass

    def plot(self,sim,var,**kwargs):
        """
        Inner callback for plotting
        """
        # Load & update kwargs
        parameters = {'figsize': (15,8),'force':'total','component':1,'axis':'z','weight':False, 'linest': '-' , 'verbose': 0, 'colormap': 'Spectral'}
        for key, value in kwargs.items():
            parameters[key] = value
        # Adjust model laser beam orientation. 'up' the beam propagates donwards, 'down' the beam propagates upwards. Both on a standar 3D xyz reference frame
        if self.orientation == 'up':
            sign = 1
        elif self.orientation == 'down':
            sign = -1
        else: 
            sign = 1
        # Load kwargs parameters
        force,component,ax,weight,colors = parameters['force'],parameters['component'],parameters['axis'],parameters['weight'], parameters['colormap']
        # Set the colormapping for the simulations
        n = len(sim.keys())
        colorm = mcp.gen_color(colors,n=n)
        # Reorder the simulation dictionaries 
        selected_sim = {}
        for key in var:
            selected_sim[key] = sim[key]
        # Plotting
        if weight == False:
            fig, ax_1 = plt.subplots(1,1,figsize=parameters['figsize'])
            i = 0
            # Iterations over the simulation sictionary items
            for var, resul in selected_sim.items():
                if parameters['verbose'] == 1 or parameters['verbose'] == 2 : print(resul['parameters'])
                power = resul['parameters']['PL']
                x = resul['positions'][ax]
                y = resul['results'][force][component]
                if not(self.variable == 'material' or self.variable == 'medium'):
                    if self.variable=='waist':
                        if self.structure=='Bessel':
                            v = r'$\rho_B$ [cm]'
                        else:
                            v = r'$W_0$ [cm]'
                        v_var = np.round(float(var)*1e3,3)
                    elif self.variable=='radii':
                        v= r'$a$ [$\mu m$]'
                        v_var = np.round(float(var)*1e6,1)
                    else:
                        v= self.variable
                        v_var = np.round(float(var),6)

                    ax_1.plot(sign*x*1e6,sign*y*1e12/power,label='{v}: {val}'.format(v=v, val = v_var), ls=parameters['linest'],c=colorm[i])
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 :print('F Boyant [pN]',B*1e12)
                    plt.legend()
                    plt.grid()
                else:
                    ax_1.plot(sign*x*1e6,sign*y*1e12/power,label='{v}: {val}'.format(v=self.variable, val=var),ls=parameters['linest'],c=colorm[i])
                    if parameters['verbose'] == 2 : print('Weight [pN]',W)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B)
                    plt.legend()
                    plt.grid()
                ax_1.set_title('{f} force - multiple {v} - {b}'.format( f=force, v=self.variable, b=self.structure))
                ax_1.set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                ax_1.set_ylabel('{f} {c}-Force [pN/mW]'.format(f=force, c=component))
                i +=1                       
        else:
            # Ordering system for the subplots axes index
            n_rows = len(sim.items())//2
            if len(sim.items())%2 ==1:
                n_rows += 1
            fig, ax_ = plt.subplots(n_rows,2,figsize=parameters['figsize'])
            i = 0
            n_int = 0            
            for var, resul in selected_sim.items():
                even = (i-1%n_rows+1)%2
                if even == 0:
                    c = (i+n_int)%n_rows
                    c_n = c
                elif even ==1:
                    c =c_n
                    n_int-=1
                if parameters['verbose'] == 1 or parameters['verbose'] == 2 : print(resul['parameters'])
                power = resul['parameters']['PL']*1e3
                x = resul['positions'][ax]
                y = resul['results'][force][component]
                r = resul['parameters']['medium']
                a = resul['parameters']['a']
                rho = self.mediums_density[r]
                B = ((4/3)*np.pi*(a**3)*9.81*rho)
                # if sign == -1:
                #     B = -B
                r = resul['parameters']['material']
                a = resul['parameters']['a']
                rho = self.materials_density[r]
                W = ((4/3)*np.pi*(a**3)*9.81*rho)
                # if sign == -1:
                #     W = -W
                W_rel = B-W   #[N]
                F_hat = abs(y/power).max()    #[N/mW]
                P_min = abs(W_rel)/F_hat    #[mW]
                print('The minimum power to trap the particle is: {p} [mW]. Given {v} = {val}'.format(p=P_min, v = self.variable, val = var))

                if not(self.variable == 'material' or self.variable == 'medium'):
                    # fig, ax_ = plt.subplots(1,1,figsize=parameters['figsize'])
                    ax_[c][even].plot(sign*x*1e6,(sign*y+W_rel)*1e12,label='{v}: {val}'.format(v=self.variable, val = np.round(float(var),6)),ls=parameters['linest'],c=colorm[i])
                    ax_[c][even].legend()
                    ax_[c][even].grid()
                    ax_[c][even].set_title('{f} force - multiple {v} - {b}'.format( f=force, v=self.variable, b=self.structure))
                    ax_[c][even].set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                    ax_[c][even].set_ylabel('{f} {c}-Force [pN]'.format(f=force, c=component))
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B*1e12)                   
                else:
                    # fig, ax_ = plt.subplots(1,1,figsize=parameters['figsize'])
                    ax_[c][even].plot(sign*x*1e6,(sign*y+W_rel)*1e12,label='{v}: {val}'.format(v=self.variable, val = var),ls=parameters['linest'],c=colorm[i])
                    ax_[c][even].legend()
                    ax_[c][even].grid()
                    ax_[c][even].set_title('{f} force - multiple {v} - {b}'.format( f=force, v=self.variable, b=self.structure))
                    ax_[c][even].set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                    ax_[c][even].set_ylabel('{f} {c}-Force [pN/mW]'.format(f=force, c=component))
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B*1e12)
                i +=1                        
    
    def plot_multiple(self,sim,var,figure,**kwargs):
        """
        Inner callback for multiple plots provided some axis objects
        """
        # Load & update kwargs
        parameters = {'figsize': (15,8),'force':'total','component':1,'axis':'z','weight':False, 'linest': '-' , 'verbose': 0, 'colormap': 'Spectral'}
        for key, value in kwargs.items():
            parameters[key] = value
        # Adjust model laser beam orientation. 'up' the beam propagates donwards, 'down' the beam propagates upwards. Both on a standar 3D xyz reference frame
        if self.orientation == 'up':
            sign = 1
        elif self.orientation == 'down':
            sign = -1
        else: 
            sign = 1
        force,component,ax,weight,colors = parameters['force'],parameters['component'],parameters['axis'],parameters['weight'], parameters['colormap']
        n = len(sim.keys())
        colorm = mcp.gen_color(colors,n=n)
        selected_sim = {}
        for key in var:
            selected_sim[key] = sim[key]
        if weight == False:
            plt.figure(figure, figsize=parameters['figsize'])
            i = 0
            for var, resul in selected_sim.items():
                if parameters['verbose'] == 1 or parameters['verbose'] == 2 : print(resul['parameters'])
                power = resul['parameters']['PL']
                x = resul['positions'][ax]
                y = resul['results'][force][component]
                if not(self.variable == 'material' or self.variable == 'medium'):
                    plt.plot(sign*x*1e6,sign*y*1e12/power,label='{v} {s}: {val}'.format(s= self.structure[0],v=self.variable, val = np.round(float(var),6)),
                     ls=parameters['linest'],c=colorm[i])
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 :print('F Boyant [pN]',B*1e12)
                    plt.legend()
                    plt.grid()
                else:
                    plt.plot(sign*x*1e6,sign*y*1e12/power,label='{v}: {val}'.format(v=self.variable, val=var),ls=parameters['linest'],c=colorm[i])
                    if parameters['verbose'] == 2 : print('Weight [pN]',W)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B)
                    plt.legend()
                    plt.grid()
                plt.title('{f} force - multiple {v}'.format( f=force, v=self.variable, b=self.structure))
                plt.xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                plt.ylabel('{f} {c}-Force [pN/mW]'.format(f=force, c=component))
                i +=1                       
        else:
            n_rows = len(sim.items())//2
            if len(sim.items())%2 ==1:
                n_rows += 1
            fig =plt.figure(figure)
            i = 0
            n_int = 0            
            for var, resul in selected_sim.items():
                if parameters['verbose'] == 1 or parameters['verbose'] == 2 : print(resul['parameters'])
                power = resul['parameters']['PL']*1e3
                x = resul['positions'][ax]
                y = resul['results'][force][component]
                r = resul['parameters']['medium']
                a = resul['parameters']['a']
                rho = self.mediums_density[r]
                B = ((4/3)*np.pi*(a**3)*9.81*rho)
                r = resul['parameters']['material']
                a = resul['parameters']['a']
                rho = self.materials_density[r]
                W = ((4/3)*np.pi*(a**3)*9.81*rho)
                W_rel = B-W   #[N]
                F_hat = abs(y/power).max()    #[N/mW]
                P_min = abs(W_rel)/F_hat    #[mW]
                print('The minimum power to trap the particle is: {p} [mW]. Given {v} {s}= {val}'.format(s= self.structure[0], p=P_min, v = self.variable, val = var))

                if not(self.variable == 'material' or self.variable == 'medium'):
                    if parameters['subplot_axis'] >0:
                        j = parameters['subplot_axis']
                        plt_axis = plt.subplot(n_rows,2,j)
                    else:
                        j = i+1
                        plt_axis = fig.add_subplot(n_rows,2,j)
                    plt_axis.plot(sign*x*1e6,(sign*y+W_rel)*1e12,label='{v} {s}: {val}'.format(s= self.structure, v=self.variable, val = np.round(float(var),6)),ls=parameters['linest'],c=colorm[i])
                    plt_axis.legend()
                    plt_axis.grid()
                    plt_axis.set_title('{f} force - multiple {v} '.format ( f=force, v=self.variable, b=self.structure))
                    plt_axis.set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                    plt_axis.set_ylabel('{f} {c}-Force [pN]'.format(f=force, c=component))
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B*1e12)                   
                else:
                    plt_axis = fig.add_subplot(n_rows,2,i+1)
                    plt_axis.plot(sign*x*1e6,(sign*y+W_rel)*1e12,label='{v}: {val}'.format(v=self.variable, val = var),ls=parameters['linest'],c=colorm[i])
                    plt_axis.legend()
                    plt_axis.grid()
                    plt_axis.set_title('{f} force - multiple {v} - {b}'.format( f=force, v=self.variable, b=self.structure))
                    plt_axis.set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
                    plt_axis.set_ylabel('{f} {c}-Force [pN/mW]'.format(f=force, c=component))
                    if parameters['verbose'] == 2 : print('Weight [pN]',W*1e12)
                    if parameters['verbose'] == 2 : print('F Boyant [pN]',B*1e12)
                i +=1                        
           
    def plot_multiple_subplots(self,sim,var,axes,**kwargs):
        """
        Inner callback for platting multiple results according to a set of subplots
        """
        # Load & update kwargs
        parameters = {'figsize': (15,8),'force':'total','component':1,'axis':'z','weight':False, 'linest': '-' , 'verbose': 0, 'colormap': 'Spectral', 
        't_param_1':'a', 't_param_2': 'W'}
        for key, value in kwargs.items():
            parameters[key] = value
        # Adjust model laser beam orientation. 'up' the beam propagates donwards, 'down' the beam propagates upwards. Both on a standar 3D xyz reference frame
        if self.orientation == 'up':
            sign = 1
        elif self.orientation == 'down':
            sign = -1
        else: 
            sign = 1
        force,component,ax,weight,colors = parameters['force'],parameters['component'],parameters['axis'],parameters['weight'], parameters['colormap']
        n = len(sim.keys())
        colorm = mcp.gen_color(colors,n=n)
        selected_sim = {}
        for key in var:
            selected_sim[key] = sim[key]        
        i = 0
        ## create a subfunction that takes the loaded simulation name and expresses the desired parameter in the grid
        ## def(string_name.json) return NA value WB value etc.
        for var, resul in selected_sim.items():
            if parameters['verbose'] == 1 or parameters['verbose'] == 2 : print(resul['parameters'])
            power = resul['parameters']['PL']
            x = resul['positions'][ax]
            y = resul['results'][force][component]
            def fancy_variable(param_name, value):
                if param_name == 'a':
                    return r'$a$',np.round(value*1e6),r'[$\mu m$]'
                elif param_name == 'W':
                    if self.structure == 'Gaussian':
                        return r'$W_0$',np.round(value*1e3,2), '[cm]'
                    else:
                        return r'$\rho_B$',value*1e3,'[cm]'
                elif param_name == 'NA':
                    return 'NA',value,''
            if not(self.variable == 'material' or self.variable == 'medium'):
                axes.plot(sign*x*1e6,sign*y*1e12/power,label='{v} {s}: {val}'.format(s= self.structure[0],v=fancy_variable(self.variable,1)[0], val = fancy_variable(self.variable,float(var))[1]),
                    ls=parameters['linest'],c=colorm[i])
                axes.legend()
                axes.grid()
            else:
                axes.plot(sign*x*1e6,sign*y*1e12/power,label='{v}: {val}'.format(v=self.variable, val=var),ls=parameters['linest'],c=colorm[i])
                axes.legend()
                axes.grid()
            params = resul['parameters']
            t_param_1, t_param_2 = parameters['t_param_1'], parameters['t_param_2']
            n1,val1,unit1 = fancy_variable(t_param_1, params[t_param_1])
            n2,val2,unit2 = fancy_variable(t_param_2, params[t_param_2])
            axes.set_title('{t_1}: {p_t_1} {u1} - {t_2}: {p_t_2} {u2}'.format(t_1=n1, p_t_1=val1, u1=unit1, 
            t_2=n2, p_t_2=val2, u2=unit2, b=self.structure))
            #axes.set_title('{f} force - multiple {v}'.format( f=force[0].upper() +force[1:], v=self.variable, b=self.structure))
            axes.set_xlabel('{pos} position '.format(pos=ax)+f'[$\mu m$]')
            component_dict = {0:'x', 1:'z'}
            axes.set_ylabel('{f} {c}-Force [pN/mW]'.format(f=force[0].upper() +force[1:], c=component_dict[component]))
            i +=1                       

    def order_result(self,result):
            self.ordered_dict = {}
            for key in sorted(result.keys(), key=float):
                self.ordered_dict[key] = result[key]
            return self.ordered_dict

    def plot_simulation(self,orientation,figure=None, ax_=None, **kwargs):
        """
        User function for plotting the results of a generated of loaded simulation
        Parameters
        -----------
        orientation (str)   := 'up', 'down' 
                                orientation of the laser beam with respect the reference frame. Up the beam propagates downwards and vice versa
        figure (fig.obj)    := matplotlib.figure
                            matplotlib figure object for the artist
        ax_ (ax.obj)        := matplotlib.axis
                            matplotlib axis object for the artist    
        """
        # Load & update kwargs
        parameters = {'figsize': (15,8),'force':'total','component':1,'axis':'z','weight':False, 'linest':'-', 'type': 'total',
         'verbose': 0, 'colormap':'Spectral','variables':'all', 'subplot_axis': -1}
        for key, value in kwargs.items():
            parameters[key] = value
        self.orientation = orientation
        try:
            sim = self.order_result(self.simulation_gen)
            if parameters['variables'] == 'all':
                var = sim.keys()
            else:
                var = parameters['variables']
        except AttributeError:
            print('A simulation was not generated, checking if loaded simulation exists')
            try:
                sim = self.order_result(self.simulation_load)
                if parameters['variables'] == 'all':
                    var = sim.keys()
                else:
                    var = parameters['variables']
            except AttributeError:
                print('A simulation is needed it was neither generated or loaded, try generating a simulation or loading one.')
            else:
                if ax_ == None:
                    if figure == None:
                        self.plot(sim,var,**parameters)
                    else:
                        self.plot_multiple(sim,var,figure,**parameters)
                else:
                    self.plot_multiple_subplots(sim,var,ax_,**parameters)    
        else:
            if ax_ == None:
                if figure == None:
                    self.plot(sim,var,**parameters)
                else:
                    self.plot_multiple(sim,var,figure,**parameters)
            else:
                self.plot_multiple_subplots(sim,var,ax_,**parameters)

    def get_equilibrium(self,sim,axis,weight):
        """
        Inner function to calculate the equilibrium positions
        """
        def find_root(F):
            sign = np.sign(F)
            sign_change = ((np.roll(sign,1)-sign)!=0).astype(int)
            sign_change[0] = 0
            roots = np.where(sign_change==1)[0]
            return roots
        if axis == 1:
            z_eq = []
            z_eq_norm = []
            var = []
            for pos, res in sim.items():
                med = res['parameters']['medium']
                rho = self.mediums_density[med]
                a = res['parameters']['a']
                B = ((4/3)*np.pi*(a**3)*9.81*rho)
                mat = res['parameters']['material']
                rho = self.materials_density[mat]
                W = ((4/3)*np.pi*(a**3)*9.81*rho)
                W_rel = B-W
                if weight == True:
                    F_t = res['results']['total'][1] + W_rel
                else:
                    F_t = res['results']['total'][1]
                index = find_root(F_t)
                eq = res['positions']['z'][index]
                rad = res['parameters']['a']
                z_eq_norm.append(abs(eq/rad))
                var.append(pos)
                z_eq.append(eq)
            z_eq = np.array(z_eq)
            z_eq_norm = np.array(z_eq_norm)
            self.z_eq = z_eq
            self.z_eq_norm = z_eq_norm
            self.z_eq_central = []
            self.z_eq_norm_central = []
            for z in z_eq:
                try:
                    self.z_eq_central.append(abs(z).min()*1e6)
                except ValueError:
                    self.z_eq_central.append(np.NaN)
            for z in z_eq_norm:
                try:
                    self.z_eq_norm_central.append(abs(z).min())
                except ValueError:
                    self.z_eq_norm_central.append(np.NaN)
            # self.z_eq_central = np.array([abs(z).min() for z in z_eq])
            # self.z_eq_norm_central = np.array([z.min() for z in z_eq_norm])
            self.var_eq = var

        elif axis==0:
            x_eq = []
            x_eq_norm = []
            var = []
            for pos, res in sim.items():
                F_t = res['results']['total'][0]
                index = find_root(F_t)
                eq = res['positions']['z'][index]
                rad = res['parameters']['a']
                x_eq_norm.append(abs(eq/rad))
                var.append(pos)
                x_eq.append(eq)
            x_eq = np.array(x_eq)
            x_eq_norm = np.array(x_eq_norm)
            self.x_eq = x_eq
            self.x_eq_norm = x_eq_norm
            self.x_eq_central = []
            self.x_eq_norm_central = []
            for x in x_eq:
                try:
                    self.x_eq_central.append(abs(x).min()*1e6)
                except ValueError:
                    self.x_eq_central.append(np.NaN)
            for x in x_eq_norm:
                try:
                    self.x_eq_norm_central.append(abs(x).min())
                except ValueError:
                    self.x_eq_norm_central.append(np.NaN)
            # self.x_eq_central = np.array([abs(x).min() for x in x_eq])
            # self.x_eq_norm_central = np.array([x.min() for x in x_eq_norm])
            self.var_eq = var

        # if (self.variable =='radii') or (self.variable == 'NA'):
        #     x,y = var, self.z_eq_central
        #     lists = sorted(itertools.zip_longest(*[x,y]))
        #     self.var_eq_sort_1, self.z_eq_central_sort = list(itertools.zip_longest(*lists))

        #     x,y = var, self.z_eq_norm_central
        #     lists = sorted(itertools.zip_longest(*[x,y]))
        #     self.var_eq_sort_2, self.z_eq_norm_central_sort = list(itertools.zip_longest(*lists))
        # #     self.var_eq = sorted(var, key=lambda x: float(x))

        
        pass
    
    def plot_eq_points(self,sim,axis):
        """
        Inner function for plotting the calculated equilibrium positions of a simulation
        """
        plt.rcParams['figure.figsize'] = (4.0, 4.0)
        plt.rcParams['figure.dpi'] = (150)
        plt.ylabel(f'Z position $\mu m$')
        plt.xlabel('Simulation variables {}'.format(self.variable))
        plt.title('Equilibrium positions for various: {}'.format(self.variable))
        plt.grid()
        if self.variable == 'radii': 
            self.var_eq = np.array([float(i) for i in self.var_eq])*1e6
        elif self.variable == 'NA':
            self.var_eq = np.array([float(i) for i in self.var_eq])

        # if (self.variable =='radii') or (self.variable == 'NA'):
        #     plt.scatter(self.var_eq_sort_1,np.array(self.z_eq_central_sort)*1e6,marker='^',c='red')
        # else:
        if axis==1:
            plt.scatter(self.var_eq[~np.isnan(self.z_eq_central)],
            np.array(self.z_eq_central)[~np.isnan(self.z_eq_central)],marker='^',c='red')
            plt.ylabel(f'Z position $\mu m$')
            plt.show()

        elif axis==0:
            plt.scatter(self.var_eq[~np.isnan(self.x_eq_central)]
            ,np.array(self.x_eq_central)[~np.isnan(self.x_eq_central)],marker='^',c='red')
            plt.ylabel(f'X position $\mu m$')
            plt.show()
        plt.rcParams['figure.figsize'] = (4.0, 4.0)
        plt.rcParams['figure.dpi'] = (150)

        plt.xlabel('Simulation variables {}'.format(self.variable))
        plt.title(' Normalized Equilibrium positions for various: {}'.format(self.variable))
        plt.grid()
        # if (self.variable =='radii') or (self.variable == 'NA'):
        #     plt.scatter(self.var_eq_sort_2,np.array(self.z_eq_norm_central_sort),marker='^',c='red')
        # else:
        if axis==1:
            plt.scatter(self.var_eq[~np.isnan(self.z_eq_norm_central)],
            np.array(self.z_eq_norm_central)[~np.isnan(self.z_eq_norm_central)],marker='^',c='red')
            plt.ylabel(f'Z position z/a')
            plt.ylim((-max(self.z_eq_norm_central)*1.5,max(self.z_eq_norm_central)*2))
            plt.show()

        elif axis==0:
            plt.scatter(self.var_eq[~np.isnan(self.x_eq_norm_central)]
            ,np.array(self.x_eq_norm_central)[~np.isnan(self.x_eq_norm_central)],marker='^',c='red')
            plt.ylabel(f'X position x/a')
            plt.ylim((-max(self.x_eq_norm_central)*1.5,max(self.x_eq_norm_central)*1.5))
            plt.show()
        pass
                
    def equilibrium(self, axis=1,weight=True):
        """
        User function for calculating the equilibrium positions of a simulation
        Parameters
        -----------
        axis (int)      :=  0,1
                        axis of the deisred equilibrium points x:0, z=1
        weight (bool)   :=  True/False
                        Flag for considering the relative weight of the particle in the selected host medium
        """
        try:
            sim = self.order_result(self.simulation_gen)
        except AttributeError:
            print('A simulation was not generated, checking if loaded simulation exists')
            try:
                sim = self.order_result(self.simulation_load)
            except AttributeError:
                print('A simulation is needed it was neither generated or loaded, try generating a simulation or loading one.')
            else:
                self.get_equilibrium(sim,axis,weight)
                self.plot_eq_points(sim,axis)
        else:
            self.get_equilibrium(sim,axis,weight)
            self.plot_eq_points(sim,axis)
        pass

    def stiffness(self,axis='z',root=0,cmap='Spectral'):
        """
        User function called to calculate and plot the stiffness of a simulation
        Parameters
        -----------
        axis (str)      := 'x','z'
                        Desired stffnes of force component x or z
        root (int)      := 0,1,2
                        Ordinal number of the force component root. This is the centered point 
                        at which the linear regression will be calculated
        """
        try:
            sim = self.order_result(self.simulation_gen)
        except AttributeError:
            print('A simulation was not generated, checking if loaded simulation exists')
            try:
                sim = self.order_result(self.simulation_load)
            except AttributeError:
                print('A simulation is needed it was neither generated or loaded, try generating a simulation or loading one.')
            else:
                self.get_stiffness(sim,axis,root)
                self.plot_stiffness(sim,axis, cmap)
        else:
            self.get_stiffness(sim,axis,root)
            self.plot_stiffness(sim,axis, cmap)
        pass

    def get_stiffness(self,sim,axis,r):
        """
        Inner function to calculate the stiffnes according to a Linear regression at the selected root
        """
        def find_root(F):
            sign = np.sign(F)
            sign_change = ((np.roll(sign,1)-sign)!=0).astype(int)
            sign_change[0] = 0
            roots = np.where(sign_change==1)[0]
            return roots
        def classify_eq(F):
            roots = {'Inestable': [], 'Estable': []}
            i = 0
            for root in find_root(F):
                if (F[root-1] < 0) and (F[root+1] > 0):
                    roots['Inestable'].append(root)
                elif(F[root-1] > 0) and (F[root+1] < 0):
                    roots['Estable'].append(root)
                i +=1
            return roots
        def stiffness(F,position,n=20):
            stiffness = {'x': [], 'y': []} 
            root = classify_eq(F)['Estable'][r]
            for i in range(root-n,root+n):
                x = position[i]
                y = -F[i]
                stiffness['x'].append(x)
                stiffness['y'].append(y)
            stiffness['x'] =  np.array(stiffness['x'])
            stiffness['y'] = np.array(stiffness['y'])
            model = LinearRegression().fit(stiffness['x'].reshape((-1,1)),stiffness['y'])
            stiffness['model'] = model
            stiffness['R'] = model.score
            stiffness['k'] = model.coef_
            return stiffness['k']
        self.stiff_values = {}
        if axis == 'z':
            force_axis = 1
        elif axis == 'x' :
            force_axis = 0
        for key, result in sim.items():
            try:
                self.stiff_values[key] = stiffness(result['results']['total'][force_axis], result['positions'][axis])
            except IndexError:
                self.stiff_values[key] = 0
        return self.stiff_values

    def plot_stiffness(self, sim, ax, cmap):
        """
        Inner function callback for plotting the calculated stiffness of a simulation 
        """
        plt.rcParams['figure.figsize'] = (8.0, 6.0)
        plt.rcParams['figure.dpi'] = (150)
        plt.ylabel(f'K{ax}'+r' [$pN/\mu m$]')
        plt.xlabel('Simulation variables {}'.format(self.variable))
        plt.title('Stiffness for various: {}'.format(self.variable))
        plt.grid()
        n = len(self.stiff_values.keys())
        colorm = mcp.gen_color(cmap=cmap,n=n)
        i=0
        for var_, k in self.stiff_values.items():
            if self.variable == 'radii': 
                var = float(var_)*1e6
            elif self.variable == 'NA':
                var = np.array([float(var_)])
            elif self.variable == 'waist':
                var = np.round(float(var_)*1e3,2)
            else:
                var = var_
            plt.plot(var,float(k)*1e6,'*',color=colorm[i])
            i+=1
        plt.show()
        plt.ylabel(f'K{ax}'+r' [$pN/\mu m mW$]')
        plt.xlabel('Simulation variables {}'.format(self.variable))
        plt.title(' Power-Normalized Stiffness for various: {}'.format(self.variable))
        plt.grid()
        i=0
        for var_, k in self.stiff_values.items():
            if self.variable == 'radii': 
                var = float(var_)*1e6
            elif self.variable == 'NA':
                var = np.array([float(i) for i in var_])
            elif self.variable == 'waist':
                var = np.round(float(var_)*1e3,2)
            else:
                var = var_
            try:
                power = self.simulation_gen[var_]['parameters']['PL']*1e3
            except AttributeError:
                print('A simulation was not generated, checking if loaded simulation exists')
                try:
                    power = self.simulation_load[var_]['parameters']['PL']*1e3
                except AttributeError:
                    print('A simulation is needed it was neither generated or loaded, try generating a simulation or loading one.')
                else:
                    plt.plot(var,float(k)*1e6/power,'*', color=colorm[i])
            else:
                plt.plot(var,float(k)*1e6/power,'*',color=colorm[i])
            i+=1
        plt.show()
        pass
