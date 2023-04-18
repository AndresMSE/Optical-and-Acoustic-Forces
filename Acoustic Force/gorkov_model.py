import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from tqdm import tqdm
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
class AcousticPT:
    
    def __init__(self, variable, condition='soft-wall',**kwargs):
        """
        The object instance is created. 
        
        The type of experimental variable and the model parameters
        are selecto choosing between particle material and size, 
        type of wall condition and microchip geometry.
        Parameters
        ------------
        variable(str) := 'radii', 'material', 'medium', 'modes', 'frequency'
                            variables for experimentation
        condition(str) := 'soft-wall', 'hard-wall'
                            type of boundary condition on the chip
        kwargs(dict) := {'a': p. radius [m], l': chip's length [m], 'w': chip's width [m], 
                            'h': chip's height [m], 'pa': preassure amplitude [Pa], 
                            'f': acoustic frecuency [Hz], 'medium': medium material,
                            'material': p. material, 'n_points': size of samplings}
        Material velocities recovered from https://www.olympus-ims.com/en/ndt-tutorials/thickness-gauge/appendices-velocities/
        https://iopscience.iop.org/article/10.1088/1361-6560/aa6226#:~:text=Our%20approach%20enables%20an%20accurate,0%25%20to%2080%25%20respectively.
        https://www.researchgate.net/figure/Physical-constants-for-phosphate-buffered-saline-with-05-w-v-albumin-solution_tbl1_258827391
        https://www.ias.ac.in/article/fulltext/seca/009/04/0312-0315
        https://www.sciencedirect.com/science/article/abs/pii/0041624X92900943
        
        """
        self.variable = variable
        self.condition = condition
        # materials sound velocity
        self.materials_velocity = {'latex':1610, 'silica':5000, 'oil':1469, 'liver': 1492, 'poly':2350}  #m/s
        self.materials_density = {'latex': 1100, 'silica': 2000, 'oil':1020, 'liver': 1070, 'poly':1050}    # kg/m3
        # mediums sound velocity
        self.mediums_velocity = {'water': 1497, 'air': 348, 'glycerol': 1957,'PBS':1505} #m/s
        self.mediums_density = {'water': 998, 'air': 1.225, 'glycerol': 1260, 'PBS':1062} #kg/m3
        self.param  = {'a': 10e-6, 'f': None, 
                            'h': 1e-3, 'pa': 1e6, 
                            'nx': 1, 'ny': 1 , 'nz': 1, 'medium': 'water',
                            'material': 'silica', 'n_points': 250, 'l':None}
        
        # The p dictionary will store and update the parameters according
        # to the user in **kwars
        for key, value in kwargs.items():
            if key in list(self.param.keys()):
                self.param[key] = value    # Update the simulation parameters
            else:
                raise KeyError('Select a valid parameter.')
        if (self.param['f'] == None) and (self.param['l']==None):
            raise KeyError('Must select parameter f or l.')
        # Get the compressibility given the material and the medium
        if self.param['material'] in self.materials_velocity.keys():
            mat = self.param['material']
            self.param['k_p'] = 1/((self.materials_velocity[mat]**2)*self.materials_density[mat])    #Compressibility of the material 1/(rho_p c_p^2)
            self.param['c_p'] = self.materials_velocity[mat]
            self.param['rho_p'] = self.materials_density[mat]

        else:
            raise KeyError('Select a valid material. The allowed materials are: poly,latex,silica, glass, oil or liver')
        if self.param['medium'] in self.mediums_velocity.keys():
            med = self.param['medium']
            self.param['k_0'] = 1/((self.mediums_velocity[med]**2)*self.mediums_density[med])    #Compressibility of the medium  1/(rho_0 c_0^2)
            self.param['rho_0'] = self.mediums_density[med]
            self.param['c_0'] = self.mediums_velocity[med]
            if self.param['l'] == None:
                denom = (4*(self.param['f']**2)/(self.mediums_velocity[med]**2)) - (self.param['nz']/self.param['h'])**2
                self.param['l'] = np.sqrt((self.param['nx']**2 + self.param['ny']**2)/denom)    #Calculated x-y dimensions
            elif self.param['f'] == None:
                arg = (self.param['nx']/self.param['l'])**2 +(self.param['ny']/self.param['l'])**2+ (self.param['nz']/self.param['h'])**2   #fixed nz**4 -> nz**2
                self.param['f'] = (self.mediums_velocity[med]/2)*np.sqrt(arg)   #Frequency of the wave to get the given resonance modes            
        else:
            raise KeyError('Select a valid medium. The allowed mediyms are: water, air, glycerol')
        
        #Verify number of wavelenghts
        if self.param['nx'] == 0 and self.condition in ['soft-wall','mixed-wall']:
            raise ValueError('In soft wall condition nx > 0')
        elif self.param['ny'] == 0 and self.condition in ['soft-wall','mixed-wall']:
            raise ValueError('In soft wall condition ny > 0')
        elif self.param['nz'] == 0 and self.condition in ['soft-wall']:
            raise ValueError('In soft wall condition nz > 0')
        #Acoustic energy density
        self.param['E_ac'] = (self.param['pa']**2)/(4*self.param['rho_0']*(self.param['c_0']**2))  #changed to explicit form of E_ac rather de k_0 version

            
    def preassure_f (self,x,y,z):
        p = self.param
        l,h = p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        c_0 = p['c_0']
        kx,ky,kz = (nx*np.pi)/l, (ny*np.pi)/l, (nz*np.pi)/h
        self.kx, self.ky, self.kz = kx, ky, kz
        if self.condition == 'soft-wall':
            p1 = self.param['pa']*np.sin(kx*x)*np.sin(ky*y)*np.sin(kz*z)
        elif self.condition == 'hard-wall':
            p1 = self.param['pa']*np.cos(kx*x)*np.cos(ky*y)*np.cos(kz*z)
        elif self.condition == 'mixed-wall':
            p1 = self.param['pa']*np.sin(kx*x)*np.sin(ky*y)*np.cos(kz*z)
        return p1
    
            
    def velocity_f(self,x,y,z):
        p = self.param
        l,h = p['l'],p['h']
        c_0 = p['c_0']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = (nx*np.pi)/l, (ny*np.pi)/l, (nz*np.pi)/h
        w = 2*np.pi*p['f']
        phi_constant = 1/(p['rho_0']*w) #Changed, added sign Re(-i \times -i*) = 1
        if self.condition == 'soft-wall':
            v1_x = kx*self.param['pa']*np.cos(kx*x)*np.sin(ky*y)*np.sin(kz*z)
            v1_y = ky*self.param['pa']*np.sin(kx*x)*np.cos(ky*y)*np.sin(kz*z)
            v1_z = kz*self.param['pa']*np.sin(kx*x)*np.sin(ky*y)*np.cos(kz*z)
            v1 = np.array([v1_x,v1_y,v1_z])
        elif self.condition == 'hard-wall':
            v1_x = -kx*self.param['pa']*np.sin(kx*x)*np.cos(ky*y)*np.cos(kz*z)
            v1_y = -ky*self.param['pa']*np.cos(kx*x)*np.sin(ky*y)*np.cos(kz*z)
            v1_z = -kz*self.param['pa']*np.cos(kx*x)*np.cos(ky*y)*np.sin(kz*z)
            v1 = np.array([v1_x,v1_y,v1_z])
        elif self.condition == 'mixed-wall':
            v1_x = kx*self.param['pa']*np.cos(kx*x)*np.sin(ky*y)*np.cos(kz*z)
            v1_y = ky*self.param['pa']*np.sin(kx*x)*np.cos(ky*y)*np.cos(kz*z)
            v1_z = -kz*self.param['pa']*np.sin(kx*x)*np.sin(ky*y)*np.sin(kz*z)
            v1 = np.array([v1_x, v1_y, v1_z])
        return v1*phi_constant
        
    def plot_preassure_slices(self, planes, s_x=1.0, s_y=2.0, s_z=3.0, snorm=False,colormap=plt.cm.hot, **kwargs):
        #Make sure these are floating point values:                                                                                                                                                                                              
        x_scale = s_x
        y_scale = s_y
        z_scale = s_z
        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0
        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)

        
        mpl.rcParams['axes.labelsize'] = 15
        mpl.rcParams['xtick.labelsize'] = 15
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        x = np.linspace(0,l,p['n_points'])
        planes_def = {'x':[0,int(p['n_points']/2),p['n_points']-1],'y':[],'z':[]}
        for key,val in planes.items():
            planes_def[key] = val
        xslice, yslice, zslice = planes['x'], planes['y'], planes['z']
        fig = plt.figure(figsize=(12,8), constrained_layout=True)
        [X,Y,Z] = np.meshgrid(x,y,z, indexing='ij')
        P1 = self.preassure_f(X,Y,Z)**2
        max_val, min_val = P1.max(), P1.min()

        ax = fig.add_subplot(111, projection='3d')
        # colormap = plt.cm.hot
        ax.get_proj=short_proj
        i = 0
        for xs in xslice:
            x_fix = xs*l/p['n_points']
            Y,Z = np.meshgrid(y,z, indexing='ij')
            [Y,Z] = np.meshgrid(y,z, indexing='ij')
            slice_x = self.preassure_f(x_fix,Y,Z)**2
            X = (l*xs/p['n_points'])*np.ones((p['n_points'],p['n_points']))
            volume_ = ax.plot_surface(X*1e3,Y*1e3,Z*1e3, rstride=1, cstride=1, shade=False, 
                            facecolors=colormap((slice_x-min_val)/(max_val-min_val)))
            ax.set_title(r'YZ plane slices of Preassure $P_{in}^2$')
            ax.set_xlabel('$x$ $[mm]$')
            ax.set_ylabel('$y$ $[mm]$')
            ax.set_zlabel('$z$ $[mm]$')
            ax.view_init(elev=30, azim=-45, )
            if i ==2:
                V = (slice_x-min_val)/(max_val-min_val)
                m = mpl.cm.ScalarMappable(cmap=colormap)
                m.set_array([V])
                fig.colorbar(m,shrink=0.3)
            i+=1


        return fig,ax
    
    def plot_velocity_slices(self, planes, s_x=1.0, s_y=2.0, s_z=3.0, norm=False, **kwargs):
                #Make sure these are floating point values:                                                                                                                                                                                              
        x_scale = s_x
        y_scale = s_y
        z_scale = s_z
        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0
        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)

        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        x = np.linspace(0,l,p['n_points'])
        [X,Y,Z] = np.meshgrid(x,y,z)
        V1 = np.linalg.norm(self.velocity_f(X,Y,Z), axis=0)
        planes_def = {'x':[0,int(p['n_points']/2),p['n_points']-1],'y':[],'z':[]}
        for key,val in planes.items():
            planes_def[key] = val
        xslice, yslice, zslice = planes['x'], planes['y'], planes['z']
        max_val, min_val = V1.max(), V1.min()
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        colormap = plt.cm.binary
        for xs in xslice:
            x_fix = xs*l/p['n_points']
            Y,Z = np.meshgrid(y,z)
            slice_x = np.linalg.norm(self.velocity_f(x_fix,Y,Z),axis=0)
            X = (l*xs/250)*np.ones((p['n_points'],p['n_points']))
            volume_ = ax.plot_surface(X*1e3,Y*1e3,Z*1e3, rstride=1, cstride=1, shade=False, 
                            facecolors=colormap((slice_x-min_val)/(max_val-min_val)))
            ax.set_title('X slice of Velocity magnitude')
            ax.set_xlabel('$X$ mm')
            ax.set_ylabel('$Y$ mm')
            ax.set_zlabel('$Z$ mm')
        plt.show()
        
    def plot_preassure(self,x_plane,norm=False,**kwargs):
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        Y,Z = np.meshgrid(y,z)
        x_fix = x_plane*l/p['n_points']
        X = self.preassure_f(x_fix,Y,Z)
        fig = plt.figure()
        if norm == True:
            norm_val = np.reshape(self.preassure_f(l/2,Y,Z),(-1,p['n_points']**2))
            levels = np.linspace(norm_val.min(),norm_val.max(),10)
            CF = plt.contourf(Z,Y,X, levels=levels)
        else:
            CF = plt.contourf(Y,Z,X)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(f'Plane x {x_plane}')
        cbar = fig.colorbar(CF)
        plt.show()
    def plot_velocity(self,x_plane, s, **kwargs):
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        Y,Z = np.meshgrid(y[::s],z[::s])
        x_fix = x_plane*l/p['n_points']
        V = self.velocity_f(x_fix,Y,Z)
        u = V[1]
        v = V[2]
        V_mag = np.sqrt(u**2 + v**2)
        fig = plt.figure()
        plt.quiver(Y*1e3,Z*1e3,u/V_mag,v/V_mag)
        plt.xlabel('y mm')
        plt.ylabel('z mm')
        plt.title(f'Plane x {x_fix}')
        plt.show()
    
    def plot_fields(self,x_plane,**kwargs):
        parameters = {'figsize': (16,6), 'colormap': 'Spectral', 'normalize': False}
        for key, value in kwargs.items():
            parameters[key] = value
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        fig,ax = plt.subplots(1,2,figsize=parameters['figsize'])
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        Y,Z = np.meshgrid(y,z)
        X = self.velocity_f(x_plane,Y,Z)
        u = X[1]
        v = X[2]
        ax[0].quiver(Y,Z,u,v)
        
        
        X = self.preassure_f(x_plane,Y,Z)
        if parameters['normalize'] == True:
            norm_val = np.reshape(self.preassure_f(l/2,Y,Z),(-1,p['n_points']**2))
            levels = np.linspace(norm_val.min(),norm_val.max(),10)
            CF = ax[1].contourf(Z,Y,X, levels=levels)
        else:
            CF = ax[1].contourf(Z,Y,X,cmap=parameters['colormap'])
        cbar = fig.colorbar(CF)
        plt.show()
        
    def plot_fields_contour(self, x_plane, s, **kwargs):
        parameters = {'figsize': (16,6), 'colormap': 'Spectral', 'normalize': False, 'color':'black'}
        for key, value in kwargs.items():
            parameters[key] = value
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        # Contour plot
        x_fix = x_plane*l/p['n_points']
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        n= p['n_points']
        x = np.linspace(0,l,p['n_points'])
        Y, Z = np.meshgrid(y, z)
        P1 = self.preassure_f(x_fix, Y, Z)
#         Y, Z = Y[:,x_plane], Z[:,x_plane]
        cp = plt.contourf(Y*1e3,Z*1e3, P1)
        cb = plt.colorbar(cp)
        # Vector Field
        Y,Z = np.meshgrid(y,z)
        V = self.velocity_f(x_fix,Y,Z)
        Vy,Vz = V[1], V[2]
        speed = np.sqrt(Vy**2 + Vz**2)
        u = Vy/speed
        v = Vz/speed
        quiv = plt.streamplot(y*1e3, z*1e3, u, v, color=parameters['color'])
        plt.xlabel('y mm')
        plt.ylabel('z mm')
        plt.title(f'Plane x {x_plane}')
        plt.show()
    
    def plot_gorkov_slices(self, planes, s_x=1.0, s_y=2.0, s_z=3.0, snorm=False, cmap_=plt.cm.hot):
        #Make sure these are floating point values:                                                                                                                                                                                              
        x_scale = s_x
        y_scale = s_y
        z_scale = s_z
        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0
        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)

        
        mpl.rcParams['axes.labelsize'] = 15
        mpl.rcParams['xtick.labelsize'] = 15
        p = self.param
        l,w,h = p['l'],p['l'],p['h']
        nx,ny,nz = self.param['nx'], self.param['ny'], self.param['nz']
        kx,ky,kz = nx*np.pi/l, ny*np.pi/w, nz*np.pi/h
        z = np.linspace(0,h,p['n_points'])
        y = np.linspace(0,w,p['n_points'])
        x = np.linspace(0,l,p['n_points'])
        planes_def = {'x':[0,int(p['n_points']/2),p['n_points']-1],'y':[],'z':[]}
        for key,val in planes.items():
            planes_def[key] = val
        xslice, yslice, zslice = planes['x'], planes['y'], planes['z']
        fig = plt.figure(figsize=(12,8), constrained_layout=True)
        [X,Y,Z] = np.meshgrid(x,y,z, indexing='ij')
        G1 = self.U(X,Y,Z)
        max_val, min_val = G1.max(), G1.min()

        ax = fig.add_subplot(111, projection='3d')
        colormap = cmap_
        ax.get_proj=short_proj
        i = 0
        for xs in xslice:
            x_fix = xs*l/p['n_points']
            Y,Z = np.meshgrid(y,z, indexing='ij')
            [Y,Z] = np.meshgrid(y,z, indexing='ij')
            slice_x = self.U(x_fix,Y,Z)
            X = (l*xs/p['n_points'])*np.ones((p['n_points'],p['n_points']))
            volume_ = ax.plot_surface(X*1e3,Y*1e3,Z*1e3, rstride=1, cstride=1, shade=False, 
                            facecolors=colormap((slice_x-min_val)/(max_val-min_val)))
            ax.set_title(r"YZ plane slices of Gor'kov potential $U$")
            ax.set_xlabel('$x$ $[mm]$')
            ax.set_ylabel('$y$ $[mm]$')
            ax.set_zlabel('$z$ $[mm]$')
            ax.view_init(elev=30, azim=-45, )
            if i ==2:
                V = (slice_x-min_val)/(max_val-min_val)
                m = mpl.cm.ScalarMappable(cmap=colormap)
                m.set_array([V])
                fig.colorbar(m,shrink=0.3)
            i+=1
        
        return fig,ax
    

    def U (self,x,y,z):
            p = self.param
            # Get the parameters from dictionary
            a, f, k_p, k_0 = p['a'], p['f'], p['k_p'], p['k_0']
            rho_p, rho_0 = p['rho_p'], p['rho_0']
            tavg_scalar = lambda f: 0.5*(f**2)
            k_hat = k_p/k_0
            rho_hat = rho_p/rho_0
            f1 = 1-k_hat
            f2 = (2*(rho_hat-1))/(2*rho_hat + 1)
            # x,y,z must be meshgrid coord
            pin = self.preassure_f(x,y,z)
            vin_vec = self.velocity_f(x,y,z)
            vin= np.linalg.norm(vin_vec, axis=0)  # operation over axis 0 
            const = (4*np.pi/3)*(a**3)
            f1_t = (f1*k_0/2)*(0.5)*(pin**2)
            f2_t = (f2*(3/4)*rho_0)*(0.5)*(vin**2)
            U = const*(f1_t - f2_t)
            return U
    
    def acoustic_force(self,r_i, r_f, **kwargs):
        #self.update_params(**kwargs)
        self.r_i = r_i
        self.r_f = r_f
        p = self.param
        '''Data preparation'''
        #  Model dictionary will store the useful results
        model = {'positions':{},'results':{}}
        # Get the positions vectors
        n = p['n_points']
        z = np.linspace(r_i[2],r_f[2],n)
        y = np.linspace(r_i[1],r_f[1],n)
        x = np.linspace(r_i[0],r_f[0],n)
        # Store them in the model dictionary
        model['positions']['x'] = x
        model['positions']['y'] = y
        model['positions']['z'] = z
        # Create the 3D-force results array
        '''U potential'''
        # Get the parameters from dictionary
        a, f, k_p, k_0 = p['a'], p['f'], p['k_p'], p['k_0']
        rho_p, rho_0 = p['rho_p'], p['rho_0']
        tavg_scalar = lambda f: 0.5*(f**2)
        k_hat = k_p/k_0
        rho_hat = rho_p/rho_0
        f1 = 1-k_hat
        f2 = (2*(rho_hat-1))/(2*rho_hat + 1)
        Phi = (f1/3) + (f2/2)
        X,Y,Z  = np.meshgrid(x,y,z,indexing='ij')
        dr =x[1]-x[0]
        model['results']['total'] = -np.array(np.gradient(self.U(X,Y,Z),dr)) #removed axis=0, added step for differentiation
        # model['results']['total'] = gradient(self.U(X,Y,Z),x,y,z)
        model['results']['potential'] = self.U(X,Y,Z)
        # Store the parameters used 
        pa = p['pa']
        vin_mag = pa/(self.mediums_density[p['medium']]*self.mediums_velocity[p['medium']])
        A = 0.64*(self.param['l']**2) #0.64 transducer parameter
        power = A*pa*vin_mag
        model['parameters'] = p
        model['parameters']['power'] = power
        model['parameters']['A_trans'] = A
        model['parameters']['Vin'] = vin_mag
        model['parameters']['PHI'] = Phi
        model['parameters']['E_ac'] = (p['pa']**2)*(k_0)/4
        model['parameters']['f1'] = f1
        model['parameters']['f2'] = f2
        self.model = model
        return model

    def stiffness(self,):
        p = self.param
        a, f, k_p, k_0 = p['a'], p['f'], p['k_p'], p['k_0']
        rho_p, rho_0 = self.materials_density[p['material']], self.mediums_density[p['medium']]
        k_hat = k_p/k_0
        rho_hat = rho_p/rho_0
        f1 = 1-k_hat
        f2 = (2*(rho_hat-1))/(2*rho_hat + 1)
        Phi = (f1/3) + (f2/2)
        k_vec = np.array([self.kx,self.ky,self.kz])     #[1/m]
        E_ac = (p['pa']**2)*(k_0)/4     #[J/m^3]
        K_vec = 8*np.pi*Phi*(a**3)*E_ac*(k_vec**2)      #[N/m]
        P_ac = E_ac*p['l']*p['h']*p['l']    #[J/s] by 1 sec
        K_vec_norm = K_vec/P_ac     #[N/W m]
        return K_vec,K_vec_norm

