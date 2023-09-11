import numpy as np
import math
import random
import json,os

from PTFEScatter import scatter_on_ptfe  # import the scattering data
from Utils import intersection_with_cylinder, calculate_position, generate_lambertian 

from sys import exit

# import ptfe_utils

XENON_GAS = 0
XENON_LIQ = 1
PTFE = 2
PMT = 3
VOID = 4

medium_names = ['GXe', 'LXe', 'PTFE', 'PMT', "VOID"]
refractive_index = np.array([1., 1.5, 1.69, 3.5, 1.0])

class OpticalPhoton:
    """
    Class to propagate an optical photon through a simplified version of a XENON detector To keep things simple, 
    the detector is modelled as a cylinder with a flat top and bottom.

    In addition, the code is not segmented into separate classes in order to keep the code fast and simple.

    A.P. Colijn
    """
    def __init__(self, **kwargs):
        """
        Initialize the optical photon

        Parameters
        ----------
        config(str) : str
            The configuration file -> this sets the default values for 
            the detector geometry. See Generator.py for documentation on the config file.

        A.P. Colijn

        """
        # Read configuration file
        self.config_file = kwargs.get('config', 'config.json')
        if os.path.isfile(self.config_file):
            print("OpticalPhoton::Reading configuration from file: {}".format(self.config_file))
            self.config = json.load(open(self.config_file, 'r'))
        else:
            raise ValueError("Config file does not exist.")

        self.R = self.config['geometry']['radius']
        self.ztop = self.config['geometry']['ztop']
        self.zliq = self.config['geometry']['zliq']
        self.zbot = self.config['geometry']['zbot']

        # exent of the ptfe cylinder. If a photon intersects with the cylinder outside 
        # this range, it is terminated.
        self.ptfe_zmin = self.config['geometry']['ptfe_zmin']
        self.ptfe_zmax = self.config['geometry']['ptfe_zmax']


        # initial and current position
        self.x0 = np.zeros(3)
        self.xc = np.zeros(3)
        # initial and current direction
        self.t0 = np.zeros(3)
        self.tc = np.zeros(3)

        self.current_medium = XENON_GAS
        self.alive = True
        self.detected = False

        if 'set_no_scatter' in self.config:
            print("'set_no_scatter' set to {}".format(self.config['set_no_scatter']))
            self.no_scattering = self.config['set_no_scatter']
        else:
            print("'set_no_scatter' not in config: setting to False")
            self.no_scattering = False

        if 'set_experimental_scatter_model' in self.config:
            print("'set_experimental_scatter_model' set to {}".format(self.config['set_experimental_scatter_model']))
            self.experimental_scatter_model = self.config['set_experimental_scatter_model']
        else:
            print("'set_experimental_scatter_model' not in config: setting to True")
            self.experimental_scatter_model = True

    def set_experimental_scatter_model(self, experimental_scatter_model):
        """
        Switch on/off the experimental scattering model for 175 nmphotons on PTFE from GXe/LXe  

        Parameters
        ----------
        experimental_scatter_model(bool) : bool
            Switch on/off the experimental scattering model

        A.P. Colijn
        """
        self.experimental_scatter_model = experimental_scatter_model
        
    def set_no_scattering(self, no_scattering):
        """
        Switch off scattering

        Parameters
        ----------
        no_scattering(bool) : bool
            Switch off scattering

        A.P. Colijn
        """
        self.no_scattering = no_scattering

    def set_photon_position(self, x):
        """
        Set the photon position

        Parameters
        ----------
        x0(float) : array_like
            The initial position of the photon

        A.P. Colijn
        """
        self.x = x
    
    def get_photon_position(self):
        """
        Get the photon position

        Returns
        -------
        x(float) : array_like
            The position of the photon

        A.P. Colijn
        """
        return self.x

    def get_medium(self, x):
        """
        Get the medium given the position x

        Parameters
        ----------
        x(float) : array_like
            The position of the photon

        Returns
        -------
        medium(int) : int

        A.P. Colijn
        """
        margin = 1e-6

        r = np.sqrt(x[0]**2 + x[1]**2)
        z = x[2]

        if r >= self.R:
            if self.ptfe_zmin < z < self.ptfe_zmax:
                medium = PTFE
            else:
                medium = VOID
        else:
            if ((z > self.zliq) and (z <= self.ztop - margin)):
                medium = XENON_GAS
            elif ((z < self.zliq) and (z >= self.zbot + margin)):
                medium = XENON_LIQ
            elif ((z > self.ztop-margin) or (z < self.zbot+margin)):
                medium = PMT
            else:
                print('error')
                exit()         

        return medium

    def set_medium(self):
        """
        Set the medium in which the photon is propagating

        A.P. Colijn
        """
        margin = 1e-6

        r = np.sqrt(self.x[0]**2 + self.x[1]**2)
        if r >= self.R:
            medium = PTFE
        else:
            z = self.x[2]
            if (z > self.zliq) and (z <= self.ztop - margin):
                medium = XENON_GAS
            if (z < self.zliq) and (z >= self.zbot + margin):
                medium = XENON_LIQ
            if (z > self.ztop-margin) or (z < self.zbot+margin):
                medium = PTFE  

        self.current_medium = medium
    
    def generate_photon(self, x0):
        """
        Generate a photon with a random direction t0=(tx,ty,tz) at position x0. The direction is isotropic.

        A.P. Colijn
        """
        # Generate random values for the angles theta and phi
        phi = random.uniform(0.0, 2.0 * math.pi)
        theta = math.acos(2 * random.uniform(0.0, 1.0) - 1.0)

        # Calculate the direction vector components
        tx = math.sin(theta) * math.cos(phi)
        ty = math.sin(theta) * math.sin(phi)
        tz = math.cos(theta)

        # Create the random unit direction vector t
        self.t0 = np.array([tx, ty, tz])
        self.t = self.t0

        # Set the initial position
        if not self.position_is_inside_detector(x0):
            print("Error: photon position is outside the detector")
            exit(0)

        self.x0 = np.array(x0)
        self.set_photon_position(self.x0)
        # the photon is alive...
        self.alive = True
        self.detected = False

        # Set the current medium
        self.set_medium()

        return 
    
    def position_is_inside_detector(self, x):
        """
        Check if the position x is inside the detector

        Parameters
        ----------
        x(float) : array_like
            The position of the photon

        Returns
        -------
        inside(bool) : bool

        A.P. Colijn
        """
        r = np.sqrt(x[0]**2 + x[1]**2)
        if r > self.R:
            return False
        else:
            z = x[2]
            if (z > self.zbot) and (z < self.ztop):
                return True
            else:
                return False

    def get_photon_direction(self):
        """
        Get the photon direction

        Returns
        -------
        t(float) : array_like
            The direction of the photon

        A.P. Colijn
        """
        return self.t

    def set_photon_direction(self, t):
        """
        Set the photon direction

        Parameters
        ----------
        t0(float) : array_like
            The initial direction of the photon

        A.P. Colijn
        """
        self.t = t / np.linalg.norm(t)

    def is_alive(self):
        """
        Check if the photon is alive

        Returns
        -------
        alive(bool) : bool

        A.P. Colijn
        """
        return self.alive
    
    def is_detected(self):
        """
        Check if the photon is detected

        Returns
        -------
        detected(bool) : bool

        A.P. Colijn
        """
        return self.detected    

    def get_refractive_index(self, medium):
        """Returns the refractive index of the medium.

        Args:
            medium (int): medium

        Returns:
            float: refractive index
        """
        return refractive_index[medium]
    
    def get_medium_name(self, medium):  
        """Returns the name of the medium.

        Args:
            medium (int): medium

        Returns:
            string: name of medium
        """
        return medium_names[medium]

    def interact_with_surface(self, xint, dir, nvec):
        """Calculates the interaction of the photon with a surface. The photon is either reflected or transmitted.
        After the interaction, the position and direction of the photon are updated as well as the medium.

        Args:
            xint (array): intersection point
            dir (array): direction of photon
            nvec (array): normal vector to surface  

        Returns:
            array, array: reflected ray, transmitted ray

        A.P. Colijn
        """    

        if np.linalg.norm(dir) != 0.0:
            dir = dir / np.linalg.norm(dir)
        else:
            print('error: direction vector has zero length')
            return 1
        
        if np.linalg.norm(nvec) != 0.0:
            nvec = nvec / np.linalg.norm(nvec)  
        else:
            print('error: normal vector has zero length')
            return 1


        # 1. get the medium in which the photon is propagating
        medium1 = self.current_medium
        n1 = self.get_refractive_index(medium1)
        # 2. get the medium on which the photon is incident
        #    calculate the material by stepping into minus the direction of the normal vector to the material we are scattering on. Then get the material at that position.
        #   This is a bit of a hack, but it works.
        medium2 = self.get_medium(calculate_position(xint, nvec, -1e-4))         
        n2 = self.get_refractive_index(medium2)

        ####if medium2 == PTFE: # test to see the effect of PTFE relfection on the photon propagation
        ####    # photon is terminated
        ####    self.alive = False
        ####    return 0

        # 3. calculate the angle of incidence
        dot_product = np.dot(-dir, nvec)
        clamped_value = min(max(dot_product, -1.0), 1.0)
        if abs(dot_product) > 1.0:
            print('error: dot_product > 1.0')
            self.print()
        theta1 = math.acos(clamped_value)
        # 4. calculate the average reflected power. I assume unpolarized light....
        
        R_diff = -1.0
        if (medium2 == PTFE) and self.experimental_scatter_model: # PTFE reflection based on experimental data 
            R_average, R_diff, _, _ = scatter_on_ptfe(theta1, medium_names[medium1])
        else:  # standard Fresnel reflection/transmission
            R_average, _ = self.fresnel_coefficients_average(n1, n2, theta1)

        # 5. decide whether the photon is reflected or transmitted based on the reflected power

        ##print('x before scatter =', xint,'direction before scatter', self.t, 'medium before scatter', medium1, 'medium after scatter', medium2, 'r_average', R_average)
        if self.no_scattering or (medium2 == VOID):
            rran = 1.0 # force transmission
        else:
            rran = random.uniform(0, 1)

        if  rran < R_average:
            # reflected
            #print("reflected")
            # 6. calculate the reflected direction

            if (rran < R_diff) and (medium2 == PTFE) and self.experimental_scatter_model:
                # diffuse reflection
                #print('... diffuse reflection from PTFE. experimental model =', self.experimental_scatter_model)
                reflected_dir = generate_lambertian(nvec)

            else:
                # specular reflection
                #print('... specular reflection')
                in_dir = self.t
                dot_product = np.dot(-in_dir, nvec)
                reflected_dir = in_dir + 2 * dot_product * nvec
                # medium does not change.....
            
            self.t = reflected_dir
            self.x = xint

        else:
            # transmitted 
            #print("transmitted")
            # 6. calculate the transmitted direction           
            in_dir = self.t
            dot_product = np.dot(-in_dir, nvec)
            refracted_dir = (n1 / n2) * (in_dir + dot_product * nvec) - np.sqrt(1.0 - (n1 / n2)**2 * (1.0 - dot_product**2)) * nvec

            self.t = refracted_dir
            self.x = xint
            self.current_medium = medium2


        ##print("x after scatter = ", self.x, "direction after scatter", self.t, "medium after scatter", self.current_medium)

        return 0

    def fresnel_coefficients(self, n1, n2, theta_i):
        """
        Calculate the reflection and transmission coefficients using the Fresnel equations.

        Parameters
        ----------
        n1(float) : float
            The refractive index of the medium in which the photon is incident
        n2(float) : float
            The refractive index of the medium in which the photon is transmitted
        theta_i(float) : float  
            The angle of incidence

        Returns
        -------
        r_parallel(float) : float
            The reflection coefficient for p-polarization
        r_perpendicular(float) : float
            The reflection coefficient for s-polarization


        A.P. Colijn
        """
        # Calculate angles of refraction (theta_t) using Snell's law
        sin_theta_i = np.sin(theta_i)
        sin_theta_t = (n1 / n2) * sin_theta_i

        # Handle total internal reflection

        if sin_theta_t > 1.0:
            r_parallel = 1.0
            r_perpendicular = 1.0
            return r_parallel, r_perpendicular # Total internal reflection, R=1, T=0
        
        cos_theta_i = np.cos(theta_i)
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)

        ##print('cos_theta_i', cos_theta_i, 'cos_theta_t', cos_theta_t,'n1',n1,'n2',n2)

        # Calculate the coefficients for p-polarization
        r_parallel = ((n1 * cos_theta_i - n2 * cos_theta_t) /
                    (n1 * cos_theta_i + n2 * cos_theta_t))
        # Calculate the coefficients for s-polarization
        r_perpendicular = ((n2 * cos_theta_i - n1 * cos_theta_t) /
                        (n2 * cos_theta_i + n1 * cos_theta_t))

        #print('r_parallel', r_parallel, 'r_perpendicular', r_perpendicular)
        return r_parallel, r_perpendicular
    
    def fresnel_coefficients_average(self, n1, n2, theta_i):
        """Calculate the average reflection and transmission coefficients for randomly polarized light.

        Parameters
        ----------
        n1(float) : float
            The refractive index of the medium in which the photon is incident
        n2(float) : float
            The refractive index of the medium in which the photon is transmitted
        theta_i(float) : float
            The angle of incidence

        Returns
        -------
        R_average(float) : float
            The average reflected power (this is not the same as the reflection coefficient!)
        T_average(float) : float
            The average transmitted power

        A.P. Colijn
        """
        r_parallel, r_perpendicular = self.fresnel_coefficients(n1, n2, theta_i)
    
        # Calculate the average coefficients for randomly polarized light
        R_average = 0.5 * (r_parallel**2 + r_perpendicular**2)
        T_average = 1 - R_average
    
        return R_average, T_average

    def propagate(self):
        """
        Propagate the photon through the detector. 

        A.P. Colijn
        """

        propagating = True
        #print("")
        #print("NEXT")    

        #self.print()    
        while propagating:

            # intersection with cylinder
            path_length = -100
            if self.current_medium == XENON_GAS:
                xint, path_length, nvec = intersection_with_cylinder(self.x, self.t, self.R, self.zliq, self.ztop)
            else:
                xint, path_length, nvec = intersection_with_cylinder(self.x, self.t, self.R, self.zbot, self.zliq)

            #
            # Let the photon interact with a surface
            # 
            # Several things can happen:
            # 1. the photon is reflected
            # 2. the photon is transmitted
            # 3. the photon is absorbed (not implemented yet - depends on mean free path of photons in xenon gas and liquid)
            # 4. if the photon is transmitted to the PMT, it is absorbed and detected
            # 5. if the photon is transmitted to the PTFE, it is terminated
            # 
            # The position and direction of the photon are updated after the interaction.
            #
            # print(xint, self.t, nvec, path_length)
            if path_length>0.0:
                istat = self.interact_with_surface(xint, self.t, nvec)
                if istat == 1:
                    print('error in interact_with_surface')
                    self.alive = False 
            else:
                # no intersection with cylinder
                print('no intersection with cylinder')
                self.alive = False
                return 0
            #self.print()
            #
            # Check if the photon should continue propagating, be terminated, or be detected
            #
            propagating = self.photon_propagation_status()

        #print('alive =',self.alive, 'detected =',self.detected, 'x =', self.x, 't =', self.t, 'medium =', self.current_medium)
        return 0

    def photon_propagation_status(self):
        """
        Check if the photon should continue propagating, be terminated, or be detected

        Returns
        -------
        propagating(bool) : bool

        A.P. Colijn
        """
        # Check if the photon is still alive
        if not self.alive:
            return False
        
        # Check if the photon is still inside the detector
        if self.current_medium == PMT:
            # photon is detected
            self.alive = False
            self.detected = True
            return False
        elif self.current_medium == PTFE:
            # photon is terminated
            self.alive = False
            return False
        else:
            # photon is still propagating
            return True
        
    def print(self):
        """
        Print the photon position and direction

        A.P. Colijn
        """
        print('x =', self.x, 't =', self.t, 'medium =', self.current_medium)
        return 0