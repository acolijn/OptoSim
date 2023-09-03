import numpy as np
import math
import random

XENON_GAS = 0
XENON_LIQ = 1
PTFE = 2
PMT = 3

refractive_index = np.array([1, 1.6, 1.38, 3.5])

class OpticalPhoton:
    """
    Class to propagate an optical photon through a simplified version of a XENON detector To keep things simple, 
    the detector is modelled as a cylinder with a flat top and bottom.

    In addition, the code is not segmented into separate classes in order to keep the code fast and simple.

    A.P. Colijn
    """
    def __init__(self, **kwargs):
        """
        _summary_

        """

        self.R = kwargs.get('R', 2.5)
        self.ztop = kwargs.get('ztop',   1.0)
        self.zliq = kwargs.get('zliq',   0.0)
        self.zbot = kwargs.get('zbot', -10.0)

        # initial and current position
        self.x0 = np.zeros(3)
        self.xc = np.zeros(3)
        # initial and current direction
        self.t0 = np.zeros(3)
        self.tc = np.zeros(3)

        self.current_medium = XENON_GAS


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
        if r >= self.R:
            medium = PTFE
        else:
            z = x[2]
            if ((z > self.zliq) and (z <= self.ztop - margin)):
                medium = XENON_GAS
            elif ((z < self.zliq) and (z >= self.zbot + margin)):
                medium = XENON_LIQ
            elif ((z > self.ztop-margin) or (z < self.zbot+margin)):
                medium = PMT
            else:
                print('error')
                exit(0)            

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
        phi = random.uniform(0, 2 * math.pi)
        theta = theta = math.acos(2 * random.uniform(0, 1) - 1)

        # Calculate the direction vector components
        tx = math.sin(theta) * math.cos(phi)
        ty = math.sin(theta) * math.sin(phi)
        tz = math.cos(theta)

        # Create the random unit direction vector t
        self.t0 = np.array([tx, ty, tz])
        self.t = self.t0
        self.nvec = np.zeros(3)

        # Set the initial position
        self.x0 = np.array(x0)
        self.set_photon_position(self.x0)
        # the photon is alive...
        self.alive = True

        # Set the current medium
        self.set_medium()

        return 


    def intersection_with_cylinder(self, x, t, R, zb, zt):
        """Finds intersection of straight line photon trajectory with cylinder. Only intersections 
        in the direction of the photon are considered.

        Args:
            x (array): start point of photon
            t (array): direction of photon
            R (float): radius of cylinder
            zb (float): z of bottom of cylinder
            zt (float): z of top of cylinder

        Returns:
            array, float, array: intersection point, path length, normal vector
        """
        # Initialize a list of path lengths
        s = []
        surface = []

        # Calculate the intersection points with the bottom horizontal plane
        t_bottom_plane = (zb - x[2]) / t[2]
        if t_bottom_plane >= 0:
            s.append(t_bottom_plane)
            surface.append("bottom")

        # Calculate the intersection points with the top horizontal plane
        t_top_plane = (zt - x[2]) / t[2]
        if t_top_plane >= 0:
            s.append(t_top_plane)
            surface.append("top")

        # Calculate coefficients for the quadratic equation for the cylinder shell
        A = t[0]**2 + t[1]**2
        B = 2 * (x[0] * t[0] + x[1] * t[1])
        C = x[0]**2 + x[1]**2 - R**2

        # Calculate the discriminant
        discriminant = B**2 - 4 * A * C

        # Check if there are real solutions for the cylinder shell
        if discriminant >= 0:
            # Calculate the solutions for t
            t1 = (-B + np.sqrt(discriminant)) / (2 * A)
            if t1 > 0:
                s.append(t1)
                surface.append("cylinder")
            t2 = (-B - np.sqrt(discriminant)) / (2 * A)
            if t2 > 0:
                s.append(t2)
                surface.append("cylinder")

        # Calculate the corresponding intersection points
        # Only find the intersection point furthest away from the start point. In this way we avoid selecting teh intersection point close to teh starting point
        # of the photon trajectory is found due to numerical imprecision. This is an isue if teh photon starts on the boundary of a volume.
        #
        intersection_points = []
        margin = 1e-6
        path_length = -100
        intersection_point = None

        # we calculate the normal vector to the surface at the intersection point (the normal vector points inward)
        normal_vec = np.zeros(3)

        for s_i in s:
            point = self.calculate_position(x, t, s_i)
            if (zb - margin <= point[2] <= zt +margin) and (point[2] - x[2]) / t[2] >= 0 and  (np.sqrt(point[0]**2 + point[1]**2) <= R + margin):
                if s_i > path_length:
                    intersection_point = point 
                    path_length = s_i
                    if surface[s.index(s_i)] == "bottom":
                        normal_vec = np.array([0, 0,  1])
                    elif surface[s.index(s_i)] == "top":
                        normal_vec = np.array([0, 0, -1])
                    else:
                        len = np.sqrt(point[0]**2 + point[1]**2)
                        normal_vec = np.array([-point[0] / len, -point[1] / len, 0])                        

        if intersection_points == None:
            print("No intersection points found")
            return None, None, None

        return intersection_point, path_length, normal_vec
    
    def calculate_position(self, x, t, s):
        """Calculates the position of the photon after propagating a distance s along the trajectory.

        Args:
            x (array): start point of photon
            t (array): direction of photon
            s (float): distance to propagate

        Returns:
            array: position of photon after propagating distance s
        """
        return (x[0] + s * t[0], x[1] + s * t[1], x[2] + s * t[2])

    def get_refractive_index(self, medium):
        """Returns the refractive index of the medium.

        Args:
            medium (int): medium

        Returns:
            float: refractive index
        """
        return refractive_index[medium]
    
    def interact_with_surface(self, xint, nvec):
        """Calculates the reflection and transmission coefficients for the photon at the surface. The reflection 
        and transmission coefficients are calculated using the Fresnel equations. The reflected and transmitted 
        rays are calculated using the Snell's law.

        Args:
            xint (array): intersection point
            nvec (array): normal vector to surface  

        Returns:
            array, array: reflected ray, transmitted ray

        A.P. Colijn
        """

        self.nvec = nvec
        
        # 1. get the medium in which the photon is propagating
        medium1 = self.current_medium
        n1 = self.get_refractive_index(medium1)
        # 2. get the medium on which the photon is incident
        #    calculate the material by stepping into minus the direction of the normal vector to the material we are scattering on. Then get the material at that position.
        #   This is a bit of a hack, but it works.
        medium2 = self.get_medium(self.calculate_position(xint, nvec, -1e-4))         
        n2 = self.get_refractive_index(medium2)
        # 3. calculate the angle of incidence
        theta1 = math.acos(np.dot(-self.t, nvec))
        # 4. calculate the average reflected power. I assume unpolarized light....
        R_average, _ = self.fresnel_coefficients_average(n1, n2, theta1)

        # 5. decide whether the photon is reflected or transmitted based on the reflected power

        print('x before scatter =', xint,'direction before scatter', self.t, 'medium before scatter', medium1, 'medium after scatter', medium2, 'r_average', R_average)
        if random.uniform(0, 1) < R_average:
            # reflected
            print("reflected")
            # 6. calculate the reflected direction
            in_dir = self.t
            dot_product = np.dot(-in_dir, nvec)
            reflected_dir = in_dir + 2 * dot_product * nvec
            # medium does not change.....
            self.t = reflected_dir
            self.x = xint
        else:
            # transmitted
            print("transmitted")
            # 6. calculate the transmitted direction
            in_dir = self.t
            dot_product = np.dot(-in_dir, nvec)
            refracted_dir = (n1 / n2) * (in_dir + dot_product * nvec) - np.sqrt(1.0 - (n1 / n2)**2 * (1.0 - dot_product**2)) * nvec
            # medium does not change.....
            self.t = refracted_dir
            self.x = xint
            self.current_medium = medium2

        print("x after scatter = ", self.x, "direction after scatter", self.t, "medium after scatter", self.current_medium)

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
            return 1.0, 1.0 # Total internal reflection, R=1, T=0

        cos_theta_i = np.cos(theta_i)
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)

        print('cos_theta_i', cos_theta_i, 'cos_theta_t', cos_theta_t,'n1',n1,'n2',n2)

        # Calculate the coefficients for p-polarization
        r_parallel = ((n1 * cos_theta_i - n2 * cos_theta_t) /
                    (n1 * cos_theta_i + n2 * cos_theta_t))
        # Calculate the coefficients for s-polarization
        r_perpendicular = ((n2 * cos_theta_i - n1 * cos_theta_t) /
                        (n2 * cos_theta_i + n1 * cos_theta_t))

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
        
        while propagating:
            print("")
            print("NEXT")
            # intersection with cylinder
            if self.current_medium == XENON_GAS:
                xint, path_length, nvec = self.intersection_with_cylinder(self.x, self.t, self.R, self.zliq, self.ztop)
            else:
                xint, path_length, nvec = self.intersection_with_cylinder(self.x, self.t, self.R, self.zbot, self.zliq)

            # use the Fresnel equations to calculate the reflection and transmission coefficients
            self.interact_with_surface(xint, nvec)

            propagating = False

        return 0

