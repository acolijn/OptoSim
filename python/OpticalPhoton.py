import numpy as np
import math
import random

XENON_GAS=0
XENON_LIQ=1

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


        self.x0 = np.zeros(3)
        self.t = np.array([0.,0.,0.])

        self.current_medium = XENON_GAS


    def set_photon_position(self, x0):
        """
        Set the photon position

        Parameters
        ----------
        x0(float) : array_like
            The initial position of the photon

        A.P. Colijn
        """
        self.x0 = x0

        if self.x0[2]>self.zliq:
            self.current_medium = XENON_GAS
        else:
            self.current_medium = XENON_LIQ
    
    def generate_photon_direction(self):
        """
        Generate a photon with a random direction t=(tx,ty,tz). The direction is isotropic.

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
        self.t = (tx, ty, tz)

        return 

    def intersection_with_horizontal_plane(self, x, t, z0):
        """

        Args:
            x (_type_): start point of photon
            t (_type_): direction of photon
            z0 (_type_): z-coordinate of the plane

        Returns:
            _type_: _description_
        """
        # Ensure that the line is not parallel to the plane
        if t[2] == 0:
            return None, None  # No intersection
        
        path_length = (z0 - x[2]) / t[2]

        # Check if the intersection point is in front of the starting point
        if path_length > 0:
            intersection_point = (x[0] + path_length * t[0], x[1] + path_length * t[1], z0)
            return intersection_point, path_length
        else:
            return None, None  # The intersection point is behind the starting point

        return None, None
    

    def intersection_with_cylinder(self, R, zb, zt):
        # Extract components of the line direction
        x = self.x0
        tx, ty, tz = self.t

        # Initialize a list to store intersection points
        intersection_points = []

        # Calculate the intersection points with the bottom horizontal plane
        t_bottom_plane = (zb - x[2]) / tz
        if t_bottom_plane >= 0:
            intersection_points.append((x[0] + t_bottom_plane * tx, x[1] + t_bottom_plane * ty, zb))

        # Calculate the intersection points with the top horizontal plane
        t_top_plane = (zt - x[2]) / tz
        if t_top_plane >= 0:
            intersection_points.append((x[0] + t_top_plane * tx, x[1] + t_top_plane * ty, zt))

        # Calculate coefficients for the quadratic equation for the cylinder shell
        A = tx**2 + ty**2
        B = 2 * (x[0] * tx + x[1] * ty)
        C = x[0]**2 + x[1]**2 - R**2

        # Calculate the discriminant
        discriminant = B**2 - 4 * A * C

        # Check if there are real solutions for the cylinder shell
        if discriminant >= 0:
            # Calculate the solutions for t
            t1 = (-B + np.sqrt(discriminant)) / (2 * A)
            t2 = (-B - np.sqrt(discriminant)) / (2 * A)

            # Calculate the corresponding intersection points
            intersection_points += [
                (x[0] + t1 * tx, x[1] + t1 * ty, x[2] + t1 * tz),
                (x[0] + t2 * tx, x[1] + t2 * ty, x[2] + t2 * tz)
            ]

        # Filter intersection points with positive path length within the height of the 
        margin = 1e-6
        valid_intersection_points = [
            point for point in intersection_points if (zb - margin <= point[2] <= zt +margin) and (point[2] - x[2]) / tz >= 0 and (np.sqrt(point[0]**2 + point[1]**2) <= R + margin)
        ]

        return valid_intersection_points
    
    def propagate(self):

        propagating = True

        while propagating:

            # intersection with cylinder
            if self.current_medium == XENON_GAS:
                intersection_points = self.intersection_with_cylinder(self.R, self.zliq, self.ztop)
            else:
                intersection_points = self.intersection_with_cylinder(self.R, self.zbot, self.zliq)


        return 0

