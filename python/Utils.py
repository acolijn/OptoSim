import numpy as np  
from numba import njit

"""
This file contains functions that are used in the OpticalPhoton simulation.

Functions:
    calculate_position: Calculates the position of the photon after propagating a distance s along the trajectory.
    intersection_with_cylinder: Finds intersection of straight line photon trajectory with cylinder. Only intersections in the direction of the photon are considered.
    generate_lambertian: Generates a random unit vector based on Lambert's cosine law
    rotation_matrix_from_z_to_n: Calculates the rotation matrix to rotate the z-axis onto the vector n

"""

@njit
def calculate_position(x, t, s):
    """Calculates the position of the photon after propagating a distance s along the trajectory.

    Args:
        x (array): start point of photon
        t (array): direction of photon
        s (float): distance to propagate

    Returns:
        array: position of photon after propagating distance s
    """
    return (x[0] + s * t[0], x[1] + s * t[1], x[2] + s * t[2])

@njit
def intersection_with_cylinder_old(x, t, R, zb, zt):
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
    margin = 1e-6

    # Calculate the intersection points with the horizontal planes
    if abs(t[2]) > margin:  # Check if the photon is not parallel to the bottom plane
        t_bottom_plane = (zb - x[2]) / t[2]
        if t_bottom_plane >= 0.:
            s.append(t_bottom_plane)
            surface.append("bottom")

        # Calculate the intersection points with the top horizontal plane
        t_top_plane = (zt - x[2]) / t[2]
        if t_top_plane >= 0.:
            s.append(t_top_plane)
            surface.append("top")

    # Calculate coefficients for the quadratic equation for the cylinder shell
    A = t[0]**2 + t[1]**2
    B = 2 * (x[0] * t[0] + x[1] * t[1])
    C = x[0]**2 + x[1]**2 - R**2

    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C

    # Check if there are real solutions for the cylinder shell
    if discriminant >= 0.:
        # Calculate the solutions for t
        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        if t1 > 0.:
            s.append(t1)
            surface.append("cylinder")
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
        if t2 > 0.:
            s.append(t2)
            surface.append("cylinder")

    # Calculate the corresponding intersection points
    # Only find the intersection point furthest away from the start point. In this way we avoid selecting teh intersection point close to teh starting point
    # of the photon trajectory is found due to numerical imprecision. This is an isue if teh photon starts on the boundary of a volume.
    #
    path_length = -100.
    intersection_point = None

    # we calculate the normal vector to the surface at the intersection point (the normal vector points inward)
    normal_vec = np.zeros(3)

    for s_i in s:
        # Calculate the intersection point
        point = calculate_position(x, t, s_i)
        # Check if the intersection point is within the cylinder
        # if (zb - margin <= point[2] <= zt +margin) and (point[2] - x[2]) / t[2] >= 0. and  (np.sqrt(point[0]**2 + point[1]**2) <= R + margin):
        if (zb - margin <= point[2] <= zt +margin) and (np.sqrt(point[0]**2 + point[1]**2) <= R + margin):

            # Check if the intersection point is further away from the start point than the previous intersection point
            if s_i > path_length:
                intersection_point = point 
                path_length = s_i
                if surface[s.index(s_i)] == "bottom":
                    normal_vec = np.array([0., 0.,  1.])
                elif surface[s.index(s_i)] == "top":
                    normal_vec = np.array([0., 0., -1.])
                else:
                    len = np.sqrt(point[0]**2 + point[1]**2)
                    normal_vec = np.array([-point[0] / len, -point[1] / len, 0.])                        

    if intersection_point == None:
        print("No intersection points found")
        return None, None, None

    return intersection_point, path_length, normal_vec

MARGIN = 1e-6

@njit
def intersection_with_cylinder(x, t, R, zb, zt):
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
    
    def calculate_intersection_plane(axis_val, t_val, plane_val):
        if abs(t_val) > MARGIN:
            return (plane_val - axis_val) / t_val
        return -1.0
    
    def calculate_position(start, direction, distance):
        return start + direction * distance

    # Check intersection with bottom and top planes
    t_bottom_plane = calculate_intersection_plane(x[2], t[2], zb)
    t_top_plane = calculate_intersection_plane(x[2], t[2], zt)
    
    # Calculate coefficients for the quadratic equation for the cylinder shell
    A = t[0]**2 + t[1]**2
    B = 2 * (x[0] * t[0] + x[1] * t[1])
    C = x[0]**2 + x[1]**2 - R**2

    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C
    t1, t2 = -1.0, -1.0
    if discriminant >= 0.:
        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    possible_ts = [t_val for t_val in [t_bottom_plane, t_top_plane, t1, t2] if t_val > 0.0]
    
    intersection_point = np.zeros(3)
    max_path_length = -100.
    normal_vec = np.zeros(3)
    
    for t_val in possible_ts:
        point = calculate_position(x, t, t_val)
        if (zb - MARGIN <= point[2] <= zt + MARGIN) and np.linalg.norm(point[:2]) <= R + MARGIN:
            if t_val > max_path_length:
                max_path_length = t_val
                intersection_point = point
                if t_val == t_bottom_plane:
                    normal_vec = np.array([0., 0.,  1.])
                elif t_val == t_top_plane:
                    normal_vec = np.array([0., 0., -1.])
                else:
                    len_point = np.linalg.norm(point[:2])
                    normal_vec = np.array([-point[0] / len_point, -point[1] / len_point, 0.])

    #if intersection_point == None:
    #    print("No intersection points found")
    #    return np.zeros(3), max_path_length, np.zeros(3)    

    return intersection_point, max_path_length, normal_vec


@njit
def generate_lambertian(n):
    """
    Generates a random unit vector based on Lambert's cosine law

    Parameters
    ----------
    n : normal vector

    Returns
    -------
    new_unit_vector : new unit vector

    A.P. Colijn
    """
    # Generate a random azimuthal angle phi
    phi = 2 * np.pi * np.random.rand()

    # Generate a random cos(theta) value following Lambert's cosine law
    theta = np.arcsin(np.random.uniform(0., 1.0)) 
    cos_theta = np.cos(theta)

    # Calculate the new unit vector based on spherical coordinates
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    # Calculate the transformation matrix to rotate the vector onto n
    rotation_matrix = rotation_matrix_from_z_to_n(n)

    # Rotate the vector onto n
    new_unit_vector = np.dot(rotation_matrix, np.array([x, y, z]))

    return new_unit_vector

@njit
def rotation_matrix_from_z_to_n(n):
    """
    Calculates the rotation matrix to rotate the z-axis onto the vector n

    Parameters
    ----------
    n : normal vector

    Returns
    -------
    rotation_matrix : rotation matrix

    A.P. Colijn
    """
    # Ensure n is a unit vector
    n = n / np.linalg.norm(n)

    # Calculate the rotation axis and angle
    rotation_axis = np.cross([0.0, 0.0, 1.0], n)
    ###rotation_angle = np.arccos(np.dot([0.0, 0.0, 1.0], n))
    rotation_angle = np.arccos(n[2])

    # Calculate the rotation matrix using Rodrigues' formula
    sin_angle = np.sin(rotation_angle)
    cos_angle = np.cos(rotation_angle)

    K = np.array([[0.0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0.0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0.0]])

    rotation_matrix = np.eye(3) + sin_angle * K + (1.0 - cos_angle) * np.dot(K, K)

    return rotation_matrix
