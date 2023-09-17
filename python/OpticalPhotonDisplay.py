
from OpticalPhoton import *

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

scatter_types = ['specular', 'diffuse', 'lambertian', 'mirror', 'dielectric', 'dielectric_metal']

class OpticalPhotonDisplay(OpticalPhoton):
    """
    Class for displaying the path of an optical photon in the detector. The class inherits from
    OpticalPhoton. 

    The class has the following attributes:
        x : np.array
            Position of the photon in the detector.
        t : np.array
            Direction of the photon.
        R : float
            Radius of the detector.
        zbot : float
            Bottom of the detector.
        zliq : float    
            Liquid level of the detector.
        ztop : float    
            Top of the detector.
        current_medium : int
            Current medium the photon is in.
        alive : bool
            Status of the photon. True if the photon is alive, False if the photon is dead.
        path : list
            List to store the photon's path.

    The class has the following methods:
        propagate()
            Propagate the photon through the detector.
        plot()
            Plot the photon's path in 3D, top view, and side view.
        plot_3d_view(ax)
            Plot the photon's path in 3D.
        plot_top_view(ax)
            Plot the photon's path in the top view (X-Y plane).
        plot_side_view(ax)
            Plot the photon's path in the side view (X-Z plane).
        plot_3d_view_animation()
            Plot the photon's path in 3D with animation.

    A.P. Colijn
    """
    def __init__(self, **kwargs):
        """
        Initializes the OpticalPhotonDisplay class. The configuration is read from the config file.

        Parameters
        ----------
        config : str
            Filename of configuration file. Default is 'config.json'.

        A.P. Colijn
        """
        super().__init__(**kwargs)
        self.path = []  # List to store the photon's path
        self._last_point = None

    def propagate(self):
        """
        Propagate the photon through the detector. 

        A.P. Colijn
        """
        self.path = []  # List to store the photon's path
        propagating = True  
        # add initial position to path
        self.path.append(self.x.copy())  # Add the photon's position to the path list

        self.print()

        while propagating and self.alive:  # Check if the photon is alive in the loop condition

            # intersection with cylinder
            path_length = -100

            if self.current_medium == XENON_GAS:
                xint, path_length, nvec = intersection_with_cylinder(self.x, self.t, self.R, self.zliq, self.ztop)
            else:
                xint, path_length, nvec = intersection_with_cylinder(self.x, self.t, self.R, self.zbot, self.zliq)

            if path_length > 0.0:
                istat = self.interact_with_surface(xint, self.t, nvec)
                if istat == 1:
                    print('error in interact_with_surface')
                    self.alive = False 
            else:
                print('no intersection with cylinder')
                self.alive = False

            self.path.append(self.x.copy())  # Add the photon's position to the path list
            propagating = self.photon_propagation_status()
            self.print()

    def plot(self):
        """
        Plot the photon's path in 3D, top view, and side view.

        A.P. Colijn
        """
        fig = plt.figure(figsize=(15, 5))

        # 3D View
        ax1 = fig.add_subplot(131, projection='3d')
        fig.canvas.mpl_connect('scroll_event', self.zoom)   
        self.plot_3d_view(ax1)

        # Top View (X-Y plane)
        ax2 = fig.add_subplot(132)
        self.plot_top_view(ax2)

        # Side View (X-Z plane)
        ax3 = fig.add_subplot(133)
        self.plot_side_view(ax3)

        plt.tight_layout()
        plt.show()

    def plot_3d_view(self, ax):
        """"
        Plot the photon's path in 3D.

        A.P. Colijn
        """ 
        # Plot the detector as two cylinders: one for gas and one for liquid
        u = np.linspace(0, 2 * np.pi, 100)

        # Gas Xenon
        v_gas = np.linspace(self.zliq, self.ztop, 100)
        U_gas, V_gas = np.meshgrid(u, v_gas)
        X_gas = self.R * np.cos(U_gas)
        Y_gas = self.R * np.sin(U_gas)
        Z_gas = V_gas
        ax.plot_surface(X_gas, Y_gas, Z_gas, color='lightblue', alpha=0.5)

        # Liquid Xenon
        v_liquid = np.linspace(self.zbot, self.zliq, 100)
        U_liquid, V_liquid = np.meshgrid(u, v_liquid)
        X_liquid = self.R * np.cos(U_liquid)
        Y_liquid = self.R * np.sin(U_liquid)
        Z_liquid = V_liquid
        ax.plot_surface(X_liquid, Y_liquid, Z_liquid, color='dodgerblue', alpha=0.5)

        # Plot the photon's path
        path = np.array(self.path)
        if path.ndim == 2:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='r', marker='o', markersize=2)
        ax.set_title('3D View')

    def draw_detector(self, ax):
        """
        Draw the detector in 3D.

        A.P. Colijn
        """
        # Plot the detector as two cylinders: one for gas and one for liquid
        u = np.linspace(0, 2 * np.pi, 100)

        # Gas Xenon
        v_gas = np.linspace(self.zliq, self.ztop, 100)
        U_gas, V_gas = np.meshgrid(u, v_gas)
        X_gas = self.R * np.cos(U_gas)
        Y_gas = self.R * np.sin(U_gas)
        Z_gas = V_gas
        ax.plot_surface(X_gas, Y_gas, Z_gas, color='lightblue', alpha=0.5)

        # Liquid Xenon
        v_liquid = np.linspace(self.zbot, self.zliq, 100)
        U_liquid, V_liquid = np.meshgrid(u, v_liquid)
        X_liquid = self.R * np.cos(U_liquid)
        Y_liquid = self.R * np.sin(U_liquid)
        Z_liquid = V_liquid
        ax.plot_surface(X_liquid, Y_liquid, Z_liquid, color='dodgerblue', alpha=0.5)

    def get_path_length(self):
        """_summary_

        Args:
            _type_:

        Returns:
            _type_:

        A.P. Colijn
        """
        path = np.array(self.path)
        if path.ndim == 2:
            return np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        else:
            return 0.0
        
    def get_number_of_reflections(self):
        """_summary_

        Args:
            _type_:

        Returns:
            _type_:

        A.P. Colijn
        """
        return len(self.path) - 1

    def plot_3d_view_animation(self):
        """
        Plot the photon's path in 3D with animation.
        A.P. Colijn
        """
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D View')

        # Draw the detector
        self.draw_detector(ax)

        # Plot the photon's path
        path = np.array(self.path)

        # Calculate segment distances
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        # Number of total steps for the entire path
        total_steps = int(total_distance*2)  # adjust as needed

        # Calculate steps per segment
        steps_per_segment = (distances / total_distance * total_steps).astype(int)
        print(steps_per_segment)
        # Interpolate the path
        interpolated_path = []
        for i in range(len(path) - 1):
            xs = np.linspace(path[i][0], path[i+1][0], steps_per_segment[i])
            ys = np.linspace(path[i][1], path[i+1][1], steps_per_segment[i])
            zs = np.linspace(path[i][2], path[i+1][2], steps_per_segment[i])
            for j in range(steps_per_segment[i]):
                interpolated_path.append([xs[j], ys[j], zs[j]])
        interpolated_path = np.array(interpolated_path)

        def update(i):
            if i == 0:
                self._last_point = interpolated_path[0]
                return

            xs = [self._last_point[0], interpolated_path[i, 0]]
            ys = [self._last_point[1], interpolated_path[i, 1]]
            zs = [self._last_point[2], interpolated_path[i, 2]]
            ax.plot(xs, ys, zs, color='r')

            self._last_point = interpolated_path[i]

            return fig,

        # Set fixed axis limits based on the detector's dimensions
        ax.set_xlim([-self.R, self.R])
        ax.set_ylim([-self.R, self.R])
        ax.set_zlim([self.zbot, self.ztop])

        ani = FuncAnimation(fig, update, frames=len(interpolated_path), interval=35, blit=False)
        plt.tight_layout()
        ani.save('photon_path.gif', writer='pillow', fps=10, dpi=200, bitrate=2000)

        plt.show()


    def plot_top_view(self, ax):
        """
        Plot the photon's path in the top view (X-Y plane).

        A.P. Colijn
        """
        path = np.array(self.path)
        if path.ndim == 2:
            ax.plot(path[:, 0], path[:, 1], color='r', marker='o', markersize=2)
        circle = plt.Circle((0, 0), self.R, color='c', fill=False)
        ax.add_artist(circle)
        ax.set_aspect('equal', 'box')
        ax.set_title('Top View (X-Y plane)')
        ax.set_xlim(-self.R - 1, self.R + 1)
        ax.set_ylim(-self.R - 1, self.R + 1)

    def plot_side_view(self, ax):
        path = np.array(self.path)
        if path.ndim == 2:
            ax.plot(path[:, 0], path[:, 2], color='r', marker='o', markersize=2)

        # Fill the region for Liquid Xenon
        ax.fill_between([-self.R, self.R], self.zbot, self.zliq, color='dodgerblue', alpha=0.5)

        # Fill the region for Gas Xenon
        ax.fill_between([-self.R, self.R], self.zliq, self.ztop, color='lightblue', alpha=0.5)

        ax.set_xlim(-self.R, self.R)
        ax.set_ylim(self.zbot, self.ztop)
        ax.set_title('Side View (X-Z plane)')

        ax.set_xlim(-self.R - 1, self.R + 1)
        ax.set_ylim(self.zbot - 1, self.ztop + 1)

    def zoom(event):
        ax = event.inaxes
        factor = 1.1 if event.button == 'up' else 0.9
        ax.dist /= factor
        event.canvas.draw()


