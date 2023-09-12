
from OpticalPhoton import *

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OpticalPhotonDisplay(OpticalPhoton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = []  # List to store the photon's path

    def propagate(self):
        """
        Propagate the photon through the detector. 

        A.P. Colijn
        """
        self.path = []  # List to store the photon's path
        propagating = True
        # add initial position to path
        self.path.append(self.x.copy())  # Add the photon's position to the path list


        print('is the photon alive? ',self.alive)
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

    def plot(self):
        fig = plt.figure(figsize=(15, 5))

        # 3D View
        ax1 = fig.add_subplot(131, projection='3d')
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


    def plot_top_view(self, ax):
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
