import numpy as np
import pandas as pd

from numba import jit

class particle:
    #@jit()

    def __init__(self, **kwargs):
        """
        Initialize a particle
        :param kwargs:
            type = particle specimen (value = gamma, <others defined later>)
            energy = energy of the particle in keV
            phys = physics class
            geometry = geometry description. Instant of cylinder class
            fiducial  = fiducial volume. Instant of cylinder class
            vrt = variance reduction technique. Values: [None, "fiducial_scatter"]
            nscatter_max = maximum number of scatters (only when vrt<>None)
            edep_max = maximum energy deposit in the xenon (keV)
            debug = show debug printout information [Default = False]
        """
        # # # print("particle::initialize")

        #np.random.seed(123456)

        self.type = kwargs.pop('type', None)
        self.energy = kwargs.pop('energy', 0.0)
        self.phys = kwargs.pop('physics',None)
        self.cryostat = kwargs.pop('geometry',None)
        self.fiducial = kwargs.pop('fiducial',None)
        self.debug = kwargs.pop('debug',False)

        #
        # vrt = variance reduction technique
        #
        self.vrt = kwargs.pop('vrt', None)
        #
        # default number of scatters = 1 (only with vrt<>None)
        #
        self.nscatter_max = kwargs.pop('nscatter_max',1)
        self.edep_max = kwargs.pop('edep_max',100000.)

        #
        # Particle weight. Will be reduced when using VRT methods
        #
        self.weight = 1.0

        if self.debug == True:
            print('particle::propagate VRT:',self.vrt)

        self.x0 = np.zeros(3)
        self.x0start = np.zeros(3)
        self.xint = []

        self.direction = np.zeros(3)
        self.edep = 0 # deposited energy

        # generate the x0 and direction of the particle
        self.generate()

    def Print(self):
        """
        Print particle
        :return:
        """
        print("particle::print PARTICLE STATUS")
        print("particle::print type = ",self.type)
        print("particle::print energy = ",self.energy," keV")
        print("particle::print origin = ",self.x0," cm")
        print("particle::print direction = ",self.direction)
        print("particle::print deposited energy = ",self.edep," keV")
        print("particle::print")
        print("particle::print variance reduction = ",self.vrt)
        print("particle::print n scatter max = ",self.nscatter_max)
        print("particle::print e deposit max = ",self.edep_max)

    def generate(self):
        """
        Generate the starting point of the particle and its direction
        :param:
        :return:
        """

        #
        # generate x0 of the particle to be at a random location on the cylinder
        #
        self.x0 = self.cryostat.generate_point()['x']
        self.x0start = self.x0

        #
        # generate a random direction for the particle
        #
        cost = np.random.uniform(-1, 1)
        sint = np.sqrt(1 - cost ** 2)
        phi = 2 * np.pi * np.random.uniform(0, 1)

        tx = np.cos(phi) * sint
        ty = np.sin(phi) * sint
        tz = cost

        #
        # store theta and phi
        #
        self.theta = np.arccos(cost)
        self.phi = phi

        #
        # store the directional unit vector
        #
        self.direction = np.array([tx, ty, tz])


        return

    def intersect(self, cylinder):
        """
        Intersect particle track with cylinder

        :param cylinder: instant of teh cylinder class, containing the geo definition
        :return number of intersections with the cylinder:
        0=particle does not hit cylinder
        1=particle probably originates somewhere on the surface, but moving away from the cylinder
        2=particle hits the cylinder at two spots

        """

        #
        # intersection with top plane of cylinder
        #
        stop = self.intersect_with_plane(cylinder, 'top')
        #
        # intersection with bottom plane of cylinder
        #
        sbot = self.intersect_with_plane(cylinder, 'bot')
        #
        # intersection with cylindrical shell
        #
        ssid = self.intersect_with_side(cylinder)
        #
        # sort the list by ascending path length, s ....
        #
        intersections = sorted([stop,sbot,ssid[0],ssid[1]],reverse=False)
        #
        # remove s=0 entries
        #
        for i in range(4):
            try:
                intersections.remove(0.)
            except:
                pass

        return intersections

    def intersect_with_side(self, cylinder):
        """
        Intersect the particle with the cylinder shell
        :param R,h:
        :return: intersections
        """

        intersections = [0,0]
        n = 0

        A = self.direction[0] ** 2 + self.direction[1] ** 2
        B = 2 * (self.x0[0] * self.direction[0] + self.x0[1] * self.direction[1])
        C = self.x0[0] ** 2 + self.x0[1] ** 2 - cylinder.radius ** 2

        discriminant = B ** 2 - 4 * A * C

        if discriminant >= 0:
            for sign in (-1,1):
                s = (-B + sign*np.sqrt(discriminant)) / (2 * A)
                xint = self.x0 + s * self.direction
                # is it hitting the cylinder? or is it outside?
                if (np.abs(xint[2]) < cylinder.height/2) & (s>1e-5): # only tracks with positive pathlength, remove intersect with zero pathlength
                    #
                    # good intersection..... add to the list
                    #
                    intersections[n] = s
                    n = n+1

        return intersections

    def intersect_with_plane(self, cylinder, type):
        """
        Intersect the particle track with the top/bottom plane
        :param R,h,type:

        :return: s
        """
        sint = 0

        zint = 0
        if type == "top":  # top plane
            zint = cylinder.height / 2
        else:  # bottom plane
            zint = -cylinder.height / 2

        tz = self.direction[2]

        # calculate the path length to the intersection point
        if tz != 0.000:
            s = (zint - self.x0[2]) / tz
        else:
            print("particle::intersect_with_plane() WARNING: tz = ", tz)
            s = 0

        # calculate the position of the intersection point from the track equation
        x = self.x0 + s * self.direction

        # calculate the radius  of teh intersect and check whether it is in range
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        if (r < cylinder.radius) & (s>1e-5): #  only tracks with positive pathlength, remove intersect with zero pathlength
        #if (r < R) & (s > 1e-5):  # only tracks with positive pathlength, remove intersect with zero pathlength
            # good intersection.... add to the list
            sint = s

        return sint

    def propagate(self):
        """
        Propagate the particle
        :return:
        """

        terminate = False
        if self.debug == True:
            print("particle::propagate Next event")

        self.nscatter = 0
        while terminate == False:
            #
            # intersection of track with cryostat
            #
            cryostat_intersections = self.intersect(self.cryostat)
            #
            # do we intersect?
            #
            if len(cryostat_intersections) > 0: # we have a track through the LXe cylinder.
                #
                # maximum path length before exiting the cryostat
                #
                s_max = cryostat_intersections[0]
                fiducial_intersections = []
                s_fiducial_entry = 0.0
                s_max_fiducial = -1.0
                #
                # * determine the path length to the fiducial volume
                #
                # * only transport to the fiducial the first time.
                #
                if self.vrt == 'fiducial_scatter':

                    if self.nscatter == 0: # transport to the fiducial volume
                        if self.debug == True:
                            print('particle::propagate VRT - transport to fiducial volume is ON')

                        fiducial_intersections = self.intersect(self.fiducial)
                        if len(fiducial_intersections) > 0:
                            s_fiducial_entry = fiducial_intersections[0]
                            s_fiducial_exit = fiducial_intersections[1]
                            self.weight = self.weight*self.phys.get_att_probability(energy=self.energy,distance=s_fiducial_entry)
                            self.update_particle('transport',s_fiducial_entry,0,0,1.0)
                            #
                            # we dont recalculate the interscetion point with the cryostat again. Just
                            # calculate: s_max' = s_max - s_fiducial_entry
                            #
                            s_max = s_max - s_fiducial_entry
                            s_max_fiducial = s_fiducial_exit-s_fiducial_entry
                        else:
                            terminate = True
                            continue # jump out of while loop
                    elif (self.nscatter>0) & (self.nscatter<self.nscatter_max):
                        #  multiple scatters as well!
                        #  determine s to boundary of fiducial volume.... (we are already inside so we will find an intersection)
                        fiducial_intersections = self.intersect(self.fiducial)
                        if len(fiducial_intersections)>0:
                            s_max_fiducial = fiducial_intersections[0]
                        else:
                            print("particle::propagate ERROR.... bad intersection. discard event.")
                            terminate = True
                            continue

                    elif self.nscatter == self.nscatter_max:
                        #
                        # path length to the outside world
                        #
                        s_max = cryostat_intersections[0]
                        #
                        # probability to reach teh outside world
                        #
                        prob = self.phys.get_att_probability(energy=self.energy, distance=s_max)
                        self.weight = self.weight*prob
                        terminate = True
                        continue

                # 1. if one intersection this gives the exit point.
                # 2. if two intersection this gives the entry point to a new volume (needed for fiducial)
                #    volume variance reduction
                # 3. s_max_fiducial is maximum path length in the fiducial volume. If no variance reduction
                #    is done it is set to -1.0. In that case it is ignored inside the routine
                s_gen = self.generate_interaction_point(smax=s_max_fiducial)

                #
                # do we have an interaction inside the cryostat?
                #
                if(s_gen < s_max): # we have a hit inside the xenon
                    #
                    # actual scattering: either Compton or Photo-electric effect
                    #
                    process = self.scatter(s_gen)
                    #
                    # if the scattering was by the Photo-electric effect, the photon is terminated
                    #
                    if process == "pho":
                        terminate = True
                else:
                    #
                    # terminate photon tracking if it exits the xenon volume
                    #
                    terminate = True
            else:
                #
                # no intersection.... terminate the propagator
                #
                terminate = True

        if self.debug == True:
            print('particle::propagate exit propagator')

        return

    def get_info(self, i):
        data = []


        return data

    def generate_interaction_point(self, **kwargs):
        """
        Propagate the particle to the next interaction location
        :param kwargs: smax = maximum path length to generate (default = -1 -> infinite)

        :return:
        """

        smax = kwargs.pop('smax',-1)

        # cdf for path length
        mu = self.phys.get_att(energy=self.energy)

        rmax = 1
        if smax > 0.0:
            rmax = 1.0 - np.exp(- smax / mu)
            self.weight = self.weight*rmax

        #
        # generate the path length
        #
        r = np.random.uniform(0,rmax)
        L =  - np.log(1-r) * mu

        return L

    def scatter(self,s):
        """
        Scatter the photon after a path-length s. Choose between Compton scatter and teh PE effect

        :return:
        """
        # select the scatter process
        process = self.select_scatter_process()

        if process == 'inc':
            #
            # Compton scatter
            #
            theta_s, phi_s, w_s = self.phys.do_compton(self.energy, self.edep_max)
            self.update_particle('inc',s,theta_s, phi_s, w_s)

            # calculate teh energy deposit in the xenon
        else:
            #
            # Photo-electric effect
            #
            self.update_particle('pho',s,None,None,1.)

        return process

    def update_particle(self, process, s_scatter, theta_scatter, phi_scatter, w):
        """
        Update the particle properties after scattering

        :param process: ['inc', 'pho', 'transport']
        :param s_scatter:
        :param theta_scatter:
        :param phi_scatter:
        :param w:

        :return:
        """

        #
        # calculate the interaction position
        #
        x0_new = self.x0 + self.direction * s_scatter

        #
        # update the particle for the different types of processes
        #
        t_new = [0,0,0]
        enew = 0
        edep = 0

        if process == 'inc':
            #
            # 1. Compton scattering
            #

            #
            # calculate the particle direction after interaction
            #
            sint = np.sin(theta_scatter)
            cost = np.cos(theta_scatter)

            t = [sint * np.cos(phi_scatter), sint * np.sin(phi_scatter), cost]
            m = self.calculate_rotation_matrix()
            t_new = np.array(m.dot(t))
            t_new = t_new.flatten()

            #
            # calculate the new photon energy
            #
            enew = self.phys.P(self.energy,cost)*self.energy
            #
            # calculate the deposited energy in this interaction
            #
            edep = self.energy - enew
            if edep<0:
                edep = 0

            self.edep_max = self.edep_max - edep
            
            if self.edep_max<0:
                self.edep_max = 0
                #print("particle::update_particle WARNING edep_max=",self.edep_max," E = ",self.energy," E_new = ",enew)

        elif process == 'pho':
            #
            # 2. Photo-electric effect
            #

            #
            # all the energy is deposited
            #
            enew = 0
            edep = self.energy
            t_new = [0,0,0]


        #
        # Overwrite particle parameters
        #
        if process != 'transport':
            self.xint.append([x0_new, edep, process])
            self.direction = t_new
            self.energy = enew
            self.edep = self.edep + edep
            self.nscatter = self.nscatter + 1

        #
        # always a new x0.....
        #
        self.x0 = x0_new
        self.weight = self.weight*w


        return

    def select_scatter_process(self):
        """
        Select either Photo-electric or Compton based on relative cross section
        :return:
        """

        # get the gamma energy
        E = self.energy
        # if we deal with Compton we will scatter, otherwise full absorption (not completely true, but I dont want to deal with pair creation)
        sigma_total = self.phys.get_sigma(process='att',energy=E)
        sigma_inc = self.phys.get_sigma(process='inc',energy=E)
        frac = sigma_inc / sigma_total

        if self.edep_max < self.energy:
            #
            # if the maximum allowed energy deposit is lower than the energy of the particle, we switch off
            # the photo-electric effect and we correct the particle weight:
            #     weight = weight* (1-sigma_PE / sigma_tot) = weight * sigma_inc/sigma_tot
            #
            process = 'inc'
            self.weight = self.weight*frac
        else:
            #
            # choose incoherent or photo-electric effect based on their relative cross sections
            #

            r = np.random.uniform(0,1)
            # # # print('stot = ',sigma_total,' sinc = ',sigma_inc,' frac = ',frac,' r= ',r)
            if r < frac:
                process = 'inc'
            else:
                process = 'pho'

            # If we require multiple scatters in the vrt MC we do not want to terminate the photon
            # prematurely.
            if (self.vrt == 'fiducial_scatter') and (self.nscatter < self.nscatter_max-1):
                process = 'inc'
                self.weight = self.weight * frac

        return process

    def calculate_rotation_matrix(self):
        """
        Calculate the rotation matrix to go from local scattering system to global coordinate system. Needed
        after a scatter.
        :return:
        """

        theta = np.arccos(self.direction[2])
        cth = np.cos(theta)
        sth = np.sin(theta)

        phi = np.arctan2(self.direction[1],self.direction[0])
        cph = np.cos(phi)
        sph = np.sin(phi)

        rot_phi = np.matrix([[cph,-sph,0],[sph,cph,0],[0,0,1]])
        rot_the = np.matrix([[cth,0,sth],[0,1,0],[-sth,0,cth]])

        mrot = rot_phi * rot_the

        return mrot
