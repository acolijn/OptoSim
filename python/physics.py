import numpy as np
import numba


class em_physics:
    """
    The electromagnetic physics driver for the simulation
    (routines routinely stole from https://github.com/ErikHogebirk/DMPlots)
    """

    def __init__(self, **kwargs):
        """
        Initialize EM physics

        :param kwargs:
        """

        # np.random.seed(123456)

        # physical constants
        self.m_electron = 510.9989461  # keV
        self.alpha = 1.0  # fine structure constant
        self.rc = 1.0  # reduced Compton wavelength rc=hbar/me/c

        # extract data from the NIST textfile
        self.extract_nist('../data/xenon_cross_section/gamma_sigma.txt')
        # extract data from the Hubbel form factor data
        self.extract_formfactors('../data/xenon_cross_section/formfactors.txt')

        self.Z_xenon = 54

        return

    def extract_formfactors(self, fn):
        """
        Extract the coherent and incoherent form factors

        :param fn: file containing the cross section information for xenon from Hubbel (1975)
        :return: cross section data
        """
        all_params = []
        with open(fn, 'r') as f:
            for i, line in enumerate(f):
                # Skip header
                if i < 2:
                    continue
                # Skip blank lines
                if len(line) < 5:
                    continue
                # Cut off the shell indication and newline character
                params = [np.float(el) for el in line.split()]

                all_params.append(params)

        d = np.array(all_params)

        self.x = d[:, 0]
        self.Fx = d[:, 1]
        self.Sx = d[:, 2]

        return

    ####@numba.njit
    def calculate_x(self, k, cost):
        """
        Calculate x
        :param k: gamma k factor E/me
        :return:
        """
        x = 2 * k * np.sqrt((1. - cost) / 2.) * 20.60774  # uit Hubbell paper
        return x

    def extract_nist(self, fn):
        """
        Extract the cross section and attenuation from the data file (stolen from Erik Hogenbirk)

        :param fn: file containing the cross section information for xenon from NIST
        :return: cross section data
        """
        all_params = []
        with open(fn, 'r') as f:
            for i, line in enumerate(f):
                # Skip header
                if i < 6:
                    continue
                # Skip blank lines
                if len(line) < 5:
                    continue
                # Cut off the shell indication and newline character
                line = line[7:-1]
                params = [np.float(el) for el in line.split(sep=' ')]

                all_params.append(params)

        d = np.array(all_params)

        self.rho = 3.0  # gram/cm3
        self.e = d[:, 0]  # energy range
        self.sigma_coh = d[:, 1]  # Coherent
        self.sigma_inc = d[:, 2]  # Incoherent. This is Compton scattering
        self.sigma_pho = d[:, 3]  # photoelectric absorption
        self.sigma_pp_nuc = d[:, 4]  # pair production nuclear field
        self.sigma_pp_el = d[:, 5]  # pair production electron field (order of magnitude below)
        self.sigma_pp = self.sigma_pp_nuc + self.sigma_pp_el
        self.sigma_att = d[:, 7]  # Total attenuation without coherent scattering
        self.att = 1 / (self.rho * self.sigma_att)  # Attenuation length

        return

    def get_sigma(self, **kwargs):
        """

        :param kwargs:
        process=["att" = total without coherent,
                 "pho" = PE absorption,
                 "inc" = incoherent scatter - Compton,
                 "pp"  = pair creation
        energy=energy of gamma in keV,

        :return: cross section in cm2/g
        """
        process = kwargs.pop('process', 'att')
        energy = kwargs.pop('energy', -1.0)
        if energy < 0:
            print('emphysics::get_sigma ERROR wrong energy. E=', energy)
            return -1

        if process == "att":  # total cross section
            mu = np.interp(energy / 1e3, self.e, self.sigma_att)
        elif process == "pho":  # PE absorption
            mu = np.interp(energy / 1e3, self.e, self.sigma_pho)
        elif process == "inc":  # Compton
            mu = np.interp(energy / 1e3, self.e, self.sigma_inc)
        elif process == "pp":  # pair creation
            mu = np.interp(energy / 1e3, self.e, self.sigma_pp)
        else:
            print('em_physics::get_sigma ERROR wrong process selected')

        return mu

    def get_att(self, **kwargs):
        """
        Calculate the attenuation length

        :param kwargs:
            E=energy of gamma in keV
        :return: attenuation length in cm
        """
        energy = kwargs.pop('energy', -1.0)

        mu = np.interp(energy / 1e3, self.e, self.sigma_att)
        return 1 / (mu * self.rho)

    def get_att_probability(self, **kwargs):
        """
        Get the probability for a particle of energy E, to travel distance dx

        :return:
        """
        energy = kwargs.pop('energy', -1.0)
        dx = kwargs.pop('distance', -1.0)

        mu = self.get_att(energy=energy)

        prob = np.exp(- dx / mu)
        return prob

    def calculate_cost_min(self, **kwargs):
        """
        Calculate the minimal scatter angle for a given maximum energy deposit

        :param kwargs:
                energy = photon energy (keV)
                de_max = maximum energy deposit (keV)

        :return: cos(theta)_min
        """
        energy = kwargs.pop('energy', -1)
        de_max = kwargs.pop('de_max', -1)

        if (energy < 0) | (de_max < 0):
            print("physics::calculate_cost_min ERROR Bad energy or de_max. E=", energy, " keV dE_max =", de_max, " keV")

        cost_min = 1.0 - self.m_electron * (1. / (energy - de_max) - 1. / energy)

        if cost_min < -1:
            cost_min = -1.0

        return cost_min

    def P(self, Eg, cost):
        """
        Calculate the Eg'/Eg as a function of cos(theta), where
        Eg' is the gamma energy after Compton scatter

        :param Eg: gamma energy
        :param cost: cosine of the scattering angle
        :return:
        """

        P = 1. / (1. + (Eg / self.m_electron) * (1 - cost))

        return P

    def KleinNishina(self, Eg, cost, **kwargs):
        """
        Calculate the Klein-Nishina differential cross section for
        Compton scattering

        :param cost: cosine of the scattering angle
               Eg: gamma energy in keV
               **kwargs: formfactor = [True (default)/False] Use Hubbel form factor or not
        :return: dsigma/dOmega
        """

        formfactor = kwargs.pop('formfactor', True)
        re2 = 0.07940775  # barn

        # Klein Nishina
        k = Eg / self.m_electron
        FF = 1 + k * (1.0 - cost)
        KN = re2 * (1.0 + cost ** 2 + k ** 2 * (1 - cost) ** 2 / FF) / FF ** 2 / 2

        # total differential cross section per atom scales with Z
        S = self.Z_xenon
        # correct with the form factor
        if formfactor == True:
            x = self.calculate_x(k, cost)
            S = self.get_S(x)

        return KN * S

    ###@numba.njit
    def get_S(self, x):
        """
        Interpolate S(x)
        :param x:
        :return: S(x)
        """
        return np.interp(x, self.x, self.Sx)

    def do_compton(self, energy, de_max):
        """
        Select the Compton scattering angle based on the Klein-Nishina differential cross section

        :param energy: gamma energy
        :param de_max: maximum energy deposit
        :return:
        """

        #
        # if the energy deposit is unrestricted then cos(theta)_min = -1.0
        #
        cost_min = -1.0
        rmin = 0.
        weight = 1.0

        #
        # make the cdf from the differential cross section
        #
        cost_range = np.linspace(-1.0, +1.0, 10001, endpoint=True)
        dsigma = self.KleinNishina(energy, cost_range, formfactor=True)
        cdf = dsigma.cumsum() / dsigma.sum()
        #
        # draw a random number from the dsigma distribution to get scatter angle
        #
        if de_max < energy:
            cost_min = self.calculate_cost_min(energy=energy, de_max=de_max)
            rmin = np.interp(cost_min, cost_range, cdf)

        weight = 1 - rmin
        # # # print('weight ..... =',weight,' rmin = ',rmin,' cost_min =',cost_min)
        theta = np.arccos(np.interp(np.random.uniform(rmin, 1.), cdf, cost_range))
        #
        # random angle in phi between 0 and 2pi
        #
        phi = 2 * np.pi * np.random.uniform(0.1)

        return theta, phi, weight
