import numpy as np

class cylinder:

    def __init__(self, **kwargs):
        """
        Initialize a cylinder - really

        :param kwargs: R = radius,
                       h = height

        """ 

        self.radius = kwargs.pop('R',-1.0)
        self.height = kwargs.pop('h',-1.0)

        if (self.radius == -1) | (self.height == -1):
            print("cylinder::__init__ ERROR. Bad radius and/or height of cylinder")
            print("cylinder::__init__    r = ",self.radius," h = ",self.height)
            return
        else:
            print("cylinder::__init__ Define cylinder with R=",self.radius," and height=",self.height)

        #.
        # Calculate the area of teh cylinder side, top and bottom
        # Needed to generate events uniformly over the surfaces
        #
        area_cyl = 2 * np.pi * self.radius * self.height
        area_top  = np.pi * self.radius**2
        area_bot  = np.pi * self.radius**2

        area_tot = area_cyl + area_top + area_bot

        self.f_cyl = area_cyl / area_tot
        self.f_top = area_top / area_tot
        self.f_bot = area_bot / area_tot

        return


    def generate_point(self):
        """
        Generate a point at a random location on the cylinder

        :return:
        """
        xyz = np.zeros(3)
        # decide on which cylinder surface to generate a hit
        r = np.random.uniform(0.0,1.0)
        surface = ""
        if r<=self.f_cyl:
            surface = "cyl"
        elif (r>self.f_cyl) & (r<self.f_cyl+self.f_top):
            surface = "top"
        else:
            surface = "bot"

        # generate a hit
        phi = np.random.uniform(0,2*np.pi)
        if surface == "cyl":
            # cylinder
            xyz[0] = self.radius*np.cos(phi)
            xyz[1] = self.radius*np.sin(phi)
            xyz[2] = np.random.uniform(-self.height/2,+self.height/2)
        else:
            r = self.radius * np.sqrt(np.random.uniform(0,1.))
            xyz[0] = r * np.cos(phi)
            xyz[1] = r * np.sin(phi)
            if surface == "top":
                # top
                xyz[2] = + self.height/2
            elif surface == "bot":
                # bottom
                xyz[2] = - self.height/2.

        return {'x': xyz, 'surface': surface}
