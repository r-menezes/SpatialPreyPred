import numpy as np
from scipy.spatial import KDTree
from scipy.stats import kstest, skew, binom
from scipy.interpolate import interp1d
import pandas as pd
# from numba import njit, cuda
from p_tqdm import p_umap

rng = np.random.default_rng(16558947)

class Environment():
    def __init__(self, Nagents, inter_distance, inter_rate, birth_rate, death_rate, sigma, centers, dispersal_kernel, tau, noise, data_interval=50, h=1e-3, env_size=1e2, seed=34567, rng=None, id=None, transient_steps=1000, check_termination_N_data=40, **kwargs):

        self.id = id
        
        # Setup random number generator
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        # Environment properties
        self.env_size = env_size
        self.h = h
        self.t = 0.
        self.steps = 0

        # Number, interactions
        self.N = Nagents
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        # self.interaction_distance = inter_distance
        self.interaction_parameter = inter_rate
        
        self.births = 0
        self.deaths = 0
        self.ts = []
        self.Ns = []

        # Dispersal kernel
        # assume a gaussian dispersal kernel with stdev dispersal_kernel
        self.dispersal_kernel = dispersal_kernel

        # Initial Position
        if centers == 'random':
            self.centers = self.rng.uniform(high=env_size, size=(self.N, 2))
        elif centers == 'origin':
            self.centers = np.zeros((self.N, 2))
        else:
            self.centers = centers

        self.x = self.centers

        # Competition Kernel
        # assume a gaussian dispersal kernel with stdev competition_kernel
        # cutoff is placed after 3 stdev
        self.competition_kernel = inter_distance
        self.interaction_max_distance = 3*inter_distance

        # interaction-related
        self.positions_tree = KDTree(self.x, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        self.possible_interacting_pairs = None
        self.n_interacting_pairs = None
        
        # Stochastic properties
        # OU model:
        # dX = -(1/tau)*(X - center)*dt + sqrt(g)*dW
        # dW is white noise
        #
        # the asymptotic position distribution is gaussian, with variance sigma**2 = g*tau/2
        self.tau = tau
        self.sigma = sigma
        self.noise = noise

        # OU_movement generator
        self.OU_mover = milshtein_BM_N_2D(rng=self.rng, x0=self.x, g=self.noise, N=self.N)

        # Gillespie variables
        self.total_rate = 0.
        self.total_birth_rate = 0.
        self.total_interaction_rate = 0.
        self.delta_t = 0.

        # Stored data
        self.data = pd.DataFrame(columns=['t', 'events', 'N', 'births','deaths']) # 'rep', 'tau', 'inter', 'interRate', 'birthRate', 
        self.data_interval = data_interval

        # Termination
        self.check_termination_N_data = check_termination_N_data
        self.theoretical_equilibrium_CDF = self.generate_binom_theoreticalCDF(self.data_interval, .5)
        #(avoid error for lack of data in function call)
        self.transient_steps = np.max([transient_steps, self.check_termination_N_data*self.data_interval])

    def run(self, max_steps):
        self.t = 0.
        self.steps = 0
        while self.steps < max_steps:
            # perform a gillespie step
            self.gillespie_step()
            
            # store variables to calculate ave_abundance
            self.ts.append(self.delta_t)
            self.Ns.append(self.N)

            self.steps += 1

            # record data 
            if self.steps % self.data_interval == 0:
                self.store_data()
                self.births = 0
                self.deaths = 0
                
                # check termination
                if self.steps > self.transient_steps:
                    test, test_value = self.stop_condition()
                    if test:
                        return self.on_exit(cause='equilibrium', testValue=test_value)

                #++ I should implement ts and Ns as some heap or stack
                self.ts = []
                self.Ns = []

            if self.N == 0:
                return self.on_exit(cause='extinct')
        else:
            return self.on_exit(cause='timeout')

    def gillespie_step(self):

        # calculate all possible interaction rates
        self.calculate_rates()

        # calculate residence time
        self.delta_t = rng.exponential(scale=1/self.total_rate)

        # choose process
        probabilites = np.array([self.total_birth_rate, self.total_death_rate, self.total_interaction_rate])
        probabilites = probabilites/np.sum(probabilites)
        chosen_process = rng.choice(['birth', 'death', 'inter'], p=probabilites)

        # implement birth
        if chosen_process == 'birth':
            # choose randomly which individual gave birth
            repr_id = rng.integers(self.N)
            # update birth
            self.update_birth(repr_id=repr_id)

        # implement death
        elif chosen_process == 'death':
            # choose randomly which individual died
            dead_id = rng.integers(self.N)
            # update death
            self.update_death(dead_id=dead_id)

        # implement interaction (death)
        elif chosen_process == 'inter':
            # choose randomly which pair interacted
            probabilites = self.interaction_rates/self.total_interaction_rate
            interacting_pair = rng.choice(range(self.n_interacting_pairs), p=probabilites)
            interacting_pair = self.possible_interacting_pairs[interacting_pair]
            # choose randomly which individual of the pair died
            dead_id = rng.choice(interacting_pair)
            # update death
            self.update_death(dead_id=dead_id)

        else:
            assert False, "No process was chosen by Gillespie algorithm. Revise 'gillespie_step()'"

        # move everyone
        #++ self.delta_t should have a maximum value for movement, if the calculated deltat is larger, several movement steps should occur
        # I am still in doubt about this, perhaps I should impose a maximum self.delta_t and use some sort of 'first reaction' method to choose between interaction or just movement.
        self.move(delta_t=self.delta_t)

        # update time
        self.t += self.delta_t

    def calculate_rates(self):
        # total birth rate
        self.total_birth_rate = self.birth_rate * self.N
        
        # total death rate
        self.total_death_rate = self.death_rate * self.N

        # total interaction rate
        self.calc_interaction_pairs()
        self.calc_interaction_rates()

        # total rate
        self.total_rate = self.total_interaction_rate + self.total_birth_rate + self.total_death_rate

    def calc_interaction_rates(self):
        # get an array with the distances between the pairs
        positions0 = self.x[self.possible_interacting_pairs[:, 0]]
        positions1 = self.x[self.possible_interacting_pairs[:, 1]]
        sqred_distances = self.sqred_distance_in_torus(positions0, positions1, self.env_size)
        
        # calculate the vector of interaction rates
        # In the simulation, this is implemented as the rate at which a pair of organisms interact and one of them dies.
        # Since in theory this is the rate at which EACH organism dies, the interaction rate in the simulation is multiplied by two.
        # The rate should be doubled because to account for both organisms experiencing a symmetrical interaction.
        compet_variance = self.competition_kernel*self.competition_kernel
        self.interaction_rates = 2*self.interaction_parameter*np.exp(-sqred_distances/(2*compet_variance))/(2*np.pi*compet_variance)
        self.total_interaction_rate = np.sum(self.interaction_rates)


    @staticmethod
    def sqred_distance_in_torus(x0, x1, dimensions):
        """returns the (squared) distance between two sets of points
        ref: https://stackoverflow.com/a/11109336/13845224

        Args:
            x0 (array): array with position of organisms
            x1 (array): array with position of organisms
            dimensions (float / array): the size of the region with periodic boundary conditions

        Returns:
            array: distances between elements in vector x0 and x1
        """
        delta = np.abs(x0 - x1)
        delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
        return np.sum(delta * delta, axis=-1)

    def calc_interaction_pairs(self):
        self.positions_tree = KDTree(self.x, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        self.possible_interacting_pairs = self.positions_tree.query_pairs(self.interaction_max_distance, output_type='ndarray')
        self.n_interacting_pairs = len(self.possible_interacting_pairs)

        
    def update_birth(self, repr_id):
        # disperse center of new individual

        # INFO Random new center
        if self.dispersal_kernel == 'random':
            new_center = self.env_size*rng.random(size=2)[np.newaxis,:]
        else:
        # INFO Gaussian Distribution Kernel
            new_center = rng.normal(loc=self.centers[repr_id], scale=self.dispersal_kernel, size=2)[np.newaxis,:]

        new_center = new_center % self.env_size
        # INFO Random new center
        # new_center = self.env_size*rng.random(size=2)[np.newaxis,:]
        # INFO Custom Kernel
        # new_center = self.dispersal_kernel(self.x[repr_id])[np.newaxis,:]
        # new_center = new_center % self.env_size

        # update centers and number of individuals
        self.centers = np.concatenate((self.x, new_center), axis=0)
        self.N += 1
        self.births += 1

        # update positions assuming the organism to be 
        self.x = np.concatenate((self.x, new_center), axis=0)

        # update the 'mover'
        self.OU_mover = milshtein_BM_N_2D(rng=self.rng, x0=self.x, g=self.noise, N=self.N)

    def update_death(self, dead_id):
        # update positions, centers and number of individuals
        self.x = np.delete(self.x, dead_id, axis=0)
        self.N -= 1
        self.deaths += 1

        # update the 'mover'
        self.OU_mover = milshtein_BM_N_2D(rng=self.rng, x0=self.x, g=self.noise, N=self.N)

    def move(self, delta_t):
        self.x = self.OU_mover(delta_t)
        self.periodic_boundary_conditions()

    def periodic_boundary_conditions(self):
        self.x = self.x % self.env_size

    def stop_condition(self):
        # identify if the number of births follows a binomial (gaussian) distribution    
        # N = self.data_interval

        X = self.data['births'][-self.check_termination_N_data:]

        # test if there is a sufficient number of observations
        # if N > 9:

        # test if skewness is sufficiently low
        # if skew(X) < 1/3:

        # test if observations follow a gaussian distribution and return
        # test if p-value is greater than .5
        # (20% of kstests with the same RVS and CDF generate p-value > .5)
        # return kstest(rvs=X, cdf='binom', args=(N, .5))[1] > .5

        # 0.065 is roughly the 17% quantile of the distribution of kolmogorov_like_test
        # when empirical CDF was generated from the theoretical CDF (scipy rvs method)
        # i.e. the distributions are considered equal under a quite restrictive assumption
        # in comparison, concluding that distributions are different by rejection of the null hypothesis 
        # at the alpha=0.05 level requires the measure to be larger than 0.21 [based on New Cambridge stat tables]

        CDFn = self.generate_CDF(X, self.data_interval)
        test = self.kolmogorov_like_test(CDFn, self.theoretical_equilibrium_CDF)
        return test <= .065, test
    
    @staticmethod
    def kolmogorov_like_test(CDFn, CDFtheo):
        # N = CDFn.shape[0]
        # assert CDFtheo.shape == CDFn.shape, "CDFs must have the same shape"

        # returns the maximum absolute difference between the theoretical and empirical CDFs
        return np.max(np.abs(CDFn-CDFtheo))

    @staticmethod
    def generate_CDF(X, N):
        """Receives a vector of observations and returns the empirical CDF with support [0,N].

        Args:
            X (np.array): Vector of observations
            N (int): upper limit of the support

        Returns:
            np.array: The vector containing the values of the empirical CDF. Output is a N+1 dimensional numpy vector.
        """
        
        # All but the last (righthand-most) bin is half-open.
        # the last bin need to be [N, N+1]
        bin_limits = np.arange(0,N+2)
        CDF = np.histogram(X, bins=bin_limits, density=True)[0].cumsum()
        return CDF

    @staticmethod
    def generate_binom_theoreticalCDF(N, p):
        """Generates the theoretical CDF of the binomial distribution with $N$ samples and probability of success $p$.

        Args:
            N (int): Number of samples
            p (float): probability of success

        Returns:
            np.array: The vector containing the values of the theoretical CDF. Output is a N+1 dimensional numpy vector.
        """
        dist = binom(N, p)
        xs = np.arange(0,N+1)
        CDF = dist.cdf(xs)
        return CDF

        
    def on_exit(self, cause = None, **argv):
        if cause == 'extinct':
            ave_N = 0
        elif cause == 'timeout':
            ave_N = self.N
        else:
            ave_N = self.calculate_ave_abundance(self.ts, self.Ns)

        PCFxs, PCFys = self.calculate_PCF('positions')
        try:
            PCF_position = interp1d(PCFxs, PCFys, kind='cubic', fill_value="extrapolate")
        except ValueError:
            PCF_position = interp1d(PCFxs, PCFys, kind='linear', fill_value="extrapolate")

        results = {'T': self.t,
                    'steps': self.steps,
                    'aveN': ave_N,
                    'id': self.id,
                    'tau': None,
                    'sigma': None,
                    'noise': self.noise,
                    'birthRate': self.birth_rate,
                    'deathRate': self.death_rate,
                    'interDistance': self.competition_kernel,
                    'interRate': self.interaction_parameter,
                    'dispersal': self.dispersal_kernel,
                    'envSize': self.env_size,
                    'cause': cause,
                    'PCFposition': np.array([PCF_position.x, PCF_position.y]),
                    'prediction_MF': self.calculate_MF_carrCap(),
                    # 'prediction_SpatLog': self.calculate_spatLogistic_carrCap(PCF_position),
                    'carrCap': ave_N/(self.env_size*self.env_size),
                    **argv}

        return results#pd.DataFrame(results, index=[0])
        # self._calculate_PCF(self.centers)

    def calculate_MF_carrCap(self):
        """Returns the carrying capacity of the system predicted by mean field.

        Returns:
            float: the carrying capacity of the system predicted by mean field.
        """
        return (self.birth_rate - self.death_rate)/(self.interaction_parameter)

    def calculate_spatLogistic_carrCap(self, PCF):
        """Calculates the carring capacity of the spatial logistic model.

        Args:
            PCF (interp1d function): The pairwise correlation function of the position of organisms in the population

        Returns:
            float: The carrying capacity predicted by the spatial logistic model
        """
        from scipy.integrate import quad

        compet_variance = self.competition_kernel*self.competition_kernel

        def inter_kernel(d):
            return np.exp(-d*d/(2*compet_variance))/(2*np.pi*compet_variance)
        
        def integrand(d):
            return d*inter_kernel(d)*PCF(d)
        
        spatial_correction = quad(integrand, 0, self.interaction_max_distance)[0]

        MFPred = self.calculate_MF_carrCap()

        return MFPred/(2*np.pi*spatial_correction)
    
    @staticmethod
    def calculate_ave_abundance(ts, Ns):
        return np.average(Ns, weights=ts)


    def store_data(self):
        self.data = self.data.append({'t': self.t, 'events': self.steps, 'N': self.N, 'births': self.births, 'deaths': self.deaths}, ignore_index=True) # , 'sigma': self.sigma, 'rep': self.id, 'tau': self.taus, 'inter': self.interaction_distance, 'interRate':self.total_interaction_rate, 'birthRate': self.total_birth_rate, 

    @staticmethod
    def _calculate_PCF(distances, max_dist, resolution, npoints, env_size):
        """Calculates the Pairwise Correlation Function from a set of precalculated distances.

        Args:
            distances (np.array): vector of distances between points
            max_dist (float): maximum distance between points
            resolution (int): number of intervals, `dr = max_dist/resolution`

        Returns:
            (np.array, np.array): X and Y values of the PCF
        """
        dr = max_dist/(resolution)
        bin_limits = np.linspace(0., max_dist, num=(resolution+1))
        bin_midpoints = bin_limits[:-1] + dr/2
        PCF = np.histogram(distances, bins=bin_limits)[0]
        
        # Normalizing (so for random points, PCF=1)
        # density
        dens = npoints/(env_size*env_size)

        # check for zero density:
        if dens == 0.:
            # PCF is probably [0, ... , 0]
            return bin_midpoints, PCF

        # expected number of particles in each interval
        PCF = PCF/(2*np.pi*bin_midpoints*dr * dens)
        # divide by the number of particles
        PCF /= npoints
        # each distance should be counted twice
        PCF *= 2

        return bin_midpoints, PCF

    @staticmethod
    def _guestimate_PCF_resolution(density, env_area, max_dist, expected_min_count = 20.):
        dr = np.sqrt(expected_min_count/(np.pi * env_area * density * density))
        return int(np.floor(max_dist/dr))

    def calculate_PCF(self, distance, resolution='auto'):
        """Convenient function to call `_calculate_PCF` with reasonable defaults.

        Args:
            distance (string): distance must either be 'centers' or 'positions'
            resolution (int): number of intervals, `dr = max_dist/resolution`. Defaults to a dr of .1.

        Returns:
            (np.array, np.array): X and Y values of the PCF
        """
        if self.N == 0:
            return np.array([0]), np.array([0])

        if resolution == 'auto':
            # defaults to a dr calculated so that C(0) == 1 <=> there are 10 particles within dr of each other
            A = self.env_size*self.env_size
            resolution = self._guestimate_PCF_resolution(density = self.N/A, env_area=A, max_dist=self.env_size/2)

        dist = self._all_distances(distance)

        return self._calculate_PCF(distances=dist, max_dist=self.env_size/2, resolution=resolution, npoints=self.N, env_size=self.env_size)

    def _all_distances(self, points):
        """Returns a vector with all non-zero distances between points

        Args:
            points (string): points must either be 'centers' or 'positions'

        Returns:
            np.array: vector with all nonzero distances between points
        """
        if points == 'centers':
            KDT = KDTree(self.centers, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        elif points == 'positions':
            KDT = KDTree(self.x, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        else:
            assert False, "not implemented. points must either be 'centers' or 'positions'."
        
        # get distance matrix between all the points in environment
        dist = KDT.sparse_distance_matrix(KDT, max_distance=self.env_size/2).toarray()
        # select the upper triangular matrix
        dist = dist[np.tril_indices_from(dist, k=-1)]
        # filter out null distances
        dist = dist[dist > 0]

        return dist

    def plot_PCF(self, max_dist, resolution='auto', distances='centers'):
        """Calculate and Plot the Pairwise Correlation Function from a set of precalculated distances.

        Args:
            max_dist (float): maximum distance between points
            resolution (int): number of intervals, `dr = max_dist/resolution`
            distances (string): distances must either be 'centers' or 'positions'
        """
        from matplotlib.pyplot import subplots

        dist = self._all_distances(distances)

        # calculate the PCF
        xs, ys = self._calculate_PCF(distances=dist, max_dist=max_dist, resolution=resolution, npoints=self.N, env_size=self.env_size)

        fig, ax = subplots()
        ax.plot(xs, ys, 'k-')

        ax.set_xlabel('distance')
        ax.set_ylabel('Pair Correlation Function - PCF')

        fig.show()

        return ax

    @staticmethod
    def costly_ghosted(points, hshift, vshift=None):
        """Return an array with points copied over all 8 ghost neighbouring cells in a toroidal geometry.

        Args:
            points (np.array): vector with the points that should be copied
            hshift (float): width of the toroidal region
            vshift (float, optional): height of the toroidal region. Defaults to hshift value.

        Returns:
            np.array: the points array appended with shifted copies
        """
        
        if vshift is None:
            vshift = hshift

        npoints = points.shape[0]
        ghosted = points.copy()
        
        for v in [-1, 0, 1]:
            for h in [-1, 0, 1]:
                # append shifted copies of the points
                ghosted = np.vstack([ghosted, points + np.ones(npoints)[:,np.newaxis]*np.array([h*hshift, v*vshift])])

        return ghosted

    @staticmethod
    def asympt_distr(x, y, centers, stdev):
        """The idea is to calculate the cumulative probability density of an individual occuppying some spot, given the center of the movement and its stdev.
        The assumption is that organisms move through an Orstein-Uhlenback process, and have a gaussian distribution with identical stdandard deviations (stdev)
        over which it is possible to define an Home Range (HR).

        Args:
            x (np.array): array with the x-coordinate of the positions at which the density will be calculated
            y (np.array): array with the y-coordinate of the positions at which the density will be calculated
            centers (np.array): array with the xy-coordinates of the HR centers
            stdev (float): the standard deviation of the indiviual gaussian distribution

        Returns:
            np.array: the density evaluated at all the points [x_i, y_i]
        """
        # We must sum the contributions from all normal distributions at each point of the N x N points [x_i, y_i].
        # To do that in a vectorial manner, we first define the tensor point_vec = [x, y] (shape = [N, N, 2])
        point_vec = np.stack((y, x), axis=-1)

        # Say we have M organisms, and therefore M centers.
        # subtracting the centers from these vectors, we create a tensor dist_from_center = [[x_i - center_z_x,  y_i - center_z_y]] (shape = [M, N, N, 2])
        # at each 'sheet' of the tensor, the distance from a different center to all the points is recorded.
        dist_from_center = point_vec - centers[:, np.newaxis, np.newaxis]

        # squaring dist_from_center, summing over the x and y coordinates and then dividing by 2*stdev^2 gives the argument of the exponential function in the normal distribution
        # exp_args (shape = [M, N, N])
        exp_args = - np.sum(dist_from_center*dist_from_center, axis=-1)/(2*stdev*stdev)

        # calculating the exponential function for each of the exp_args matrix elements and then normalizing
        # normal_dis (shape = [M, N, N])
        normal_dis = np.exp(exp_args)
        normal_dis = normal_dis/(stdev*np.sqrt(2.*np.pi))
        
        # Acumulate all the densities
        # total_dens_at_each_point (shape = [N, N])
        total_dens_at_each_point = np.sum(normal_dis, axis=0)
        
        return total_dens_at_each_point

    def plot_heatmap(self, resolution):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize, ListedColormap


        # make these smaller to increase the resolution
        dx = self.env_size/resolution
        dy = dx

        x = np.arange(0., self.env_size, dx)
        y = np.arange(0., self.env_size, dy)
        x, y = np.meshgrid(x, y)
        y = y[::-1]

        # when layering multiple images, the images need to have the same
        # extent.  This does not mean they need to have the same shape, but
        # they both need to render to the same coordinate system determined by
        # xmin, xmax, ymin, ymax.  Note if you use different interpolations
        # for the images their apparent extent could be different due to
        # interpolation edge effects

        # nx, ny = X.shape

        # Z = np.zeros((nx,ny))

        # periodic boundary conditions:
        centers = self.costly_ghosted(points=self.centers, hshift=self.env_size)

        g = 1.
        tau = 0.001
        stdev = np.sqrt(g*tau/2)

        asympt_distr = self.asympt_distr(x=x, y=y, centers=centers, stdev=stdev)

        # extent = 
        
        fig, (ax, cax) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [0.95, 0.05]}, figsize=(7.5,6))

        # color
        cmap = ListedColormap(cm.get_cmap('viridis')(np.linspace(0., 0.75, 256)))
        norm = Normalize(vmin=np.min(asympt_distr), vmax=np.max(asympt_distr))
        fig.set_facecolor("#fff9db")

        # plot heatmap
        im = ax.imshow(asympt_distr, cmap=cmap, norm=norm, extent=tuple([0., self.env_size, 0., self.env_size]), origin='lower')

        # ColorBar
        psm = cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(psm, cax=cax, label='asymptotic density')

        return fig

class milshtein_BM_N_2D:
    #
    # BM model:
    # dX = sqrt(g)*dW
    # dW is white noise
    #
    def __init__(self, rng, x0, g, N, h=1e-3):
        # random number generator
        self.rng = rng
        # time-step
        self.h = h
        # number of independent 'movers'
        self.N = int(N)
        # Process parameters (all of them are numpy vectors!)
        # self.tau = infinity # shape = (N,2)
        self.coef = np.sqrt(g) # shape = (N,2) 
        self.xi = x0 # shape = (N,2)

        # Auxiliary Variables
        self.h_sqrt = np.sqrt(h)
        # self.sigma = np.sqrt(g)

    def __call__(self, delta_t):
        #BUG the ceil function is increasing the time interval slightly. An alternative is to consider several h steps and a final smaller step.
        # number of time intervals to be considered
        intervals = int(np.ceil(delta_t/self.h))

        # generate all random values at once
        uh = self.h_sqrt*self.rng.normal(size=(self.N, 2, intervals))
        
        for i in range(intervals):
            self.xi += uh[:,:,i]*self.coef
        return self.xi

def run_environment(dic):
    id = 'tau='+str(dic['taus']) +'_sigma='+str(dic['sigma'])+'_rep='+str(dic['rep'])
    env = Environment(Nagents=100, inter_distance=dic['inter'], birth_rate=1., inter_rate=1., taus=dic['taus'], sigma=dic['sigma'], centers='random', dispersal_kernel='random', data_interval=50, env_size=1., id=dic['rep'])
    env.run(int(1e4))
    # np.save('np_centers/'+id, env.centers)
    # env.data.to_pickle("results_full"+id+".zip")
    return env.data

def fix_df(df):
    ls = []
    for id, it in df.iterrows():
        ls.append(it[0])
    
    return pd.DataFrame(ls)

def store_data(data, columns = ['t', 'N', 'rep', 'tau', 'inter', 'interRate', 'birthRate', 'births', 'deaths']):
    df = pd.DataFrame(columns=columns)
    for d in data:
        # print(d)
        # input()
        auxdf = pd.DataFrame(d)#, index=[0])
        # print(auxdf)
        # input()
        df = pd.concat([df, auxdf], ignore_index=True)
        # for d in container:
    return df


def main():
    # disp = np.linspace(0.1, 5., num=20)
    taus = np.geomspace(1e-3, 1e2, num=20)
    sigma = np.geomspace(1e-6, 1e0, num=20)
    inters = np.geomspace(1e-6, 1e0, num=20)
    reps = 20
    # sigma = 1.

    sims = []
    for inter in inters:
        for t in taus:
            for s in sigma:
                for rep in range(reps):
                    sims.append({'sigma':s, 'taus':t, 'rep':rep, 'inter': inter})

    # np.random.shuffle(sims)

    results = p_umap(run_environment, sims, num_cpus=22)
    # df = pd.DataFrame(results)
    # df = fix_df(df)
    df = store_data(results)
    df.to_pickle("results_full.zip")

if __name__ == '__main__':
    main()
