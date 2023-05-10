import numpy as np
from scipy.spatial import KDTree
import uuid
import pandas as pd
# from numba import njit, cuda

rng = np.random.default_rng(16558947)

class Environment():
    def __init__(self,
                Nresource, # number of resources
                Nconsumer, # number of consumers
                R_birth_rate, # birth rate of resources
                C_death_rate, # death rate of consumers
                C_conversion_efficiency, # conversion efficiency of consumers
                pred_kernel, # predation kernel
                pred_max_dist, # maximum distance for predation
                pred_rate, # predation rate
                comp_kernel, # competition kernel
                comp_max_dist, # maximum distance for competition
                R_MF_carrCap, # carrying capacity of resources
                R_mover_class, # class of the resource mover
                R_asymptotic_pos_dist, # asymptotic movement of resources
                C_mover_class, # class of the consumer mover
                C_asymptotic_pos_dist, # asymptotic movement of consumers
                init_centers_dist, # initial distribution of the centers of resources and consumers
                                    # it can be 'random', 'origin' or a predetermined array of centers with R and C centers at [0] and [1] respectively
                dispersal_kernel, # dispersal kernel
                data_interval=50, # interval at which data is stored
                env_size=1., # size of the environment
                seed=34567, # seed for the random number generator
                rng=None, # random number generator
                id=None, # an unique id for the environment
                ):

        # This is the spatial stochastic simulation of a consumer-resource system.
        # The system is composed of a resource and a consumer.
        # The resource is a single population that grows logistically.
        # The consumer is a single population that decays exponentially and consumes the resource.

        # the MF equations of the system are:
        # dR/dt = R*r*(1 - R/K) - alpha*C*R
        # dC/dt = epsilon*alpha*C*R - dC

        # generate a unique id for the simulation
        if id is None:
            self.id = str(uuid.uuid4())
        
        # Setup random number generator
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        # Environment properties
        self.env_size = env_size
        self.t = 0.
        self.steps = 0

        # Resource properties
        self.Nr = Nresource # number of resources
        self.R_birth_rate = R_birth_rate # birth rate of resources
        self.competition_kernel = comp_kernel # this is the kernel that describes the competition between individuals of the resource
                                            # a function that receives a vector of squared distances and returns the kernel value for each distance
        self.competition_max_dist = comp_max_dist # this is the maximum distance for which the competition kernel is non-zero
        self.R_MF_carrCap = R_MF_carrCap # the carrying capacity of the resource at the Mean Field
        self.competition_rate = R_birth_rate/R_MF_carrCap # this is the rate of competition between resource organisms
        self.R_mover_class = R_mover_class#  this is a class that updates the position of the resource individuals
                                        #    the class should be initialized with the HR centers, current positions of the individuals and a random number generator
                                        #    the class should have a __call__ method that receives a time interval and returns the updated positions of the individuals
        self.R_asymptotic_pos_dist = R_asymptotic_pos_dist # this is the asymptotic distribution of the positions of the resource individuals
                                        #    a function that receives the HR center and return a position sampled from the asymptotic distribution

        # Consumer properties
        self.Nc = Nconsumer # number of consumers
        self.conversion_efficiency = C_conversion_efficiency # conversion efficiency of consumers (the proportion of killed resource individuals that are converted into consumer individuals)
        self.C_death_rate = C_death_rate # death rate of consumers
        self.predation_kernel = pred_kernel # this is the kernel that describes the predation of the consumer on the resource
                                            # a function that receives a vector of squared distances and returns the kernel value for each distance
        self.predation_max_dist = pred_max_dist # this is the maximum distance for which the predation kernel is non-zero
        self.predation_rate = pred_rate # this is the rate of predation of the consumer on the resource
        self.C_mover_class = C_mover_class#  this is a class that updates the position of the consumer individuals
                                        #    the class should be initialized with the HR centers, current positions of the individuals and a random number generator
                                        #    the class should have a __call__ method that receives a time interval and returns the updated positions of the individuals
        self.C_asymptotic_pos_dist = C_asymptotic_pos_dist # this is the asymptotic distribution of the positions of the consumer individuals
                                        #    a function that receives the HR center and return a position sampled from the asymptotic distribution

        # Dispersal kernel
        # this is the kernel that describes the dispersal of the consumer and resource
        # a function that receives the position of the ancestor organism and outputs the position of the offspring
        # this position should be a random number drawn from the dispersal kernel distribution
        self.R_dispersal_kernel = dispersal_kernel
        self.C_dispersal_kernel = dispersal_kernel

        # Inital HR center positions
        if type(init_centers_dist) == str:
            if init_centers_dist == 'random':
                self.R_centers = self.rng.uniform(high=self.env_size, size=(self.Nr, 2))
                self.C_centers = self.rng.uniform(high=self.env_size, size=(self.Nc, 2))

            elif init_centers_dist == 'origin':
                self.R_centers = np.zeros((self.Nr, 2))
                self.C_centers = np.zeros((self.Nc, 2))
        else:
            self.R_centers = init_centers_dist[0]
            self.C_centers = init_centers_dist[1]
        
        # Relax each organism to its asymptotic position distribution (this is done to avoid the initial transient)
        self.Rx = self.R_asymptotic_pos_dist(self.R_centers, self.rng)
        self.Rx = self.periodic_boundary_conditions(self.Rx)
        self.Cx = self.C_asymptotic_pos_dist(self.C_centers, self.rng)
        self.Cx = self.periodic_boundary_conditions(self.Cx)

        # Initialize the resource and consumer movers
        self.R_mover = self.R_mover_class(centers=self.R_centers, pos=self.Rx, rng=self.rng)
        self.C_mover = self.C_mover_class(centers=self.C_centers, pos=self.Cx, rng=self.rng)

        # interaction-related
        self.Rpositions_tree = KDTree(self.R_centers, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        self.Cpositions_tree = KDTree(self.C_centers, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        self.possible_competing_pairs = None
        self.possible_predating_pairs = None
        self.n_competing_pairs = None
        self.n_predating_pairs = None
        
        # Gillespie variables
        self.total_rate = 0.
        self.total_R_birth_rate = 0.
        # self.total_C_birth_rate = 0.
        self.total_C_death_rate = 0.
        self.total_competition_rate = 0.
        self.total_predation_rate = 0.
        self.delta_t = 0.

        # Stored data
        self.data = pd.DataFrame(columns=['t', 'events', 'Nr', 'Nc', 'R_births', 'R_deaths', 'C_births', 'C_deaths', 'competitions', 'predations'])
        self.data.loc[0] = [self.t, 0, self.Nr, self.Nc, 0, 0, 0, 0, 0, 0]

        self.data_interval = data_interval

        # Storage properties
        self.R_births = 0
        self.R_deaths = 0
        self.C_births = 0
        self.C_deaths = 0
        self.competitions = 0
        self.predations = 0


    def run(self, max_steps):
        self.steps = 0
        while self.steps < max_steps:
            # perform a gillespie step
            self.gillespie_step()
            self.steps += 1

            # record data 
            if self.steps % self.data_interval == 0:
                self.store_data()
                self.R_births = 0
                self.R_deaths = 0
                self.C_births = 0
                self.C_deaths = 0
                self.competitions = 0
                self.predations = 0

                # print(self.steps)
                # print('R:', self.Nr)
                # print('C:',self.Nc)
                # check termination
                # if self.steps > self.transient_steps:
                #     test, test_value = self.stop_condition()
                #     if test:
                #         return self.on_exit(cause='equilibrium', testValue=test_value)

            if self.Nr == 0:
                return self.on_exit(cause='collapse')

            # if self.Nc == 0:
            #     return self.on_exit(cause='prey-release')
        return self.on_exit(cause='timeout')

    def run_time(self, max_time):
        self.steps = 0
        while self.t < max_time:
            # perform a gillespie step
            self.gillespie_step()
            self.steps += 1

            # record data 
            if self.steps % self.data_interval == 0:
                self.store_data()
                self.R_births = 0
                self.R_deaths = 0
                self.C_births = 0
                self.C_deaths = 0
                self.competitions = 0
                self.predations = 0

                # print(self.steps)
                # print('R:', self.Nr)
                # print('C:',self.Nc)
                # check termination
                # if self.steps > self.transient_steps:
                #     test, test_value = self.stop_condition()
                #     if test:
                #         return self.on_exit(cause='equilibrium', testValue=test_value)

            if self.Nr == 0:
                return self.on_exit(cause='collapse')

            # if self.Nc == 0:
            #     return self.on_exit(cause='prey-release')
        return self.on_exit(cause='timeout')


    @staticmethod
    def from_listsOflists_to_array(listsOflists):
        """This function receives a list of lists.
        the ith list stores the indices of the possible interactiong partners of i.
        The output of this function is one array with the indices of possible interacting pairs.
        """
        n = len(listsOflists)
        n_pairs = sum(len(l) for l in listsOflists)
        pairs = np.zeros((n_pairs, 2), dtype=int)
        k = 0
        for i in range(n):
            for j in listsOflists[i]:
                pairs[k, 0] = i
                pairs[k, 1] = j
                k += 1
        return pairs


    def gillespie_step(self):

        # calculate all possible interaction rates
        self.calculate_rates()

        # calculate residence time
        self.delta_t = rng.exponential(scale=1/self.total_rate)
        
        # choose process
        probabilites = np.array([self.total_R_birth_rate, self.total_C_death_rate, self.total_competition_rate, self.total_predation_rate])
        probabilites = probabilites/np.sum(probabilites)
        chosen_process = rng.choice(['Rbirth', 'Cdeath', 'compet', 'pred'], p=probabilites)

        # implement birth
        if chosen_process == 'Rbirth':
            # choose randomly which individual gave birth
            repr_id = rng.integers(self.Nr)
            # update birth
            self.update_birth(repr_id=repr_id, species='R')

        # implement death
        elif chosen_process == 'Cdeath':
            # choose randomly which individual died
            dead_id = rng.integers(self.Nc)
            # update death
            self.update_death(dead_id=dead_id, species='C')

        # implement interaction (death)
        elif chosen_process == 'compet':
            # choose randomly which pair interacted
            probabilites = self.competition_rates/self.total_competition_rate
            competing_pair = rng.choice(range(self.n_competing_pairs), p=probabilites)
            competing_pair = self.possible_competing_pairs[competing_pair]
            # choose randomly which individual of the pair died
            # TODO: implement the different 'fitnesses'
            # TODO: it should be more likely that the 'invading' organisms loose
            dead_id = rng.choice(competing_pair)
            # update death
            # TODO: change update death to dispersal
            self.update_death(dead_id=dead_id, species='R')

            self.competitions += 1

        elif chosen_process == 'pred':
            # choose randomly which pair interacted
            probabilites = self.predation_rates/self.total_predation_rate
            predating_pair = rng.choice(range(self.n_predating_pairs), p=probabilites)
            predating_pair = self.possible_predating_pairs[predating_pair]
            # implement death of the prey
            prey_id = predating_pair[0]
            # update death
            self.update_death(dead_id=prey_id, species='R')
            # implement birth of the predator based on the conversion efficiency
            if rng.random() < self.conversion_efficiency:
                repr_id = predating_pair[1]
                # update birth
                self.update_birth(repr_id=repr_id, species='C')

            self.predations += 1

        else:
            assert False, "No valid process was chosen by Gillespie algorithm. Revise 'gillespie_step()'"

        # move everyone
        #++ self.delta_t should have a maximum value for movement, if the calculated deltat is larger, several movement steps should occur
        # I am still in doubt about this, perhaps I should impose a maximum self.delta_t and use some sort of 'first reaction' method to choose between interaction or just movement.
        self.move(delta_t=self.delta_t)

        # update time
        self.t += self.delta_t


    def calculate_rates(self):
        # total birth rate
        self.total_R_birth_rate = self.R_birth_rate * self.Nr
        
        # total death rate
        self.total_C_death_rate = self.C_death_rate * self.Nc

        # create positions tree
        self.Rpositions_tree = KDTree(self.Rx, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)
        self.Cpositions_tree = KDTree(self.Cx, compact_nodes=False, balanced_tree=False, boxsize=self.env_size)

        # total interaction rate
        self.calc_interaction_pairs('competition')
        self.calc_interaction_rates('competition')

        self.calc_interaction_pairs('predation')
        self.calc_interaction_rates('predation')

        # total rate
        self.total_rate = self.total_competition_rate + self.total_predation_rate + self.total_R_birth_rate + self.total_C_death_rate


    def calc_interaction_rates(self, interaction_type):
        if interaction_type == 'competition':
            # get an array with the distances between the pairs
            positions0 = self.Rx[self.possible_competing_pairs[:, 0]]
            positions1 = self.Rx[self.possible_competing_pairs[:, 1]]
            sqred_distances = self.sqred_distance_in_torus(positions0, positions1, self.env_size)
            
            # calculate the vector of interaction rates
            # In the simulation, this is implemented as the rate at which a pair of organisms interact and one of them dies.
            # Since in theory this is the rate at which EACH organism dies, the interaction rate in the simulation is multiplied by two.
            # The rate should be doubled because to account for both organisms experiencing a symmetrical interaction.
            self.competition_rates = 2*self.competition_rate*self.competition_kernel(sqred_distances)
            self.total_competition_rate = np.sum(self.competition_rates)

        elif interaction_type == 'predation':
            # get an array with the distances between the pairs
            positions0 = self.Rx[self.possible_predating_pairs[:, 0]]
            positions1 = self.Cx[self.possible_predating_pairs[:, 1]]
            sqred_distances = self.sqred_distance_in_torus(positions0, positions1, self.env_size)
            
            # calculate the vector of interaction rates
            # In the simulation, this is implemented as the rate at which a pair of organisms interact and one of them dies.
            # Since in theory this is the rate at which EACH organism dies, the interaction rate in the simulation is multiplied by two.
            # The rate should be doubled because to account for both organisms experiencing a symmetrical interaction.
            self.predation_rates = 2*self.predation_rate*self.predation_kernel(sqred_distances)
            self.total_predation_rate = np.sum(self.predation_rates)

        else:
            assert False, "Interaction type not recognized"


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


    def calc_interaction_pairs(self, interaction_type):
        # calculate the possible pairs of organisms that can interact
        # the pairs are stored in an array with two columns, each row is a pair with the index of the two organisms

        if interaction_type == 'competition':
            self.possible_competing_pairs = self.Rpositions_tree.query_pairs(r=self.competition_max_dist, output_type='ndarray')
            self.n_competing_pairs = len(self.possible_competing_pairs)

        if interaction_type == 'predation':
            aux = self.Rpositions_tree.query_ball_tree(other=self.Cpositions_tree, r=self.predation_max_dist)
            # convert the list of lists into an array
            self.possible_predating_pairs = self.from_listsOflists_to_array(aux)
            self.n_predating_pairs = len(self.possible_predating_pairs)

    @staticmethod
    def convert_list_of_lists_to_array(list_of_lists):
        """converts a list of lists into an array

        Args:
            list_of_lists (list): list of lists

        Returns:
            array: array with the elements of the list of lists
        """
        array = np.empty((0, 2), dtype=int)
        for i, sublist in enumerate(list_of_lists):
            for j in sublist:
                array = np.append(array, np.array([[i, j]]), axis=0)
        return array
    
    def update_birth(self, repr_id, species):

        if species == 'R':
            # disperse center of new individual
            new_center = self.R_dispersal_kernel(self.R_centers[repr_id])[np.newaxis,:]
            new_center = self.periodic_boundary_conditions(new_center)

            # update centers and number of individuals
            self.R_centers = np.concatenate((self.R_centers, new_center), axis=0)

            # update position
            new_pos = self.R_asymptotic_pos_dist(new_center, self.rng)#[np.newaxis,:]
            new_pos = self.periodic_boundary_conditions(new_pos)
            self.Rx = np.concatenate((self.Rx, new_pos), axis=0)

            # increment number of individuals and birth counter
            self.Nr += 1
            self.R_births += 1

            # update the 'mover'
            self.R_mover = self.R_mover_class(centers=self.R_centers, pos=self.Rx, rng=self.rng)

        elif species == 'C':
            # disperse center of new individual
            new_center = self.C_dispersal_kernel(self.C_centers[repr_id])[np.newaxis,:]
            new_center = self.periodic_boundary_conditions(new_center)

            # update centers and number of individuals
            self.C_centers = np.concatenate((self.C_centers, new_center), axis=0)

            # update position
            new_pos = self.C_asymptotic_pos_dist(new_center, self.rng)#[np.newaxis,:]
            new_pos = self.periodic_boundary_conditions(new_pos)
            self.Cx = np.concatenate((self.Cx, new_pos), axis=0)

            # increment number of individuals and birth counter
            self.Nc += 1
            self.C_births += 1

            # update the 'mover'
            self.C_mover = self.C_mover_class(centers=self.C_centers, pos=self.Cx, rng=self.rng)
        
        else:
            assert False, "Species not recognized"


    def update_death(self, dead_id, species):
        if species == 'R':
            # remove dead individual from centers and positions
            self.R_centers = np.delete(self.R_centers, dead_id, axis=0)
            self.Rx = np.delete(self.Rx, dead_id, axis=0)

            # decrement number of individuals and death counter
            self.Nr -= 1
            self.R_deaths += 1

            # update the 'mover'
            self.R_mover = self.R_mover_class(centers=self.R_centers, pos=self.Rx, rng=self.rng)
        
        elif species == 'C':
            # remove dead individual from centers and positions
            self.C_centers = np.delete(self.C_centers, dead_id, axis=0)
            self.Cx = np.delete(self.Cx, dead_id, axis=0)

            # decrement number of individuals and death counter
            self.Nc -= 1
            self.C_deaths += 1

            # update the 'mover'
            self.C_mover = self.C_mover_class(centers=self.C_centers, pos=self.Cx, rng=self.rng)

        else:
            assert False, "Species not recognized"


    def move(self, delta_t):
        # move all individuals
        self.Rx = self.R_mover(delta_t)
        self.Rx = self.periodic_boundary_conditions(self.Rx)
        self.Cx = self.C_mover(delta_t)
        self.Cx = self.periodic_boundary_conditions(self.Cx)


    def periodic_boundary_conditions(self, vector):
        return vector % self.env_size


    def on_exit(self, cause = None, **argv):
        # sourcery skip: inline-immediately-returned-variable
        if cause == 'collapse':
            ave_Nr = 0.
            ave_Nc = 0.
        elif cause == 'prey-release':
            ave_Nr, ave_Nc = self.calculate_ave_abundance()
            # ave_Nr = self.R_MF_carrCap
            ave_Nc = 0.
        else:
            ave_Nr, ave_Nc = self.calculate_ave_abundance()

        # PCFxs, PCFys = self.calculate_PCF('centers')#, resolution=40)
        # try:
        #     PCF_center = interp1d(PCFxs, PCFys, kind='cubic', fill_value="extrapolate")
        # except ValueError:
        #     PCF_center = interp1d(PCFxs, PCFys, kind='linear', fill_value="extrapolate")

        # PCFxs, PCFys = self.calculate_PCF('positions')#, resolution=40)
        # try:
        #     PCF_position = interp1d(PCFxs, PCFys, kind='cubic', fill_value="extrapolate")
        # except ValueError:
        #     PCF_position = interp1d(PCFxs, PCFys, kind='linear', fill_value="extrapolate")
        
        MFprey, MFpred = self.calculate_MF_abundances()

        results = {'finalTime': self.t,
                    'id': self.id,
                    'steps': self.steps,
                    'aveNr': ave_Nr,
                    'aveNc': ave_Nc,
                    'preyGrowthRate': self.R_birth_rate,
                    'PredationCoefficient': self.predation_rate,
                    'ConversionEfficiency': self.conversion_efficiency,
                    'PredatorDeathRate': self.C_death_rate,
                    'interDistance': self.competition_kernel,
                    'envSize': self.env_size,
                    'cause': cause,
                    'NCompetitions': self.competitions,
                    'NPredations': self.predations,
                    'Rcenters': self.R_centers,
                    'Rpositions': self.Rx,
                    'Ccenters': self.C_centers,
                    'Cpositions': self.Cx,
                    'MeanFieldR': MFprey,
                    'MeanFieldC': MFpred,
                    'temporalData': self.data,
                    **argv}

        return results


    def calculate_MF_abundances(self):
        """Returns the equilibrium population size of prey and predators predicted by the stable solution of the Mean Field model"""

        prey_pop = (self.C_death_rate)/(self.conversion_efficiency*self.predation_rate)
        pred_pop = (self.R_birth_rate/self.predation_rate)*(1 - prey_pop/self.R_MF_carrCap)

        if prey_pop < self.R_MF_carrCap:
            return np.array([prey_pop, pred_pop])
        else:
            return np.array([self.R_MF_carrCap, 0.])


    def calculate_ave_abundance(self):
        """Calculates the average abundance of a population over last 30*data_interval events"""
        auxdf = self.data.iloc[-31:]
        dts = auxdf['t'].diff().dropna().values
        auxdf = self.data.iloc[-30:]
        Nrs = auxdf['Nr'].values
        Ncs = auxdf['Nc'].values

        return np.sum(Nrs*dts)/np.sum(dts), np.sum(Ncs*dts)/np.sum(dts)
        # print(Nrs.shape, dts.shape)

        # return np.average(Nrs, weights=dts, axis=), np.average(Ncs, weights=dts)


    def store_data(self):
        # append data to store data dataframe
        self.data = self.data.append({'t': self.t,
                                'events': self.steps,
                                'Nr': self.Nr,
                                'Nc': self.Nc,
                                'R_births': self.R_births,
                                'R_deaths': self.R_deaths,
                                'C_births': self.C_births,
                                'C_deaths': self.C_deaths,
                                'competitions': self.competitions,
                                'predations': self.predations}, ignore_index=True)




class milshtein_OU_N_2D:
    #
    # OU model:
    # dX = -(1/tau)*(X - center)*dt + sqrt(g)*dW
    # dW is white noise
    #
    def __init__(self, rng, x0, tau, center, sigma, h, N, env_size):
        # random number generator
        self.rng = rng
        # time-step
        self.h = h
        # number of independent 'movers'
        self.N = int(N)
        # Process parameters (all of them are numpy vectors!)
        self.tau = tau # shape = (N,2)
        self.xi = x0 # shape = (N,2)
        self.center = center # shape = (N,2)
        self.envsize = env_size # float

        # Auxiliary Variables
        self.coef = np.sqrt(2*sigma*sigma/tau) # shape = (N,2) (sqrt of g = sqrt(2*sigma^2/tau))
        self.invtau = h/tau # shape = (N,2)
        self.half_envsize = env_size/2. # float
        self.h_sqrt = np.sqrt(h) # float
        # self.coef = np.sqrt(g)
        
        assert self.invtau < 1e-1, 'The time-step is too large!'    

    def __call__(self, delta_t):
        #BUG the ceil function is increasing the time interval slightly. An alternative is to consider several h steps and a final smaller step.
        # this could be implemented by intervals = delta_t//self.h and final_step = delta_t%self.h
        # the problem is that the final step might be expensive to recalculate invtau and coef every time
        # number of time intervals to be considered
        intervals = int(np.ceil(delta_t/self.h))

        # generate all random values at once
        # uh = self.h_sqrt*self.rng.normal(size=(self.N, 2, intervals))

        for _ in range(intervals):
            # generate all random values for this time step
            uh = self.h_sqrt*self.rng.normal(size=(self.N, 2))
            # update the positions
            # take into consideration the periodic boundary conditions:
            # to account for the periodic boundary conditions, we note that if the displacement from the center is larger than half the environment size,
            # then the closest path to the center is through the opposite side of the environment.
            # This is equivalent to subtracting the environment size from the displacement if the displacement is positive and adding the environment size if the displacement is negative.
            #
            delta = self.center - self.xi
            drift = np.where(np.abs(delta) > self.half_envsize, delta - np.sign(delta)*self.envsize, delta)
            self.xi += uh*self.coef + self.invtau*drift
        return self.xi

if __name__ == '__main__':
    assert False, 'This is a module, not a script. Please import it and use the classes and functions defined in it.'
