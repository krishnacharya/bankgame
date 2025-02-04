from game.helpers import *
from game.distributions import *
from game.hedgealgs import HedgeSimultaneous
import nashpy as nash

class GameTrueMatrix:
    def __init__(self, gammas:list[float], taus: list[float], dist:Dist): # both banks have the same strategy spaces
        self.gammas = sorted(gammas)  # important to sort
        self.taus = sorted(taus)
        self.A = generate_utility_matrix(gammas=self.gammas, taus=self.taus, c_f=dist.c_f)
        self.dist = dist
        self.saveget_NE_vertexenum() # saves all NE, numerical algorithm
        self.instance_name = self.dist.name + str(self.gammas) # this defines a unique utility matrix

    def run_hedge(self, T:int, p_b1:np.array, p_b2:np.array, eta:float):
        '''
            T: number of rounds of Hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 
            eta: learning rate for hedge

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        b1_record = [p_b1]
        b2_record = [p_b2]
        for t in range(T):
            p_b1, p_b2 = HedgeSimultaneous(p_b1, p_b2, eta, self.A)
            b1_record.append(p_b1)
            b2_record.append(p_b2)
        return np.array(b1_record), np.array(b2_record), self.gammas, self.taus

    def get_closest_euclidean_NE(self, p_b1, p_b2):
        """
        p_b1: Bank 1's strategy (numpy array)
        p_b2: Bank 2's strategy (numpy array)

        Finds the closest Nash equilibrium (NE) based on L2 distance.
        
        Returns:
        - The closest NE (NE_b1, NE_b2) in terms of Euclidean distance, the corresponding sum of distance of ||NE_b1-p_b1||_2 + ||NE_b2-p_b2||_2
        """
        closest_NE = None
        min_distance = float("inf")
        for NE_b2, NE_b1 in self.NE_ve:
            dist_b1 = np.linalg.norm(p_b1 - NE_b1) # Compute L2 distances separately for each bank to NE strat
            dist_b2 = np.linalg.norm(p_b2 - NE_b2)
            total_distance = dist_b1 + dist_b2
            if total_distance < min_distance:
                min_distance = total_distance
                closest_NE = (NE_b1, NE_b2)   # Store the closest NE
        return closest_NE, total_distance

    def saveget_NE_vertexenum(self):
        npygame = nash.Game(self.A.T, self.A)  # nashpy first takes the row players matrix, then column players;
        self.NE_ve = list(npygame.vertex_enumeration())  # return list of all equilbria, each equilbrium is a tuple of numpy arrays, first element is the row player array, second is column player array
        return self.NE_ve

    def saveget_NE_supportenum(self): # runs support enumeration to get all NE, (Slow)
        npygame = nash.Game(self.A.T, self.A) # nashpy first takes the row players matrix, then column players;
        self.NE_se = list(npygame.support_enumeration()) # return list of all equilbria, each equilbrium is a tuple of numpy arrays, first row player(Bank2) strat then column player strat (Bank1)
        return self.NE_se

class GameFreshEstimate:
    def __init__(self, gammas:list[float], taus: list[float], num_samples:int, dist:Dist):
        '''
            num_samples: number of customers to draw in each round.
        '''
        self.gammas = sorted(gammas)  # important to sort
        self.taus = sorted(taus)
        self.num_samples = num_samples
        self.dist = dist

    def get_PayoffMat_est(self):
        y_samples = self.dist.get_samples(self.num_samples)
        self.A_current = matrix_from_samples(y_samples=y_samples, gammas=self.gammas, taus=self.taus)
        return self.A_current

    def run_hedge(self, T:int, p_b1, p_b2, eta):
        '''
            T: number of rounds of hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        b1_record = [p_b1]
        b2_record = [p_b2]
        for t in range(T):
            A_est = self.get_PayoffMat_est()
            p_b1, p_b2 = HedgeSimultaneous(p_b1, p_b2, eta, A_est)
            b1_record.append(p_b1)
            b2_record.append(p_b2)
        return np.array(b1_record), np.array(b2_record), self.gammas, self.taus


class GameMovingAvg:
    def __init__(self, gammas:list[float], taus: list[float], num_samples:int, dist:Dist):
        '''
            num_samples: number of customers to draw in each round.
        '''
        self.gammas = sorted(gammas) # important to sort in ascending
        self.taus = sorted(taus)
        self.num_samples = num_samples
        self.dist = dist
        self.num_actions = len(gammas) * len(taus)

    def update_PayoffMat_est(self):
        y_samples = self.dist.get_samples(self.num_samples)
        current_matrix = matrix_from_samples(y_samples=y_samples, gammas=self.gammas, taus=self.taus)
        self.num_rounds += 1
        self.A_est = self.A_est + (current_matrix - self.A_est) / self.num_rounds
        return self.A_est

    def run_hedge(self, T:int, p_b1, p_b2, eta):
        '''
            T: number of rounds of hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        self.num_rounds = 0 # tracks number of rounds of hedge
        self.A_est = np.zeros((self.num_actions, self.num_actions)) # set back to zero if running hedge again on same object
        b1_record = [p_b1]
        b2_record = [p_b2]
        for t in range(T):
            self.update_PayoffMat_est()
            p_b1, p_b2 = HedgeSimultaneous(p_b1, p_b2, eta, self.A_est)
            b1_record.append(p_b1)
            b2_record.append(p_b2)
        return np.array(b1_record), np.array(b2_record), self.gammas, self.taus

class GameTrueMatrix2by2: # TODO change name
    '''
        Special case of the GameTrueMatrix for 2 gammas and 2 taus, for which we have theoretical characterization for all NE
    '''
    def __init__(self, gammas: list[float], taus: list[float], dist: Dist):  # both banks have the same strategy spaces
        assert len(gammas) == len(taus) == 2
        self.gammas = sorted(gammas) # important to sort
        self.taus = sorted(taus)
        self.gl, self.gh = self.gammas
        self.taul, self.tauh = self.taus
        self.A = generate_utility_matrix(gammas=self.gammas,taus=self.taus, c_f=dist.c_f)
        self.dist = dist
        self.am = {
            'tlgl': [1, 0, 0, 0],
            'tlgh': [0, 1, 0, 0],
            'thgl': [0, 0, 1, 0],
            'thgh': [0, 0, 0, 1],
        }  # action map for code we index as given in the docstring, slightly diff than the paper index 1 and 2 swapped
        self.save_NE_theory()
        self.saveget_NE_supportenum()

        # add instance name and epsilon sign case
        self.eps_case = f"{'+' if self.eps1 > 0 else '-'}{'+' if self.eps2 > 0 else '-'}"
        self.instance_name = self.dist.name + str(self.gammas) + self.eps_case


    def save_NE_theory(self):
        '''
            analytically characterized NE in the paper depending on epsilon conditions
        '''
        gammal_tauh_1 = self.dist.c_f(self.gl, self.tauh, 1.0)
        gammah_taul_tauh = self.dist.c_f(self.gh, self.taul, self.tauh)
        gammah_tauh_1 = self.dist.c_f(self.gh, self.tauh, 1.0)
        gammah_taul_1 = self.dist.c_f(self.gh, self.taul, 1.0)
        self.eps1 = 0.5 * gammah_taul_1 - gammal_tauh_1
        self.eps2 = gammah_taul_tauh - 0.5 * gammal_tauh_1
        self.c = (gammal_tauh_1 - 2 * gammah_taul_tauh) / (gammah_tauh_1 - gammal_tauh_1 - gammah_taul_tauh)  # useful for mixed NE

        self.NE_theory = []
        if self.eps1 > 0 and self.eps2 > 0:  # only 1 symmetric pure NE is (gammah, taul)
            self.NE_theory.append([self.am['tlgh'], self.am['tlgh']])

        elif self.eps1 < 0 and self.eps2 < 0:  # only 1 symmetric pure NE is (gammal, tauh)
            self.NE_theory.append([self.am['thgl'], self.am['thgl']])

        elif self.eps1 < 0 and self.eps2 > 0:  # 3 NE, 2 assymetric pure, 1 mixed NE
            self.NE_theory.append([self.am['tlgh'], self.am['thgl']])
            self.NE_theory.append([self.am['thgl'], self.am['tlgh']])
            self.NE_theory.append([[0, self.c, 1 - self.c, 0],
                                   [0, self.c, 1 - self.c, 0]])

        elif self.eps1 > 0 and self.eps2 < 0:  # 3 NE, 2 symmetric pure, 1 mixed NE
            self.NE_theory.append([self.am['thgl'], self.am['thgl']])
            self.NE_theory.append([self.am['tlgh'], self.am['tlgh']])
            self.NE_theory.append([[0, self.c, 1 - self.c, 0],
                                   [0, self.c, 1 - self.c, 0]])
        else:
            raise Exception  # anyone of them exactly zero?

    def get_closest_euclidean_NE(self, p_b1, p_b2):
        """
        p_b1: Bank 1's strategy (numpy array)
        p_b2: Bank 2's strategy (numpy array)

        Finds the closest Nash equilibrium (NE) based on L2 distance.
        
        Returns:
        - The closest NE (NE_b1, NE_b2) in terms of Euclidean distance, the corresponding sum of distance of ||NE_b1-p_b1||_2 + ||NE_b2-p_b2||_2
        """
        closest_NE = None
        min_distance = float("inf")
        for NE_b2, NE_b1 in self.NE_se: # nashpy outputs row strat first so this is used
            assert p_b1.shape == NE_b1.shape == p_b2.shape == NE_b2.shape
            dist_b1 = np.linalg.norm(p_b1 - NE_b1) # Compute L2 distances separately for each bank to NE strat
            dist_b2 = np.linalg.norm(p_b2 - NE_b2)
            total_distance = dist_b1 + dist_b2
            if total_distance < min_distance:
                min_distance = total_distance
                closest_NE = (NE_b1, NE_b2)  # Bank 1 then Bank2
        return closest_NE, min_distance

    # def get_closest_elementwise_NE(self, p_b1, p_b2, epsilon=1e-8):
    #     """
    #     p_b1 bank 1's stategy
    #     p_b2 bank 2's stategy

    #     Checks if there exists a Nash equilibrium (of the normal formal game) where each element is within `epsilon` tolerance.

    #     Returns:
    #     - (True, closes_NE list ) if there is a NE is within the element wise tolerance
    #     - (False, []) if there is no NE within element wise tolerance
    #     """
    #     profile = np.concatenate((p_b2, p_b1))
    #     close_NEs = []
    #     for NE_b2, NE_b1 in self.NE_se:
    #         NE_vector = np.concatenate((NE_b2, NE_b1))
    #         if np.allclose(profile, NE_vector, atol=epsilon):
    #             close_NEs.append((NE_b2, NE_b1)) # tuple of numpy arrays
    #     return len(close_NEs) >= 1, close_NEs  # Returns a list (empty if no NE is within tolerance)

    def saveget_NE_supportenum(self):  # runs support enumeration to get all NE, pure, mixed
        '''
            Save and return NE obtained via support enumeration for the bank game
        '''
        npygame = nash.Game(self.A.T, self.A)  # nashpy first takes the row players matrix, then column players;
        self.NE_se = list(npygame.support_enumeration())  # return list of all equilbria, each equilbrium is a tuple of numpy arrays, first row player(Bank2) strat then column player strat (Bank1)
        return self.NE_se

    def run_hedge(self, T:int, p_b1:np.array, p_b2:np.array, eta:float):
        '''
            T: number of rounds of Hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 
            eta: learning rate for hedge

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        b1_record = [p_b1]
        b2_record = [p_b2]
        for t in range(T):
            p_b1, p_b2 = HedgeSimultaneous(p_b1, p_b2, eta, self.A)
            b1_record.append(p_b1)
            b2_record.append(p_b2)
        return np.array(b1_record), np.array(b2_record), self.gammas, self.taus

    def check_convergence_to_theoryeq(self, p_b1, p_b2):
        '''
            check if both banks have converged to any of the NE for the game
        '''
        pass