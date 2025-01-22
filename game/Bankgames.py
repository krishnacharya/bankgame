from game.helpers import *
from game.distributions_integrals import *

class GameTrueMatrix:
    def __init__(self, gammas:list[float], taus: list[float]): # both banks have the same strategy spaces
        self.A = generate_utility_matrix(gammas=gammas, taus=taus)
        self.gammas = gammas
        self.taus = taus

    def run_hedge(self, T:int, p_b1, p_b2):
        '''
            T: number of rounds of Hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        pass


class GameFreshEstimate:
    def __init__(self, gammas:list[float], taus: list[float], num_samples:int, dist:Dist):
        '''
            num_samples: number of customers to draw in each round.
        '''
        self.gammas = gammas
        self.taus = taus
        self.num_samples = num_samples
        self.dist = dist

    def get_PayoffMat_est(self):
        y_samples = self.dist.get_samples(self.num_samples)
        self.Aest = matrix_from_samples(y_samples=y_samples, gammas=self.gammas, taus=self.taus)
        

    def run_hedge(self, T:int, p_b1, p_b2):
        '''
            T: number of rounds of hedge
            p_b1: initial probability weights on each action for bank1 
            p_b2: initial probability weights on each action for bank1 

            returns 
            probability on each action in each round
            
            return type -  (T, #actions), (T, #actions)
            Bank 1, Bank2
        '''
        for t in range(T):
            


class GameRunningEstimate:
    pass
