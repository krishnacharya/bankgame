from game.helpers import *

class GameTrue:
    def __init__(self, gammas:list[float], taus: list[float]): # both banks have the same strategy spaces
        self.A = generate_utility_matrix(gammas=gammas, taus=taus)


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
        pass


class GameFreshEstimate:
    def get_PayoffMat_est(self, number_samples:int):
        self.number_samples = number_samples

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
        pass


class GameRunningEstimate:
    pass
