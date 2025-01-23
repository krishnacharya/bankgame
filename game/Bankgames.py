from game.helpers import *
from game.distributions import *
from game.hedgealgs import HedgeSimultaneous

class GameTrueMatrix2by2:
    def __init__(self, gammas:list[float], taus: list[float], dist:Dist): # both banks have the same strategy spaces
        assert len(gammas) == len(taus) == 2
        self.gl, self.gh = sorted(gammas)
        self.taul, self.tauh = sorted(taus)
        self.A = generate_utility_matrix(gammas=gammas, taus=taus, c_f=dist.c_f)
        self.dist = dist
        self.save_eps12_c()

        # TODO resmaining
    def save_eps12_c():
        # self.eps1
        # self.eps2
        # self.cval
        pass
    
    def run_hedge():
        pass
    
    def get_theory_eq():
        '''
            Get the NE for the game depedning on eps_1, eps_2, c etc
        '''
        pass
    
    def check_convergence_to_theoryeq(p_b1, p_b2):
        '''
            check if both banks have converged to any of the NE for the game
        '''
        pass



class GameTrueMatrix:
    def __init__(self, gammas:list[float], taus: list[float], dist:Dist): # both banks have the same strategy spaces
        self.A = generate_utility_matrix(gammas=gammas, taus=taus, c_f=dist.c_f)
        self.dist = dist
        self.gammas = sorted(gammas)
        self.taus = sorted(taus)

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

class GameFreshEstimate:
    def __init__(self, gammas:list[float], taus: list[float], num_samples:int, dist:Dist):
        '''
            num_samples: number of customers to draw in each round.
        '''
        self.gammas = sorted(gammas)
        self.taus = sorted(taus)
        self.num_samples = num_samples
        self.dist = dist

    def get_PayoffMat_est(self):
        y_samples = self.dist.get_samples(self.num_samples)
        return matrix_from_samples(y_samples=y_samples, gammas=self.gammas, taus=self.taus)

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
        self.gammas = sorted(gammas)
        self.taus = sorted(taus)
        self.num_samples = num_samples
        self.dist = dist
        self.num_rounds = 0 # number of rounds of hedge
        self.num_actions = len(gammas) * len(taus)
        self.A_est = np.zeros((self.num_actions, self.num_actions))

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
        b1_record = [p_b1]
        b2_record = [p_b2]
        for t in range(T):
            self.update_PayoffMat_est()
            p_b1, p_b2 = HedgeSimultaneous(p_b1, p_b2, eta, self.A_est)
            b1_record.append(p_b1)
            b2_record.append(p_b2)
        return np.array(b1_record), np.array(b2_record), self.gammas, self.taus
