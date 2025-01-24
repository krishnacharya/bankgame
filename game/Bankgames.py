from game.helpers import *
from game.distributions import *
from game.hedgealgs import HedgeSimultaneous
import nashpy as nash

class GameTrueMatrix2by2:
    def __init__(self, gammas:list[float], taus: list[float], dist:Dist): # both banks have the same strategy spaces
        '''
            Note action index 0 - (gammal, taul); 1 - (gammal, tauh) ; 2- (gammah, taul) ; 3 - (gammah, tauh)
        '''
        assert len(gammas) == len(taus) == 2
        self.gammas = sorted(gammas)
        self.taus = sorted(taus)
        self.gl, self.gh = self.gammas
        self.taul, self.tauh = self.taus
        self.A = generate_utility_matrix(gammas=self.gammas, taus=self.taus, c_f=dist.c_f)
        self.dist = dist
        self.am = {'tlgl': [1,0,0,0],
        'tlgh': [0,1,0,0], # this has support usually
        'thgl': [0,0,1,0], # this has support usually
        'thgh': [0,0,0,1],
         } # action map for code we index as given in the docstring, slightly diff than the paper index 1 and 2 swapped
        self.save_NE_theory()
        self.save_NE_supportenum()


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
        self.c = (gammal_tauh_1 - 2*gammah_taul_tauh) / (gammah_tauh_1 - gammal_tauh_1 - gammah_taul_tauh) # useful for mixed NE
        
        self.NE_theory = []
        if self.eps1 > 0 and self.eps2 > 0: # only 1 symmetric pure NE is (gammah, taul)
            self.NE_theory.append([self.am['tlgh'], self.am['tlgh']])

        elif self.eps1 < 0 and self.eps2 < 0: # only 1 symmetric pure NE is (gammal, tauh)
            self.NE_theory.append([self.am['thgl'], self.am['thgl']])

        elif self.eps1 < 0 and self.eps2 > 0: # 3 NE, 2 assymetric pure, 1 mixed NE
            self.NE_theory.append([self.am['tlgh'], self.am['thgl']])
            self.NE_theory.append([self.am['thgl'], self.am['tlgh']])
            self.NE_theory.append([[0,self.c,1-self.c, 0], [0,self.c,1-self.c, 0]])

        elif self.eps1 > 0 and self.eps2 < 0: # 3 NE, 2 symmetric pure, 1 mixed NE
            self.NE_theory.append([self.am['thgl'], self.am['thgl']])
            self.NE_theory.append([self.am['tlgh'], self.am['tlgh']])
            self.NE_theory.append([[0,self.c,1-self.c, 0], [0,self.c,1-self.c, 0]])
        else:
            raise Exception # anyone of them exactly zero?

    def save_NE_supportenum(self): # runs support enumeration to get all NE, pure, mixed 
        npygame = nash.Game(self.A.T, self.A) # nashpy first takes the row players matrix, then column players; 
        self.NE_se = list(npygame.support_enumeration()) # return list of all equilbria, each equilbrium is a tuple of numpy arrays, first row player(Bank2) strat then column player strat (Bank1)

    def run_hedge(self):
        pass

    def check_convergence_to_theoryeq(self, p_b1, p_b2):
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
    
    def save_NE_supportenum(self): # runs support enumeration to get all NE, pure, mixed 
        npygame = nash.Game(self.A.T, self.A) # nashpy first takes the row players matrix, then column players; 
        self.NE_se = list(npygame.support_enumeration()) # return list of all equilbria, each equilbrium is a tuple of numpy arrays, first row player(Bank2) strat then column player strat (Bank1)
        return self.NE_se

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
