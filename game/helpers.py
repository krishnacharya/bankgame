import numpy as np
from fractions import Fraction
from scipy.stats import truncnorm, beta
from scipy.integrate import quad
from itertools import product
from game.Bankgames import *
from game.plotting import *
from game.distributions import TruncatedGaussian, Uniform, PiecewiseUniform
from tqdm import tqdm

# def c_f(gamma, tau_a, tau_b, c = 2):

#     if c==1:
#         ####### Uniform56
#         # epsilon_1>0, epsilon_2 < 0 33
#         return 0.5 * (2 + gamma) * (tau_b**2- tau_a**2) - (tau_b - tau_a)
#     if c==2:
#         ####### Trancate Gaussian
#         ## sigma = 0.2, ga_l = 0.2 ga_h = 0.9
#         # mu = 0.1 epsion_1 < 0, epsilon_2 >0 22
#         # mu = 0.4 epsilon_1 <0, epsilon_2 <0 22 33
#         # mu = 0.6 epsilon_1 <0, epsilon_2 <0 22 33
#         # mu = 0.7 epsilon_1 >0, epsilon_2 <0 33
#         #
#         ## sigma  = 0.1, mu = 0.55, ga_l = 0.5 ga_h = 0.6
#         sigma = 0.2
#         mu = 0.6
#         def truncated_normal_pdf(y):
#             a, b = 0, 1
#             a_scaled = (a - mu) / sigma
#             b_scaled = (b - mu) / sigma
#             return truncnorm.pdf(y, a_scaled, b_scaled, loc=mu, scale=sigma)

#         def integrand(y):
#             return ((2 + gamma) * y - 1) * truncated_normal_pdf(y)

#         result, _ = quad(integrand, tau_a, tau_b)
#         return result
#     if c==3:
#         ##### Beta
#         alpha = 1
#         beta_param = 2
#         def integrand(y, gamma, alpha, beta_param):
#             return ((2 + gamma) * y - 1) * beta.pdf(y, alpha, beta_param)
#         result, _ = quad(integrand, tau_a, tau_b, args=(gamma, alpha, beta_param))
#         return result
#     if c ==4:
#         ####### pisewise uniform
#         ga_l = 0.5
#         ga_h = 0.6
#         tau_l = 1/(2+ga_h)
#         tau_h = 1/(2+ga_l)

#         def piecewise_uniform_pdf(y):
#             if 0 <= y < tau_l:
#                 return (0.01) / tau_l
#             elif tau_l <= y < tau_h:
#                 return (0.95) / (tau_h - tau_l)
#             elif tau_h <= y <= 1:
#                 return 0.049 / (1 - tau_h)
#             else:
#                 return 0  # 超出 [0, 1] 范围

#         def integrand(y):
#             return ((2 + gamma) * y - 1) * piecewise_uniform_pdf(y)

#         result, _ = quad(integrand, tau_a, tau_b)
#         return result

def single_sample_from_simplex(dimension) -> np.array:
    '''
    Sample a SINGLE user from the probability simplex of dimension=dimension \sum_j x_j = 1
    shape (dim=#of actions, )
    '''
    #unifrom sampling form the probability simplex
    #reference : https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    return np.diff([0] + sorted(np.random.uniform(size=dimension-1)) + [1])

def sample_simplex_profile(dimension, num_banks = 2) -> np.array:
    '''
        each row in the returned is a vector on the probability simplex
    '''
    return np.array([single_sample_from_simplex(dimension=dimension) for _ in range(num_banks)]) # shape (2,d)

def generate_utility_matrix(gammas, taus, c_f):
    '''
        Return utility matrix for bank1, column index represents bank1's strategy, row index represents bank2's strat
        c_f is a function that can do integral tau_a to tau_b of [(2+gamma)y - 1] p(y) dy

        action indexing is of the form
        tau_1, (gamma_1....gamma_n); tau_2 (gamma_1 ...gamma_n); ... ; tau_n (gamma_1 ... gamma_n)
    '''
    g = len(gammas)
    n = len(taus)
    size =  n*g  # Matrix size based on all possible gamma-tau pairs
    matrix = np.zeros((size, size))

    # Helper function to get index for gamma-tau pair
    def get_pair_index(tau_idx, gamma_idx):
        return tau_idx * g + gamma_idx

    # Iterate through all combinations for both banks
    for g1 in range(g):  # Bank 1 gamma
        for t1 in range(n):  # Bank 1 tau
            for g2 in range(g):  # Bank 2 gamma
                for t2 in range(n):  # Bank 2 tau
                    row = get_pair_index(t2, g2)  # Bank 2's choice
                    col = get_pair_index(t1, g1)  # Bank 1's choice
                    
                    tau1, tau2 = taus[t1], taus[t2]
                    ga1, ga2 = gammas[g1], gammas[g2]
                    if tau1 > tau2:
                        if ga1 > ga2:  # Higher gamma
                            matrix[row, col] = 0
                        elif ga1 < ga2:
                            matrix[row, col] = c_f(gammas[g1], taus[t1], 1)
                        else:
                            matrix[row, col] = 0.5 * c_f(gammas[g1], taus[t1], 1)
                    elif tau1 == tau2:
                        if ga1 > ga2:
                            matrix[row, col] = 0
                        elif ga1 < ga2:
                            matrix[row, col] = c_f(gammas[g1], taus[t1], 1)
                        else:
                            matrix[row, col] = 0.5 * c_f(gammas[g1], taus[t1], 1)
                    elif tau1 < tau2:
                        if ga1 > ga2:
                            matrix[row, col] = c_f(gammas[g1], taus[t1], taus[t2])
                        elif ga1 < ga2:
                            matrix[row, col] = c_f(gammas[g1], taus[t1], 1)
                        else:
                            matrix[row, col] = c_f(gammas[g1], taus[t1], taus[t2]) + 0.5 * c_f(gammas[g1], taus[t2], 1)
    return matrix


def calculate_utility_from_sample(gamma1:float, tau1:float, gamma2:float, tau2:float, y:float):
    '''
    Calculate utility for Bank 1 from this customer with credit score y
    y is the sampled credit score for a customer

    Bank 1 accepts y only if its credit score is higher than its threshold
  '''
    if 0 < y < tau1:
        return 0
    elif tau1 <= y < 1:
        if y < tau2:  # number line [0, tau1, y, tau2, 1]
            return (2 + gamma1) * y - 1
        if tau2 <= y:  # number line [0, tau1, tau2, y, 1]
            if gamma1 < gamma2:
                return (2 + gamma1) * y - 1
            elif gamma1 == gamma2:
                return 0.5 * ((2 + gamma1) * y - 1)
            else:
                return 0
    else:
        raise Exception


def matrix_from_samples(y_samples, gammas, taus):
    '''
        action indexing of the form tau_1, (gamma_1....gamma_n); tau_2 (gamma_1 ...gamma_n); ... ; tau_n (gamma_1 ... gamma_n)
    '''
    n = len(gammas)
    size = n * n  # Matrix size based on all possible gamma-tau pairs
    matrix = np.zeros((size, size))

    def get_pair_index(tau_idx, gamma_idx):
        return tau_idx * n + gamma_idx

    for g1 in range(n):
        for t1 in range(n):
            for g2 in range(n):
                for t2 in range(n):
                    row = get_pair_index(t2, g2)  # Bank 2's choice
                    col = get_pair_index(t1, g1)  # Bank 1's choice

                    tau1, tau2 = taus[t1], taus[t2]
                    ga1, ga2 = gammas[g1], gammas[g2]
                    matrix[row, col] = np.mean([
                        calculate_utility_from_sample(ga1, tau1, ga2, tau2, y)
                        for y in y_samples
                    ])
    return matrix


def search_eps_signs(dist = 'truncated_gaussian', sigma_value = 0.2, mu_values = [0.1, 0.2, 0.6, 0.8, 0.9]):
    from game.Bankgames import GameTrueMatrix2by2
    if dist == 'truncated_gaussian':
        gamma_values = np.linspace(0.1, 1.0, 10)  # Choosing 5 values for gamma
        mu_values = mu_values
        sigma_values = [sigma_value]
    elif dist == 'piecewise_uniform':
        gamma_values = np.linspace(0.01, 0.99, 100)
    else:
        raise ValueError(f"Distribution {dist} not supported")

    found_cases = {}
    if dist == 'truncated_gaussian':
        # Iterate over all possible combinations
        for gamma_l, gamma_h, mu, sigma in product(gamma_values, gamma_values, mu_values, sigma_values):
            if gamma_l >= gamma_h:  # Ensure gamma_l < gamma_h
                continue
            
            taus = sorted([1 / (2 + gamma) for gamma in [gamma_l, gamma_h]])
            tg = TruncatedGaussian(mu=mu, sigma=sigma)
            gtm = GameTrueMatrix2by2(gammas=[gamma_l, gamma_h], taus=taus, dist=tg)
            
            eps1, eps2 = gtm.eps1, gtm.eps2
            sign_pair = (np.sign(eps1), np.sign(eps2))
            pair = (int(sign_pair[0]), int(sign_pair[1]))

            # Store unique cases for each sign combination
            if pair not in found_cases:
                found_cases[pair] = [(gamma_l, gamma_h, mu, sigma, eps1, eps2)]
            else:
                found_cases[pair].append((gamma_l, gamma_h, mu, sigma, eps1, eps2))

            # Stop early if we found all sign cases
            if len(found_cases) == 4:
                break
    elif dist == 'piecewise_uniform':
        for gamma_l, gamma_h in tqdm(product(gamma_values, gamma_values)):
            if gamma_l >= gamma_h:  # Ensure gamma_l < gamma_h
                continue
            taus = sorted([1 / (2 + gamma) for gamma in [gamma_l, gamma_h]])
            puf = PiecewiseUniform(ga_l = gamma_l, ga_h = gamma_h)
            gtm = GameTrueMatrix2by2(gammas=[gamma_l, gamma_h], taus=taus, dist=puf)
            
            eps1, eps2 = gtm.eps1, gtm.eps2
            sign_pair = (np.sign(eps1), np.sign(eps2))
            pair = (int(sign_pair[0]), int(sign_pair[1]))
            # Store unique cases for each sign combination
            if sign_pair not in found_cases:
                found_cases[pair] = [(gamma_l, gamma_h)]
            else:
                found_cases[pair].append((gamma_l, gamma_h))

            # Stop early if we found all sign cases
            if len(found_cases) == 4:
                break
    else:
        raise ValueError(f"Distribution {dist} not supported")
    # Print results
    #for sign_pair, values in found_cases.items():
    #    gamma_l, gamma_h, mu, sigma, eps1, eps2 = values[0]
    #    print(f"Sign: {sign_pair}, γ_l: {gamma_l:.2f}, γ_h: {gamma_h:.2f}, μ: {mu:.2f}, σ: {sigma:.2f}, eps1: {eps1:.4f}, eps2: {eps2:.4f}")
    
    return found_cases