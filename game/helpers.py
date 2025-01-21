import numpy as np

def generate_utility_matrix(gammas, taus):
    '''
        Return utility matrix for bank1, column index represents bank1's strategy, row index represents bank2's strat
    '''
    n = len(gammas)
    size = n * n  # Matrix size based on all possible gamma-tau pairs
    matrix = np.zeros((size, size))

    # Helper function to get index for gamma-tau pair
    def get_pair_index(gamma_idx, tau_idx):
        return gamma_idx * n + tau_idx

    # Iterate through all combinations for both banks
    for g1 in range(n):  # Bank 1 gamma
        for t1 in range(n):  # Bank 1 tau
            for g2 in range(n):  # Bank 2 gamma
                for t2 in range(n):  # Bank 2 tau
                    row = get_pair_index(t2, g2)  # Bank 2's choice
                    col = get_pair_index(t1, g1)  # Bank 1's choice

                    tau1 = taus[t1]
                    tau2 = taus[t2]

                    ga1 = gammas[g1]
                    ga2 = gammas[g2]

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


def calculate_utility_from_sample(gamma1, tau1, gamma2, tau2, y):
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
    n = len(gammas)
    size = n * n  # Matrix size based on all possible gamma-tau pairs
    matrix = np.zeros((size, size))

    def get_pair_index(gamma_idx, tau_idx):
        return gamma_idx * n + tau_idx

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