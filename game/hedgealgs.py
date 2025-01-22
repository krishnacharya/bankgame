import numpy as np

def HedgeSimultaneous(p_b1:np.array, p_b2:np.array, eta:float, A:np.array):
    '''
        Description: Simultaneous update

        p_b1 is the probability that (Bank 1) currently places on all its actions
        p_b2 is the probability that (Bank 2) currently places on all its actions
        eta is the learning rate for hedge
        A is either the true or estimated payoff matrix for Bank1
    
        Returns Updated p_b1 and p_b2
    '''
    assert A.shape[0] == A.shape[1] == p_b1.shape[0] == p_b2.shape[0]
    mw_1 = np.exp(eta * (p_b2 @ A))
    mw_2 = np.exp(eta * (p_b1 @ A))
    p_b1 = (mw_1 * p_b1)
    p_b2 = (mw_2 * p_b2)
    return p_b1 / p_b1.sum(), p_b2 / p_b2.sum()

def Hedge_B1thenB2(p_b1, p_b2, eta, A):
    '''        
        p_b1 is the probability that (Bank 1) currently places on all the actions
        p_b2 is the probability that (Bank 2) currently places on all the actions\
        eta is the learning rate for hedge
        A is either the true or estimated payoff matrix for Bank1

        Description: First bank 1 is updated using hedge, then bank 2 vieweing this update performs it's hedge update

        Returns Updated p_b1 and p_b2
    '''
    pass

# def HedgeA(p_be, A, p_ot, eta):
#     ell_t = np.dot(p_ot.T,A)
#     w = np.array([0.0,0.0,0.0,0.0])
#     p_f = np.array([0.0,0.0,0.0,0.0])

#     for i in range(4):
#         w[i] = p_be[i] * np.exp(eta* ell_t[i])


#     for i in range(4):
#         p_f[i] = w[i]/(np.sum(w))
#     return p_f
