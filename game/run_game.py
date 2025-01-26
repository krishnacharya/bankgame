from game.distributions import Dist
from game.Bankgames import GameTrueMatrix, GameFreshEstimate, GameMovingAvg, GameTrueMatrix2by2
import numpy as np
from game.helpers import sample_simplex_profile
from tqdm import tqdm
import pandas as pd

def run_across_initializations(gtm: GameTrueMatrix2by2, save_dest:str, num_startprofiles=5, T=10000, eta=0.1, seed=21):
    '''
        This is for the n = 2 gammas case

        T is the number of rounds of hedge
        num_startprofiles is the number of different start profiles we want to start the hedge algorithm from
        run hedge across different initialization for bank1 and bank2 on the probability simplex
            - on the true matrix 
            - on the fresh estimation each time
            - on the moving average version
    '''
    n = len(gtm.gammas)
    np.random.seed(seed=seed)
    start_profiles = [sample_simplex_profile(dimension=n**2, num_banks=2) for _ in range(num_startprofiles)]  # n^2 actions for each bank

    # Different number of samples in each round
    gf1 = GameFreshEstimate(gammas=gtm.gammas, taus=gtm.taus, num_samples=1, dist=gtm.dist)
    gmv1 = GameMovingAvg(gammas=gtm.gammas, taus=gtm.taus, num_samples=1, dist=gtm.dist)

    gf10 = GameFreshEstimate(gammas=gtm.gammas, taus=gtm.taus, num_samples=10, dist=gtm.dist)
    gmv10 = GameMovingAvg(gammas=gtm.gammas, taus=gtm.taus, num_samples=10, dist=gtm.dist)

    gf20 = GameFreshEstimate(gammas=gtm.gammas, taus=gtm.taus, num_samples=20, dist=gtm.dist)
    gmv20 = GameMovingAvg(gammas=gtm.gammas, taus=gtm.taus, num_samples=20, dist=gtm.dist)
    res = []
    with tqdm(total=len(start_profiles), mininterval=5) as pbar:
        for profile in start_profiles:
            s1, s2 = profile[0], profile[1]  # Each is a numpy array of shape (n^2,)
            di = {'Bank1_start': s1, 'Bank2_start': s2, 'eps1': gtm.eps1, 'eps2': gtm.eps2}

            # Run hedge on each of the games and get the strategy profiles over time
            bank1_gtm, bank2_gtm, _, _ = gtm.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)  # Game true matrix

            bank1_gf1, bank2_gf1, _, _ = gf1.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)  # Game fresh estimates
            # bank1_gf10, bank2_gf10, _, _ = gf10.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)
            # bank1_gf20, bank2_gf20, _, _ = gf20.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)

            bank1_gmv1, bank2_gmv1, _, _ = gmv1.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)  # Game moving estimates
            # bank1_gmv10, bank2_gmv10, _, _ = gmv10.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)
            # bank1_gmv20, bank2_gmv20, _, _ = gmv20.run_hedge(T=T, p_b1=s1, p_b2=s2, eta=eta)

            # Check convergence of last iterate to NE for each of the games # TODO check for typos
            di['closestNE_knownmat'], di['closestNEdist_knownmat'] = gtm.get_closest_eucliedean_NE(p_b1=bank1_gtm[-1], p_b2=bank2_gtm[-1])

            di['closestNE_fresh1'], di['closestNEdist_fresh1'] = gtm.get_closest_eucliedean_NE(p_b1=bank1_gf1[-1], p_b2=bank2_gf1[-1])
            # di['converged_fresh_10'], _  = gtm.get_closest_elementwise_NE(p_b1=bank1_gf10[-1], p_b2=bank2_gf10[-1])
            # di['converged_fresh_20'], _ = gtm.get_closest_elementwise_NE(p_b1=bank1_gf20[-1], p_b2=bank2_gf20[-1])

            di['closestNE_moving1'], di['closestNEdist_moving1']  = gtm.get_closest_eucliedean_NE(p_b1=bank1_gmv1[-1], p_b2=bank2_gmv1[-1])
            # di['converged_moving_10'], _ = gtm.get_closest_elementwise_NE(p_b1=bank1_gmv10[-1], p_b2=bank2_gmv10[-1])
            # di['converged_moving_20'], _ = gtm.get_closest_elementwise_NE(p_b1=bank1_gmv20[-1], p_b2=bank2_gmv20[-1])
            pbar.update(1)
            res.append(di)
    df = pd.DataFrame(res)
    df.to_pickle(save_dest)

