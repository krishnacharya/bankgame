from game.run_game import run_across_initializations
from game.distributions import *
from game.Bankgames import *
from utils.project_dirs import save_df_n_dist

gammas = sorted([0.1, 0.8])
taus = sorted([1/(2+gamma) for gamma in gammas])
mu = 0.6
sigma = 0.2
tg = TruncatedGaussian(mu = mu, sigma = sigma)
gtm = GameTrueMatrix2by2(gammas=gammas, taus=taus, dist=tg)

print(gtm.eps1, gtm.eps2)

save_dir = str(save_df_n_dist(n = len(gammas), distribution=f'TG_mu{mu}_sig{sigma}') / f'gammas_{str(gammas)}') # get the save directory

run_across_initializations(gtm=gtm, save_dest = save_dir)
