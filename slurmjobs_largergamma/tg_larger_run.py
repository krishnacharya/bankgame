import argparse
from game.run_game import run_across_init_largegamma
from game.Bankgames import *
from game.distributions import TruncatedGaussian
from pathlib import Path
from utils.project_dirs import *


def main():
    parser = argparse.ArgumentParser(description='Run experiments with Truncated Gaussian')
    parser.add_argument('--mu',type=float,required=True,help='Mean of truncated Gaussian')
    parser.add_argument('--sigma',type=float,required=True, help='Standard deviation of truncated Gaussian')
    parser.add_argument('--gamma', nargs='+', type=float,required=True,help='List of gamma values in (0,1)')
    parser.add_argument('--horizon',type=int,default=100000,help='Hedge horizon')
    parser.add_argument('--eta', type=float, default=0.1, help='Hedge step size')
    parser.add_argument('--num_startprofiles',type=int,default=5,help='Number of random initializations for both banks')
    args = parser.parse_args()
    assert all(0.0 < g < 1.0 for g in args.gamma), "Gamma values must be in (0,1)"
    gammas = sorted(args.gamma)
    taus = sorted([1.0 / (2 + gamma) for gamma in gammas])

    tg = TruncatedGaussian(mu=args.mu, sigma=args.sigma)
    gtm = GameTrueMatrix(gammas=gammas, taus=taus, dist=tg)
    print(f'Mu: {args.mu}, Sigma: {args.sigma}, Gammas: {gammas}')
    print(tg.name, gtm.instance_name)
    save_dir_full = saved_df_n_dist_full_T(n=len(gammas), distribution='truncated_gaussian', T=args.horizon)
    save_dir_concise = saved_df_n_dist_concise_T(n=len(gammas), distribution='truncated_gaussian', T=args.horizon)

    df, df_conc = run_across_init_largegamma(gtm=gtm, save_dir_full=save_dir_full, save_dir_conc=save_dir_concise,
        num_startprofiles=args.num_startprofiles, T=args.horizon, eta=args.eta)


if __name__ == '__main__':
    main()
