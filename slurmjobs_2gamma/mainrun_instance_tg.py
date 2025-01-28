import argparse
from game.run_game import run_across_init_2gamma
from game.Bankgames import *
from game.distributions import TruncatedGaussian
from pathlib import Path
from utils.project_dirs import saved_df_2gamma

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--mu', type = float, help = 'has to be in (0,1)')
    parser.add_argument('--sigma', type = float, help = 'has to be in (0,1)')
    parser.add_argument('--gamma_l', type = float, help = 'has to be in (0,1)')
    parser.add_argument('--gamma_h', type = float, help = 'has to be in (0,1)')
    parser.add_argument('--horizon', type = int, help='Hedge horizon', default = 100000) # 100k steps of hedge
    parser.add_argument('--eta', type = float, help='Hedge step size', default = 0.1)
    parser.add_argument('--num_startprofiles', type = int, help='Number of random initializations for both banks', default = 5)
    args = parser.parse_args()

    assert 0.0 < args.gamma_l < args.gamma_h < 1.0
    assert 0.0 < args.mu < 1.0
    assert 0.0 < args.sigma < 1.0

    gammas = sorted([args.gamma_l, args.gamma_h])
    taus = sorted([1.0/(2+gamma) for gamma in gammas])

    tg = TruncatedGaussian(mu = args.mu, sigma = args.sigma)
    gtm = GameTrueMatrix2by2(gammas=gammas, taus=taus, dist=tg)

    epsigns = f'sign{gtm.eps_case}'
    save_dir = saved_df_2gamma(epsigns=epsigns)

    print(f'TG mean is {args.mu}')
    print(f'TG sigma is {args.sigma}')
    print(f'gamma_l, gamma_h are {args.gamma_l}, {args.gamma_h}')
    print(tg.name, gtm.instance_name, gtm.eps_case)
    
    df, df_conc = run_across_init_2gamma(gtm=gtm, save_dir=save_dir, num_startprofiles=args.num_startprofiles, T=args.horizon, eta = args.eta)
    
if __name__ == '__main__':
    main()