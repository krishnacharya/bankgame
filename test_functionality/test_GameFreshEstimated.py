from game.Bankgames import *
from game.distributions import *

gammas = [0.5, 0.6]
taus = sorted([1/(2+ga) for ga in gammas])

tg = TruncatedGaussian(mu = 0.6, sigma = 0.2)
num_samples = 10 # 10 per iteration
gfe = GameFreshEstimate(gammas=gammas, taus=taus, num_samples=num_samples,  dist = tg)


T = 10000
eta = 1
p_b1 = np.array([0.1, 0.5, 0.3, 0.1])
p_b2 = np.array([0.1, 0.5, 0.3, 0.1])

b1_rec, b2_rec = gfe.run_hedge(T=T, p_b1=p_b1, p_b2=p_b2, eta=1)
