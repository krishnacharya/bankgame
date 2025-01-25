from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import truncnorm
from scipy.integrate import quad


class Dist(ABC):
    @abstractmethod
    def get_samples(self, num_samples: int):
        """
        Generate `num_samples` from the probability distribution.
        """
        pass

    @abstractmethod
    def c_f(self, gamma, tau_a, tau_b):
        """
        Computes the integral from tau_a to tau_b of 
        [(2+gamma)y - 1] * p(y) dy.
        """
        pass


class Uniform(Dist):
    def get_samples(self, num_samples: int):
        return np.random.uniform(0, 1, num_samples)

    def c_f(self, gamma, tau_a, tau_b):
        return 0.5 * (2 + gamma) * (tau_b**2 - tau_a**2) - (tau_b - tau_a)


class TruncatedGaussian(Dist):
    def __init__(self, mu=0.6, sigma=0.2):
        self.mu = mu
        self.sigma = sigma
        self.a, self.b = 0, 1  # Truncation limits
        self.a_scaled = (self.a - mu) / sigma
        self.b_scaled = (self.b - mu) / sigma

    def get_samples(self, num_samples: int):
        return truncnorm.rvs(self.a_scaled,
                             self.b_scaled,
                             loc=self.mu,
                             scale=self.sigma,
                             size=num_samples)

    def c_f(self, gamma, tau_a, tau_b):
        def truncated_normal_pdf(y):
            return truncnorm.pdf(y,
                                 self.a_scaled,
                                 self.b_scaled,
                                 loc=self.mu,
                                 scale=self.sigma)

        def integrand(y):
            return ((2 + gamma) * y - 1) * truncated_normal_pdf(y)

        result, _ = quad(integrand, tau_a, tau_b)
        return result


class PiecewiseUniform(Dist):
    def __init__(self, ga_l=0.5, ga_h=0.6):
        self.ga_l = ga_l
        self.ga_h = ga_h
        self.tau_l = 1 / (2 + self.ga_h)
        self.tau_h = 1 / (2 + self.ga_l)

    def get_samples(self, num_samples: int):
        bins = np.array([0, self.tau_l, self.tau_h, 1])
        probs = np.array([0.01 / self.tau_l, 0.95 / (self.tau_h - self.tau_l),0.049 / (1 - self.tau_h)])
        probs /= probs.sum()
        samples = np.random.choice([0, 1, 2], size=num_samples, p=probs)
        lower_bounds = bins[samples]
        upper_bounds = bins[samples + 1]
        return np.random.uniform(lower_bounds, upper_bounds)

    def c_f(self, gamma, tau_a, tau_b):
        def piecewise_uniform_pdf(y):
            if 0 <= y < self.tau_l:
                return 0.001 / self.tau_l
            elif self.tau_l <= y < self.tau_h:
                return 0.95 / (self.tau_h - self.tau_l)
            elif self.tau_h <= y <= 1:
                return 0.049 / (1 - self.tau_h)
            else:
                return 0

        def integrand(y):
            return ((2 + gamma) * y - 1) * piecewise_uniform_pdf(y)

        result, _ = quad(integrand, tau_a, tau_b)
        return result
