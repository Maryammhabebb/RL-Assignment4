import numpy as np

class GaussianNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def sample(self, size):
        return np.random.normal(0, self.sigma, size)
