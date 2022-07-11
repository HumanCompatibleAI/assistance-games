from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_probability(self, x):
        pass

    def sample(self):
        sample = np.random.random()
        for x in self.support():
            total_prob += self.get_probability(x)
            if total_prob >= sample:
                return x

        raise ValueError("Total probability was less than 1")


class ContinuousDistribution(Distribution):
    def support(self):
        raise ValueError("Cannot ask for support of a continuous distribution")

    def get_probability(self, x):
        raise ValueError("Cannot get probability of an element of a continuous distribution")


class KroneckerDistribution(Distribution):
    def __init__(self, x):
        self.x = x

    def support(self):
        yield self.x

    def get_probability(self, x):
        return 1.0 if x == self.x else 0.0

    def sample(self):
        return self.x


class DiscreteDistribution(Distribution):
    def __init__(self, option_prob_map):
        if type(option_prob_map) == np.ndarray:
            assert len(option_prob_map.shape) == 1
            option_prob_map = dict(zip(range(len(option_prob_map)), option_prob_map))
        self.option_prob_map = option_prob_map

    def support(self):
        for x, p in self.option_prob_map.items():
            if p > 0:
                yield x

    def get_probability(self, option):
        return self.option_prob_map.get(option, 0.0)

    def sample(self):
        options, probs = zip(*self.option_prob_map.items())
        idx = np.random.choice(len(options), p=probs)
        return options[idx]


class UniformDiscreteDistribution(DiscreteDistribution):
    def __init__(self, options):
        p = 1.0 / len(options)
        super().__init__({ option:p for option in options })


class UniformContinuousDistribution(ContinuousDistribution):
    def __init__(self, lows, highs):
        self.lows = lows
        self.highs = highs

    def sample(self):
        return np.random.uniform(self.lows, self.highs)


class MapDistribution(Distribution):
    def __init__(self, f, finv, base_dist):
        self.f = f
        self.finv = f
        self.base_dist = base_dist

    def support(self):
        for x in self.base_dist.support():
            yield self.f(x)

    def get_probability(self, option):
        return self.base_dist.get_probability(self.finv(option))

    def sample(self):
        return self.f(self.base_dist.sample())
