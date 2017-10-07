__author__ = 'Jacob Montiel'

import numpy as np
from skmultiflow.core.utils.statistics import normal_probability

class GaussianEstimator(object):
    """ GaussianEstimator
    Gaussian incremental estimator that uses incremental method that is more resistant to floating point imprecision.
    For more info, Donald Knuth's "The Art of Computer Programming, Volume 2: Seminumerical Algorithms", section 4.2.2.
    """
    def __init__(self):
        self._weight_sum = 0.0
        self._mean = 0.0
        self._variance_sum = 0.0
        self._NORMAL_CONSTANT = np.sqrt(2 * np.pi)

    def add_observation(self, value, weight):
        """ add_observation

        Adds a new observation and updates statistics

        Parameters
        ----------
        value: The value
        weight: The weight of the instance

        Returns
        -------
        self

        """
        if value is None or np.isinf(value):
            return
        if self._weight_sum > 0.0:
            self._weight_sum += weight
            last_mean = self._mean
            self._mean += weight * (value - last_mean) / self._weight_sum
            self._variance_sum += weight * (value - last_mean) * (value - self._mean)
        else:
            self._mean = value
            self._weight_sum = weight

    def get_total_weight_observed(self):
        return self._weight_sum

    def get_mean(self):
        return self._mean

    def get_std_dev(self):
        return np.sqrt(self.get_variance())

    def get_variance(self):
        return self._variance_sum / (self._weight_sum - 1.0) if self._weight_sum > 1.0 else 0.0

    def probability_density(self, value):
        """ probability_density

        Calculates the normal distribution

        Parameters
        ----------
        value: The value

        Returns
        -------
        Probability density (normal distribution)

        """
        if self._weight_sum > 0.0:
            std_dev = self.get_std_dev()
            if std_dev > 0.0:
                diff = value - self.get_mean()
                return (1.0 / (self._NORMAL_CONSTANT * std_dev)) * np.exp(-(diff**2 / (2.0 * std_dev**2)))
            else:
                return 1.0 if value == self.get_mean() else 0.0
        else:
            return 0.0

    def estimated_weight_lessthan_equalto_greaterthan_value(self, value):
        equalto_weight = self.probability_density(value) * self._weight_sum
        std_dev = self.get_std_dev()
        if std_dev > 0.0:
            lessthan_weight = normal_probability((value * self.get_mean()) / std_dev) * self._weight_sum \
                              - equalto_weight
        else:
            if value < self.get_mean():
                lessthan_weight = self._weight_sum - equalto_weight
            else:
                lessthan_weight = 0.0
        greaterthan_weight = self._weight_sum - equalto_weight - lessthan_weight
        if greaterthan_weight < 0.0:
            greaterthan_weight = 0.0

        return [lessthan_weight, equalto_weight, greaterthan_weight]
