#!/usr/bin/env python

"""Implementation of the Generalized Mallow Model. It's used for modeling
 temporal relations within video collection of one complex activity. """

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np


class Mallow(object):
    """The Generalized Mallows Model"""
    def __init__(self, K, rho_0=1.0, nu_0=0.1):
        """
        Args:
            K: number of subactions in current complex activity
        """
        self._canon_ordering = None
        # number of subactions
        self._K = K
        self.k = 0
        self.rho = [1e-8] * (K - 1)
        self.rho_0 = rho_0
        self._nu_0 = nu_0
        self._dispersion = np.zeros((self._K, 1))
        self._v_j_0 = {}
        self._init_v_j_0()

        self._v_j_sample = 0
        self._nu_sample = 0

    def _init_v_j_0(self):
        for k in range(self._K):
            v_j_0 = 1. / (np.exp(self.rho_0) - 1) - \
                    (self._K - k + 1) / (np.exp((self._K - k + 1) * self.rho_0) - 1)
            self._v_j_0[k] = v_j_0

    def set_sample_params(self, sum_inv_vals, k, N):
        """
        Args:
            sum_inv_vals: summation over all videos in collection for certain
                position in inverse count vectors
            k: current position for computations
            N: number of videos in collection
        """
        self._k = k
        self._nu_sample = self._nu_0 + N
        self._v_j_sample = (sum_inv_vals + self._v_j_0[k] * self._nu_0)  # / (self._nu_0 + N)

    def logpdf(self, ro_j):
        norm_factor = np.log(self._normalization_factor(self.k, ro_j))
        result = -ro_j * self._v_j_sample - norm_factor * self._nu_sample
        return np.array(result)

    def _normalization_factor(self, k, rho_k):
        power = (self._K - k + 1) * rho_k
        numerator = 1. - np.exp(-power)
        denominator = 1. - np.exp(-rho_k)
        return numerator / denominator

    def single_term_prob(self, count, k):
        result = -(self.rho[k] * count) - \
                 np.log(self._normalization_factor(k, self.rho[k]))
        return result

    @staticmethod
    def inversion_counts(ordering):
        """Compute inverse count vector from ordering"""
        ordering = np.array(ordering)
        inversion_counts_v = []
        for idx, val in enumerate(ordering):
            idx_end = int(np.where(ordering == idx)[0])
            inversion_counts_v.append(np.sum(ordering[:idx_end] > idx))
        return inversion_counts_v[:-1]

    def ordering(self, inverse_count):
        """Compute ordering from inverse count vector"""
        ordering = np.ones(self._K, dtype=int) * -1
        for action, val in enumerate(inverse_count):
            for idx, established in enumerate(ordering):
                if established > -1:
                    continue
                if val == 0:
                    ordering[idx] = action
                    break
                if established == -1:
                    val -= 1
        # last action
        ordering[np.where(ordering == -1)] = self._K - 1
        return ordering
