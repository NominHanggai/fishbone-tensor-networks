import numpy as np
from fishbonett.common import *
from fishbonett.stuff import sigma_z
from scipy.linalg import svd as csvd
from scipy.linalg import expm
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from numpy import exp
import fishbonett.recurrence_coefficients as rc
from copy import deepcopy as dcopy
from fishbonett.stuff import temp_factor, sigma_z, sigma_x

class SpinBoson:

    def __init__(self, v_x, v_z, pd_spin, pd_boson, boson_domain, sd, coup, freq, temperature):
        self.pd_spin = pd_spin
        self.pd_boson = pd_boson
        self.len_boson = len(self.pd_boson)
        self.domain = boson_domain
        self.w_list, self.k_list = get_bath_nn_paras(sd, self.len_boson, domain=self.domain)

    def get_boson_h_onsite(self):
        w_list = self.w_list
        h1 = []
        for i, w in enumerate(w_list):
            c = c_(self.pd_boson[i])
            h1.append(w * c.T @ c)
        return h1

    def get_boson_h_full(self):
        h_onsite = self.get_boson_h_onsite()
        kn = self.k_list[1:]
        h_boson_full = []
        for i, k in enumerate(kn):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = c_(d1)
            c2 = c_(d2)
            nn_coupling = k * (kron(c1.T, c2) + kron(c1, c2.T))
            on_site = kron(h_onsite[i], np.eye(d2))
            h_boson_full.append((nn_coupling + on_site, d1, d2))
        return h_boson_full

    def get_full_h(self, t):
        k0 = self.k_list[0]
        d1 = self.pd_spin
        d2 = self.pd_boson[0]
        c = c_(d2)
        spin_momentum_term = kron(sigma_z, 1j * t * k0 (c - c.T))
        exponent = -2j * t * kron(sigma_z, k0 * (c+c.T))
        spin_exp_term = kron(self.v_x * sigma_x, np.eye(d2)) @ calc_U(exponent, t)
        h_sb = spin_exp_term + spin_momentum_term
        h = [(h_sb, d1, d2)] + self.get_boson_h_onsite()
        return h

    def get_u(self, dt):
        U = [0] * len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = (d1, d2, u)
            print("Exponential", i, r0 * s0, r1 * s1)
        return U
