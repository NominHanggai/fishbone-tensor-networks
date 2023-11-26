import numpy as np
from fishbonett.common import *
from fishbonett.stuff import sigma_z
from scipy.linalg import svd as csvd
from scipy.linalg import expm
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from numpy import exp
from copy import deepcopy as dcopy
from fishbonett.stuff import temp_factor, sigma_z, sigma_x

class SpinBosonModel:

    def __init__(self, v_x, v_z, pd_spin, pd_boson, boson_domain, sd, dt):
        self.v_x = v_x
        self.v_z = v_z
        self.pd_spin = pd_spin
        self.pd_boson = pd_boson
        self.len_boson = len(self.pd_boson)
        self.domain = boson_domain
        self.w_list, self.k_list = get_bath_nn_paras(sd, self.len_boson, domain=self.domain)
        self.h_boson_onsite = self.get_boson_h_onsite()
        self.h_boson_full = self.get_boson_h_full()
        self.h_full = []
        self.U1, self.U2 = self.get_time_independent_u(dt)

    def get_boson_h_onsite(self):
        w_list = self.w_list
        h_onsite = []
        for i, w in enumerate(w_list):
            c = c_(self.pd_boson[i])
            h_onsite.append(w * c.T @ c)
        return h_onsite

    def get_boson_h_full(self):
        h_onsite = self.h_boson_onsite
        kn = self.k_list[1:]
        h_boson_full = []
        for i, k in enumerate(kn):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = c_(d1)
            c2 = c_(d2)
            nn_coupling = k * (kron(c1.T, c2) + kron(c1, c2.T))
            on_site = kron(h_onsite[i+1], np.eye(d2))
            h_boson_full.append((nn_coupling + on_site, d1, d2))
        return h_boson_full

    def get_full_h(self, t):
        k0 = self.k_list[0]
        d1 = self.pd_spin
        d2 = self.pd_boson[0]
        c = c_(d2)
        spin_momentum_term = kron(sigma_z, 1j * t * k0 * (c - c.T))
        exponent = -2j * t * kron(sigma_z, k0 * (c+c.T))
        spin_exp_term = kron(self.v_x * sigma_x, np.eye(d2)) @ calc_u_sp(exponent, t)
        vib0_onsite = kron(np.eye(d1), self.h_boson_onsite[0])
        h_sb = spin_exp_term + spin_momentum_term + vib0_onsite
        h = [(h_sb, d1, d2)] + self.get_boson_h_full()
        return h
    
    def get_time_independent_u(self, dt):
        self.h_full = self.get_full_h(0)
        U1 = dcopy(self.h_full)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.h_full):
            h, d1, d2 = h_d1_d2
            u = calc_U(h.toarray(), dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1,0,3,2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2
    
    def get_u(self, t, dt):
        self.h_full = self.get_full_h(t)
        i, h_d1_d2 = 0, self.h_full[0]
        h, d1, d2 = h_d1_d2
        u = calc_U(h.toarray(), dt)
        r0 = r1 = d1  # physical dimension for site A
        s0 = s1 = d2  # physical dimension for site B
        u1 = u.reshape([r0, s0, r1, s1])
        u2 = np.transpose(u1, [1,0,3,2])
        self.U1[i] = u1
        self.U2[i] = u2
        return self.U1, self.U2
    
if __name__ == '__main__':
    
    pass
