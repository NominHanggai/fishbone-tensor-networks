import numpy as np

from scipy.linalg import svd as csvd
from scipy.linalg import expm
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from scipy.sparse.linalg import expm as sparseExpm
from scipy.sparse import csc_matrix
from numpy import exp
import fishbonett.recurrence_coefficients as rc
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import scipy
from fishbonett.lanczos import lanczos
from fishbonett.stuff import temp_factor, sigma_z


def _c(dim: int):
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def kron(a, b):
    if a is None or b is None:
        return None
    if type(a) is list and type(b) is list:
        return skron(*a, *b, format='csc')
    if type(a) is list and type(b) is not list:
        return skron(*a, b, format='csc')
    if type(a) is not list and type(b) is list:
        return skron(a, *b, format='csc')
    else:
        return skron(a, b, format='csc')


def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    H_sparse = csc_matrix(H)
    return sparseExpm(-dt * 1j * H_sparse)


class SpinBoson:

    def __init__(self, pd, coup, freq, temp):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = [lambda x: np.heaviside(x, 1) / 1. * exp(-x / 100)] * self.pd_spin
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_spin)
        self.h1e = np.eye(self.pd_spin)
        self.temp = temp
        freq = np.array(freq)
        self.freq = np.concatenate((-freq, freq))
        print("coup_mat Zero Temp", [c for c in coup])
        coup = np.concatenate((coup, coup))
        self.coup = [c * np.sqrt(np.abs(temp_factor(temp, self.freq[n]))) for n, c in enumerate(coup)]
        print(f"coup {temp} Temp", [c for c in self.coup])
        index = self.freq.argsort()
        self.freq = self.freq[index]
        # print(f"self.freq {self.freq}")
        self.coup = np.array(self.coup)[index]
        #  ↑ A list of coupling constants c_k. H_i = A_sys \otimes \sum_k c_k * (a+a^\dagger)
        self.H = []

    def build(self):
        def tri_diag(self):
            v0 = [c for c in self.coup]
            print("Initial Vector", v0)
            h = np.diag(self.freq)
            tri_mat, coef = lanczos(h, v0)
            return tri_mat, coef, np.linalg.norm(v0)

        print("Coupling Over")
        tri_mat, Q, k0 = tri_diag(self)
        res = np.diagonal(Q.T @ Q - np.eye(Q.shape[0]))
        print('Lanczos Residual:', res @ res)
        self.w_list = np.diagonal(tri_mat)
        k_list = np.diagonal(tri_mat, -1)
        self.k_list = np.array([k0] + list(k_list))

        hee = self.get_h2()
        print("Hamiltonian Over")
        self.H = hee

    def get_h1(self):
        w_list = self.w_list[::-1]
        h1 = []
        for i, w in enumerate(w_list):
            c = _c(self.pd_boson[i])
            h1.append(w * c.T @ c)
        h1.append(self.h1e)
        return h1

    def get_h2(self):
        h1 = self.get_h1()
        k_list = self.k_list[::-1]
        k0 = k_list[-1]
        k_list = k_list[0:-1]
        h2 = []
        for i, k in enumerate(k_list):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = _c(d1)
            c2 = _c(d2)
            coup = k * (kron(c1.T, c2) + kron(c1, c2.T))
            site = kron(h1[i], np.eye(d2))
            h2.append((coup + site, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        c0 = _c(d1)
        coup = k0 * kron(c0 + c0.T, self.he_dy)
        site = kron(h1[-2], np.eye(d2)) + kron(np.eye(d1), h1[-1])
        h20 = coup + site
        h2.append((h20, d1, d2))
        return h2

    def get_u(self, dt):
        U = [0] * len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = u.toarray().reshape([r0, s0, r1, s1])
            print("Exponential", i, r0 * s0, r1 * s1)
        return U
