import numpy as np
from fishbonett.stuff import temp_factor
from scipy.linalg import svd as csvd
from scipy.linalg import expm
from fbpca import pca as rsvd
from opt_einsum import contract as einsum
from scipy.sparse.linalg import expm as sparseExpm
from scipy.sparse import csc_matrix
from numpy import exp
import fishbonett.recurrence_coefficients as rc
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import scipy.integrate as integrate
import sympy
import scipy
from sympy.utilities.lambdify import lambdify

def _c(dim: int):
    """
    Creates the annihilation operator.
    This fuction is from the package py-tedopa/tedopa.
    https://github.com/MoritzLange/py-tedopa/tedopa/

    The BSD 3-Clause License
    Copyright (c) 2018, the py-tedopa developers.
    All rights reserved.

    :param dim: Dimension of the site it should act on
    :type dim: int
    :return: The annihilation operator
    :rtype: numpy.ndarray
    """
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def eye(d):
    if d == [] or None:
        return None
    elif type(d) is int or d is str:
        return np.eye(int(d))
    elif type(d) is list or np.ndarray:
        return np.eye(*d)


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


def svd(A, b, full_matrices=False):
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        # print("CSVD", A.shape, b)
        # cs = csvd(A, full_matrices=False)
        print("RRSVD", A.shape, b)
        rs = rsvd(A, b, True, n_iter=2, l=2 * b)
        # print("Difference", diffsnorm(A, *B))
        # print(cs[1] - rs[1])
        return rs
    else:
        return csvd(A, full_matrices=False)


def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return scipy.linalg.expm(-dt * 1j * H)


def _to_list(x):
    """
    Converts x to [x] if x is a np.ndarray. If x is None,
    convert x(=None) to []. If x is already a list of a
    np.ndarray return x itself. Else if x is not a list of
    just one np.ndarray, raise TypeError.
    :param x: an np.array or a list of one np.ndarray
    :type x:
    :return:
    :rtype:
    """
    if x is None:
        return []
    elif x is list:
        return x
    else:
        return [x]

class SpinBoson:

    def __init__(self, pd, coup_mat, freq, temp):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_spin)
        self.h1e = np.eye(self.pd_spin)
        self.temp = temp
        freq = np.array(freq)
        print(freq)
        self.freq = np.concatenate((-freq, freq))
        self.coup_mat = [mat * np.sqrt(np.abs(temp_factor(temp, self.freq[n]))) for n, mat in
                         enumerate(coup_mat + coup_mat)]
        # self.freq = np.array(freq)
        # self.coup_mat = [mat  for n, mat in enumerate(coup_mat)]
        print('temp', temp)
        print("coup_mat", [mat[0, 0] for mat in self.coup_mat])
        self.size = self.coup_mat[0].shape[0]
        index = np.abs(self.freq).argsort()
        self.freq = self.freq[index]
        self.coup_mat_np = np.array(self.coup_mat)[index]

    def get_coupling(self, n, j, domain, g, ncap=20000):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        _, _, self.h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def poly(self):
        k = self.k_list
        w = self.w_list
        pn_list = [0, 1/k[0]]
        x = sympy.symbols("x")
        for i in range(1, len(k)):
            pi_1 = pn_list[i]
            pi_2 = pn_list[i - 1]
            pi = ((1 / k[i] * x - w[i - 1] / k[i]) * pi_1 - k[i - 1] / k[i] * pi_2).expand()
            pn_list.append(pi)
        pn_list = pn_list[1:]
        return [lambdify(x,pn) for pn in pn_list]

    def diag(self):
        w= self.w_list
        k = self.k_list
        self.coup = np.diag(w) + np.diag(k[1:], 1) + np.diag(k[1:], -1)
        freq, coef = np.linalg.eig(self.coup)
        return freq, coef 

    def get_h2(self, t, delta):
        print("Geting h2")
        freq = self.freq
        # coef = self.coef
        #e = self.phase
        mat_list = self.coup_mat_np
        print("Geting d's")
        # Permutation
        #indexes = np.abs(freq).argsort()
        #freq = freq[indexes]
        # j0 = j0[indexes]
        # END Permutation
        # freq = freq[::-1]
        h2 = []
        for i, mat in enumerate(mat_list):
            print(f"Chain Len {len(self.pd_boson)}; mat_list Len {len(mat_list)}")
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            f = freq[i]
            coup = np.kron(c1 + c1.T, mat)
            site = np.kron(f*c1.T@c1, np.eye(d2))
            h2.append((delta*(coup+site), d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        site = delta*np.kron(np.eye(d1), self.h1e)
        h2[-1] = (h2[-1][0] + site, d1, d2)
        return h2

    def build(self, g, ncap=20000):
        self.build_coupling(g, ncap)
        print("Coupling Over")
        self.freq, self.coef = self.diag()
        # self.pn_list = self.poly()
        # hee = self.get_h2(t)
        # print("Hamiltonian Over")
        # self.H = hee

    def get_u(self, t, dt, factor=1, mode='normal'):
        self.H = self.get_h2(t, dt)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h/factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1,0,3,2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2
