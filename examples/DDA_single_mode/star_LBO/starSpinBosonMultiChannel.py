import numpy as np
from fishbonett.stuff import temp_factor
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



def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return scipy.linalg.expm(-dt * 1j * H)


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
        print('temp', temp)
        print("coup_mat", [mat[0, 0] for mat in self.coup_mat])
        self.size = self.coup_mat[0].shape[0]
        index = np.abs(np.array(self.coup_mat)[:,0,0]**2/self.freq).argsort()
        self.freq = self.freq[index]
        self.coup_mat_np = np.array(self.coup_mat)[index]
        print("Order Reference",
              np.abs(np.array(self.coup_mat_np)[:,0,0]**2/self.freq)
              )

    def get_h2(self, delta):
        print("Geting h2")
        freq = self.freq
        mat_list = self.coup_mat_np
        print("Geting d's")
        h2 = []
        for i, mat in enumerate(mat_list):
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

    def get_u(self, dt, factor=1, mode='normal'):
        self.H = self.get_h2(dt)
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
