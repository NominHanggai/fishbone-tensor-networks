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
import scipy.integrate as integrate
import sympy
import scipy
from sympy.utilities.lambdify import lambdify
from fishbonett.stuff import drude, temp_factor, _c


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


class SpinBoson:

    def __init__(self, pd):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_spin)
        self.h1e = np.eye(self.pd_spin)
        self.k_list = []
        self.w_lsit = []
        self.H = []
        self.coef= []
        self.freq = []
        self.phase = lambda lam, t, delta: (np.exp(-1j*lam*(t+delta)) - np.exp(-1j*lam*t))/(-1j*lam)
        self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))
        # self.phase = lambda lam, t, delta: np.exp(-1j * lam * (t+delta/2)) * delta

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
        freq, coef = np.linalg.eigh(self.coup)
        sign = np.sign(coef[0,:])
        coef = coef.dot(np.diag(sign))
        return freq, coef

    def get_dk(self, t, star=False):
        freq = self.freq
        coef = self.coef
        e = self.phase_func
        k0 = self.k_list[0]
        j0 = k0 * coef[0, :]  # interaction strength in the diagonal representation
        if star:
            indexes = freq.argsort()
            freq = freq[indexes]
            j0 = j0[indexes]
            reorg = sum([j0[i]**2/ freq[i] for i in range(len(j0))])
            return j0, freq, coef, reorg
        else:
            phase_factor = np.array([e(w, t) for w in freq])
            print("Geting d's")
            j = lambda w: np.pi*4*drude(w, lam=4.0*78.53981499999999/2, gam=0.25*4*19.634953749999998
                                  ) * temp_factor(226.00253972894595*0.5*1,w)
            g=1
            def h_squared(x):
                return j(g * x) * g / np.pi
            j_list = np.array([np.sqrt(h_squared(x)) for x in freq])
            # print(freq)
            # print(j_list)
            # print(j0)
            perm = np.abs(j0).argsort()
            shuffle = coef.T
            d_nt = [einsum('k,k,k', j0, shuffle[:, n], phase_factor) for n in range(len(freq))]
            # d_nt_p = [einsum('k,k,k', j_list, shuffle[:, n], phase_factor) for n in range(len(freq))]
            d_nt = d_nt #+ d_nt_p
            # print(f'd_nt{d_nt}')
            d_nt = d_nt[::-1]
            return d_nt

    def get_h2(self, t, delta, inc_sys=True):
        print("Geting h2")
        freq = self.freq
        coef = self.coef
        e = self.phase
        k0 = self.k_list[0]
        j0 = k0 * coef[0,:] # interaction strength in the diagonal representation
        phase_factor = np.array([e(w, t, delta) for w in freq])
        print("Geting d's")
        perm = np.abs(j0).argsort()
        shuffle = coef.T#[perm]
        d_nt = [einsum('k,k,k', j0, shuffle[:,n], phase_factor) for n in range(len(freq))]
        # print(f'd_nt{d_nt}')
        d_nt = d_nt[::-1]
        h2 = []
        # ul = calc_U(self.h1e, -t)
        # he_dy = ul @ self.he_dy @ (ul.T.conj())
        he_dy = self.he_dy
        for i, k in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            kc = k.conjugate()
            coup = kron(k*c1 + kc* c1.T, he_dy)
            h2.append((coup, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        site = delta*kron(np.eye(d1), self.h1e)
        if inc_sys is True:
            h2[-1] = (h2[-1][0] + site, d1, d2)
        else:
            h2[-1] = (h2[-1][0], d1, d2)
        return h2

    def build(self, g, ncap=20000):
        self.build_coupling(g, ncap)
        print("Coupling Over")
        self.freq, self.coef = self.diag()
        # self.pn_list = self.poly()
        # hee = self.get_h2(t)
        # print("Hamiltonian Over")
        # self.H = hee

    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        self.H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h.toarray()/factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # print(u)
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1,0,3,2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2
