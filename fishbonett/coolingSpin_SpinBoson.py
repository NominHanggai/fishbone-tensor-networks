import numpy as np

from scipy.linalg import svd as csvd
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import sympy
import scipy
import fishbonett.recurrence_coefficients as rc
from fishbonett.stuff import temp_factor, sigma_z


def _c(dim: int):
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def kron(a, b):
    return skron(a, b, format='csc')


def svd(A, b, full_matrices=False):
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        print("RRSVD", A.shape, b)
        return rsvd(A, b, True, n_iter=2, l=2 * b)
    else:
        return csvd(A, full_matrices=False)


def calc_U(H, dt):
    return scipy.linalg.expm(-dt * 1j * H)


class SpinBoson:
    def __init__(self, pd, betaOmega=2.):
        def g_state(dim):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1]) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        self.H = [np.zeros(0) for _ in pd[1:]]
        self.betaOmega = betaOmega

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * np.exp(-x / 1)
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



    def get_theta1(self, i: int):
        return np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])

    def get_theta2(self, i: int):
        j = (i + 1)
        return np.tensordot(self.get_theta1(i), self.B[j], [2, 0])

    def get_rdm(self):
        theta = self.get_theta1(self.len_boson)
        # theta = einsum('ij,PjQ->PiQ', self.heating_op, theta)
        # rho = einsum('PiQ,PjQ->ij', theta, theta.conj())
        rho = einsum('PiQ, ij, kl, PlQ->jk', theta, self.heating_op, self.heating_op, theta.conj())
        return rho/np.trace(rho)

    def split_truncate_theta(self, theta, i: int, chi_max: int, eps: float):
        (chi_left_on_left, phys_left,
         phys_right, chi_right_on_right) = theta.shape
        theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                   phys_right * chi_right_on_right])
        A, S, B = svd(theta, chi_max, full_matrices=False)
        chivC = min(chi_max, np.sum(S > eps))
        print("Error Is", np.sum(S > eps), chi_max, S[chivC:] @ S[chivC:], chivC)
        # keep the largest `chivC` singular values
        piv = np.argsort(S)[::-1][:chivC]
        A, S, B = A[:, piv], S[piv], B[piv, :]
        S = S / np.linalg.norm(S)
        # A: {vL*i, chivC} -> vL i vR=chivC
        A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
        # B: {chivC, j*vR} -> vL==chivC j vR
        B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
        # vL [vL'] * [vL] i vR -> vL i vR
        A = np.tensordot(np.diag(self.S[i] ** (-1)), A, [1, 0])
        # vL i [vR] * [vR] vR -> vL i vR
        A = np.tensordot(A, np.diag(S), [2, 0])
        self.S[i + 1] = S
        self.B[i] = A
        self.B[i + 1] = B

    def update_bond(self, i: int, chi_max: int, eps: float, swap):
        theta = self.get_theta2(i)
        U_bond = self.U[i]
        # i j [i*] [j*], vL [i] [j] vR
        print(theta.shape, U_bond.shape)
        if swap == 1:
            print("swap: on")
            Utheta = einsum('ijkl,PklQ->PjiQ', U_bond, theta)

        elif swap == 0:
            print("swap: off")
            Utheta = einsum('ijkl,PklQ->PijQ', U_bond, theta)
        else:
            print(swap)
            raise ValueError
        self.split_truncate_theta(Utheta, i, chi_max, eps)

    def get_coupling(self, n, j, domain, g, ncap=20000):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def diag(self):
        w= self.w_list
        k = self.k_list
        coup = np.diag(w) + np.diag(k[1:], 1) + np.diag(k[1:], -1)
        freq, coef = np.linalg.eigh(coup)
        sign = np.sign(coef[0, :])
        coef = coef.dot(np.diag(sign))
        return freq, coef

    def build(self, g, ncap=20000):
        self.build_coupling(g, ncap)
        print("Coupling Over")
        self.freq, self.coef = self.diag()
        op = scipy.linalg.expm(-1* sigma_z * self.betaOmega)
        self.heating_op = op / np.linalg.norm(op)

    def get_h2(self, delta):
        print("Geting h2")
        freq = self.freq
        coef = self.coef
        e = self.phase
        k0 = self.k_list[0]
        j0 = k0 * coef[0,:] # interaction strength in the diagonal representation
        print("Geting d's")
        # Permutation
        indexes = np.abs(freq).argsort()
        freq = freq[indexes]
        j0 = j0[indexes]
        # END Permutation
        d_nt = j0
        d_nt = d_nt[::-1]
        freq = freq[::-1]
        h2 = []
        for i, k in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            kc = k.conjugate()
            print(f'k {k}; kc {kc}')
            f = freq[i]
            coup = np.kron(k  * c1 + kc * c1.T, self.he_dy)
            site = np.kron(f*c1.T@c1, np.eye(d2))
            h2.append((delta*(coup+site), d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin

        h1e = scipy.linalg.expm(self.betaOmega * sigma_z) @ self.h1e @ scipy.linalg.expm(-self.betaOmega*sigma_z)
        site = delta*np.kron(np.eye(d1), h1e)
        h2[-1] = (h2[-1][0] + site, d1, d2)
        return h2

    def get_u(self, dt):
        self.H = self.get_h2(dt)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # print(u)
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1,0,3,2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2

if __name__ == '__main__':
    from fishbonett.stuff import drude, entang, sigma_z, sigma_x
    bath_length = 200
    phys_dim = 20
    threshold = 1e-4
    coup = 4.0
    bond_dim = 1000
    tmp = 2.0
    bath_freq = 1.0

    pd = [phys_dim] * bath_length + [2]
    bo = 0
    etn = SpinBoson(pd=pd, betaOmega=bo)
    g = 500 + bath_freq * 5000
    etn.domain = [-g, g]
    temp = 226.00253972894595 * 0.5 * tmp

    j = lambda w: drude(w, lam=coup * 78.53981499999999 / 2, gam=bath_freq * 4 * 19.634953749999998) * temp_factor(temp,w)
    etn.sd = j
    etn.he_dy = sigma_z
    etn.h1e = (78.53981499999999) * sigma_x

    etn.build(g=1, ncap=20000)

    dt = 0.001 / int(np.ceil(bath_freq)) / 10
    num_steps = 100 * int(np.ceil(bath_freq)) *1

    p1 = []
    p2 = []
    s_dim = np.empty([0, 0])
    s_ent = np.empty([0, 0])

    from time import time

    U1, U2 = etn.get_u(dt)

    for tn in range(num_steps):
        t0 = time()
        etn.U = U1
        for j in range(bath_length - 1, 0, -1):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)

        etn.update_bond(0, bond_dim, threshold, swap=0)
        etn.update_bond(0, bond_dim, threshold, swap=0)

        etn.U = U2
        for j in range(1, bath_length):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)

        theta = etn.get_theta1(bath_length)  # c.shape vL i vR
        rho1 = etn.get_rdm()
        rho2 = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pop1 = np.einsum('ij,ji', rho1, sigma_z)
        pop2 = np.einsum('ij,ji', rho2, sigma_z)
        p1 = p1 + [pop1]
        p2 = p2 + [pop2]


        dim = [len(s) for s in etn.S]
        ent = [entang(s) for s in etn.S]
        s_dim = np.append(s_dim, dim)
        s_ent = np.append(s_ent, ent)

    pop1 = [x.real for x in p1]
    pop2 = [x.real for x in p2]
    print("getRDM", pop1)
    print("noGetRDM", pop2)
    # p.astype('float32').tofile(f'./output/pop_cooling_BO{bo}.dat')
    # s_dim.astype('float32').tofile(f'./output/sDim_cooling_BO{bo}.dat')
    # s_ent.astype('float32').tofile(f'./output/entropy_cooling_BO{bo}.dat')


