import numpy as np

from scipy.linalg import svd as csvd
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
import fishbonett.recurrence_coefficients as rc
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import scipy
from fishbonett.stuff import drude, temp_factor, _c


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

    def __init__(self, pd, domain, ncap, sd, h1e, he_dy):
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

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = sd
        self.domain = domain
        self.he_dy = np.array(he_dy)
        self.h1e = np.array(h1e)
        self.k_list = []
        self.w_lsit = []
        self.H = []
        self.coef = []
        self.freq = []
        self.phase = lambda lam, t, delta: (np.exp(-1j * lam * (t + delta)) - np.exp(-1j * lam * t)) / (-1j * lam)
        self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))

        def get_coupling( n, j, domain, g, ncap=20000):
            alphaL, betaL = rc.recurrenceCoefficients(
                n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
            )
            w_list = g * np.array(alphaL)
            k_list = g * np.sqrt(np.array(betaL))
            k_list[0] = k_list[0] / g
            _, _, self.h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
            return w_list, k_list

        def build_coupling(g, ncap):
            n = len(self.pd_boson)
            self.w_list, self.k_list = get_coupling(n, self.sd, self.domain, g, ncap)

        def build(g=1, ncap=20000):
            build_coupling(g, ncap)
            print("Coupling Over")
            self.freq, self.coef = self.diag()

        build()
        self.shuffle = self.coef.T
        # self.phase = lambda lam, t, delta: np.exp(-1j * lam * (t+delta/2)) * delta



    def get_theta1(self, i: int):
        return np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])

    def get_theta2(self, i: int):
        j = (i + 1)
        return np.tensordot(self.get_theta1(i), self.B[j], [2, 0])


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
        _, _, self.h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
        return w_list, k_list

    def build_coupling(self, g, ncap):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def diag(self):
        w= self.w_list[::-1]
        k = self.k_list[::-1]
        self.coup = np.diag(w) + np.diag(k[:-1], 1) + np.diag(k[:-1], -1)
        freq, coef = np.linalg.eigh(self.coup)
        sign = np.sign(coef[0,:])
        coef = coef.dot(np.diag(sign))
        return freq, coef

    def rotate_matrix_complex(self, dnt):
        dim = len(dnt)
        B = dnt.copy()
        rot_mat_lt = []
        U = np.eye(dim)
        # print("B=",B)
        for i in range(len(dnt)-1):
            pos_0 = i
            pos_1 = pos_0 + 2
            vec = B[i:i+2]
            vec = vec / np.linalg.norm(vec)
            if np.abs(vec[0]) >0.1:
                vec = vec / np.linalg.norm(vec)
                u = np.array([[vec[1], vec[0].conj()], [-vec[0], vec[1].conj()]])
            else:
                print("No rotation")
                u = np.array([[1, 0], [0, 1]])

            mat = np.eye(dim, dtype=np.complex128)
            mat[pos_0:pos_1, pos_0:pos_1] = u
            U = U@mat
            B = np.einsum('j,jk->k', B, mat)
            # print('B=', B)
            rot_mat_lt.append(u)
        return U, rot_mat_lt, B

    def get_h2(self, t, delta, inc_sys=True):
        print("Geting h2")
        freq = self.freq
        coef = self.coef
        e = self.phase
        k0 = self.k_list[0]
        j0 = k0 * coef[0,:] # interaction strength in the diagonal representation
        print("j0", j0)
        print("freq", freq)
        phase_factor = np.array([e(w, t, delta) for w in freq])
        print("Geting d's")

        d_nt = [einsum('k,k,k', j0, self.shuffle[:, n], phase_factor) for n in range(len(freq))]
        rot_mat, rot_mat_lt, d_nt = self.rotate_matrix_complex(d_nt)
        self.shuffle = self.shuffle@rot_mat
        d_nt = [einsum('k,k,k', j0, self.shuffle[:, n], phase_factor) for n in range(len(freq))]

        h2 = []
        he_dy = np.array(self.he_dy, dtype=np.complex128)
        for i in range(self.len_boson-1):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i+1]
            c1 = _c(d1)
            c2 = _c(d2)
            op = np.array([[np.kron(c1.T @ c1, np.eye(d2)), np.kron(c1, c2.T)],
                           [np.kron(c1.T, c2), np.kron(np.eye(d1), c2.T @ c2)]])
            log_u_dagger = scipy.linalg.logm(rot_mat_lt[i].conj())
            fock_U = np.einsum('ij,ijpq->pq',log_u_dagger, op)
            h2.append((fock_U, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        coup = np.array(np.kron(np.eye(d1), self.h1e), dtype=np.complex128) * delta
        k = d_nt[-1]
        print("k=", d_nt)
        c = np.array(_c(d1), dtype=np.complex128)
        print(coup)
        coup += np.kron(k*c+k.conj()*c.T, he_dy)
        h2.append((coup,d1,d2))
        return h2


    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        self.H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h/factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            u1 = u.reshape([r0, s0, r1, s1])
            U1[i] = u1
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2


if __name__ == '__main__':
    from fishbonett.stuff import drude, entang, sigma_z, sigma_x

    bath_length = 20
    phys_dim = 10
    threshold = 1e-3
    coup = 0.5
    bond_dim = 1000
    tmp = 2.0
    bath_freq = 1.0

    pd = [phys_dim] * bath_length + [2]

    g = 500 #+ bath_freq * 5000
    domain = [-g,g]
    temp = 226.00253972894595 * 0.5 * tmp
    j = lambda w: drude(w, lam=coup * 78.53981499999999 / 2, gam=bath_freq * 4 * 19.634953749999998) * temp_factor(temp, w)

    he_dy = sigma_z
    h1e = (78.53981499999999) * sigma_x
    etn = SpinBoson(pd, domain, 20000, j, h1e, he_dy)

    dt = 0.001 / int(np.ceil(bath_freq)) / 10 *2
    num_steps = 50

    p2 = []
    s_dim = np.empty([0, 0])
    s_ent = np.empty([0, 0])

    from time import time



    for tn in range(num_steps):
        U1, U2 = etn.get_u(tn*dt, dt)
        t0 = time()
        etn.U = U1
        for j in range(bath_length):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=0)

        theta = etn.get_theta1(bath_length)  # c.shape vL i vR
        rho2 = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pop2 = np.einsum('ij,ji', rho2, sigma_z)
        p2 = p2 + [pop2]

        dim = [len(s) for s in etn.S]
        ent = [entang(s) for s in etn.S]
        s_dim = np.append(s_dim, dim)
        s_ent = np.append(s_ent, ent)

    pop2 = [x.real for x in p2]
    print("getRDM", pop2)
