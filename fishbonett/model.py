import numpy as np
import sys
from scipy.linalg import expm
from numpy import exp
import recurrence_coefficients as rc
from copy import deepcopy as dcopy


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
    if d is [] or None:
        return None
    elif type(d) is int or d is str:
        return np.eye(int(d))
    elif type(d) is list:
        return np.eye(*d)


def kron(a, b):
    if a is None or b is None:
        return None
    if type(a) is list and type(b) is list:
        return np.kron(*a, *b)
    if type(a) is list and type(b) is not list:
        return np.kron(*a, b)
    if type(a) is not list and type(b) is list:
        return np.kron(a, *b)
    else:
        return np.kron(a, b)


def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    #print(H)
    return expm(-dt * 1j * H)


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


class FishBoneH:

    @property
    def H(self):
        if not self.w_list or not self.k_list:
            print("K and W dont exist. ", file=sys.stderr)
            raise IndexError
        else:
            return self._H

    @property
    def _sd(self):
        return [[_to_list(x) for x in y] for y in self.sd]

    # @sd.setter
    # def _sd(self, m):
    #     m = self.sd

    @property
    def h1e(self):
        return [_to_list(x) for x in self._h1e]

    @h1e.setter
    def h1e(self, m):
        # TODO check type
        self._h1e = m

    @property
    def h1v(self):
        return [_to_list(x) for x in self._h1v]

    @h1v.setter
    def h1v(self, m):
        # TODO check type
        self._h1v = m

    @property
    def h2ee(self):
        return [_to_list(x) for x in self._h2ee]

    @h2ee.setter
    def h2ee(self, m):
        # TODO check type
        self._h2ee = m

    @property
    def h2ev(self):
        return [_to_list(x) for x in self._h2ev]

    @h2ev.setter
    def h2ev(self, m):
        # check type
        self._h2ev = m

    @property
    def he_dy(self):
        return self._he_dy

    @he_dy.setter
    def he_dy(self, m):
        # TODO check type
        self._he_dy = m

    @property
    def hv_dy(self):
        return self._hv_dy

    @hv_dy.setter
    def hv_dy(self, m):
        # TODO check type
        self._hv_dy = m

    def __init__(self, pd: np.ndarray, ):
        """
        TODO
        :type pd: nd.ndarray
        :param pd: is a list.
         pD[0] contains physical dimensions of eb, ev, vb on the first chain,
         pD[1] contains physical dimensions of eb, ev, vb on the second chain,
         etc.
        """
        self._pd = pd
        self._nc = len(pd)  # an int
        # pD is a np.ndarray.
        self._ebL = [len(x) for x in self._pd[:, 0]]
        # pD[:,0] is the first column of the array, the eb column
        self._eL = [len(x) for x in self._pd[:, 1]]
        self._vL = [len(x) for x in self._pd[:, 2]]
        self._evL = [x + y for x, y in zip(self._eL, self._vL)]
        # pD[:,1] is the second column of the array, the ev column
        self._vbL = [len(x) for x in pd[:, 3]]
        # pD[:,2] is the third column of the array, the vb column
        self._L = [sum(x) for x in zip(self._ebL, self._evL, self._vbL)]
        self._ebD = self._pd[:, 0]
        self._eD = self._pd[:, 1]
        self._vD = self._pd[:, 2]
        self._vbD = self._pd[:, 3]
        # PLEASE NOTE THE SHAPE of pd and nd.array structure.
        # pd = nd.array([
        # [eb0, ev0, vb0], [eb1, ev1, vb1], [eb2, ev2, vb2]
        # ])
        # | eb0 ev0 vb0 |
        # | eb1 ev1 vb1 |
        # | eb2 ev2 vb2 | is the same as the structure depicted in SimpleTTS class.

        self.sd = np.empty([self._nc, 2], dtype=object)
        self.domain = [-1, 1]
        # TODO two lists. w is frequency, k is coupling.
        #  Get them from the function `get_coupling`

        self.w_list = [[[None] * self._ebL[n], [None] * self._vbL[n]] for n in range(self._nc)]
        self.k_list = [[[None] * self._ebL[n], [None] * self._vbL[n]] for n in range(self._nc)]

        # initialize spectral densities.
        for n in range(self._nc):
            if self._evL[n] == 2:
                if self._vbL[n] != 0:
                    self.sd[n, 0] = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
                    self.sd[n, 1] = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
                else:
                    self.sd[n, 0] = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
                    self.sd[n, 1] = None
            elif self._evL[n] == 1:
                self.sd[n, 0] = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
                self.sd[n, 1] = None
            else:
                raise SystemError  # TODO tell users what happens.
        # TODO Must have p-leg dims for e and v. Use [] if v not existent.

        # Assign the matrices below according to self.pd
        self._H = []  # list -> all bond Hamiltonians.
        # _H = [ [Heb00, Heb01, ..., Hev0, Hvb00, Hvb01, ..., Hvb0N, Hee0],
        #        [Heb10, Heb11, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, Hee1],
        #        [Heb00, Heb01, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, None]
        #      ] in the case of 3 chains.
        self._h1e = [eye(d) for d in self._eD]
        # list -> single Hamiltonian on e site. None as a placeholder if the p-leg is [].
        self._h1v = [eye(d) for d in self._vD]
        # list -> single Hamiltonian on v site. None as a placeholder if the p-leg is [].
        self._h2ee = [kron(eye(m), eye(n)) for (m, n) in zip(self._eD[:-1], self._eD[1:])]
        # list -> coupling Hamiltonian on e and e
        self._h2ev = [kron(eye(m), eye(n)) for (m, n) in
                      zip(self._eD, self._vD)]  # list -> coupling Hamiltonian on e and v
        self._he_dy = [eye(d) for d in self._eD]  # list -> e dynamic variables coupled to eb
        self._hv_dy = [eye(d) for d in self._vD]  # list -> v dynamic variables coupled to vb
        self.build()

    def get_coupling(self, n, j, domain, g, ncap=600):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        return w_list, k_list

    def build_coupling(self):
        L = [[a, b] for a, b in zip(self._ebL, self._vbL)]
        # print("L", L)
        for n, sdn in enumerate(self._sd):
            for i, sdn_il in enumerate(sdn):
                for a, sdn_i in enumerate(sdn_il):
                    # print("ni",n,i)
                    self.w_list[n][i], self.k_list[n][i] = \
                        self.get_coupling(L[n][i], sdn_i, self.domain, g=1., ncap=600)

    def get_h1(self, n, c=None) -> tuple:
        """

        :param c:
        :type c:
        :param n:
        :type n:
        :return:
        :rtype:
        """
        if 0 <= n < self._nc:
            """
            Generates h1eb
            """
            w_list = self.w_list[n][0]
            pd = self._pd[n, 0]  # -> Physical dimensions of sites ->
            # on eb of the nth chain.
            # n -> the nth chain, 0 -> the 1st element -> w_list for eb.
            # h1eb: EB Hamiltonian list

            h1eb = [None] * len(pd)
            for i, w in enumerate(w_list):
                c = _c(pd[-1 - i])
                print("w is", w, n, i)
                h1eb[-1 - i] = w * c.T @ c
                #print("n i and w", n, i, w)
            # If w_list = [], so as pd = [],then h1eb becomes []

            """
            Generates h1vb
            """
            w_list = self.w_list[n][1]
            pd = self._pd[n, 3]
            # print("pd and w_list are", w_list, pd)
            # n -> the nth chain, 0 -> the 3rd element -> w_list for vb.
            h1vb = [None] * len(pd)  # VB Hamiltonian list on the chain n
            for i, w in enumerate(w_list):
                c = _c(pd[i])
                #print("w=", w, n, i)
                h1vb[i] = w * c.T @ c
            # EV single Hamiltonian list on the chain n
            # print(self.h1e, self.h1v)
            h1ev = [self.h1e[n], self.h1v[n]]
            return h1eb, h1ev, h1vb
        else:
            raise ValueError

    def get_h2(self, n):
        if n == -1:
            e = self.h1e
            for i, d in enumerate(self._eD[1:]):
                e[i] = kron(e[i][0], eye(d))
            e[-1] = kron(eye(self._eD[-1]), e[-1][0])

            ee = self.h2ee
            # print("e=", e)
            # print("ee=", ee)

            h2ee = [(e[n][0] + ee[n][0], self._eD[i][0], self._eD[i + 1][0]) for i in range(self._nc - 1)]
            h2ee[-1] = (h2ee[-1][0] + e[-1], self._eD[-2][0], self._eD[-1][0])
            # print("Total h2ee is", h2ee[-2])

            return h2ee

        if 0 <= n <= self._nc - 1:
            h1eb, _, h1vb = self.get_h1(n)
            # print("h1eb", h1eb)
            pd = self._pd[n, 0]
            kL = self.k_list[n][0]
            # kL is a list of k's (coupling constants). Index 0 indicates eb
            # Start to generate ev Hamiltonian lists
            if kL is not [] and pd is not []:
                k0, kn = kL[-1], kL[1:]
                #print("k0", k0)
                w0 = self.w_list[n][0][0]
                # print("kn is", kn,type(kn))
                kn = kn[::-1]
                h2eb = []
                for i, k in enumerate(kn):

                    r0, r1 = pd[i], pd[i + 1]
                    print("r0,r1", r0, r1)
                    c1 = _c(r0);
                    c2 = _c(r1)
                    h1 = h1eb[i]
                    print("k=", k, n, i)
                    h2 = kron(h1, eye(r1)) + k * (kron(c1.T, c2) + kron(c1, c2.T))
                    h2eb.append((h2, r0, r1))
                # The following requires that we must have a e site.
                c0 = _c(pd[-1])
                pdE = self._pd[n, 1][0]
                # TODO: add an condition to determine if the dimensions match.
                #print("Chain n h1e", self.h1e[n])
                h2eb0 = np.kron(h1eb[-1], np.eye(pdE)) + k0 * np.kron((c0 + c0.T), self.he_dy[n])
                h2eb.append((h2eb0, pd[-1], pdE))
            else:
                h2eb = []

            pd = self._pd[n, 3]  # 3 indicates the vb list
            kL = self.k_list[n][1];
            wL = self.w_list[n][1]
            # kL is a list of k's (coupling constants) 0 indicates eb
            if kL is not [] and pd is not []:
                k0, kn = kL[0], kL[1:];
                w0 = wL[0]
                c0 = _c(pd[0])
                pdV = self._pd[n, 2][0]
                dvb1 = h1vb[0].shape[0]
                #print("vdyn shape",k0, self.hv_dy[n].shape, c0.shape, self.h1v[n][0].shape)
                #print("vdyn shape", self.k_list)
                #print("_vD",self._vD,pd)
                h2vb0 = k0 * np.kron(self.hv_dy[n], c0 + c0.T) + \
                        kron(self.h1v[n], eye(dvb1))

                h2vb = [(h2vb0, pdV, pd[0])]
                for i, k in enumerate(kn):
                    r0, r1 = pd[i], pd[i + 1]
                    c0 = _c(r0);
                    c1 = _c(r1)
                    h1 = h1vb[i]
                    h2 = kron(h1, eye(r1)) + k * (np.kron(c0.T, c1) + np.kron(c0, c1.T))
                    # h2.shape is (m*n, m*n)
                    h2vb.append((h2, r0, r1))

            else:
                h2vb = []

            # TODO. Calculate h2ev
            h2ev = []
            for h in self.h2ev[n]:
                r0 = self._eD[n][0]
                r1 = self._vD[n][0]
                h = h + kron(self.h1e[n], eye(self._vD[n]))
                h2ev.append((h, r0, r1))

            return h2eb + h2ev + h2vb

    def build(self):
        self.build_coupling()
        # TODO Gotta check the existences of Hee,
        #  Hev, H_dy's, sd and stuff.
        H = []
        hee = self.get_h2(-1)
        for n in range(self._nc):
            # print("n======", n)
            h = self.get_h2(n)
            # print("get_h2 is", h)
            H.append(h)
        # print("H is", H)
        for n in range(self._nc - 1):
            # print("Hee_n is", hee[n])
            H[n].append(hee[n])
        self._H = H

    def get_u(self, dt):
        # TODO, change the definition of h2e, to strictly follow the even-odd pattern.
        #  Done but need double check ↑.
        # print("H0 is", self.H[0])
        U = dcopy(self.H)
        for i, r in enumerate(self.H):
            for j, s in enumerate(r):
                h = self.H[i][j][0]
                # print("ij=====", i, j)
                # print("h======", self.H[i][j], self.H[i][j][0].shape)
                u = calc_U(h, dt)
                r0 = r1 = self.H[i][j][1]  # physical dimension for site A
                s0 = s1 = self.H[i][j][2]  # physical dimension for site B
                u = u.reshape([r0, s0, r1, s1])
                U[i][j] = u
        return U


if __name__ == "__main__":
    a = [3, 3, 3]
    b = [2]
    pd = np.array([[a, b, b, a], [a, b, b, a]], dtype=object)
    tri = FishBoneH(pd)
    # tri.H
