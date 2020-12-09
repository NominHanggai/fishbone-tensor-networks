import numpy as np
import sys
from numpy import exp


def _c(dim: int):
    """
    Creates the annihilation operator.
    From the package py-tedopa/tedopa.
    https://github.com/MoritzLange/py-tedopa

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


def eye(*args):
    if not args:
        return None
    else:
        return np.eye(*args)


class FishBone:

    @property
    def H(self):
        if not self.w_list or not self.k_list:
            print("K and W dont exist. ", file=sys.stderr)
            raise IndexError
        else:
            self._H = self.build()
            return self._H

    @property
    def sd(self):
        return self._sd

    #
    @property
    def h1e(self):
        return self._h1e

    @h1e.setter
    def h1e(self, m):
        # TODO check type
        self._h1e = m

    @property
    def h1v(self):
        self._h1v = []
        return self._h1v

    @h1v.setter
    def h1v(self, m):
        # TODO check type
        self._h1v = m

    @property
    def h2ee(self):
        return self._h2ee

    @h2ee.setter
    def h2ee(self, m):
        # TODO check type
        self._h2ee = m

    @property
    def h2ev(self):
        return self._h2ev

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

    def __init__(self, pd: np.ndarray):
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
        self._evL = self._eL + self._vL
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

        self._sd = np.empty([2, self._nc], dtype=object)

        # TODO two lists. w is frequency, k is coupling.
        #  Get them from the function `get_coupling`
        self.w_list = []
        self.k_list = []

        # initialize spectral densities.
        for n in range(self._nc):
            if self._evL[n] == 2:
                self.sd[n, 0] = self.sd[n, 1] = lambda x: 1. / 1. * exp(-x / 1)
            elif self._evL[n] == 1:
                self.sd[n, 0] = lambda x: 1. / 1. * exp(-x / 1)
                self.sd[n, 1] = None
            else:
                raise SystemError  # TODO tell users what happens.
        # TODO Must have p-leg dims for e and v. Use 0 if v not existent.

        # Assign the matrices below according to self.pd
        self._H = []  # list -> all bond Hamiltonians.
        # _H = [ [Heb00, Heb01, ..., Hev0, Hvb00, Hvb01, ..., Hvb0N, Hee0],
        #        [Heb10, Heb11, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, Hee1],
        #        [Heb00, Heb01, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, None]
        #      ] in the case of 3 chains.
        self._h1e = [[eye(*d) for d in self._eD]]
        # list -> single Hamiltonian on e site. None as a placeholder if the p-leg is [].
        self._h1v = [[eye(*d) for d in self._vD]]
        # list -> single Hamiltonian on v site. None as a placeholder if the p-leg is [].
        self._h2ee = []  # list -> coupling Hamiltonian on e and e
        self._h2ev = []  # list -> coupling Hamiltonian on e and v
        self._he_dy = []  # list -> e dynamic variables coupled to eb
        self._hv_dy = []  # list -> v dynamic variables coupled to vb

    def get_coupling(self):
        # TODO Get w and k for each spectral density
        # TODO w and k have the same structures as  self.sd (spectral densities)
        self.w_list = []
        self.k_list = []
        return self.w_list, self.k_list

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
                h1eb[-1 - i] = w * c @ c.T
            # If w_list = [], so as pd = [],then h1eb becomes []

            """
            Generates h1vb
            """
            w_list = self.w_list[n][1]
            pd = self._pd[n, 2]
            # n -> the nth chain, 0 -> the 3rd element -> w_list for vb.
            h1vb = [None] * len(pd)  # VB Hamiltonian list on the chain n
            for i, w in enumerate(w_list):
                c = _c(pd[i])
                h1vb[i] = w * c @ c.T
            # EV single Hamiltonian list on the chain n
            h1ev = self.h1e[n], self.h1v[n]
            return h1eb, h1ev, h1vb
        else:
            raise ValueError

    def get_h2(self, n):
        if n == -1:
            e = self.h1e[:-1]
            ee = self.h2ee
            h2ee = [e[n] + ee[n] for i in range(self._nc - 1)]
            h2ee[-1] = h2ee[-1] + e[-1]
            return h2ee

        if 0 <= n <= self._nc < 1:
            h1eb, h1ev, h1vb = self.get_h1(n)

            pd = self._pd[n, 0]
            kL = self.k_list[n][0]
            # kL is a list of k's (coupling constants). Index 0 indicates eb
            if kL is not [] and pd is not []:
                k0, kn = kL[0], kL[1:]
                w0 = self.w_list[n][0][0]
                kn.reverse()
                h2eb = []
                for i, k in enumerate(kn):
                    m, n = pd[i], pd[i + 1]
                    cm = _c(m);
                    cn = _c(n)
                    h1 = h1eb[i]
                    h2 = h1 + k * (np.kron(cm, cn.T)) + np.kron(cm.T, cn)
                    h2eb.append(h2)
                h2eb.append(w0 * np.eye(pd[-1]))
            else:
                h2eb = []

            pd = self._pd[n, 2]  # 2 indicates the vb list
            kL = self.k_list[n][1];
            wL = self.w_list[n][1]
            # kL is a list of k's (coupling constants) 0 indicates eb
            if kL is not [] and pd is not []:
                k0, kn = kL[0], kL[1:];
                w0 = wL[0]
                h2vb = [w0 * _c(1) @ _c(pd[0])]
                for i, k in enumerate(kn):
                    m, n = pd[i], pd[i + 1]
                    cm = _c(m);
                    cn = _c(n)
                    h1 = h1vb[i]
                    h2 = h1 + k * (np.kron(cm, cn.T)) + np.kron(cm.T, cn)
                    # h2.shape is (m*n, m*n)
                    h2vb.append(h2)
            else:
                h2vb = []

            # TODO. Calculate h2ev
            h2ev = [self.h2ev[n]]

            return h2eb + h2ev + h2vb

    def build(self):
        # TODO Gotta check the existences of Hee,
        #  Hev, H_dy's, sd and stuff.
        H = []
        hee = self.get_h2(-1)
        for n in range(self._nc):
            h = self.get_h2(n)
            H.append(h)
        for n in range(self._nc - 1):
            H[n].append(hee[n])
        return H


if __name__ == "__main__":
    a = [3, 3, 3]
    b = [2]
    pd = np.array([[a, b, b, a], [a, b, b, a]], dtype=object)
    tri = FinshBone(pd)
    tri.H
