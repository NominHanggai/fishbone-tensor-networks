import numpy as np
import sys
from scipy.linalg import expm
from scipy.linalg import svd as csvd
from scipy.sparse.linalg import svds as sparsesvd
from copy import deepcopy as dcopy
from fbpca import pca as rsvd
from opt_einsum import contract as einsum


def svd(A, b, full_matrices=False):
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        # print("CSVD", A.shape, b)
        # cs = csvd(A, full_matrices=False)
        print("RRSVD", A.shape, b)
        rs = rsvd(A, b, True, n_iter=2, l=2 * b)
        #print("Difference", diffsnorm(A, *B))
        # print(cs[1] - rs[1])
        return rs
    else:
        return csvd(A, full_matrices=False)




class FishBoneNet:
    """ Simple Tree Like Tensor-Product States
                         ⋮
        --d33--d32--d31--E3--V3--b31--b32--b33--  0
                         |
        --d23--d22--d21--E2--V2--b21--b22--b23--  1
                         |
        --d13--d12--d11--E1--V1--b11--b12--b13--  2
                         |
        --d03--d02--d01--E0--V0--b01--b02--b03--  3
                         ⋮
                        -1
    Bond d03--d02 will be the bond 0 on the chain 0.
    Bond d12--d11 will be the bond 1 on the china 1.
    Bond E0-E1 will be the bond 0 on the chain -1.

    """

    def __init__(self, B, S):
        """
        :param B:
        :type B:
        :param S:
        :type S:
        """
        (eb, e, v, vb) = B
        (eb_s, e_s, v_s, vb_s) = S
        try:
            assert len(eb) == len(e) == len(v) == len(vb)
        except AssertionError:
            print("Lengths of the tensor lists don't match. Look back the "
                  "diagram in the comment of the class SimpleTTPS. The "
                  "tensors in the lists should compose a comb-like network.",
                  file=sys.stderr
                  )
            raise

        self._nc = len(e)
        self._eL = [len(ev_n) for ev_n in e]
        self._vL = [len(ev_n) for ev_n in v]
        self._ebL = [len(eb_n) for eb_n in eb]
        self._vbL = [len(vb_n) for vb_n in vb]
        self._L = [sum(x)
                   for x in zip(self._ebL, self._eL, self._vL, self._vbL)]

        self.ttnB = []
        # ttn means tree tensor network.
        # The self.ttn array stores the tensors in the whole network.
        for n in range(self._nc):
            self.ttnB.append(
                eb[n] + e[n] + v[n] + vb[n]
            )
        self._pD = []  # dimensions of physical legs
        for n in range(self._nc):
            self._pD.append(
                [self.ttnB[n][i].shape[1] for i in range(self._L[n])]
            )
        self.ttnS = []

        """ ttnS = [ [ S_11, S_12, ..., S_1M, S_backbone_1 ],
                     [ S_21, S_22, ..., S_2N, S_backbone_2 ],
                   ]
        """
        for n in range(self._nc):
            self.ttnS.append(eb_s[n] + e_s[n] + v_s[n] + vb_s[n])

        self.U = []
        """ ttnH = [ [ H_11, H_12, ..., H_1M, H_backbone_1 ],
                     [ H_21, H_22, ..., H_2N, H_backbone_2 ],
                     ...
                   ]
        """
        # for n in range(self._nc):
        #     if self._evL[n] == 2:
        #         H_eb, H_ev, H_vb = H[n]
        #         self.ttnH.append(H_eb + H_ev + H_vb)
        #     elif self._evL[n] == 1:
        #         H_eb, H_ev = H[n]
        #         self.ttnH.append(H_ev + H_ev)
        self.pre_factor = 1.2
    def get_theta1(self, n: int, i: int):
        """Calculate effective single-site wave function on sites i in B
        canonical form.

        :return:
        :rtype:
        :param n: Which chain
        :param i: Which site
        :return: S*B
        """
        assert 0 <= n <= self._nc - 1
        assert 0 <= i <= self._L[n] - 1
        return np.tensordot(
            np.diag(self.ttnS[n][i]),
            self.ttnB[n][i],
            [1, 0]
        )  # vL [vL'], [vL] i vU vD  vR -> vL i vU vD vR

    def get_theta2(self, n: int, i: int):
        """Calculate effective two-site wave function on sites i,j=(i+1) in 
        mixed canonical form.

        :param n:
        :type n: int
        :param i:
        :type i:
        :return:
        :rtype:
        """
        # n=-1 means the backbone chain. When n=-1, i is the bond number
        # bottom up
        max_index_main = self._nc - 2
        max_index_n = self._L[n] - 2
        number_of_chains = self._nc

        if n == -1 and 0 <= i <= max_index_main:
            theta_lower = self.get_theta1(i + 1, self._ebL[i + 1])
            s_inverse_middle = np.diag(self.ttnS[i + 1][-1] ** (-1))
            lower_gamma_down_canonical = np.tensordot(
                theta_lower,
                s_inverse_middle,
                [2, 0]
            )  # vL i [vU] vD vR, [vU] vU -> vL i vD vR vU
            # vL i vD vR vU -> vL i vU vD vR
            lower_gamma_down_canonical = np.transpose(
                lower_gamma_down_canonical, [0, 1, 4, 2, 3])
            higher_theta = self.get_theta1(i, self._ebL[i])
            # vL i vU [vD] vR , vL' j [vU'] vD' vR' ->
            # {vL i vU vR; VL' j vD' vR'}
            return np.tensordot(
                higher_theta,
                lower_gamma_down_canonical,
                [3, 2]
            )
        elif self._nc - 1 >= 0 and 0 <= i <= max_index_n:
            # vL i _vU_ _vD_ [vR],  [vL] j _vU_ _vD_ vR ->
            # {vL i _vU_ _vD_; j _vU_ _vD_ vR}
            return np.tensordot(
                self.get_theta1(n, i), self.ttnB[n][i + 1], axes=1
            )
        else:
            raise ValueError("Check the values of n and i. "
                             "They should be in the range of the network")

    def split_truncate_theta(self, theta, n, i, chi_max, eps):
        """Split the contracted two-site wave function and truncate the number
        of singular values.

        :param theta:
        :type theta:
        :param n: Which chain. n=-1 means the backbone.
        In this case (n=-1), i means which bond: 0 -> the first.
        :type n: int
        :param i: which bond on the chain
        :type i: int
        :param chi_max: int, Maximum number of singular values to keep
        :type chi_max: int
        :param eps:
        :type eps: float
        :return: No return. Just updates the tensors in self.ttnS and self.ttnB
        """
        # theta = self.get_theta2(n, i)
        max_index_n = self._L[n] - 2
        max_index_main = self._nc - 1
        e_index = self._ebL[n] - 1
        v_index = self._ebL[n]
        if n == -1 and 0 <= i <= max_index_main:
            # {Up part: vL i vU vR; Down part: VL' j vD' vR'}
            (chi_left_higher, phys_higher, chi_up_higher,
             chi_right_higher, chi_left_lower, phys_lower,
             chi_down_lower, chi_right_lower) = theta.shape
            theta = np.reshape(theta, [chi_left_higher * phys_higher *
                                       chi_up_higher * chi_right_higher,
                                       chi_left_lower * phys_lower *
                                       chi_down_lower * chi_right_lower])
            chi_try = int(self.pre_factor * len(self.ttnS[i + 1][-1])) + 10
            higher_gamma_up_canonical, S, lower_gamma_down_canonical = svd(
                theta, chi_try, full_matrices=False)
            chivC = min(chi_max, np.sum(S > eps), chi_try)
            while chivC == chi_try and chi_try < min(*theta.shape):
                print(f"Expanding chi_try by{self.pre_factor}")
                chi_try = int(round(self.pre_factor * chi_try))
                higher_gamma_up_canonical, S, lower_gamma_down_canonical = svd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps), chi_try)
            print("Error Is", np.sum(S > eps), chi_try, S[chivC:]@S[chivC:], chivC)
            # keep the largest `chivC` singular values
            piv = np.argsort(S)[::-1][:chivC]
            higher_gamma_up_canonical, S, lower_gamma_down_canonical = (
                higher_gamma_up_canonical[:, piv], S[piv],
                lower_gamma_down_canonical[piv, :])
            S = S / np.linalg.norm(S)
            self.ttnS[i + 1][-1] = S
            # gamma: vL*i*vU*vR*chivC -> vL i vU vR vU=chivC
            higher_gamma_up_canonical = np.reshape(higher_gamma_up_canonical, [
                chi_left_higher, phys_higher, chi_up_higher,
                chi_right_higher, chivC])
            higher_gamma_up_canonical = np.transpose(
                higher_gamma_up_canonical, [0, 1, 2, 4, 3])
            # vL i vU vR vD -> vL i vU vD vR
            higher_theta = np.tensordot(
                higher_gamma_up_canonical, np.diag(S),
                [3, 0]
            )  # vL i vU [vD] vR, [vD'] vD -> vL i vU vR vD
            higher_theta = np.transpose(higher_theta, [0, 1, 2, 4, 3])
            higher_gamma_right_canonical = np.tensordot(
                # S on the left of higher gamma.
                np.diag(self.ttnS[i][self._ebL[i]] ** -1),
                higher_theta,
                [1, 0]
            )  # vL [vL'], [vL'] i vU vD vR -> vL i vD vR vU
            self.ttnB[i][self._ebL[i]] = higher_gamma_right_canonical

            lower_gamma_down_canonical = np.reshape(
                lower_gamma_down_canonical,
                [chivC, chi_left_lower, phys_lower,
                 chi_down_lower, chi_right_lower]
            )  # gamma: chivC*vL*j*vD*vR -> vD==chivC vL i vD vR
            # vU vL i vD vR -> vL i vU vD vR
            lower_gamma_down_canonical = np.transpose(
                lower_gamma_down_canonical, [1, 2, 0, 3, 4])

            lower_theta = np.tensordot(
                lower_gamma_down_canonical, np.diag(S),
                [2, 0]
            )  # vL i [vU] vD vR, [vU'] vU -> vL i vD vR vU
            lower_theta = np.transpose(lower_theta, [0, 1, 4, 2, 3])
            # vL i vD vR vU -> vL i vU vD vR
            gamma_lower_right_canonical = np.tensordot(
                np.diag(self.ttnS[i + 1][self._ebL[i + 1]] ** -1), lower_theta,
                [1, 0]
            )  # vL [vL'], [vL'] i vU vD vR -> vL i vU vD vR
            self.ttnB[i + 1][self._ebL[i + 1]] = gamma_lower_right_canonical

        elif 0 <= n < self._nc and i >= 0:
            if i == self._ebL[n] - 1:
                (chi_left_on_left, phys_left, phys_right, chi_up_on_right,
                 chi_down_on_right, chi_right_on_right) = theta.shape
                theta = np.reshape(theta, [
                    chi_left_on_left * phys_left,
                    phys_right * chi_up_on_right *
                    chi_down_on_right * chi_right_on_right])
                chi_try = int(self.pre_factor * len(self.ttnS[n][i+1])) + 10
                A, S, B = svd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps), chi_try)
                while chivC == chi_try:
                    print(f"Expanding chi_try by{self.pre_factor}")
                    chi_try = int(round(self.pre_factor * chi_try))
                    A, S, B = svd(theta, chi_try, full_matrices=False)
                    chivC = min(chi_max, np.sum(S > eps), chi_try)
                print("Error Is", np.sum(S > eps), chi_try, S[chivC:]@S[chivC:], chivC)
                # keep the largest `chivC` singular values
                piv = np.argsort(S)[::-1][:chivC]
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
                # A: vL*i*vR -> vL i vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_up_on_right,
                                   chi_down_on_right, chi_right_on_right])
                # B: chivC*j*vU*vD*vR -> vL==chivC j vU vD vR
                A = np.tensordot(
                    np.diag(self.ttnS[n][i] ** (-1)), A, axes=[1, 0])
                A = np.tensordot(A, np.diag(S), axes=[2, 0])
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B

            elif i == self._ebL[n]:
                (chi_left_on_left, phys_left, chiU_l, chiD_l,
                 phys_right, chi_right_on_right) = theta.shape
                theta = np.reshape(
                    theta, [chi_left_on_left * phys_left * chiU_l * chiD_l,
                            phys_right * chi_right_on_right])
                chi_try = int(self.pre_factor * len(self.ttnS[n][i+1])) + 10
                A, S, B = svd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps), chi_try)
                while chivC == chi_try:
                    print(f"Expanding chi_try by{self.pre_factor}")
                    chi_try = int(round(self.pre_factor * chi_try))
                    A, S, B = svd(theta, chi_try, full_matrices=False)
                    chivC = min(chi_max, np.sum(S > eps), chi_try)
                print("Error Is", np.sum(S > eps), chi_try, S[chivC:]@S[chivC:], chivC)
                # keep the largest `chivC` singular values
                piv = np.argsort(S)[::-1][:chivC]
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                A = np.reshape(
                    A, [chi_left_on_left, phys_left, chiU_l, chiD_l, chivC])
                # A: {vL*i*vU*vD, chivC} -> vL i vU vD vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
                # B: {chivC, j*vR} -> vL==chivC j vR
                A = np.tensordot(
                    np.diag(self.ttnS[n][i] ** (-1)), A, axes=[1, 0])
                # vL [vL'] * [vL] i vU vD vR -> vL i vU vD vR
                A = np.tensordot(A, np.diag(S), [4, 0])
                # vL i vU vD [vR] * [vR] vR -> vL i vU vD vR
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B

            else:
                (chi_left_on_left, phys_left,
                 phys_right, chi_right_on_right) = theta.shape
                theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                           phys_right * chi_right_on_right])
                chi_try = int(self.pre_factor * len(self.ttnS[n][i+1])) + 10
                A, S, B = svd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps), chi_try)
                while chivC == chi_try:
                    print(f"Expanding chi_try by{self.pre_factor}")
                    chi_try = int(round(self.pre_factor * chi_try))
                    A, S, B = svd(theta, chi_try, full_matrices=False)
                    chivC = min(chi_max, np.sum(S > eps), chi_try)
                print("Error Is", np.sum(S > eps), chi_try, S[chivC:]@S[chivC:], chivC)
                # keep the largest `chivC` singular values
                piv = np.argsort(S)[::-1][:chivC]
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
                # A: {vL*i, chivC} -> vL i vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
                # B: {chivC, j*vR} -> vL==chivC j vR
                A = np.tensordot(np.diag(self.ttnS[n][i] ** (-1)), A, [1, 0])
                # vL [vL'] * [vL] i vR -> vL i vR
                A = np.tensordot(A, np.diag(S), [2, 0])
                # vL i [vR] * [vR] vR -> vL i vR
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B
        else:
            raise ValueError

    def update_bond(self, n, i, chi_max, eps):
        """
        :param n:
        :type n:
        :param i:
        :type i:
        :param chi_max:
        :type chi_max:
        :param eps:
        :type eps:
        """
        theta = self.get_theta2(n, i)
        # ("HERE", n, i)
        max_index_n = self._L[n] - 2
        max_index_main = self._nc - 2
        e_index = self._ebL[n] - 1
        v_index = self._ebL[n]
        if n == -1 and 0 <= i <= max_index_main:
            # {Down part: vL i VD vR; Up part: VL' j vU' vR'}
            Utheta = einsum('IJKL, aKcdeLgh->aIcdeJgh',
                            self.U[i][-1], theta)
            # {i j [i*] [j*]} * {vL [i] vD vR, vL' [j] vD vR}
            # {i j   k   l}     {a   b  c  d ,  e  f  g  h}
            self.split_truncate_theta(Utheta, n, i, chi_max, eps)
        elif 0 <= n < self._nc and 0 <= i <= max_index_n:
            if i == e_index:
                Utheta = einsum('IJKL, aKLfgh->aIJfgh', self.U[n][i], theta)
                # {i j [i*] [j*]} * {vL [i]  [j] vU vD vR}
                # {i j  k   l}      {a   b   e   f  g  h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            elif i == v_index:
                Utheta = einsum('IJKL, aKcdLh->aIcdJh', self.U[n][i], theta)
                # {i j [i*] [j*]} * {vL [i] vU vD,  [j]  vR}
                # {i j  k   l}      {a   b  c  d ,   e   h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            elif 0 <= i <= max_index_n:
                Utheta = einsum('IJKL,aKLh->aIJh', self.U[n][i], theta)
                # {i j [i*] [j*]} * {vL [i], [j] vR}
                # {I J  K   L}      {a   b,   e   h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)
            else:
                raise ValueError
                # TODO error that i is out of range. Should echo n and i
        else:
            raise ValueError
            # TODO error that n is out of range.
            # TODO Should echo n and number of chains


def init_ttn(nc, L, d1, de, dv):
    """
    Initialize the SimpleTTN class.
    Fill all the relevant lists, including ttnS, ttnB, ttnH.
    :param de:
    :type de:
    :param dv:
    :type dv:
    :param nc:
    :type nc:
    :param L:
    :type L:
    :param d1:
    :type d1:
    :param d2:
    :type d2:
    :return:
    :rtype:
    """
    eb = np.zeros([1, d1, 1], np.float)  # vL i vR
    eb[0, 0, 0] = 1.
    ebs = [eb.copy() for i in range(L)]
    ebss = [dcopy(ebs) for i in range(nc)]
    #
    e = np.zeros([1, de, 1, 1, 1], np.float)  # vL i vU vD vR
    e[0, 0, 0, 0, 0] = 1.
    v = np.zeros([1, dv, 1], np.float)  # vL i vR
    v[0, 0, 0] = 1.

    ess = [[e.copy()] for i in range(nc)]
    vss = [[v.copy()] for i in range(nc)]
    #
    vb = np.zeros([1, d1, 1], np.float)  # vL i vR
    vb[0, 0, 0] = 1.
    vbs = [vb.copy() for i in range(L)]
    vbss = [dcopy(vbs) for i in range(nc)]

    eb_s = np.ones([1], np.float)
    eb_ss = [eb_s.copy() for i in range(L)]
    eb_sss = [dcopy(eb_ss) for i in range(nc)]

    e_s = np.ones([1], np.float)
    v_s = np.ones([1], np.float)
    e_ss = [e_s.copy()]  # we have two sites here, e site and v site.
    v_ss = [v_s.copy()]
    e_sss = [dcopy(e_ss) for i in range(nc)]
    v_sss = [dcopy(v_ss) for i in range(nc)]

    vb_s = np.ones([1], np.float)
    # L+1 is because we will store the main-bone S in s[n][-1].
    vb_ss = [vb_s.copy() for i in range(L + 1)]
    vb_sss = [dcopy(vb_ss) for i in range(nc)]
    return FishBoneNet(
        (ebss, ess, vss, vbss),
        (eb_sss, e_sss, v_sss, vb_sss)
    )


def init(pd):
    def g_state(dim):
        tensor = np.zeros(dim)
        tensor[(0,) * len(dim)] = 1.
        return tensor

    nc = len(pd)
    length_chain = [len(sum(chain_n, [])) for chain_n in pd]
    eb_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 0][i]]
        for i in range(nc)]
    e_tensor = [
        [g_state([1, d, 1, 1, 1]) for d in pd[:, 1][i]]
        for i in range(nc)]
    v_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 2][i]]
        for i in range(nc)]
    vb_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 3][i]]
        for i in range(nc)]
    eb_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in eb_tensor
    ]
    e_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in e_tensor
    ]
    v_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in v_tensor
    ]
    vb_s = [
        list(np.ones([1], np.float) for b in chain_n)
        for chain_n in vb_tensor
    ]
    main_s = [[np.ones([1], np.float)] for chain_n in vb_tensor]
    vb_s_and_main_s = [vb_s[i] + vb_tensor[i] for i in range(nc)]
    # eb_tensor[0][0][0,0,0] = 1.
    e_tensor[0][0][0, 0, 0, 0, 0] = 1 / np.sqrt(2)
    e_tensor[0][0][0, 1, 0, 0, 0] = 1 / np.sqrt(2)
    return FishBoneNet(
        (eb_tensor, e_tensor, v_tensor, vb_tensor),
        (eb_s, e_s, v_s, vb_s_and_main_s)
    )

try:
    import cupy as cp
    CUPY_SUCCESS = True
    print("Success1")
    import rsvd_cupy.rsvd as cursvd

    print("Success2")
    def cusvd(A, b, full_matrices=False):
        dim = min(A.shape[0], A.shape[1])
        b = min(b, dim)
        # print("CSVD", A.shape, b)
        # cs = csvd(A, full_matrices=False)
        print("RRSVD", A.shape, b)
        rs = cursvd(A, b, True, n_iter=2, l=2 * b)
        # print("Difference", diffsnorm(A, *B))
        # print(cs[1] - rs[1])
        return rs


    print("Success3")
except ImportError:
    print("CuPy is not imported. Will use CPUs")
    CUPY_SUCCESS = False
else:
    print("CuPy is not imported. Will use CPUs")
    CUPY_SUCCESS = False

class SpinBoson1D:

    def __init__(self, pd):
        def g_state(dim: int):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.pd = pd
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1], np.float) for d in pd]
        self.U = [np.zeros(0) for d in pd[1:]]

    def get_theta1(self, i: int):
        return np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])

    def get_theta2(self, i: int):
        j = (i + 1)
        return np.tensordot(self.get_theta1(i), self.B[j], [2, 0])

    def split_truncate_theta(self, theta, i: int, chi_max: int, eps: float, gpu=False):
        if gpu is False or CUPY_SUCCESS is False:
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
        elif gpu is True and CUPY_SUCCESS is True:
            (chi_left_on_left, phys_left,
             phys_right, chi_right_on_right) = theta.shape
            theta = cp.reshape(theta, [chi_left_on_left * phys_left,
                                       phys_right * chi_right_on_right])
            A, S, B = cusvd(theta, chi_max, full_matrices=False)
            chivC = min(chi_max, cp.sum(S > eps))
            print("Error Is", cp.sum(S > eps), chi_max, S[chivC:] @ S[chivC:], chivC)
            # keep the largest `chivC` singular values
            piv = cp.argsort(S)[::-1][:chivC]
            A, S, B = A[:, piv], S[piv], B[piv, :]
            S = S / cp.linalg.norm(S)
            # A: {vL*i, chivC} -> vL i vR=chivC
            A = cp.reshape(A, [chi_left_on_left, phys_left, chivC])
            # B: {chivC, j*vR} -> vL==chivC j vR
            B = cp.reshape(B, [chivC, phys_right, chi_right_on_right])
            # vL [vL'] * [vL] i vR -> vL i vR
            A = cp.tensordot(cp.diag(self.S[i] ** (-1)), A, [1, 0])
            # vL i [vR] * [vR] vR -> vL i vR
            A = cp.tensordot(A, np.diag(S), [2, 0])
            self.S[i + 1] = S.get()
            self.B[i] = A.get()
            self.B[i + 1] = B.get()


    def update_bond(self, i: int, chi_max: int, eps: float, gpu=False):
        if not gpu or CUPY_SUCCESS is False:
            theta = self.get_theta2(i)
            d1 = self.pd[i]
            d2 = self.pd[i+1]
            U_bond = self.U[i].toarray()
            U_bond = U_bond.reshape([d1, d2, d1, d2])
            # i j [i*] [j*], vL [i] [j] vR
            Utheta = np.tensordot(U_bond, theta,
                                  axes=([2, 3], [1, 2]))
            Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
            self.split_truncate_theta(Utheta, i, chi_max, eps)
        else:
            theta = cp.array(self.get_theta2(i))
            d1 = self.pd[i]
            d2 = self.pd[i + 1]
            U_bond = cp.array(self.U[i].toarray())
            U_bond = U_bond.reshape([d1, d2, d1, d2])
            # i j [i*] [j*], vL [i] [j] vR
            Utheta = cp.tensordot(U_bond, theta,
                                  axes=([2, 3], [1, 2]))
            Utheta = cp.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
            self.split_truncate_theta(Utheta, i, chi_max, eps, gpu=True)


if __name__ == "__main__":
    ttn = init(nc=4, L=3, d1=5, de=2, dv=10)
    ttn.get_theta2(-1, 1)
