import numpy as np
import sys
from scipy.linalg import svd, expm
from copy import deepcopy as dcopy
import itertools


class FishBoneNet:
    """ Simple Tree Like Tensor-Product States
                         ⋮
        --d33--d32--d31--E3--V3--b31--b32--b33--  3
                         |
        --d23--d22--d21--E2--V2--b21--b22--b23--  2
                         |
        --d13--d12--d11--E1--V1--b11--b12--b13--  1
                         |
        --d03--d02--d01--E0--V0--b01--b02--b03--  0
                         ⋮
                        -1
    Let's look at what's already in the diagram and ignore the ellipsis.
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
        # self.e = ele_list
        # self.v = vib_list
        # self.eb = e_bath_list
        # self.vb = v_bath_list
        (eb, e, v, vb) = B
        (eb_s, e_s, v_s, vb_s) = S
        # print("eb", eb, "len", len(eb))
        # print("ev", ev, "len", len(ev))
        # print("vb", vb, "len", len(vb))
        try:
            assert len(eb) == len(e) ==len(v) == len(vb)
        except AssertionError:
            print("Lengths of the tensor lists don't match. "
                  "Look back the diagram in the comment of the class SimpleTTPS. "
                  "The tensors in the lists should compose a comb-like network.",
                  file=sys.stderr
                  )
            raise

        self._nc = len(e)
        self._eL = [len(ev_n) for ev_n in e]
        self._vL = [len(ev_n) for ev_n in v]
        self._ebL = [len(eb_n) for eb_n in eb]
        self._vbL = [len(vb_n) for vb_n in vb]
        self._L = [sum(x) for x in zip(self._ebL, self._eL, self._vL, self._vbL)]

        self.ttnB = []
        # ttn means tree tensor network. The self.ttn array stores the tensors in the whole network.
        for n in range(self._nc):
            self.ttnB.append(
                eb[n] + e[n] + v[n] + vb[n]
            )
        self._pD = []  # dimensions of physical legs
        for n in range(self._nc):
            print(n, self.ttnB[n])
            self._pD.append(
                [self.ttnB[n][i].shape[1] for i in range(self._L[n])]
            )
        self.ttnS = []

        """ ttnS = [ [ S_11, S_12, ..., S_1M, S_backbone_1 ],
                     [ S_21, S_22, ..., S_2N, S_backbone_2 ],
                     ...
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

    def get_theta1(self, n: int, i: int):
        """
        Calculate effective single-site wave function on sites i in B canonical form.
        :return:
        :rtype:
        :param n: Which chain
        :param i: Which site
        :return: S*B
        """
        assert 0 <= n <= self._nc - 1
        assert 0 <= i < self._L[n] - 1
        return np.tensordot(
            np.diag(self.ttnS[n][i]),
            self.ttnB[n][i],
            [1, 0]
        )  # vL [vL'], [vL] i vU vD  vR -> vL i vU vD vR

    def get_theta2(self, n: int, i: int):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.
        :param n:
        :type n: int
        :param i:
        :type i:
        :return:
        :rtype:
        """
        # n=-1 means the backbone chain. When n=-1, i is the bond number bottom up
        max_index_main = self._nc - 2
        max_index_n = self._L[n] - 2
        number_of_chains = self._nc

        if n == -1 and 0 <= i <= max_index_main:
            theta_lower = self.get_theta1(i + 1, self._ebL[i + 1])
            s_inverse_middle = np.diag(self.ttnS[i][-1] ** (-1))
            print("SHAPE", theta_lower.shape, s_inverse_middle.shape)
            lower_gamma_down_canonical = np.tensordot(
                theta_lower,
                s_inverse_middle,
                [2, 0]
            )  # vL i [vU] vD vR, [vU] vU -> vL i vD vR vU
            lower_gamma_down_canonical = np.transpose(lower_gamma_down_canonical,
                                                      [0, 1, 4, 2, 3])  # vL i vD vR vU -> vL i vU vD vR
            higher_theta = self.get_theta1(i, self._ebL[i])
            return np.tensordot(
                higher_theta,
                lower_gamma_down_canonical,
                [3, 2]
            )  # vL i vU [vD] vR , vL' j [vU'] vD' vR' -> {vL i vU vR; VL' j vD' vR'}
        elif self._nc - 1 >= 0 and 0 <= i <= max_index_n:
            return np.tensordot(
                self.get_theta1(n, i), self.ttnB[n][i + 1], axes=1
            )  # vL i _vU_ _vD_ [vR],  [vL] j _vU_ _vD_ vR -> {vL i _vU_ _vD_; j _vU_ _vD_ vR}
        else:
            raise ValueError("Check the values of n and i. They should be in the range of the network")

    def split_truncate_theta(self, theta, n, i, chi_max, eps):
        """
        Split the contracted two-site wave function and truncate the number of singular values.
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
        if n == -1 and 0<=i<= max_index_main:
            # {Down part: vL i VD vR; Up part: VL' j vU' vR'}
            chi_left_higher, phys_higher, chi_up_higher, chi_right_higher, \
            chi_left_on_left, phys_lower, chi_down_lower, chi_right_lower = theta.shape
            theta = np.reshape(theta, [chi_left_higher * phys_higher * chi_up_higher * chi_right_higher,
                                       chi_left_on_left * phys_lower * chi_down_lower * chi_right_lower])
            higher_gamma_up_canonical, S, lower_gamma_down_canonical = svd(theta, full_matrices=False)
            chivC = min(chi_max, np.sum(S > eps))
            piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
            higher_gamma_up_canonical, S, lower_gamma_down_canonical = higher_gamma_up_canonical[:, piv], S[
                piv], lower_gamma_down_canonical[piv, :]
            S = S / np.linalg.norm(S)
            self.ttnS[i][-1] = S
            higher_gamma_up_canonical = np.reshape(
                higher_gamma_up_canonical,
                [chi_left_higher, phys_higher, chi_up_higher, chi_right_higher, chivC]
            )  # gamma: vL*i*vU*vR*chivC -> vL i vU vR vU=chivC
            higher_gamma_up_canonical = np.transpose(higher_gamma_up_canonical, [0, 1, 2, 4, 3])
            # vL i vU vR vD -> vL i vU vD vR
            # print("DSHAPE", D.shape)
            higher_theta = np.tensordot(
                higher_gamma_up_canonical, np.diag(S),
                [3, 0]
            )  # vL i vU [vD] vR, [vD'] vD -> vL i vU vR vD
            higher_theta = np.transpose(higher_theta, [0, 1, 2, 4, 3])
            higher_gamma_right_canonical = np.tensordot(
                np.diag(self.ttnS[i][self._ebL[i]] ** -1),  # S on the left of higher gamma.
                higher_theta,
                [1, 0]
            )  # vL [vL'], [vL'] i vU vD vR -> vL i vD vR vU
            self.ttnB[i][self._ebL[i + 1]] = higher_gamma_right_canonical

            lower_gamma_down_canonical = np.reshape(
                lower_gamma_down_canonical,
                [chivC, chi_left_on_left, phys_lower, chi_down_lower, chi_right_lower]
            )  # gamma: chivC*vL*j*vD*vR -> vD==chivC vL i vD vR
            lower_gamma_down_canonical = np.transpose(lower_gamma_down_canonical,
                                                      [1, 2, 0, 3, 4])  # vU vL i vD vR -> vL i vU vD vR

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
            self.ttnB[i + 1][self._ebL[i]] = gamma_lower_right_canonical

        elif 0 <= n < self._nc and i >= 0:
            if i == self._ebL[n] - 1:
                # print("ni", n, i, theta.shape)
                chi_left_on_left, phys_left, \
                phys_right, chi_up_on_right, chi_down_on_right, chi_right_on_right = theta.shape
                theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                           phys_right * chi_up_on_right * chi_down_on_right * chi_right_on_right])
                A, S, B = svd(theta, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                print("Shape of middle S", np.diag(self.ttnS[n][i + 1]).shape)
                A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
                # A: vL*i*vR -> vL i vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_up_on_right, chi_down_on_right, chi_right_on_right])
                # B: chivC*j*vU*vD*vR -> vL==chivC j vU vD vR
                print("Shape of left S and A", np.diag(self.ttnS[n][i] ** (-1)).shape, A.shape)
                A = np.tensordot(np.diag(self.ttnS[n][i] ** (-1)), A, axes=[1, 0])
                A = np.tensordot(A, np.diag(S), axes=[2, 0])
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B

            elif i == self._ebL[n]:
                chi_left_on_left, phys_left, chiU_l, chiD_l, phys_right, chi_right_on_right = theta.shape
                theta = np.reshape(theta, [chi_left_on_left * phys_left * chiU_l * chiD_l,
                                           phys_right * chi_right_on_right])
                A, S, B = svd(theta, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                A, S, B = A[:, piv], S[piv], B[piv, :]
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                # print("Shape of middle S", np.diag(self.ttnS[n][i + 1]).shape)
                A = np.reshape(A, [chi_left_on_left, phys_left, chiU_l, chiD_l, chivC])
                # A: {vL*i*vU*vD, chivC} -> vL i vU vD vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
                # B: {chivC, j*vR} -> vL==chivC j vR
                # print("Shape of left S and A", np.diag(self.ttnS[n][i] ** (-1)).shape, A.shape)
                A = np.tensordot(np.diag(self.ttnS[n][i] ** (-1)), A, axes=[1, 0])
                # vL [vL'] * [vL] i vU vD vR -> vL i vU vD vR
                A = np.tensordot(A, np.diag(S), [4, 0])
                # vL i vU vD [vR] * [vR] vR -> vL i vU vD vR
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B

            else:
                chi_left_on_left, phys_left, phys_right, chi_right_on_right = theta.shape
                # print("theta shape", theta, theta.shape)
                theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                           phys_right * chi_right_on_right])
                A, S, B = svd(theta, full_matrices=False)
                # print("ABS", A.shape, B.shape, S)
                chivC = min(chi_max, np.sum(S > eps))
                piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
                # print("piv", chivC, piv, np.sum(S > eps))
                A, S, B = A[:, piv], S[piv], B[piv, :]
                # print("AB", A.shape, B.shape, S.shape)
                S = S / np.linalg.norm(S)
                self.ttnS[n][i + 1] = S
                A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
                # print("Areshape", A, S)
                # A: {vL*i, chivC} -> vL i vR=chivC
                B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
                # B: {chivC, j*vR} -> vL==chivC j vR
                A = np.tensordot(np.diag(self.ttnS[n][i] ** (-1)), A, [1, 0])
                # print("Adot", A)
                # vL [vL'] * [vL] i vR -> vL i vR
                A = np.tensordot(A, np.diag(S), [2, 0])
                # vL i [vR] * [vR] vR -> vL i vR
                # print("ABfinal", "A",A, "B", B)
                self.ttnB[n][i] = A
                self.ttnB[n][i + 1] = B
                # print("ttnB n= ",n , self.ttnB[n])
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
            Utheta = np.einsum('IJKL, aKcdeLgh->aIcdeJgh', self.U[i][-1], theta)
            print("Utheta.shape", Utheta.shape, self.U[i][-1].shape, theta.shape)
            # {i j [i*] [j*]} * {vL [i] vD vR, vL' [j] vD vR}
            # {i j   k   l}     {a   b  c  d ,  e  f  g  h}
            self.split_truncate_theta(Utheta, n, i, chi_max, eps)
        elif 0 <= n < self._nc and 0 <= i <= max_index_n:
            if i == e_index:
                print("Theta Shape", theta.shape, "n and i", n, i, self.U[n][i].shape, n,i,self._ebL[n] )
                Utheta = np.einsum('IJKL, aKLfgh->aIJfgh', self.U[n][i], theta)
                # {i j [i*] [j*]} * {vL [i]  [j] vU vD vR}
                # {i j  k   l}      {a   b   e   f  g  h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            elif i == v_index:
                print("index", n, i)
                print(self.U[n][i].shape, theta.shape)
                Utheta = np.einsum('IJKL, aKcdLh->aIcdJh', self.U[n][i], theta)
                # {i j [i*] [j*]} * {vL [i] vU vD,  [j]  vR}
                # {i j  k   l}      {a   b  c  d ,   e   h}
                self.split_truncate_theta(Utheta, n, i, chi_max, eps)

            elif 0 <= i <= max_index_n:
                print("HEREE", theta.shape, n, i)
                # print("Hshape", self.ttnH[n][i].reshape(4,4), n, i)
                Utheta = np.einsum('IJKL,aKLh->aIJh', self.U[n][i], theta)
                # print("theta", theta)
                # print("Utheta", Utheta)
                # print("U", self.ttnH[n][i])
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

def init(pd):
    def g_state(dim):
        tensor = np.zeros(dim)
        tensor[(0,)*len(dim)] = 1.
        return tensor
    nc = len(pd)
    length_chain = [len(sum(chain_n, []))  for chain_n in pd]
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
    e_tensor[0][0][0,0,0,0,0] = 1/np.sqrt(2)
    e_tensor[0][0][0,1,0,0,0] = 1/np.sqrt(2)
    return FishBoneNet(
        (eb_tensor, e_tensor, v_tensor, vb_tensor),
        (eb_s, e_s, v_s, vb_s_and_main_s)
    )

if __name__ == "__main__":
    ttn = init(nc=4, L=3, d1=5, de=2, dv=10)
    ttn.get_theta2(-1, 1)
