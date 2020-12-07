import numpy as np
import sys
from numpy import exp

class FinshBone:
    @property
    def H(self):
        self._H = self.fill_H()
        return self._H
    @property
    def sd(self):
        return self._sd
    #
    # @property
    # def He(self):
    #     return self._He
    # @He.setter
    # def He(self,m):
    #     if len(m) == sum(self._eL):
    #         for e in enumerate(m:
    #             if sum(e.shape)/2 == self.pd[:]
    #
    #
    #     self._He = m
    #
    # @property
    # def Hv(self):
    #     return self.Hv
    # @Hv.setter
    # def Hv(self,m):
    #     # check type
    #     self._Hv = m
    #
    # @property
    # def Hee(self):
    #     return self._Hee
    # @Hee.setter
    # def Hee(self,m):
    #     # check type
    #     self._Hee = m
    #
    # @property
    # def Hev(self):
    #     return self._Hev
    # @Hev.setter
    # def Hev(self,m):
    #     # check type
    #     self._Hev = m
    #
    # @property
    # def He_dy(self):
    #     return self._He_dy
    # @He_dy.setter
    # def He_dy(self,m):
    #     # check type
    #     self._He_dy = m
    #
    # @property
    # def Hv_dy(self):
    #     return self._Hv_dy
    # @Hv_dy.setter
    # def Hv_dy(self,m):
    #     # check type
    #     self._Hv_dy = m

    def __init__(self, pd: np.ndarray):
        """
        TODO
        :type pd: nd.ndarray
        :param pd: is a list.
         pD[0] contains physical dimensions of eb, ev, vb on the first chain,
         pD[1] contains physical dimensions of eb, ev, vb on the second chain,
         etc.
        """
        self.pd = pd
        self._nc  = len(pd) # an int
        # pD is a np.ndarray.
        self._ebL = [len(x) for x in pd[:, 0]]
        # pD[:,0] is the first column of the array, the eb column
        self._eL = [len(x) for x in pd[:, 1]]
        self._vl = [len(x) for x in pd[:, 2]]
        # pD[:,1] is the second column of the array, the ev column
        self._vbL = [len(x) for x in pd[:, 3]]
        # pD[:,2] is the third column of the array, the vb column
        self._L = [sum(x) for x in zip(self._ebL,self._evL,self._vbL)]
        # PLEASE NOTE THE SHAPE of pd and nd.array structure.
        # pd = nd.array([
        # [eb0, eb1, eb2], [ev0, ev1, ev2], [vb0, vb1, vb2]
        # ])
        # | eb0 ev0 vb0 |
        # | eb1 ev1 vb1 |
        # | eb2 ev2 vb2 | is different from the structure depicted in SimpleTTS class.

        self.sd = np.empty([2, self._nc], dtype=object)

        #TODO two lists. w is frequency, k is coupling. Get them
        # from the get_coupling function below
        self.w_list = np.empty([2, self._nc], dtype=object)
        self.k_list = np.empty([2, self._nc], dtype=object)

        # initialize spectral densities.
        # for n in range(self._nc):
        #     if   0 not in self._evL[n] :
        #         self.sd[0, n] = self.sd[0, n] = lambda x: 1./1. * exp(-x/1)
        #     elif 0 in     self._evL[n]:
        #         self.sd[0,n] = lambda x: 1./1. * exp(-x/1)
        #     else:
        #         raise ValueError

        # Assign the matrices below according to self.pd
        self._H     = []
        self._He    = []  # list -> single Hamiltonian on e site
        self._Hv    = []  # list -> single Hamiltonian on v site
        self._Hee   = [] # list -> coupling Hamiltonian on e and e
        self._Hev   = [] # list -> coupling Hamiltonian on e and v
        self._He_dy = [] # list -> e dynamic variables coupled to eb
        self._Hv_dy = [] # list -> v dynamic variables coupled to vb

    def get_coupling(self):
        # TODO Get w and k for each spectral density
        # TODO w and k have the same structures as  self.sd (spectral densities)
        self.w_list = []
        self.k_list = []
        return self.w_list, self.k_list

    def get_h1(self, n):
        # TODO k and w
        if 0<= n <= self._nc -1:
            w_list = self.w_list[n,0]
            pd = self.pd[n, 0]
            # n -> the nth chain, 0 -> the 1st element -> w_list for eb.

            try:
                assert len(pd) == len(w_list)
            except AssertionError:
                print("Lengths of the w and pd don't match. ", file=sys.stderr)
                raise
                sys.exit(1)
            # EB Hamiltonian list
            H_eb = [None] * len(pd)
            for i, w in enumerate(w_list):
                H_eb[-1-i] = w_list[i] * np.eye(pd[-1-i])

            w_list = self.w_list[n,1]
            pd = self.pd[n, 2]
            # n -> the nth chain, 0 -> the 3rd element -> w_list for vb.
            try:
                assert len(pd) == len(w_list)
            except AssertionError:
                print("Lengths of the w and pd don't match. ", file=sys.stderr)
                raise
                sys.exit(1)
            # VB Hamiltonian list on the chain n
            H_vb = [None] * len(pd)
            for i, w in enumerate(w_list):
                H_vb[i] = w_list[i] * 'c_n^+ c_n'

            # EV single Hamiltonian list on the chain n
            H_ev = self.He[n] + self.Hv[n]
            return H_eb + H_ev + H_vb
        else:
            raise ValueError


    def fill_H(self, k_list, w_list):
        self.fbH = []
        for n in range(self._nc):
            H_eb = [None]*self._ebL[n]
            H_vb = [None]*self._vbL[n]
            H_ev = [None]
            if range(self._nc) == 2:
                H_eb[-1] = w_list[0] * 'c_n^+ c_n'
                H_vb[0] = w_list[0]
                for k in k_list:
                    H_eb[-k-1] = w_list[k] * 'c_n^+ c_n'
                    H_ev = 
                    H_vb =  
                return H_eb + H_ev + H_vb
            elif range(self._nc) == 1:
                for k in k_list:
                    H_eb =  
                return H_eb
            self.fbH.append(L)

        


class VibronicBath:
    pass
