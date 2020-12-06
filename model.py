import numpy as np


class FinshBone:
    _nc  = 0
    _evL = []
    _ebL = []
    _vbL = []
    sd = []
    ## TODO. _L should change accodring to the parameters above.
    _L = [sum(x) for x in zip(_ebL, _evL, _vbL)]
    _pd = []
    _H = []
    
    @property
    def H(self):
        self._H = self.fill_H()
        return self._H
    
    
    def __init__(self, ncL: int, evL: list, ebL: list, vbL: list, pD: list):
        """
        TODO
        :param pD: is a list. pD[0] contains peb, pev, pvb. pD[1] contains peb, pev, pvb etc.
        """
        self._nc = ncL
        self._ebL = ebL
        self._evL = evL
        self._vbL = vbL

        # init spectral desnity 
        for n in range(self._nc):
            if len(self.evL[n]) == 2 :
                self.sd.append(
                    [lambda x: 1./1. * exp(-x/1), lambda x: 1./1. * exp(-x/1)]
                )
            elif len(self.evL[n]) == 1:
                self.sd.append(
                    [lambda x: 1./1. * exp(-x/1)]
                )
            else:
                raise ValueError
        self.H = []
    def get_coupling(self):
        return ω_list, k_list
    def get_H_list(self, ch: str, k_list, w_list):
        if ch == 'eb':
            H_eb = [None]*self._ebL[n]
            for i, k in enumerate(k_list):
                H_eb[-1] = w_list*'c_n^+ c_n'
                H_eb[]
        if ch == 'vb':
            H_vb = [None]*self._vbL[n]
        if ch == 'ev':
            H_ev = [None]

    def fill_H(self, k_list, w_list):
        self.fbH = []
        for n in range(self._nc):
            H_eb = [None]*self._ebL[n]
            H_vb = [None]*self._vbL[n]
            H_ev = [None]
            if range(self._nc) == 2:
                H_eb[-1] = w_list[0] 'c_n^+ c_n'
                H_vb[0] = w_list[0]
                for k in k_list:
                    H_eb[-k-1] = w_list[k] 'c_n^+ c_n' +  
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
