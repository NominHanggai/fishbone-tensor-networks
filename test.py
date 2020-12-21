from fishbonett.model import _c
import numpy as np

lowerTheta = self.get_theta1(i + 1, self._ebL[i + 1])
sInverse = np.diag(self.ttnS[i][-1] ** (-1)),
lowerGammaDownCanon = np.tensordot(
    lowerTheta,
    sInverse,
    [2, 0]
)  # vL i [vU] vD vR, [vU] vU -> vL i vD vR vU
lowerGammaDownCanon = np.transpose(lowerGammaDownCanon, [0, 1, 4, 2, 3])  # vL i vD vR vU -> vL i vU vD vR
upperTheta = self.get_theta1(i, self._ebL[i])
print("upperTheta shape", upperTheta.shape)
print("lowerGammaDownCanon", lowerGammaDownCanon.shape)
return np.tensordot(
    upperTheta,
    lowerGammaDownCanon,
    [3, 2]
)  # vL i vU [vD] vR , vL' j [vU'] vD' vR' -> {vL i vU vR; VL' j vD' vR'}


class C:
    def __init__(self):
        self._x = [1, 2, 3]

    @property
    def x(self):
        print("getter")
        return self._x
    # @x.setter
    # def x(self, value):
    #     print("setter")
    #     self._x = value


print(_c(2))

def eye(d):
    if d is [] or None:
        print("1",d)
        return None
    elif type(d) is int or d is str:
        print("2",d)
        return np.eye(int(d))
    elif type(d) is list:
        print("3", d)
        return np.eye(*d)

print("eye3=", eye(3))
# a = [3, 3, 3]
# b = [2]
# pd = np.array([[a, b, b, a], [a, b, b, []]], dtype=object)
# # TODO handle the case above
# tri = FishBoneH(pd)
# tri.domain = [-1, 1]
# print("sd", tri.sd)
# print("sd",tri.sd)
#
# print("evL",tri._evL)
# print("evL",tri._ebL)
# print("!",tri._evL)
# tri.build_coupling()
# print(tri.k_list)
# print(tri.w_list)