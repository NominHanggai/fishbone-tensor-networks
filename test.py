# from fishbonett.model import _c
import numpy as np

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

print(5*_c(2)@_c(2).T)


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