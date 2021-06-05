import numba
import numba.pycc
import numpy as np

cc = numba.pycc.CC("matching_test")


@cc.export("matching_test", "i8[:](i8)")
def matching_test(num_doctors):
    return np.zeros(num_doctors, dtype=numba.i8)


def compile():
    cc.compile()
