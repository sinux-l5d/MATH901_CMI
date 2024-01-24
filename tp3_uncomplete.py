import numpy as np
np.seterr(divide='ignore')
np.set_printoptions(precision=3, suppress=True)

mat = [[3, 2, 1, 0, 0, 1800],
       [1, 0, 0, 1, 0, 400],
       [0, 1, 0, 0, 1, 600],
       [30, 50, 0, 0, 0, 0]]
mat = np.float64(mat)


def getK(mat):
    feco = mat[-1, :-1]  # don't take the bottom right value
    return feco.argmax()


def getL(mat):
    secmembre = mat[:-1, -1]
    k = getK(mat)
    coef = mat[:, k]
    out = np.zeros(len(secmembre))

    for i in range(len(secmembre)):
        try:
            out[i] = secmembre[i] / coef[i]
        except ZeroDivisionError:
            out[i] = float("inf")
    return out.argmin()


def simplex(mat):
    while mat[-1, :].max() > 0:
        k = getK(mat)
        print(f"K = {k+1}")
        print(f"coef K = {mat[-1, k]}")
        l = getL(mat)
        print(f"L = {l+1}")
        print(f"coef L = {mat[l, -1]}")
        pivot = mat[l, k]
        print(f"pivot = {pivot}")

        mat[l, :] /= pivot

        for i in range(len(mat)):
            if i != l:
                mat[i, :] -= (mat[i, k] * mat[l, :]) / pivot


if __name__ == "__main__":
    simplex(mat)
    print(mat)
