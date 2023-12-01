#!/usr/bin/env python3

# import PyQt5
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

#####
# 1 #
#####

# Orienté
mat1 = np.array([[0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 1, 1, 0],
                 [0, 1, 0, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0]])

# Non-orienté
mat2 = np.array([[0, 1, 1, 0, 0, 0, 0],
                 [1, 0, 0, 1, 1, 1, 1],
                 [1, 0, 0, 1, 1, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1, 1, 0],
                 [0, 1, 0, 0, 1, 0, 1],
                 [0, 1, 0, 0, 0, 1, 0]])


#########
# UTILS #
#########

def show(mat, title=""):
    plt.figure()
    if (mat == mat.T).all():
        G = nx.from_numpy_array(mat)
    else:
        G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
    plt.title(title)
    nx.draw(G, with_labels=True)
    plt.show(block=False)

#####
# 2 #
#####


def deg_non_oriented(mat):
    if not (mat == mat.T).all():
        raise Exception("not a symetric matrix!")
    for n in range(len(mat)):
        print(f"{n} deg={sum(mat[n,:])}")


def deg_oriented(mat):
    for n in range(len(mat)):
        print(f"{n} deg_out={sum(mat[n,:])} deg_in={sum(mat[:,n])}")


#####
# 3 #
#####


def dist_from(mat, n, dist):
    # mat = puis(mat, dist)
    mat = np.linalg.matrix_power(mat, dist)
    print(f"from {n+1} in {dist} steps:", end=" ")
    for i in range(len(mat)):
        if mat[n, i] > 0:
            print(i+1, end=" ")
    print()


#####
# 4 #
#####


def dist_min(mat, dep, arr):
    matO = mat.copy()
    dist = 0
    while mat[dep, arr] == 0:
        dist += 1
        mat = np.matmul(mat, matO)
        if dist > len(mat):
            return -1
    return dist

#####
# 5 #
#####


def nb_composantes_connexes(mat):
    dist = mat.astype(float)
    dist += np.identity(len(mat))  # To avaoid just rotating the matrix
    dist = np.linalg.matrix_power(dist, len(mat))  # to get all ateinable nodes
    cc_count = 1  # 1 because we assume the graph is connected in best case
    for i in range(len(mat)):
        if dist[i, cc_count-1] == 0:
            cc_count += 1
    return cc_count


if __name__ == "__main__":
    show(mat1, "mat1 (oriented)")
    show(mat2, "mat2 (non-oriented)")
    print("Non-oriented matrix:")
    deg_non_oriented(mat2)
    print("Oriented matrix:")
    deg_oriented(mat1)
    print("Distance for mat1 in 2 steps from 1:")
    dist_from(mat1, 0, 2)
    print("Distance min for mat1 between 3 and 7:")
    print(dist_min(mat1, 2, 6))
    print("Number of connected components in mat1:")
    print(nb_composantes_connexes(mat1))
    print("Number of connected components in mat2:")
    print(nb_composantes_connexes(mat2))

    input("Press enter to exit...")
