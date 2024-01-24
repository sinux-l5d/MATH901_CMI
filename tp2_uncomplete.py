#!/usr/bin/env python3

matCouts = np.array([[0,8,2,0,0,0,0],
                    [0,0,0,0,3,0,10],
                    [0,0,0,3,0,0,0],
                    [0,2,4,0,0,0,0],
                    [0,0,5,0,0,4,0],
                    [0,5,0,0,2,0,2],
                    [0,0,0,0,0,0,0]])

# put infinity where there is no path (but not in the diagonal)
matCouts[matCouts == 0] = np.inf
np.fill_diagonal(matCouts, 0)

def dijkstra(mat, sommet):
    # TODO
    pass


# vim: ts=4 sts=4 sw=4 et
