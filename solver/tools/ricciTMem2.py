import numpy as np
from solver.tools.takeFiniteDifference1_2 import takeFiniteDifference1_2
from solver.tools.takeFiniteDifference2_2 import takeFiniteDifference2_2

def ricciTMem2(gu, gl, delta):
    # Calculate the size of gl{1,1}
    s = np.shape(gl[0][0])

    # Ricci tensor
    R_munu = [[None for _ in range(4)] for _ in range(4)]

    # Precalculate metric derivatives
    diff_1_gl = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(i, 4):
            for k in range(4):
                diff_1_gl[i][j][k] = takeFiniteDifference1_2(gl[i][j], k, delta)

    # Assign symmetric values
    for k in range(4):
        diff_1_gl[1][0][k] = diff_1_gl[0][1][k]
        diff_1_gl[2][0][k] = diff_1_gl[0][2][k]
        diff_1_gl[2][1][k] = diff_1_gl[1][2][k]
        diff_1_gl[3][0][k] = diff_1_gl[0][3][k]
        diff_1_gl[3][1][k] = diff_1_gl[1][3][k]
        diff_1_gl[3][2][k] = diff_1_gl[2][3][k]

    # Construct Ricci tensor
    for i in range(4):
        for j in range(i, 4):
            R_munu_temp = np.zeros_like(gl[0][0])
            for a in range(4):
                for b in range(4):
                    R_munu_temp -= 0.5 * (takeFiniteDifference2_2(gl[i][j], a, b, delta) +
                                          takeFiniteDifference2_2(gl[a][b], i, j, delta) -
                                          takeFiniteDifference2_2(gl[i][b], j, a, delta) -
                                          takeFiniteDifference2_2(gl[j][b], i, a, delta)) * gu[a][b]
                    for c in range(4):
                        for d in range(4):
                            R_munu_temp += 0.5 * (0.5 * diff_1_gl[a][c][i] * diff_1_gl[b][d][j] +
                                                  diff_1_gl[i][c][a] * diff_1_gl[j][d][b] -
                                                  diff_1_gl[i][c][a] * diff_1_gl[j][b][d]) * gu[a][b] * gu[c][d]
                            R_munu_temp -= 0.25 * (diff_1_gl[j][c][i] + diff_1_gl[i][c][j] -
                                                    diff_1_gl[i][j][c]) * (2 * diff_1_gl[b][d][a] -
                                                                           diff_1_gl[a][b][d]) * gu[a][b] * gu[c][d]
            R_munu[i][j] = R_munu_temp

    # Assign symmetric values
    R_munu[1][0] = R_munu[0][1]
    R_munu[2][0] = R_munu[0][2]
    R_munu[2][1] = R_munu[1][2]
    R_munu[3][0] = R_munu[0][3]
    R_munu[3][1] = R_munu[1][3]
    R_munu[3][2] = R_munu[2][3]

    return R_munu

