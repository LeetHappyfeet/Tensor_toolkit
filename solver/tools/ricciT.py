def ricciT2(gu, gl, delta):
    # Initialize Ricci tensor
    R_munu = [[None for _ in range(4)] for _ in range(4)]

    # Precalculate metric derivatives
    diff_1_gl = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
    diff_2_gl = [[[[None for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    for i in range(4):
        for j in range(i, 4):
            for k in range(4):
                diff_1_gl[i][j][k] = takeFiniteDifference1_2(gl[i][j], k, delta)
                if k == 0:
                    diff_1_gl[i][j][k] /= c
                for n in range(k, 4):
                    diff_2_gl[i][j][k][n] = takeFiniteDifference2_2(gl[i][j], k, n, delta)
                    if (n == 0 and k != 0) or (n != 0 and k == 0):
                        diff_2_gl[i][j][k][n] /= c
                    elif n == 0 and k == 0:
                        diff_2_gl[i][j][k][n] /= c**2
                    if k != n:
                        diff_2_gl[i][j][n][k] = diff_2_gl[i][j][k][n]

    # Assign symmetric values
    for k in range(4):
        diff_1_gl[1][0][k] = diff_1_gl[0][1][k]
        diff_1_gl[2][0][k] = diff_1_gl[0][2][k]
        diff_1_gl[2][1][k] = diff_1_gl[1][2][k]
        diff_1_gl[3][0][k] = diff_1_gl[0][3][k]
        diff_1_gl[3][1][k] = diff_1_gl[1][3][k]
        diff_1_gl[3][2][k] = diff_1_gl[2][3][k]
        for n in range(4):
            diff_2_gl[1][0][k][n] = diff_2_gl[0][1][k][n]
            diff_2_gl[2][0][k][n] = diff_2_gl[0][2][k][n]
            diff_2_gl[2][1][k][n] = diff_2_gl[1][2][k][n]
            diff_2_gl[3][0][k][n] = diff_2_gl[0][3][k][n]
            diff_2_gl[3][1][k][n] = diff_2_gl[1][3][k][n]
            diff_2_gl[3][2][k][n] = diff_2_gl[2][3][k][n]

    # Construct Ricci tensor
    for i in range(4):
        for j in range(i, 4):
            R_munu_temp = np.zeros_like(gl[0][0])  # Assuming all components have the same shape
            for a in range(4):
                for b in range(4):
                    gab = gu[a][b]
                    R_munu_temp -= 0.5 * (diff_2_gl[i][j][a][b] + diff_2_gl[a][b][i][j] - diff_2_gl[i][b][j][a] - diff_2_gl[j][b][i][a]) * gab
                    for r in range(4):
                        for d in range(4):
                            R_munu_temp += 0.5 * (0.5 * diff_1_gl[a][r][i] * diff_1_gl[b][d][j] + diff_1_gl[i][r][a] * diff_1_gl[j][d][b] - diff_1_gl[i][r][a] * diff_1_gl[j][b][d]) * gab * gu[r][d]
                            R_munu_temp -= 0.25 * (diff_1_gl[j][r][i] + diff_1_gl[i][r][j] - diff_1_gl[i][j][r]) * (2 * diff_1_gl[b][d][a] - diff_1_gl[a][b][d]) * gab * gu[r][d]
            R_munu[i][j] = R_munu_temp

    # Assign symmetric values
    R_munu[1][0] = R_munu[0][1]
    R_munu[2][0] = R_munu[0][2]
    R_munu[2][1] = R_munu[1][2]
    R_munu[3][0] = R_munu[0][3]
    R_munu[3][1] = R_munu[1][3]
    R_munu[3][2] = R_munu[2][3]

    return R_munu

# Example usage:
# R_munu = ricciT2(gu, gl, delta)
