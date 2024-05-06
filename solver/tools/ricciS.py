def ricciS2(R_munu, gu):
    # Initialize the Ricci scalar
    R = 0
    
    # Calculate the Ricci scalar
    for mu in range(4):
        for nu in range(4):
            R += gu[mu][nu] * R_munu[mu][nu]
    
    return R

# Example usage:
# R = ricciS2(R_munu, gu)
