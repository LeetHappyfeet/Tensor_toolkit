import unittest
from setMinkowskiThreePlusOne import setMinkowskiThreePlusOne
import numpy as np

class TestSetMinkowskiThreePlusOne(unittest.TestCase):
    def test_output_shapes(self):
        # Define grid size for testing
        gridSize = (2, 3, 3, 3)  # Arbitrary grid size
        
        # Call the function to get alpha, beta, and gamma
        alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)
        
        # Check shapes
        self.assertEqual(alpha.shape, gridSize, "Alpha shape mismatch")
        self.assertEqual(len(beta), 3, "Number of beta components mismatch")
        for i in range(3):
            self.assertEqual(beta[i].shape, gridSize, f"Beta[{i}] shape mismatch")
        for i in range(3):
            for j in range(3):
                self.assertEqual(gamma[i][j].shape, gridSize, f"Gamma[{i}][{j}] shape mismatch")

    def test_initial_values(self):
        # Define grid size for testing
        gridSize = (2, 3, 3, 3)  # Arbitrary grid size
        
        # Call the function to get alpha, beta, and gamma
        alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)
        
        # Check initial values
        self.assertTrue(np.all(alpha == 1), "Incorrect initial values for alpha")
        for i in range(3):
            self.assertTrue(np.all(beta[i] == 0), f"Incorrect initial values for beta[{i}]")
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertTrue(np.all(gamma[i][j] == 1), f"Incorrect initial values for gamma[{i}][{j}] diagonal")
                else:
                    self.assertTrue(np.all(gamma[i][j] == 0), f"Incorrect initial values for gamma[{i}][{j}] off-diagonal")

if __name__ == '__main__':
    unittest.main()