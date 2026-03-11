import unittest

from matrix_calculator import (
    MatrixError,
    add_matrices,
    determinant,
    multiply_matrices,
    rank,
    solve_slae_cramer,
    solve_slae_gauss,
    solve_slae_matrix_method,
)


class MatrixCalculatorTests(unittest.TestCase):
    def test_add_matrices(self):
        result = add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(result.result, [[6, 8], [10, 12]])

    def test_multiply_matrices(self):
        result = multiply_matrices([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]])
        self.assertEqual(result.result, [[50], [122]])

    def test_determinant(self):
        result = determinant([[2, 3], [1, 4]])
        self.assertAlmostEqual(result.result, 5.0)

    def test_rank(self):
        result = rank([[1, 2], [2, 4], [1, 1]])
        self.assertEqual(result.result, 2)

    def test_slae_methods(self):
        a = [[2, 1], [5, 7]]
        b = [11, 13]
        expected = [64 / 9, -29 / 9]
        for solver in (solve_slae_matrix_method, solve_slae_cramer, solve_slae_gauss):
            x = solver(a, b).result
            self.assertAlmostEqual(x[0], expected[0])
            self.assertAlmostEqual(x[1], expected[1])

    def test_cramer_failure(self):
        with self.assertRaises(MatrixError):
            solve_slae_cramer([[1, 2], [2, 4]], [1, 1])


if __name__ == "__main__":
    unittest.main()
