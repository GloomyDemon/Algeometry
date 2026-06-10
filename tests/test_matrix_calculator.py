import unittest

from matrix_calculator import (
    MatrixError,
    add_matrices,
    add_vectors,
    determinant,
    cross_product,
    dot_product,
    mixed_product,
    multiply_matrices,
    multiply_vector_by_scalar,
    rank,
    subtract_vectors,
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

    def test_vector_operations(self):
        self.assertEqual(add_vectors([1, 2, 3], [4, 5, 6]).result, [5, 7, 9])
        self.assertEqual(subtract_vectors([4, 5, 6], [1, 2, 3]).result, [3, 3, 3])
        self.assertEqual(multiply_vector_by_scalar([1, -2, 3], 2).result, [2, -4, 6])
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]).result, 32)
        self.assertEqual(cross_product([1, 0, 0], [0, 1, 0]).result, [0, 0, 1])
        self.assertEqual(mixed_product([1, 0, 0], [0, 1, 0], [0, 0, 1]).result, 1)

    def test_cross_product_requires_3d(self):
        with self.assertRaises(MatrixError):
            cross_product([1, 2], [3, 4])


if __name__ == "__main__":
    unittest.main()
