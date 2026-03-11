from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


class MatrixError(ValueError):
    """Ошибка операций над матрицами."""


Matrix = List[List[float]]
Vector = List[float]


@dataclass
class CalculationResult:
    steps: List[str]
    result: object


def _shape(matrix: Matrix) -> Tuple[int, int]:
    if not matrix or not matrix[0]:
        raise MatrixError("Матрица не должна быть пустой")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise MatrixError("У матрицы разные длины строк")
    return len(matrix), cols


def _copy(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def add_matrices(a: Matrix, b: Matrix) -> CalculationResult:
    ra, ca = _shape(a)
    rb, cb = _shape(b)
    if (ra, ca) != (rb, cb):
        raise MatrixError("Сложение невозможно: разные размерности")
    result = [[a[i][j] + b[i][j] for j in range(ca)] for i in range(ra)]
    return CalculationResult(
        steps=[f"Проверка размеров: {ra}x{ca} и {rb}x{cb}", "Складываем поэлементно"],
        result=result,
    )


def subtract_matrices(a: Matrix, b: Matrix) -> CalculationResult:
    ra, ca = _shape(a)
    rb, cb = _shape(b)
    if (ra, ca) != (rb, cb):
        raise MatrixError("Вычитание невозможно: разные размерности")
    result = [[a[i][j] - b[i][j] for j in range(ca)] for i in range(ra)]
    return CalculationResult(
        steps=[f"Проверка размеров: {ra}x{ca} и {rb}x{cb}", "Вычитаем поэлементно"],
        result=result,
    )


def multiply_by_scalar(a: Matrix, scalar: float) -> CalculationResult:
    r, c = _shape(a)
    result = [[scalar * value for value in row] for row in a]
    return CalculationResult(
        steps=[f"Матрица {r}x{c}", f"Умножаем каждый элемент на {scalar}"],
        result=result,
    )


def multiply_matrices(a: Matrix, b: Matrix) -> CalculationResult:
    ra, ca = _shape(a)
    rb, cb = _shape(b)
    if ca != rb:
        raise MatrixError("Умножение невозможно: число столбцов A != числу строк B")
    result = [[0.0 for _ in range(cb)] for _ in range(ra)]
    steps = [f"Размеры: A={ra}x{ca}, B={rb}x{cb}"]
    for i in range(ra):
        for j in range(cb):
            products = [a[i][k] * b[k][j] for k in range(ca)]
            result[i][j] = sum(products)
            steps.append(f"C[{i+1},{j+1}] = {' + '.join(f'{p:.4g}' for p in products)} = {result[i][j]:.4g}")
    return CalculationResult(steps=steps, result=result)


def transpose(a: Matrix) -> CalculationResult:
    r, c = _shape(a)
    result = [[a[i][j] for i in range(r)] for j in range(c)]
    return CalculationResult(steps=[f"Транспонируем {r}x{c} -> {c}x{r}"], result=result)


def determinant(a: Matrix) -> CalculationResult:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Определитель существует только у квадратной матрицы")
    mat = _copy(a)
    det = 1.0
    steps = ["Вычисляем определитель методом Гаусса"]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            return CalculationResult(steps=steps + [f"Нулевой ведущий элемент в столбце {col+1}"], result=0.0)
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det *= -1
            steps.append(f"Меняем местами строки {col+1} и {pivot+1}")
        pivot_val = mat[col][col]
        det *= pivot_val
        steps.append(f"Ведущий элемент a[{col+1},{col+1}]={pivot_val:.6g}, накопленный det={det:.6g}")
        for row in range(col + 1, n):
            factor = mat[row][col] / pivot_val
            for k in range(col, n):
                mat[row][k] -= factor * mat[col][k]
    return CalculationResult(steps=steps, result=det)


def rank(a: Matrix) -> CalculationResult:
    rows, cols = _shape(a)
    mat = _copy(a)
    r = 0
    steps = ["Приводим матрицу к ступенчатому виду"]
    for col in range(cols):
        pivot = None
        for row in range(r, rows):
            if abs(mat[row][col]) > 1e-12:
                pivot = row
                break
        if pivot is None:
            continue
        mat[r], mat[pivot] = mat[pivot], mat[r]
        pivot_val = mat[r][col]
        for j in range(col, cols):
            mat[r][j] /= pivot_val
        for i in range(rows):
            if i != r and abs(mat[i][col]) > 1e-12:
                factor = mat[i][col]
                for j in range(col, cols):
                    mat[i][j] -= factor * mat[r][j]
        steps.append(f"Опорный столбец {col+1}, текущий ранг {r+1}")
        r += 1
        if r == rows:
            break
    return CalculationResult(steps=steps, result=r)


def solve_slae_matrix_method(a: Matrix, b: Vector) -> CalculationResult:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Матричный метод применим только к квадратной матрице")
    if len(b) != n:
        raise MatrixError("Размер вектора b не совпадает с размером матрицы")
    det_a = determinant(a)
    if abs(det_a.result) < 1e-12:
        raise MatrixError("Система не имеет единственного решения (det(A)=0)")
    inverse = _inverse(a)
    x = [sum(inverse[i][j] * b[j] for j in range(n)) for i in range(n)]
    return CalculationResult(
        steps=det_a.steps + ["det(A) != 0, находим A^-1", "x = A^-1 * b"],
        result=x,
    )


def _inverse(a: Matrix) -> Matrix:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Обратимая матрица должна быть квадратной")
    mat = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(_copy(a))]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise MatrixError("Матрица вырожденная, обратной не существует")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        pivot_val = mat[col][col]
        for j in range(2 * n):
            mat[col][j] /= pivot_val
        for i in range(n):
            if i != col:
                factor = mat[i][col]
                for j in range(2 * n):
                    mat[i][j] -= factor * mat[col][j]
    return [row[n:] for row in mat]


def solve_slae_cramer(a: Matrix, b: Vector) -> CalculationResult:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Метод Крамера применим только к квадратной матрице")
    if len(b) != n:
        raise MatrixError("Размер вектора b не совпадает с размером матрицы")
    det_a = determinant(a).result
    if abs(det_a) < 1e-12:
        raise MatrixError("Метод Крамера неприменим: det(A)=0")
    x = []
    steps = [f"det(A)={det_a:.6g}"]
    for col in range(n):
        modified = _copy(a)
        for row in range(n):
            modified[row][col] = b[row]
        det_i = determinant(modified).result
        xi = det_i / det_a
        x.append(xi)
        steps.append(f"x{col+1} = det(A{col+1})/det(A) = {det_i:.6g}/{det_a:.6g} = {xi:.6g}")
    return CalculationResult(steps=steps, result=x)


def solve_slae_gauss(a: Matrix, b: Vector) -> CalculationResult:
    n, m = _shape(a)
    if len(b) != n:
        raise MatrixError("Размер вектора b не совпадает с числом строк матрицы")
    if n != m:
        raise MatrixError("Для первой контрольной точки поддерживается квадратная СЛАУ")
    mat = [row[:] + [b[i]] for i, row in enumerate(_copy(a))]
    steps = ["Прямой ход метода Гаусса"]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise MatrixError("Система не имеет единственного решения")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        steps.append(f"Опорный элемент в столбце {col+1}: строка {pivot+1}")
        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for k in range(col, n + 1):
                mat[row][k] -= factor * mat[col][k]
    x = [0.0] * n
    steps.append("Обратный ход")
    for i in range(n - 1, -1, -1):
        rhs = mat[i][n] - sum(mat[i][j] * x[j] for j in range(i + 1, n))
        x[i] = rhs / mat[i][i]
        steps.append(f"x{i+1} = {x[i]:.6g}")
    return CalculationResult(steps=steps, result=x)
