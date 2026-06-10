from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


class MatrixError(ValueError):
    """Ошибка операций над матрицами и векторами."""


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


def _vector_size(vector: Vector) -> int:
    if not vector:
        raise MatrixError("Вектор не должен быть пустым")
    return len(vector)


def _require_same_vector_size(a: Vector, b: Vector) -> int:
    size_a = _vector_size(a)
    size_b = _vector_size(b)
    if size_a != size_b:
        raise MatrixError("Операция невозможна: векторы имеют разную размерность")
    return size_a


def _require_3d(vector: Vector, name: str) -> None:
    if _vector_size(vector) != 3:
        raise MatrixError(f"{name} должен быть трёхмерным вектором")


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


def add_vectors(a: Vector, b: Vector) -> CalculationResult:
    size = _require_same_vector_size(a, b)
    result = [a[i] + b[i] for i in range(size)]
    return CalculationResult([f"Проверка размерности: {size}", "Складываем координаты векторов"], result)


def subtract_vectors(a: Vector, b: Vector) -> CalculationResult:
    size = _require_same_vector_size(a, b)
    result = [a[i] - b[i] for i in range(size)]
    return CalculationResult([f"Проверка размерности: {size}", "Вычитаем координаты векторов"], result)


def multiply_vector_by_scalar(vector: Vector, scalar: float) -> CalculationResult:
    size = _vector_size(vector)
    result = [scalar * value for value in vector]
    return CalculationResult([f"Вектор размерности {size}", f"Умножаем каждую координату на {scalar}"], result)


def dot_product(a: Vector, b: Vector) -> CalculationResult:
    size = _require_same_vector_size(a, b)
    products = [a[i] * b[i] for i in range(size)]
    return CalculationResult(
        [f"Проверка размерности: {size}", f"Скалярное произведение = {' + '.join(f'{p:.4g}' for p in products)}"],
        sum(products),
    )


def cross_product(a: Vector, b: Vector) -> CalculationResult:
    _require_3d(a, "Первый множитель")
    _require_3d(b, "Второй множитель")
    result = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
    return CalculationResult(
        ["Векторное произведение определено для 3D-векторов", "Вычисляем координаты через миноры определителя"],
        result,
    )


def mixed_product(a: Vector, b: Vector, c: Vector) -> CalculationResult:
    _require_3d(a, "Первый вектор")
    _require_3d(b, "Второй вектор")
    _require_3d(c, "Третий вектор")
    cross = cross_product(b, c).result
    value = sum(a[i] * cross[i] for i in range(3))
    return CalculationResult(
        ["Смешанное произведение [a,b,c] = a · (b × c)", f"b × c = {cross}", f"a · (b × c) = {value:.6g}"],
        value,
    )


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
        steps=["Проверяем det(A) != 0", f"det(A)={det_a.result:.6g}", "Находим обратную матрицу A^-1", "Умножаем A^-1 на b"],
        result=x,
    )


def solve_slae_cramer(a: Matrix, b: Vector) -> CalculationResult:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Метод Крамера применим только к квадратной матрице")
    if len(b) != n:
        raise MatrixError("Размер вектора b не совпадает с размером матрицы")
    det_a = determinant(a).result
    if abs(det_a) < 1e-12:
        raise MatrixError("Метод Крамера невозможен: det(A)=0")
    x = []
    steps = [f"det(A)={det_a:.6g}"]
    for col in range(n):
        replaced = _copy(a)
        for row in range(n):
            replaced[row][col] = b[row]
        det_i = determinant(replaced).result
        x_i = det_i / det_a
        x.append(x_i)
        steps.append(f"det(A_{col+1})={det_i:.6g}; x{col+1}=det(A_{col+1})/det(A)={x_i:.6g}")
    return CalculationResult(steps=steps, result=x)


def solve_slae_gauss(a: Matrix, b: Vector) -> CalculationResult:
    n, m = _shape(a)
    if len(b) != n:
        raise MatrixError("Размер вектора b не совпадает с числом строк матрицы")
    mat = [a[i][:] + [b[i]] for i in range(n)]
    steps = ["Формируем расширенную матрицу [A|b]"]
    row = 0
    pivots = []
    for col in range(m):
        pivot = max(range(row, n), key=lambda r: abs(mat[r][col])) if row < n else row
        if row >= n or abs(mat[pivot][col]) < 1e-12:
            continue
        mat[row], mat[pivot] = mat[pivot], mat[row]
        pivot_val = mat[row][col]
        for j in range(col, m + 1):
            mat[row][j] /= pivot_val
        for i in range(n):
            if i != row and abs(mat[i][col]) > 1e-12:
                factor = mat[i][col]
                for j in range(col, m + 1):
                    mat[i][j] -= factor * mat[row][j]
        pivots.append((row, col))
        steps.append(f"Ведущий элемент в столбце {col+1}; нормируем строку {row+1}")
        row += 1
    for i in range(n):
        if all(abs(mat[i][j]) < 1e-12 for j in range(m)) and abs(mat[i][m]) > 1e-12:
            raise MatrixError("Система несовместна")
    if len(pivots) < m:
        raise MatrixError("Система имеет бесконечно много решений; единственное решение не найдено")
    x = [0.0 for _ in range(m)]
    for pivot_row, pivot_col in pivots:
        x[pivot_col] = mat[pivot_row][m]
    steps.append("Считываем решение из приведённой матрицы")
    return CalculationResult(steps=steps, result=x)


def _inverse(a: Matrix) -> Matrix:
    n, m = _shape(a)
    if n != m:
        raise MatrixError("Обратная матрица существует только для квадратной")
    mat = [a[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise MatrixError("Матрица вырожденная, обратной матрицы нет")
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
