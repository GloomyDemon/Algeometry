import ast
import tkinter as tk
from tkinter import ttk

from matrix_calculator import (
    MatrixError,
    add_matrices,
    determinant,
    multiply_by_scalar,
    multiply_matrices,
    rank,
    solve_slae_cramer,
    solve_slae_gauss,
    solve_slae_matrix_method,
    subtract_matrices,
    transpose,
)


def parse_matrix(text: str):
    try:
        value = ast.literal_eval(text)
    except Exception as exc:
        raise MatrixError(f"Ошибка чтения матрицы: {exc}") from exc
    if not isinstance(value, list):
        raise MatrixError("Матрица должна быть списком списков")
    return [[float(x) for x in row] for row in value]


def parse_vector(text: str):
    try:
        value = ast.literal_eval(text)
    except Exception as exc:
        raise MatrixError(f"Ошибка чтения вектора: {exc}") from exc
    if not isinstance(value, list):
        raise MatrixError("Вектор должен быть списком")
    return [float(x) for x in value]


class CalculatorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Калькулятор: первая контрольная точка")
        self.geometry("980x700")

        self.input_a = tk.Text(self, height=7)
        self.input_b = tk.Text(self, height=7)
        self.input_extra = tk.Entry(self)

        ttk.Label(self, text="Матрица A (например [[1,2],[3,4]]):").pack(anchor="w")
        self.input_a.pack(fill="x", padx=8)
        ttk.Label(self, text="Матрица B или вектор b (например [[5,6],[7,8]] / [5,6]):").pack(anchor="w")
        self.input_b.pack(fill="x", padx=8)
        ttk.Label(self, text="Скаляр (для умножения на число):").pack(anchor="w")
        self.input_extra.pack(fill="x", padx=8)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=8)

        operations = [
            ("A + B", self.do_add),
            ("A - B", self.do_sub),
            ("k * A", self.do_scalar),
            ("A * B", self.do_mul),
            ("A^T", self.do_transpose),
            ("det(A)", self.do_det),
            ("rank(A)", self.do_rank),
            ("СЛАУ: матричный", self.do_slae_matrix),
            ("СЛАУ: Крамер", self.do_slae_cramer),
            ("СЛАУ: Гаусс", self.do_slae_gauss),
        ]

        for idx, (title, callback) in enumerate(operations):
            ttk.Button(buttons, text=title, command=callback).grid(row=idx // 5, column=idx % 5, padx=4, pady=4, sticky="ew")
            buttons.grid_columnconfigure(idx % 5, weight=1)

        ttk.Label(self, text="Промежуточные шаги и результат:").pack(anchor="w")
        self.output = tk.Text(self, height=20)
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

        self.bind("<Return>", lambda _: self.do_add())

    def show(self, title, calc_result):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, title + "\n")
        self.output.insert(tk.END, "-" * 60 + "\n")
        for step in calc_result.steps:
            self.output.insert(tk.END, f"• {step}\n")
        self.output.insert(tk.END, "\nРезультат:\n")
        self.output.insert(tk.END, str(calc_result.result))

    def show_error(self, err: Exception):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Сбой или невозможность выполнить операцию:\n{err}")

    def do_add(self):
        try:
            self.show("Сложение матриц", add_matrices(parse_matrix(self.input_a.get("1.0", tk.END)), parse_matrix(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_sub(self):
        try:
            self.show("Вычитание матриц", subtract_matrices(parse_matrix(self.input_a.get("1.0", tk.END)), parse_matrix(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_scalar(self):
        try:
            self.show("Умножение матрицы на число", multiply_by_scalar(parse_matrix(self.input_a.get("1.0", tk.END)), float(self.input_extra.get())))
        except Exception as err:
            self.show_error(err)

    def do_mul(self):
        try:
            self.show("Умножение матриц", multiply_matrices(parse_matrix(self.input_a.get("1.0", tk.END)), parse_matrix(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_transpose(self):
        try:
            self.show("Транспонирование", transpose(parse_matrix(self.input_a.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_det(self):
        try:
            self.show("Определитель", determinant(parse_matrix(self.input_a.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_rank(self):
        try:
            self.show("Ранг матрицы", rank(parse_matrix(self.input_a.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_slae_matrix(self):
        try:
            self.show("СЛАУ матричным методом", solve_slae_matrix_method(parse_matrix(self.input_a.get("1.0", tk.END)), parse_vector(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_slae_cramer(self):
        try:
            self.show("СЛАУ методом Крамера", solve_slae_cramer(parse_matrix(self.input_a.get("1.0", tk.END)), parse_vector(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)

    def do_slae_gauss(self):
        try:
            self.show("СЛАУ методом Гаусса", solve_slae_gauss(parse_matrix(self.input_a.get("1.0", tk.END)), parse_vector(self.input_b.get("1.0", tk.END))))
        except Exception as err:
            self.show_error(err)


if __name__ == "__main__":
    CalculatorUI().mainloop()
