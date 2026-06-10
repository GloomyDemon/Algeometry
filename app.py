import ast
import os
import random
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from matrix_calculator import (
    MatrixError,
    add_matrices,
    add_vectors,
    cross_product,
    determinant,
    dot_product,
    mixed_product,
    multiply_by_scalar,
    multiply_matrices,
    multiply_vector_by_scalar,
    rank,
    solve_slae_cramer,
    solve_slae_gauss,
    solve_slae_matrix_method,
    subtract_matrices,
    subtract_vectors,
    transpose,
)

THEORY_CARDS = [
    (
        "Каноническое уравнение прямой в 3D",
        "(x-x0)/l = (y-y0)/m = (z-z0)/n, где (l,m,n) — направляющий вектор.",
    ),
    (
        "Общее уравнение плоскости",
        "Ax + By + Cz + D = 0, нормальный вектор плоскости n=(A,B,C).",
    ),
    (
        "Угол между плоскостями",
        "cos φ = |n1·n2| / (|n1| |n2|), где n1 и n2 — нормали плоскостей.",
    ),
    (
        "Расстояние от точки до плоскости",
        "d = |Ax0 + By0 + Cz0 + D| / sqrt(A²+B²+C²).",
    ),
]

SAMPLE_INPUTS = [
    ("Матрицы 2×2", "[[1,2],[3,4]]", "[[5,6],[7,8]]", "2"),
    ("СЛАУ", "[[2,1],[5,7]]", "[11,13]", ""),
    ("Векторы 3D", "[1,2,3]", "[4,5,6]", "3"),
]


def parse_literal(text: str, label: str):
    try:
        return ast.literal_eval(text.strip())
    except Exception as exc:
        raise MatrixError(f"Ошибка чтения {label}: {exc}") from exc


def parse_matrix(text: str):
    value = parse_literal(text, "матрицы")
    if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
        raise MatrixError("Матрица должна быть списком списков, например [[1,2],[3,4]]")
    return [[float(x) for x in row] for row in value]


def parse_vector(text: str):
    value = parse_literal(text, "вектора")
    if not isinstance(value, list) or any(isinstance(item, list) for item in value):
        raise MatrixError("Вектор должен быть списком чисел, например [1,2,3]")
    return [float(x) for x in value]


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def hex_to_rgb(value: str):
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise MatrixError("Цвет должен быть в формате #RRGGBB")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def normalize_rgb(value):
    if isinstance(value, tuple):
        return tuple(int(channel) for channel in value[:3])
    if isinstance(value, str):
        return hex_to_rgb(value)
    raise MatrixError("Не удалось прочитать цвет пикселя")


def scaled_photo_image(source: tk.PhotoImage, scale: float) -> tk.PhotoImage:
    scale = max(scale, 0.01)
    source_width = source.width()
    source_height = source.height()
    target_width = max(1, int(source_width * scale))
    target_height = max(1, int(source_height * scale))
    if target_width == source_width and target_height == source_height:
        return source

    image = tk.PhotoImage(width=target_width, height=target_height)
    for target_y in range(target_height):
        source_y = min(source_height - 1, int(target_y / scale))
        row = []
        for target_x in range(target_width):
            source_x = min(source_width - 1, int(target_x / scale))
            row.append(rgb_to_hex(normalize_rgb(source.get(source_x, source_y))))
        image.put("{" + " ".join(row) + "}", to=(0, target_y))
    return image


class CalculatorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Алгебра и геометрия — подготовка к КТ №3")
        self.geometry("1120x760")
        self.minsize(980, 680)

        self.style = ttk.Style(self)
        self.style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        self.configure(bg="#f5f7fb")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.calculator_tab = ttk.Frame(notebook, padding=10)
        self.lab_tab = ttk.Frame(notebook, padding=10)
        self.theory_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.calculator_tab, text="🧮 Калькулятор")
        notebook.add(self.lab_tab, text="🎨 Лабораторная №1")
        notebook.add(self.theory_tab, text="📌 КТ №3 теория")

        self._build_calculator_tab()
        self._build_lab_tab()
        self._build_theory_tab()

    def _build_calculator_tab(self):
        header = ttk.Label(
            self.calculator_tab,
            text="Матрицы, СЛАУ и векторы с промежуточными шагами",
            font=("TkDefaultFont", 16, "bold"),
        )
        header.pack(anchor="w", pady=(0, 8))

        input_frame = ttk.Frame(self.calculator_tab)
        input_frame.pack(fill="x")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)

        self.input_a = tk.Text(input_frame, height=6, wrap="word")
        self.input_b = tk.Text(input_frame, height=6, wrap="word")
        self.input_extra = ttk.Entry(input_frame)

        ttk.Label(input_frame, text="A: матрица или вектор").grid(row=0, column=0, sticky="w")
        ttk.Label(input_frame, text="B: матрица, вектор b или второй вектор").grid(row=0, column=1, sticky="w")
        self.input_a.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.input_b.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(input_frame, text="Скаляр k / третий вектор C для смешанного произведения:").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.input_extra.grid(row=3, column=0, columnspan=2, sticky="ew")

        quick = ttk.Frame(self.calculator_tab)
        quick.pack(fill="x", pady=(8, 2))
        ttk.Label(quick, text="Быстрые примеры:").pack(side="left")
        for title, a, b, scalar in SAMPLE_INPUTS:
            ttk.Button(quick, text=title, command=lambda x=(a, b, scalar): self.fill_sample(*x)).pack(side="left", padx=4)
        ttk.Button(quick, text="Очистить", command=self.clear_calculator).pack(side="right")

        buttons = ttk.LabelFrame(self.calculator_tab, text="Команды")
        buttons.pack(fill="x", pady=8)
        operations = [
            ("A + B", self.do_add),
            ("A - B", self.do_sub),
            ("k · A", self.do_scalar),
            ("A · B", self.do_mul),
            ("Aᵀ", self.do_transpose),
            ("det(A)", self.do_det),
            ("rank(A)", self.do_rank),
            ("СЛАУ матричный", self.do_slae_matrix),
            ("СЛАУ Крамер", self.do_slae_cramer),
            ("СЛАУ Гаусс", self.do_slae_gauss),
            ("a + b", self.do_vector_add),
            ("a - b", self.do_vector_sub),
            ("k · a", self.do_vector_scalar),
            ("a · b", self.do_dot),
            ("a × b", self.do_cross),
            ("[a,b,c]", self.do_mixed),
        ]
        for idx, (title, callback) in enumerate(operations):
            ttk.Button(buttons, text=title, command=callback, style="Accent.TButton" if idx in (7, 10, 13) else "TButton").grid(
                row=idx // 8, column=idx % 8, padx=4, pady=4, sticky="ew"
            )
            buttons.grid_columnconfigure(idx % 8, weight=1)

        ttk.Label(self.calculator_tab, text="Промежуточные действия и результат:").pack(anchor="w")
        self.output = tk.Text(self.calculator_tab, height=16, wrap="word", bg="#101827", fg="#e5eefc", insertbackground="#e5eefc")
        self.output.pack(fill="both", expand=True, pady=(4, 0))
        self.fill_sample(*SAMPLE_INPUTS[0][1:])
        self.show_info("Готово", "Выберите пример или введите свои данные. Все команды доступны мышью, ввод — с клавиатуры.")

    def _build_lab_tab(self):
        ttk.Label(
            self.lab_tab,
            text="Лабораторная работа №1: замена цвета в растровом изображении",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            self.lab_tab,
            text="Загрузите PNG/GIF/PPM, выберите цвет палитрой или настоящей пипеткой по изображению, задайте допуск и сохраните результат.",
        ).pack(anchor="w")

        controls = ttk.LabelFrame(self.lab_tab, text="Настройки обработки")
        controls.pack(fill="x", pady=10)
        self.image_path = tk.StringVar(value="Изображение не выбрано")
        self.target_color = tk.StringVar(value="#ff0000")
        self.replace_color = tk.StringVar(value="#2f80ed")
        self.tolerance = tk.IntVar(value=45)
        self.replaced_pixels = tk.StringVar(value="Заменено пикселей: 0")
        self.preview_status = tk.StringVar(value="Масштаб: по размеру окна. Для пипетки нажмите кнопку и кликните по исходной картинке.")
        self.preview_zoom = tk.StringVar(value="fit")
        self.pipette_enabled = False
        self.original_display_image = None
        self.processed_display_image = None
        self.preview_scales = {"original": 1.0, "processed": 1.0}

        ttk.Button(controls, text="Открыть изображение", command=self.open_image).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Label(controls, textvariable=self.image_path).grid(row=0, column=1, columnspan=7, sticky="w")
        ttk.Label(controls, text="Целевой цвет:").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.target_color, width=10).grid(row=1, column=1, sticky="w")
        ttk.Button(controls, text="Палитра", command=lambda: self.pick_color(self.target_color)).grid(row=1, column=2, padx=4)
        ttk.Button(controls, text="Пипетка", command=self.enable_pipette).grid(row=1, column=3, padx=4)
        ttk.Label(controls, text="Новый цвет:").grid(row=1, column=4, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.replace_color, width=10).grid(row=1, column=5, sticky="w")
        ttk.Button(controls, text="Палитра", command=lambda: self.pick_color(self.replace_color)).grid(row=1, column=6, padx=4)
        ttk.Label(controls, text="Допуск:").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Scale(controls, from_=0, to=255, variable=self.tolerance, orient="horizontal").grid(row=2, column=1, columnspan=4, sticky="ew")
        ttk.Label(controls, textvariable=self.replaced_pixels).grid(row=2, column=5, columnspan=3, sticky="w", padx=4)
        ttk.Button(controls, text="Заменить цвет", command=self.replace_image_color, style="Accent.TButton").grid(row=3, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Сохранить результат", command=self.save_processed_image).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Создать демо-картинку", command=self.create_demo_image).grid(row=3, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Вписать в окно", command=self.fit_previews).grid(row=3, column=3, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="100% + прокрутка", command=self.show_previews_full_size).grid(row=3, column=4, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="−", command=lambda: self.change_preview_zoom(0.8)).grid(row=3, column=5, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="+", command=lambda: self.change_preview_zoom(1.25)).grid(row=3, column=6, padx=4, pady=4, sticky="ew")
        ttk.Label(controls, textvariable=self.preview_status).grid(row=4, column=0, columnspan=8, sticky="w", padx=4, pady=(0, 4))
        for col in range(8):
            controls.grid_columnconfigure(col, weight=1)

        preview = ttk.PanedWindow(self.lab_tab, orient="horizontal")
        preview.pack(fill="both", expand=True)
        self.original_canvas = self._make_image_view(preview, "Исходное изображение")
        self.processed_canvas = self._make_image_view(preview, "Результат")
        self.original_canvas.bind("<Button-1>", self.pick_color_from_image)
        self.original_canvas.bind("<Configure>", lambda event: self._refresh_previews())
        self.processed_canvas.bind("<Configure>", lambda event: self._refresh_previews())
        self.original_image = None
        self.processed_image = None
        self.create_demo_image()

    def _make_image_view(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        parent.add(frame, weight=1)
        canvas = tk.Canvas(frame, bg="#111827", highlightthickness=0)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return canvas

    def _build_theory_tab(self):
        ttk.Label(self.theory_tab, text="Мини-тренажёр по теме «Прямая и плоскость»", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(
            self.theory_tab,
            text="Фишка для защиты: случайные карточки с формулами, которые часто спрашивают на КТ №3.",
        ).pack(anchor="w", pady=(0, 10))
        self.card_title = ttk.Label(self.theory_tab, font=("TkDefaultFont", 14, "bold"))
        self.card_title.pack(anchor="w", pady=(10, 4))
        self.card_body = tk.Text(self.theory_tab, height=8, wrap="word", bg="#fff8dd")
        self.card_body.pack(fill="x")
        ttk.Button(self.theory_tab, text="Показать случайную формулу", command=self.random_theory_card, style="Accent.TButton").pack(anchor="w", pady=8)
        checklist = tk.Text(self.theory_tab, height=14, wrap="word")
        checklist.pack(fill="both", expand=True)
        checklist.insert(
            tk.END,
            "Чек-лист на КТ №3:\n"
            "✓ Калькулятор показывает шаги и ошибки математической невозможности.\n"
            "✓ Есть операции над матрицами, СЛАУ и векторами.\n"
            "✓ Лабораторная №1 загружает растровое изображение, строит цветовую маску, заменяет цвет и сохраняет результат.\n"
            "✓ Интерфейс поддерживает ввод с клавиатуры и команды кнопками мышью.\n"
            "✓ Можно объяснить теорию прямой и плоскости по карточкам выше.\n",
        )
        checklist.configure(state="disabled")
        self.random_theory_card()

    def fill_sample(self, a, b, scalar):
        self.input_a.delete("1.0", tk.END)
        self.input_b.delete("1.0", tk.END)
        self.input_extra.delete(0, tk.END)
        self.input_a.insert(tk.END, a)
        self.input_b.insert(tk.END, b)
        self.input_extra.insert(0, scalar)

    def clear_calculator(self):
        self.fill_sample("", "", "")
        self.show_info("Очищено", "Поля ввода очищены.")

    def show(self, title, calc_result):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, title + "\n")
        self.output.insert(tk.END, "-" * 72 + "\n")
        for step in calc_result.steps:
            self.output.insert(tk.END, f"• {step}\n")
        self.output.insert(tk.END, "\nРезультат:\n")
        self.output.insert(tk.END, str(calc_result.result))

    def show_info(self, title, text):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"{title}\n{'-' * 72}\n{text}")

    def show_error(self, err: Exception):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Сбой или невозможность выполнить операцию:\n{err}")

    def _a_matrix(self):
        return parse_matrix(self.input_a.get("1.0", tk.END))

    def _b_matrix(self):
        return parse_matrix(self.input_b.get("1.0", tk.END))

    def _a_vector(self):
        return parse_vector(self.input_a.get("1.0", tk.END))

    def _b_vector(self):
        return parse_vector(self.input_b.get("1.0", tk.END))

    def _scalar(self):
        try:
            return float(self.input_extra.get())
        except ValueError as exc:
            raise MatrixError("Скаляр должен быть числом") from exc

    def do_add(self):
        self._safe_show("Сложение матриц", lambda: add_matrices(self._a_matrix(), self._b_matrix()))

    def do_sub(self):
        self._safe_show("Вычитание матриц", lambda: subtract_matrices(self._a_matrix(), self._b_matrix()))

    def do_scalar(self):
        self._safe_show("Умножение матрицы на число", lambda: multiply_by_scalar(self._a_matrix(), self._scalar()))

    def do_mul(self):
        self._safe_show("Умножение матриц", lambda: multiply_matrices(self._a_matrix(), self._b_matrix()))

    def do_transpose(self):
        self._safe_show("Транспонирование", lambda: transpose(self._a_matrix()))

    def do_det(self):
        self._safe_show("Определитель", lambda: determinant(self._a_matrix()))

    def do_rank(self):
        self._safe_show("Ранг матрицы", lambda: rank(self._a_matrix()))

    def do_slae_matrix(self):
        self._safe_show("СЛАУ матричным методом", lambda: solve_slae_matrix_method(self._a_matrix(), self._b_vector()))

    def do_slae_cramer(self):
        self._safe_show("СЛАУ методом Крамера", lambda: solve_slae_cramer(self._a_matrix(), self._b_vector()))

    def do_slae_gauss(self):
        self._safe_show("СЛАУ методом Гаусса", lambda: solve_slae_gauss(self._a_matrix(), self._b_vector()))

    def do_vector_add(self):
        self._safe_show("Сложение векторов", lambda: add_vectors(self._a_vector(), self._b_vector()))

    def do_vector_sub(self):
        self._safe_show("Вычитание векторов", lambda: subtract_vectors(self._a_vector(), self._b_vector()))

    def do_vector_scalar(self):
        self._safe_show("Умножение вектора на число", lambda: multiply_vector_by_scalar(self._a_vector(), self._scalar()))

    def do_dot(self):
        self._safe_show("Скалярное произведение", lambda: dot_product(self._a_vector(), self._b_vector()))

    def do_cross(self):
        self._safe_show("Векторное произведение", lambda: cross_product(self._a_vector(), self._b_vector()))

    def do_mixed(self):
        self._safe_show("Смешанное произведение", lambda: mixed_product(self._a_vector(), self._b_vector(), parse_vector(self.input_extra.get())))

    def _safe_show(self, title, callback):
        try:
            self.show(title, callback())
        except Exception as err:
            self.show_error(err)

    def pick_color(self, variable):
        color = colorchooser.askcolor(color=variable.get())[1]
        if color:
            variable.set(color)

    def enable_pipette(self):
        if self.original_image is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте или создайте изображение.")
            return
        self.pipette_enabled = True
        self.original_canvas.configure(cursor="crosshair")
        self.preview_status.set("Пипетка активна: кликните по нужному пикселю на исходном изображении.")

    def fit_previews(self):
        self.preview_zoom.set("fit")
        self._refresh_previews()

    def show_previews_full_size(self):
        self.preview_zoom.set("1.0")
        self._refresh_previews()

    def change_preview_zoom(self, factor):
        if self.preview_zoom.get() == "fit":
            current = self.preview_scales.get("original", 1.0)
        else:
            current = float(self.preview_zoom.get())
        next_zoom = min(4.0, max(0.05, current * factor))
        self.preview_zoom.set(str(next_zoom))
        self._refresh_previews()

    def create_demo_image(self):
        image = tk.PhotoImage(width=260, height=180)
        for y in range(180):
            for x in range(260):
                color = "#ff0000" if 35 < x < 225 and 35 < y < 145 else "#eef2ff"
                if (x - 130) ** 2 + (y - 90) ** 2 < 45**2:
                    color = "#22c55e"
                image.put(color, (x, y))
        self.original_image = image
        self.processed_image = image.copy()
        self.image_path.set("Демо-картинка создана внутри приложения")
        self.replaced_pixels.set("Заменено пикселей: 0")
        self.fit_previews()

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите растровое изображение",
            filetypes=[("Изображения", "*.png *.gif *.ppm *.pgm"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            image = tk.PhotoImage(file=path)
        except tk.TclError as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение: {exc}")
            return
        self.original_image = image
        self.processed_image = image.copy()
        self.image_path.set(f"{os.path.basename(path)} ({image.width()}×{image.height()})")
        self.replaced_pixels.set("Заменено пикселей: 0")
        self.fit_previews()

    def replace_image_color(self):
        if self.original_image is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте изображение.")
            return
        try:
            target = hex_to_rgb(self.target_color.get())
            replacement = rgb_to_hex(hex_to_rgb(self.replace_color.get()))
        except Exception as exc:
            messagebox.showerror("Ошибка цвета", str(exc))
            return
        tolerance = self.tolerance.get()
        result = self.original_image.copy()
        changed = 0
        for y in range(result.height()):
            for x in range(result.width()):
                rgb = normalize_rgb(self.original_image.get(x, y))
                if sum((rgb[i] - target[i]) ** 2 for i in range(3)) ** 0.5 <= tolerance:
                    result.put(replacement, (x, y))
                    changed += 1
        self.processed_image = result
        self.replaced_pixels.set(f"Заменено пикселей: {changed}")
        self._refresh_previews()

    def save_processed_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Нет результата", "Сначала выполните замену цвета.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("GIF", "*.gif"), ("PPM", "*.ppm")],
            title="Сохранить результат",
        )
        if not path:
            return
        try:
            self.processed_image.write(path)
        except tk.TclError as exc:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл: {exc}")
            return
        messagebox.showinfo("Готово", f"Результат сохранён:\n{path}")

    def pick_color_from_image(self, event):
        if not self.pipette_enabled or self.original_image is None:
            return
        scale = self.preview_scales.get("original", 1.0)
        canvas_x = self.original_canvas.canvasx(event.x)
        canvas_y = self.original_canvas.canvasy(event.y)
        source_x = int(canvas_x / scale)
        source_y = int(canvas_y / scale)
        if not (0 <= source_x < self.original_image.width() and 0 <= source_y < self.original_image.height()):
            return
        color = rgb_to_hex(normalize_rgb(self.original_image.get(source_x, source_y)))
        self.target_color.set(color)
        self.pipette_enabled = False
        self.original_canvas.configure(cursor="")
        self.preview_status.set(f"Пипетка выбрала цвет {color} в точке ({source_x}, {source_y}).")

    def _preview_scale_for(self, image, canvas):
        if image is None:
            return 1.0
        zoom = self.preview_zoom.get()
        if zoom != "fit":
            return float(zoom)
        canvas_width = max(1, canvas.winfo_width() - 8)
        canvas_height = max(1, canvas.winfo_height() - 8)
        if canvas_width <= 1 or canvas_height <= 1:
            return 1.0
        return min(1.0, canvas_width / image.width(), canvas_height / image.height())

    def _draw_preview(self, canvas, image, attr_name, scale_key, title):
        canvas.delete("all")
        if image is None:
            canvas.configure(scrollregion=(0, 0, 1, 1))
            return
        scale = self._preview_scale_for(image, canvas)
        display = scaled_photo_image(image, scale)
        setattr(self, attr_name, display)
        canvas.create_image(0, 0, anchor="nw", image=display)
        canvas.configure(scrollregion=(0, 0, display.width(), display.height()))
        self.preview_scales[scale_key] = scale
        if scale_key == "original":
            mode = "вписано в окно" if self.preview_zoom.get() == "fit" else f"{scale * 100:.0f}%"
            self.preview_status.set(f"{title}: {image.width()}×{image.height()}, показ: {mode}. В режиме 100% доступны полосы прокрутки.")

    def _refresh_previews(self):
        if not hasattr(self, "original_canvas"):
            return
        self._draw_preview(self.original_canvas, self.original_image, "original_display_image", "original", "Исходное изображение")
        self._draw_preview(self.processed_canvas, self.processed_image, "processed_display_image", "processed", "Результат")

    def random_theory_card(self):
        title, body = random.choice(THEORY_CARDS)
        self.card_title.configure(text=title)
        self.card_body.configure(state="normal")
        self.card_body.delete("1.0", tk.END)
        self.card_body.insert(tk.END, body)
        self.card_body.configure(state="disabled")


if __name__ == "__main__":
    CalculatorUI().mainloop()
