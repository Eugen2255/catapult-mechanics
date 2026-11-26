"""Электронная модель катапульты (раздел 8 курсового проекта по механике).

Слева задаются исходные данные (разделы 3–5), справа строится траектория
полёта снаряда; кнопка «Пуск» проигрывает анимацию выстрела.

Запуск:  python app.py
"""
from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont, messagebox, ttk

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import physics as ph

# Поля ввода: (ключ, подпись, значение по умолчанию)
PROJECTILE = [("m", "Масса снаряда m, кг", 0.008)]
SPRING = [("k", "Жёсткость пружины k, Н/м", 60.0),
          ("dx0", "Растяжение при взводе Δx₀, м", 0.08),
          ("dx1", "Растяжение при отрыве Δx₁, м", 0.02)]
ARM = [("m_arm", "Масса ложки M, кг", 0.020),
       ("L", "Длина ложки (ось → снаряд) L, м", 0.25),
       ("H", "Высота оси над столом H, м", 0.10),
       ("a0", "Угол ложки при взводе α₀, °", 10.0),
       ("a1", "Угол ложки у упора α₁, °", 45.0)]
MANUAL = [("v0", "Скорость вылета V₀, м/с", 5.54),
          ("theta", "Угол вылета θ, °", 10.0),
          ("h0", "Высота вылета h₀, м", 0.127)]
GRAVITY = [("g", "Ускорение свободного падения g, м/с²", 9.8)]
PROTOTYPE_SPRING = [("k", "Рабочая жёсткость k, Н/м", 342),
                    ("l0", "Длина резинки без груза l₀, м", 0.029)]
PROTOTYPE_ARM = [("l_cocked", "Длина резинки при взводе, м", 0.065),
                 ("l_release", "Длина резинки при отделении, м", 0.040),
                 ("m", "Масса снаряда m, кг", 0.008),
                 ("m_arm", "Масса ложки M, кг", 0.011),
                 ("radius", "От оси до снаряда L, м", 0.0605),
                 ("spoon_length", "Полная длина ложки l, м", 0.133),
                 ("axis_offset", "От оси до середины ложки a, м", 0.0115),
                 ("angle_cocked", "Угол ложки при взводе α₀, °", 0),
                 ("angle_release", "Угол ложки у упора α₁, °", 45)]
PROTOTYPE_LOADS = (("1,175", "0,062"), ("1,539", "0,074"))
TASK = [("x1", "Минимальная дальность x₁, м", 1.2),
        ("x2", "Максимальная дальность x₂, м", 1.5)]
EXPERIMENTAL_RANGE = 1.40

SPEEDS = {"×1": 1.0, "×0.5": 0.5, "×0.25": 0.25, "×0.1": 0.1}
FONT = "Times New Roman"
INK, MUTED = "#322927", "#63514a"
PAPER, PANEL, GRID = "#fffcf7", "#f5eee5", "#ded1c4"
TERRACOTTA, OLIVE = "#a04435", "#67754d"
PETG, HARDWARE = "#4c4340", "#2d2928"


def decimal(value: float, places: int) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def parse_number(text: str, name: str) -> float:
    """Число из поля ввода; допускается десятичная запятая."""
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"«{name}»: «{text}» не является числом.") from None
    if not math.isfinite(value):
        raise ValueError(f"«{name}»: недопустимое значение «{text}».")
    return value


def verdict(x: float, x1: float, x2: float) -> str:
    if x < x1:
        return f"недолёт {decimal(x1 - x, 2)} м"
    if x > x2:
        return f"перелёт {decimal(x - x2, 2)} м"
    return "попадание в зону"


@dataclass
class Shot:
    """Результат одного расчёта."""
    catapult: ph.Catapult | None  # None — параметры отрыва заданы вручную
    launch: ph.Launch
    trajectory: ph.Trajectory
    x1: float
    x2: float


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Катапульта — расчёт траектории снаряда")
        self.minsize(1000, 650)
        self._set_style()
        self.vars: dict[str, tk.StringVar] = {}
        self.labels: dict[str, str] = {}
        self.prototype_values = {
            name: f"{value:g}".replace(".", ",") for name, _, value in PROTOTYPE_SPRING + PROTOTYPE_ARM
        }
        self.prototype_loads = PROTOTYPE_LOADS
        self.mode = tk.StringVar(value="manual")
        self.speed = tk.StringVar(value="×0.5")
        self.shot: Shot | None = None
        self._view: tuple[float, float, float, float] | None = None
        self._anim_job: str | None = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_inputs()
        self._build_output()
        self._update_controls()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.calculate()

    # ----------------------------------------------------------- интерфейс

    def _set_style(self) -> None:
        self.configure(background=PANEL)
        for name in tkfont.names(self):
            tkfont.nametofont(name, root=self).configure(family=FONT)
        theme = ttk.Style(self)
        theme.theme_use("clam")
        theme.configure(".", background=PANEL, foreground=INK, bordercolor=GRID,
                        lightcolor=GRID, darkcolor=GRID, selectbackground=TERRACOTTA,
                        selectforeground=PAPER, fieldbackground=PAPER, font=(FONT, 11))
        theme.configure("TLabelframe.Label", foreground=TERRACOTTA, font=(FONT, 12, "bold"))
        theme.configure("TButton", background="#eaded0", padding=(7, 4))
        theme.configure("Run.TButton", background=TERRACOTTA, foreground=PAPER)
        theme.map("TButton", background=[("active", "#dfcec0")])
        theme.map("Run.TButton", background=[("active", "#853a2d")])
        rcParams.update({"font.family": "serif", "font.serif": [FONT, "Liberation Serif", "DejaVu Serif"],
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED})

    def _build_inputs(self) -> None:
        self._panel = ttk.Frame(self, padding=8)
        self._panel.grid(row=0, column=0, sticky="ns")
        self._row = 0

        box = self._box("Режим расчёта")
        for row, (text, value) in enumerate([("Наша модель · параметры вылета", "manual"),
                                             ("Расчёт для стержня с осью на конце", "lever")]):
            ttk.Radiobutton(box, text=text, variable=self.mode, value=value,
                            command=self._on_option).grid(row=row, column=0, columnspan=2, sticky="w")

        self.projectile_box = self._box("Снаряд")
        self._fields(self.projectile_box, PROJECTILE)

        self.spring_box = self._box("Пружина (раздел 3)")
        row = self._fields(self.spring_box, SPRING)
        ttk.Button(self.spring_box, text="Рассчитать k по эксперименту…",
                   command=lambda: StiffnessDialog(self)).grid(row=row, column=0, columnspan=2,
                                                               sticky="ew", pady=(4, 2))

        self.arm_box = self._box("Ложка (раздел 4)")
        row = self._fields(self.arm_box, ARM)
        ttk.Label(self.arm_box, foreground="gray40", font="TkSmallCaptionFont",
                  text="Углы ложки отсчитываются от горизонтали позади оси:\n"
                       "0° — ложка лежит, 90° — стоит вертикально."
                  ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.manual_box = self._box("Параметры отрыва")
        row = self._fields(self.manual_box, MANUAL)
        ttk.Label(self.manual_box, text="Шарик в опытах · пластилин · 8 г · Ø 20 мм",
                  foreground=MUTED).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(self.manual_box, text="Параметры разделов 3–5…", command=lambda: PrototypeDialog(self)).grid(
            row=row + 1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        box = self._box("Полёт (раздел 5)")
        self._fields(box, GRAVITY)

        self._fields(self._box("Целевой диапазон"), TASK)

        buttons = ttk.Frame(self._panel)
        buttons.grid(row=self._row, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Рассчитать", command=self.calculate).grid(
            row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(buttons, text="Пуск", style="Run.TButton", command=self.animate).grid(
            row=0, column=1, sticky="ew", padx=(2, 0))
        ttk.Label(buttons, text="Скорость анимации").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(buttons, textvariable=self.speed, values=list(SPEEDS), state="readonly",
                     width=6).grid(row=1, column=1, sticky="e", pady=(6, 0))

    def _box(self, title: str) -> ttk.LabelFrame:
        box = ttk.LabelFrame(self._panel, text=title, padding=(8, 2, 8, 6))
        box.grid(row=self._row, column=0, sticky="ew", pady=3)
        box.columnconfigure(0, weight=1)
        self._row += 1
        return box

    def _fields(self, box: ttk.LabelFrame, fields: list, row: int = 0) -> int:
        """Добавляет в группу строки «подпись — поле ввода», возвращает следующую строку."""
        for key, label, default in fields:
            self.vars[key] = tk.StringVar(value=f"{default:g}".replace(".", ","))
            self.labels[key] = label
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(box, textvariable=self.vars[key], width=9, justify="right")
            entry.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=1)
            entry.bind("<Return>", lambda _event: self.calculate())
            row += 1
        return row

    def _build_output(self) -> None:
        out = ttk.Frame(self, padding=(0, 8, 8, 8))
        out.grid(row=0, column=1, sticky="nsew")
        out.columnconfigure(0, weight=1)
        out.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(9, 5), layout="constrained", facecolor=PANEL)
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=out)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.canvas.mpl_connect("resize_event", self._on_resize)
        toolbar = NavigationToolbar2Tk(self.canvas, out, pack_toolbar=False)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.report = tk.Text(out, height=11, wrap="word", font=(FONT, 11),
                              background=PAPER, foreground=INK, selectbackground=TERRACOTTA,
                              relief="flat", padx=12, pady=8, state="disabled")
        scroll = ttk.Scrollbar(out, orient="vertical", command=self.report.yview)
        self.report.configure(yscrollcommand=scroll.set)
        self.report.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        scroll.grid(row=2, column=1, sticky="ns", pady=(6, 0))

    def _on_option(self) -> None:
        self._update_controls()
        self.calculate()

    def _update_controls(self) -> None:
        """Показывает поля, нужные в выбранном режиме расчёта."""
        lever = self.mode.get() == "lever"
        for box, visible in ((self.projectile_box, lever), (self.spring_box, lever), (self.arm_box, lever),
                             (self.manual_box, not lever)):
            if visible:
                box.grid()
            else:
                box.grid_remove()

    def _value(self, key: str) -> float:
        return parse_number(self.vars[key].get(), self.labels[key])

    def _target_range(self) -> tuple[float, float]:
        x1, x2 = self._value("x1"), self._value("x2")
        if not 0 <= x1 <= x2:
            raise ValueError("Дальности по заданию должны удовлетворять 0 ≤ x₁ ≤ x₂.")
        return x1, x2

    def _on_close(self) -> None:
        self._stop_animation()
        self.destroy()

    # --------------------------------------------------------------- расчёт

    def calculate(self) -> bool:
        """Считывает исходные данные, выполняет расчёт и перерисовывает график."""
        self._stop_animation()
        v = self._value
        try:
            g = v("g")
            if self.mode.get() == "lever":
                catapult = ph.Catapult(
                    k=v("k"), dx_cocked=v("dx0"), dx_release=v("dx1"), m=v("m"),
                    m_arm=v("m_arm"), arm_length=v("L"), pivot_height=v("H"),
                    angle_cocked=v("a0"), angle_release=v("a1"))
                launch = ph.launch(catapult, g)
            else:
                catapult = None
                launch = ph.manual_launch(v("v0"), v("theta"), v("h0"))
            trajectory = ph.flight(launch, g)
            x1, x2 = self._target_range()
        except ValueError as err:
            messagebox.showerror("Ошибка в исходных данных", str(err), parent=self)
            return False
        self.shot = Shot(catapult, launch, trajectory, x1, x2)
        self._draw()
        self._write_report()
        return True

    # --------------------------------------------------------------- график

    def _draw(self) -> None:
        s, ax, start = self.shot, self.ax, self.shot.launch
        tr = s.trajectory
        ax.clear()
        ax.set_facecolor(PAPER)
        ax.set_title("Траектория полёта снаряда")
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        ax.grid(True, color=GRID, alpha=0.55)
        ax.set_aspect("equal")

        # Поверхность, на которой стоит катапульта, и зона попадания по заданию
        ax.axhspan(-1e3, 0, color="#efe9dd", zorder=0)
        ax.axhline(0, color="#8c7b62", lw=1)
        ax.axvspan(s.x1, s.x2, color=OLIVE, alpha=0.14, lw=0,
                   label=f"Цель  {decimal(s.x1, 2)}–{decimal(s.x2, 2)} м")
        self.path = ax.plot(tr.x, tr.y, color=TERRACOTTA, lw=2.4,
                            label=f"Расчёт  {decimal(tr.distance, 2)} м")[0]
        self.trail = ax.plot([], [], color=TERRACOTTA, lw=2.4)[0]
        self.ball = ax.plot([0], [start.h0], "o", color=TERRACOTTA, ms=7,
                           mec=PAPER, zorder=6)[0]
        if s.catapult is None:
            ax.plot([EXPERIMENTAL_RANGE], [0], "D", color=OLIVE, ms=6, zorder=6,
                    label=f"Среднее по опытам  ≈ {decimal(EXPERIMENTAL_RANGE, 2)} м")

        if tr.max_height > start.h0 + 1e-6:
            ax.plot([tr.x_at_max], [tr.max_height], ".", color=TERRACOTTA)
            ax.annotate(f"Высота ≈ {decimal(tr.max_height, 3)} м", (tr.x_at_max, tr.max_height),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
                        color=TERRACOTTA)

        # Высота точки отрыва и вектор начальной скорости
        xs = [0.0, s.x2, tr.distance]
        ys = [start.h0, tr.max_height]
        if s.catapult is None:
            xs.append(EXPERIMENTAL_RANGE)
        ax.plot([0, 0], [0, start.h0], ":", color="gray")
        arrow = 0.12 * (max(xs) - min(xs))
        th = math.radians(start.angle)
        tip = (arrow * math.cos(th), start.h0 + arrow * math.sin(th))
        ax.annotate("", xy=tip, xytext=(0, start.h0), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.25))

        self.arm = None
        if s.catapult:
            c = s.catapult
            px, py = c.pivot
            cocked = c.tip(c.angle_cocked)
            base = (min(cocked[0], px - 0.5 * c.arm_length), px + 0.3 * c.arm_length)
            ax.plot(base, [0, 0], color=PETG, lw=6, solid_capstyle="butt", zorder=3)
            ax.plot([px, px], [0, py], color=PETG, lw=4, zorder=3)
            ax.plot([px, cocked[0]], [py, cocked[1]], color=HARDWARE, lw=3, ls="--",
                    alpha=0.35, zorder=3)                                             # ложка при взводе
            self.arm = ax.plot([px, 0], [py, start.h0], color=HARDWARE, lw=4,
                               solid_capstyle="round", zorder=4)[0]                   # ложка у упора
            ax.plot([px], [py], "o", ms=7, mfc="white", mec="k", zorder=5)            # ось
            xs += base
            ys.append(py + c.arm_length)
        self.clock = ax.text(.98, .69, "", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.legend(loc="upper right", fontsize=10, facecolor=PAPER, edgecolor=GRID,
                  framealpha=1, borderpad=.7)

        pad = 0.05 * (max(xs) - min(xs))
        self._view = (min(xs) - pad, max(xs) + pad, -pad, 1.25 * max(ys) + pad)
        self._fit_view()
        self.canvas.draw_idle()

    def _fit_view(self) -> None:
        """Пределы осей под размер графика при равном масштабе по x и y.

        Свободное место добавляется сверху (а не под поверхностью) или по бокам.
        """
        if self._view is None:
            return
        x0, x1, y0, y1 = self._view
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.fig.get_layout_engine().execute(self.fig)  # размер области графика после компоновки
        pos = self.ax.get_position(original=True)
        w, h = pos.width * self.fig.bbox.width, pos.height * self.fig.bbox.height
        if w <= 0 or h <= 0:
            return
        if (x1 - x0) * h > (y1 - y0) * w:
            self.ax.set_ylim(y0, y0 + (x1 - x0) * h / w)
        else:
            extra = ((y1 - y0) * w / h - (x1 - x0)) / 2
            self.ax.set_xlim(x0 - extra, x1 + extra)

    def _on_resize(self, _event) -> None:
        self._stop_animation()  # сохранённый для анимации фон не подходит к новому размеру
        self._fit_view()

    # ------------------------------------------------------------- анимация

    def animate(self) -> None:
        """Анимация выстрела: разгон ложки, затем полёт снаряда."""
        if not self.calculate():
            return
        s = self.shot
        self._swing = 0.0
        if s.catapult:
            # Поворот ложки считаем равноускоренным: ω растёт от 0 до V₀/L
            c = s.catapult
            self._swing = (2 * math.radians(c.angle_release - c.angle_cocked)
                           * c.arm_length / s.launch.v0)
        self._duration = self._swing + s.trajectory.flight_time
        self._rate = SPEEDS[self.speed.get()]
        self.path.set_alpha(0.25)
        # Неподвижная часть рисунка сохраняется один раз, в каждом кадре
        # поверх неё перерисовываются только движущиеся объекты
        self._moving = [self.trail, self.ball, self.clock] + ([self.arm] if self.arm else [])
        for artist in self._moving:
            artist.set_animated(True)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.fig.bbox)
        self._t0 = time.perf_counter()
        self._tick()

    def _tick(self) -> None:
        s = self.shot
        t = min((time.perf_counter() - self._t0) * self._rate, self._duration)
        if t < self._swing:
            c = s.catapult
            angle = c.angle_cocked + (c.angle_release - c.angle_cocked) * (t / self._swing) ** 2
            x, y = c.tip(angle)
            self.arm.set_data([c.pivot[0], x], [c.pivot[1], y])
            self.ball.set_data([x], [y])
            self.clock.set_text("разгон снаряда…")
        else:
            if self.arm:
                c = s.catapult
                self.arm.set_data([c.pivot[0], 0], [c.pivot[1], s.launch.h0])
            tf = t - self._swing
            traj = s.trajectory
            tt = min(tf, traj.flight_time)
            x, y = np.interp(tt, traj.t, traj.x), np.interp(tt, traj.t, traj.y)
            n = np.searchsorted(traj.t, tt)
            self.trail.set_data(np.append(traj.x[:n], x), np.append(traj.y[:n], y))
            self.ball.set_data([x], [y])
            self.clock.set_text(f"t = {decimal(tf, 2)} с")
        self.canvas.restore_region(self._background)
        for artist in self._moving:
            self.ax.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)
        if t < self._duration:
            self._anim_job = self.after(15, self._tick)
        else:
            self._anim_job = None
            self._finish_animation()

    def _stop_animation(self) -> None:
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None
            self._finish_animation()

    def _finish_animation(self) -> None:
        for artist in self._moving:
            artist.set_animated(False)
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- отчёт

    def _write_report(self) -> None:
        s, st, tr = self.shot, self.shot.launch, self.shot.trajectory
        result = verdict(tr.distance, s.x1, s.x2)
        lines = [f"Целевой диапазон {decimal(s.x1, 2)}–{decimal(s.x2, 2)} м · {result}", ""]
        if s.catapult:
            lines += [
                "РАЗГОН СНАРЯДА (раздел 4) — баланс энергии",
                f"  Eпр  = k(Δx₀² − Δx₁²)/2 = {st.e_spring:.4f} Дж — энергия пружины",
                f"  Eп   = m·g·Δh           = {st.e_pot:.4f} Дж — подъём снаряда, "
                f"Δh = L(sin α₁ − sin α₀) = {st.dh:.3f} м",
                f"  Eп.л = M·g·Δh/2         = {st.e_arm_pot:.4f} Дж — подъём ложки",
                f"  Eк.л = M·V₀²/6          = {st.e_arm_kin:.4f} Дж — кинетическая энергия ложки "
                f"(теряется при ударе об упор, теорема Карно)",
                f"  Eк   = m·V₀²/2          = {st.e_kin:.4f} Дж — кинетическая энергия снаряда "
                f"({st.e_kin / st.e_spring:.0%} энергии пружины)",
                f"  V₀ = √(2(Eпр − Eп − Eп.л)/(m + M/3)) = {st.v0:.3f} м/с,   "
                f"θ = 90° − α₁ = {st.angle:.1f}°,   h₀ = H + L·sin α₁ = {st.h0:.3f} м",
            ]
        else:
            lines += [f"Средняя дальность трёх пусков ≈ {decimal(EXPERIMENTAL_RANGE, 2)} м от точки вылета."]

        a, b, c = ph.parabola(st, self._value("g"))
        lines += [
            "",
            "Расчёт полёта",
            f"Дальность {decimal(tr.distance, 3)} м  ·  время {decimal(tr.flight_time, 3)} с  ·  "
            f"максимальная высота {decimal(tr.max_height, 3)} м",
            f"y(x) = {decimal(a, 3)} {'+' if b >= 0 else '−'} {decimal(abs(b), 3)}x − {decimal(c, 3)}x²",
            f"Скорость при падении {decimal(tr.impact_speed, 2)} м/с, угол к горизонту {decimal(tr.impact_angle, 1)}°.",
        ]

        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", "\n".join(lines))
        self.report.tag_configure("summary", font=(FONT, 12, "bold"), foreground=TERRACOTTA)
        self.report.tag_add("summary", "1.0", "1.end")
        self.report.configure(state="disabled")


class PrototypeDialog(tk.Toplevel):
    """Редактирование исходных данных разделов 3–5 для сборки команды."""

    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.app = app
        self.title("Параметры разделов 3–5")
        self.transient(app)
        self.measurements = app.prototype_loads
        self.vars: dict[str, tk.StringVar] = {}
        self.labels: dict[str, str] = {}

        content = ttk.Frame(self, padding=12)
        content.pack(fill="both", expand=True)
        for title, fields in (
            ("3. Резинка", PROTOTYPE_SPRING),
            ("4. Разгон", PROTOTYPE_ARM),
            ("5. Полёт", MANUAL[1:] + GRAVITY),
        ):
            box = ttk.LabelFrame(content, text=title, padding=(8, 4))
            box.pack(fill="x", pady=(0, 5))
            box.columnconfigure(0, weight=1)
            for row, (key, label, _) in enumerate(fields):
                initial = app.prototype_values[key] if key in app.prototype_values else app.vars[key].get()
                self.vars[key] = tk.StringVar(master=self, value=initial)
                self.labels[key] = label
                ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12))
                ttk.Entry(box, textvariable=self.vars[key], width=10, justify="right").grid(
                    row=row, column=1, pady=1, sticky="e")
            if fields is PROTOTYPE_SPRING:
                ttk.Button(box, text="Рассчитать k по замерам…", command=lambda: StiffnessDialog(app, self)).grid(
                    row=len(fields), column=0, columnspan=2, sticky="ew", pady=(4, 2))

        ttk.Label(content, text="Скорость вылета рассчитывается по балансу энергии.").pack(anchor="w", pady=(2, 6))
        ttk.Button(content, text="Применить и рассчитать", style="Run.TButton", command=self._apply).pack(fill="x")
        self.bind("<Escape>", lambda _event: self.destroy())

    def _apply(self) -> None:
        app = self.app
        try:
            values = {key: parse_number(var.get(), self.labels[key]) for key, var in self.vars.items()}
            prototype = ph.Prototype(**{key: values[key] for key in app.prototype_values})
            speed = ph.prototype_speed(prototype, values["g"])
            entered_speed = f"{speed:.3g}".replace(".", ",")
            ph.manual_launch(parse_number(entered_speed, "Скорость вылета"), values["theta"], values["h0"])
            app._target_range()
        except ValueError as err:
            messagebox.showerror("Ошибка в исходных данных", str(err), parent=self)
            return

        app.prototype_values = {key: self.vars[key].get() for key in app.prototype_values}
        app.prototype_loads = self.measurements
        app.vars["v0"].set(entered_speed)
        for key in ("theta", "h0", "g"):
            app.vars[key].set(self.vars[key].get())
        app.mode.set("manual")
        app._on_option()
        self.destroy()


class StiffnessDialog(tk.Toplevel):
    """Раздел 3: коэффициент жёсткости пружины по опыту с подвешенными грузами."""

    def __init__(self, app: App, prototype: PrototypeDialog | None = None) -> None:
        parent = prototype if prototype is not None else app
        super().__init__(parent)
        self.app = app
        self.prototype = prototype
        self.title("Жёсткость упругого элемента (раздел 3)")
        self.transient(parent)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Длина без груза l₀, м").grid(row=0, column=0, sticky="w")
        self.l0 = tk.StringVar(master=self, value=prototype.vars["l0"].get() if prototype else "")
        ttk.Entry(frame, textvariable=self.l0, width=12, justify="right").grid(
            row=0, column=1, sticky="w", padx=6, pady=(0, 6))

        self.table = ttk.Frame(frame)
        self.table.grid(row=1, column=0, columnspan=2, sticky="w")
        for col, text in enumerate(("№", "Масса груза m, кг", "Длина пружины l, м")):
            ttk.Label(self.table, text=text).grid(row=0, column=col, padx=4)
        self.rows: list[tuple[tk.StringVar, tk.StringVar]] = []
        loads = prototype.measurements if prototype else ()
        for _ in range(max(5, len(loads))):
            self._add_row()
        for (mass, length), (m, l) in zip(self.rows, loads):
            mass.set(m)
            length.set(l)

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(buttons, text="+ строка", command=self._add_row).pack(side="left")
        ttk.Button(buttons, text="Рассчитать", command=self._compute).pack(side="left", padx=4)
        ttk.Button(buttons, text="Подставить среднее k", command=self._apply).pack(side="left")

        self.out = tk.Text(frame, height=11, width=56, font=("Consolas", 10), state="disabled")
        self.out.grid(row=3, column=0, columnspan=2, pady=(8, 0))

    def _add_row(self) -> None:
        i = len(self.rows) + 1
        m, l = tk.StringVar(), tk.StringVar()
        ttk.Label(self.table, text=str(i)).grid(row=i, column=0)
        for col, var in ((1, m), (2, l)):
            ttk.Entry(self.table, textvariable=var, width=14, justify="right").grid(
                row=i, column=col, padx=4, pady=1)
        self.rows.append((m, l))

    def _compute(self) -> ph.Stiffness | None:
        try:
            l0 = parse_number(self.l0.get(), "Длина пружины без груза l₀")
            masses, lengths = [], []
            for i, (m, l) in enumerate(self.rows, start=1):
                if not m.get().strip() and not l.get().strip():
                    continue  # пустые строки таблицы пропускаем
                masses.append(parse_number(m.get(), f"Масса груза №{i}"))
                lengths.append(parse_number(l.get(), f"Длина пружины №{i}"))
            if self.prototype:
                g = parse_number(self.prototype.vars["g"].get(), self.prototype.labels["g"])
            else:
                try:
                    g = self.app._value("g")
                except ValueError:
                    g = ph.G
            if g <= 0:
                raise ValueError("Ускорение свободного падения g должно быть положительным.")
            res = ph.spring_stiffness(l0, masses, lengths, g)
        except ValueError as err:
            messagebox.showerror("Ошибка", str(err), parent=self)
            return None

        lines = [f"{'№':>2} {'m, кг':>9} {'Δl, м':>9} {'F = mg, Н':>10} {'k = F/Δl, Н/м':>14}"]
        for i, (m, dl, f, k) in enumerate(zip(masses, res.elongations, res.forces, res.k_values), 1):
            lines.append(f"{i:>2} {m:>9.4g} {dl:>9.4f} {f:>10.4f} {k:>14.2f}")
        lines += ["",
                  f"Среднее значение   k = {res.k_mean:.2f} Н/м",
                  f"СКО измерений      σ = {res.k_std:.2f} Н/м",
                  f"МНК (F = k·Δl)     k = {res.k_lsq:.2f} Н/м"]
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", "\n".join(lines))
        self.out.configure(state="disabled")
        return res

    def _apply(self) -> None:
        res = self._compute()
        if not res:
            return
        if self.prototype:
            self.prototype.vars["k"].set(f"{res.k_mean:.3g}".replace(".", ","))
            self.prototype.vars["l0"].set(self.l0.get())
            self.prototype.measurements = tuple((m.get(), l.get()) for m, l in self.rows if m.get() or l.get())
            self.destroy()
        else:
            self.app.vars["k"].set(f"{res.k_mean:.4g}")
            self.destroy()
            self.app.calculate()


def main() -> None:
    app = App()
    try:
        app.state("zoomed")  # развернуть на весь экран (Windows)
    except tk.TclError:
        pass
    app.mainloop()


if __name__ == "__main__":
    main()
