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
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import physics as ph

# Поля ввода: (ключ, подпись, значение по умолчанию)
PROJECTILE = [("m", "Масса снаряда m, кг", 0.010),
              ("d", "Диаметр снаряда d, м", 0.04)]
SPRING = [("k", "Жёсткость пружины k, Н/м", 60.0),
          ("dx0", "Растяжение при взводе Δx₀, м", 0.08),
          ("dx1", "Растяжение при отрыве Δx₁, м", 0.02)]
ARM = [("m_arm", "Масса ложки M, кг", 0.020),
       ("L", "Длина ложки (ось → снаряд) L, м", 0.25),
       ("H", "Высота оси над столом H, м", 0.10),
       ("a0", "Угол ложки при взводе α₀, °", 10.0),
       ("a1", "Угол ложки у упора α₁, °", 45.0)]
MANUAL = [("v0", "Скорость отрыва V₀, м/с", 4.0),
          ("theta", "Угол вылета θ, °", 45.0),
          ("h0", "Высота точки отрыва h₀, м", 0.30)]
GRAVITY = [("g", "Ускорение свободного падения g, м/с²", 9.8)]
AIR = [("cd", "Коэффициент сопротивления Cx", 0.47),
       ("rho", "Плотность воздуха ρ, кг/м³", 1.2)]
TASK = [("x1", "Минимальная дальность x₁, м", 2.0),
        ("x2", "Максимальная дальность x₂, м", 3.0)]

SPEEDS = {"×1": 1.0, "×0.5": 0.5, "×0.25": 0.25, "×0.1": 0.1}
BLUE, RED, GREEN, WOOD, DARK_WOOD = "#1f77b4", "#d62728", "#2ca02c", "#a0703c", "#5c3a1a"


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
        return f"недолёт {x1 - x:.2f} м"
    if x > x2:
        return f"перелёт {x - x2:.2f} м"
    return "попадание в зону"


@dataclass
class Shot:
    """Результат одного расчёта."""
    catapult: ph.Catapult | None  # None — параметры отрыва заданы вручную
    launch: ph.Launch
    ideal: ph.Trajectory          # без сопротивления воздуха
    drag: ph.Trajectory | None    # с сопротивлением воздуха
    x1: float
    x2: float

    @property
    def trajectories(self) -> list[ph.Trajectory]:
        return [self.ideal] + ([self.drag] if self.drag else [])


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Катапульта — расчёт траектории снаряда")
        self.minsize(1000, 650)
        self.vars: dict[str, tk.StringVar] = {}
        self.labels: dict[str, str] = {}
        self.entries: dict[str, ttk.Entry] = {}
        self.mode = tk.StringVar(value="lever")
        self.use_drag = tk.BooleanVar(value=True)
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

    def _build_inputs(self) -> None:
        self._panel = ttk.Frame(self, padding=8)
        self._panel.grid(row=0, column=0, sticky="ns")
        self._row = 0

        box = self._box("Скорость снаряда при отрыве")
        for row, (text, value) in enumerate([("рассчитать по параметрам катапульты", "lever"),
                                             ("задать вручную", "manual")]):
            ttk.Radiobutton(box, text=text, variable=self.mode, value=value,
                            command=self._on_option).grid(row=row, column=0, columnspan=2, sticky="w")

        self._fields(self._box("Снаряд"), PROJECTILE)

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
        self._fields(self.manual_box, MANUAL)

        box = self._box("Полёт (раздел 5)")
        self._fields(box, GRAVITY)
        ttk.Checkbutton(box, text="учитывать сопротивление воздуха", variable=self.use_drag,
                        command=self._on_option).grid(row=1, column=0, columnspan=2, sticky="w")
        self._fields(box, AIR, row=2)

        self._fields(self._box("Задание преподавателя"), TASK)

        buttons = ttk.Frame(self._panel)
        buttons.grid(row=self._row, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Рассчитать", command=self.calculate).grid(
            row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(buttons, text="▶ Пуск", command=self.animate).grid(
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
            self.vars[key] = tk.StringVar(value=f"{default:g}")
            self.labels[key] = label
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(box, textvariable=self.vars[key], width=9, justify="right")
            entry.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=1)
            entry.bind("<Return>", lambda _event: self.calculate())
            self.entries[key] = entry
            row += 1
        return row

    def _build_output(self) -> None:
        out = ttk.Frame(self, padding=(0, 8, 8, 8))
        out.grid(row=0, column=1, sticky="nsew")
        out.columnconfigure(0, weight=1)
        out.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(9, 5), layout="constrained")
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=out)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.canvas.mpl_connect("resize_event", self._on_resize)
        toolbar = NavigationToolbar2Tk(self.canvas, out, pack_toolbar=False)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.report = tk.Text(out, height=17, wrap="none", font=("Consolas", 10),
                              relief="flat", padx=6, pady=4, state="disabled")
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
        for box, visible in ((self.spring_box, lever), (self.arm_box, lever),
                             (self.manual_box, not lever)):
            if visible:
                box.grid()
            else:
                box.grid_remove()
        state = ["!disabled"] if self.use_drag.get() else ["disabled"]
        for key, *_ in AIR:
            self.entries[key].state(state)

    def _value(self, key: str) -> float:
        return parse_number(self.vars[key].get(), self.labels[key])

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
            ideal = ph.flight(launch, g)
            drag = (ph.flight_with_drag(launch, v("m"), v("d"), v("cd"), v("rho"), g)
                    if self.use_drag.get() else None)
            x1, x2 = v("x1"), v("x2")
            if not 0 <= x1 <= x2:
                raise ValueError("Дальности по заданию должны удовлетворять 0 ≤ x₁ ≤ x₂.")
        except ValueError as err:
            messagebox.showerror("Ошибка в исходных данных", str(err), parent=self)
            return False
        self.shot = Shot(catapult, launch, ideal, drag, x1, x2)
        self._draw()
        self._write_report()
        return True

    # --------------------------------------------------------------- график

    def _draw(self) -> None:
        s, ax, start = self.shot, self.ax, self.shot.launch
        ax.clear()
        ax.set_title("Траектория полёта снаряда")
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Поверхность, на которой стоит катапульта, и зона попадания по заданию
        ax.axhspan(-1e3, 0, color="#efe9dd", zorder=0)
        ax.axhline(0, color="#8c7b62", lw=1)
        ax.axvspan(s.x1, s.x2, color=GREEN, alpha=0.12, lw=0,
                   label=f"зона попадания {s.x1:g}–{s.x2:g} м")

        styles = [(BLUE, "-", "без сопротивления воздуха"), (RED, "--", "с сопротивлением воздуха")]
        self.paths, self.trails, self.balls = [], [], []
        for traj, (color, ls, name) in zip(s.trajectories, styles):
            self.paths.append(ax.plot(traj.x, traj.y, color=color, ls=ls, lw=2,
                                      label=f"{name}: x = {traj.distance:.2f} м")[0])
            self.trails.append(ax.plot([], [], color=color, ls=ls, lw=2.2)[0])
            self.balls.append(ax.plot([0], [start.h0], "o", color=color, ms=9, mec="k", zorder=6)[0])

        top = s.ideal
        if top.max_height > start.h0 + 1e-6:
            ax.plot([top.x_at_max], [top.max_height], ".", color=BLUE)
            ax.annotate(f"y_max = {top.max_height:.2f} м", (top.x_at_max, top.max_height),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
                        color=BLUE)

        # Высота точки отрыва и вектор начальной скорости
        xs = [0.0, s.x2] + [t.distance for t in s.trajectories]
        ys = [start.h0] + [t.max_height for t in s.trajectories]
        ax.plot([0, 0], [0, start.h0], ":", color="gray")
        arrow = 0.12 * (max(xs) - min(xs))
        th = math.radians(start.angle)
        tip = (arrow * math.cos(th), start.h0 + arrow * math.sin(th))
        ax.annotate("", xy=tip, xytext=(0, start.h0), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
        ax.annotate(f"V₀ = {start.v0:.2f} м/с\nθ = {start.angle:.1f}°\nh₀ = {start.h0:.2f} м",
                    xy=tip, xytext=(-6, 6), textcoords="offset points", ha="right",
                    va="bottom", fontsize=9)

        self.arm = None
        if s.catapult:
            c = s.catapult
            px, py = c.pivot
            cocked = c.tip(c.angle_cocked)
            base = (min(cocked[0], px - 0.5 * c.arm_length), px + 0.3 * c.arm_length)
            ax.plot(base, [0, 0], color=WOOD, lw=6, solid_capstyle="butt", zorder=3)  # основание
            ax.plot([px, px], [0, py], color=WOOD, lw=4, zorder=3)                    # стойка
            ax.plot([px, cocked[0]], [py, cocked[1]], color=DARK_WOOD, lw=3, ls="--",
                    alpha=0.35, zorder=3)                                             # ложка при взводе
            self.arm = ax.plot([px, 0], [py, start.h0], color=DARK_WOOD, lw=4,
                               solid_capstyle="round", zorder=4)[0]                   # ложка у упора
            ax.plot([px], [py], "o", ms=7, mfc="white", mec="k", zorder=5)            # ось
            xs += base
            ys.append(py + c.arm_length)

        self.clock = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

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
        self._duration = self._swing + max(t.flight_time for t in s.trajectories)
        self._rate = SPEEDS[self.speed.get()]
        for path in self.paths:
            path.set_alpha(0.25)
        # Неподвижная часть рисунка сохраняется один раз, в каждом кадре
        # поверх неё перерисовываются только движущиеся объекты
        self._moving = [*self.trails, *self.balls, self.clock] + ([self.arm] if self.arm else [])
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
            for ball in self.balls:
                ball.set_data([x], [y])
            self.clock.set_text("разгон снаряда…")
        else:
            if self.arm:
                c = s.catapult
                self.arm.set_data([c.pivot[0], 0], [c.pivot[1], s.launch.h0])
            tf = t - self._swing
            for traj, trail, ball in zip(s.trajectories, self.trails, self.balls):
                tt = min(tf, traj.flight_time)
                x, y = np.interp(tt, traj.t, traj.x), np.interp(tt, traj.t, traj.y)
                n = np.searchsorted(traj.t, tt)
                trail.set_data(np.append(traj.x[:n], x), np.append(traj.y[:n], y))
                ball.set_data([x], [y])
            self.clock.set_text(f"t = {tf:.2f} с")
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
        s, st, tr = self.shot, self.shot.launch, self.shot.ideal
        result = f"без сопротивления воздуха — {verdict(tr.distance, s.x1, s.x2)}"
        if s.drag:
            result += f";   с сопротивлением — {verdict(s.drag.distance, s.x1, s.x2)}"
        lines = [f"ЗАДАНИЕ  x₁ = {s.x1:g} м, x₂ = {s.x2:g} м:   {result}", ""]
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
            lines += ["ПАРАМЕТРЫ ОТРЫВА (заданы вручную)",
                      f"  V₀ = {st.v0:.3f} м/с,   θ = {st.angle:.1f}°,   h₀ = {st.h0:.3f} м"]

        a, b, c = ph.parabola(st, self._value("g"))
        lines += [
            "",
            "ПОЛЁТ БЕЗ СОПРОТИВЛЕНИЯ ВОЗДУХА (раздел 5)",
            f"  траектория — парабола:  y(x) = {a:.3f} {'+' if b >= 0 else '−'} {abs(b):.3f}·x "
            f"− {c:.3f}·x²",
            f"  время полёта   t = (V₀·sinθ + √(V₀²·sin²θ + 2·g·h₀))/g = {tr.flight_time:.3f} с",
            f"  дальность      x = V₀·cosθ·t = {tr.distance:.3f} м",
            f"  наивысшая точка  y_max = {tr.max_height:.3f} м  при x = {tr.x_at_max:.3f} м",
            f"  скорость в момент падения {tr.impact_speed:.2f} м/с под углом "
            f"{tr.impact_angle:.1f}° к горизонту",
        ]
        if s.drag:
            d = s.drag
            lines += [
                "",
                "ПОЛЁТ С СОПРОТИВЛЕНИЕМ ВОЗДУХА (F = ½·ρ·Cx·S·v², численный расчёт)",
                f"  t = {d.flight_time:.3f} с,   x = {d.distance:.3f} м "
                f"({(d.distance - tr.distance) / tr.distance:+.1%} к идеальному),   "
                f"y_max = {d.max_height:.3f} м",
            ]

        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", "\n".join(lines))
        self.report.configure(state="disabled")


class StiffnessDialog(tk.Toplevel):
    """Раздел 3: коэффициент жёсткости пружины по опыту с подвешенными грузами."""

    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.app = app
        self.title("Жёсткость пружины (раздел 3)")
        self.transient(app)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Длина пружины без груза l₀, м").grid(row=0, column=0, sticky="w")
        self.l0 = tk.StringVar()
        ttk.Entry(frame, textvariable=self.l0, width=12, justify="right").grid(
            row=0, column=1, sticky="w", padx=6, pady=(0, 6))

        self.table = ttk.Frame(frame)
        self.table.grid(row=1, column=0, columnspan=2, sticky="w")
        for col, text in enumerate(("№", "Масса груза m, кг", "Длина пружины l, м")):
            ttk.Label(self.table, text=text).grid(row=0, column=col, padx=4)
        self.rows: list[tuple[tk.StringVar, tk.StringVar]] = []
        for _ in range(5):
            self._add_row()

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
            try:
                g = self.app._value("g")
            except ValueError:
                g = ph.G
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
        if res:
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
