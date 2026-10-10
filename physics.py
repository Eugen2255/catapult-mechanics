"""Расчётная часть электронной модели катапульты.

Раздел 3 — коэффициент жёсткости пружины по закону Гука;
раздел 4 — скорость снаряда в момент отрыва от ложки (баланс энергии);
раздел 5 — траектория полёта снаряда (уравнения кинематики).

Все величины — в СИ (м, кг, с, Н, Дж), углы — в градусах.
Система координат: ось x направлена по направлению выстрела, начало отсчёта
лежит на поверхности (стол/пол) под точкой отрыва снаряда, ось y — вверх.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.8  # ускорение свободного падения, м/с² (как в методическом пособии)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Раздел 3. Экспериментальное определение коэффициента жёсткости
# ---------------------------------------------------------------------------

@dataclass
class Stiffness:
    elongations: list[float]  # удлинения Δl_i = l_i − l₀, м
    forces: list[float]       # силы тяжести грузов F_i = m_i·g, Н
    k_values: list[float]     # k_i = F_i / Δl_i, Н/м
    k_mean: float             # среднее значение k, Н/м
    k_std: float              # среднеквадратичное отклонение k_i, Н/м
    k_lsq: float              # k по методу наименьших квадратов для F = k·Δl, Н/м


def spring_stiffness(l0: float, masses: list[float], lengths: list[float],
                     g: float = G) -> Stiffness:
    """Закон Гука F = k·Δl: k_i = m_i·g / (l_i − l₀) и среднее по всем грузам."""
    _require(len(masses) > 0 and len(masses) == len(lengths),
             "Нужно хотя бы одно измерение: масса груза и длина пружины.")
    elongations, forces, k_values = [], [], []
    for i, (m, l) in enumerate(zip(masses, lengths), start=1):
        dl = l - l0
        _require(m > 0, f"Измерение {i}: масса груза должна быть положительной.")
        _require(dl > 0, f"Измерение {i}: длина l должна быть больше l₀.")
        elongations.append(dl)
        forces.append(m * g)
        k_values.append(m * g / dl)
    n = len(k_values)
    k_mean = sum(k_values) / n
    k_std = math.sqrt(sum((k - k_mean) ** 2 for k in k_values) / (n - 1)) if n > 1 else 0.0
    k_lsq = (sum(f * dl for f, dl in zip(forces, elongations))
             / sum(dl * dl for dl in elongations))
    return Stiffness(elongations, forces, k_values, k_mean, k_std, k_lsq)


# ---------------------------------------------------------------------------
# Раздел 4. Динамический анализ: разгон снаряда
# ---------------------------------------------------------------------------

@dataclass
class Catapult:
    """Рычажная катапульта: ложка — однородный стержень, вращающийся вокруг оси.

    Углы ложки отсчитываются от горизонтали в сторону, противоположную
    выстрелу: 0° — ложка лежит горизонтально позади оси, 90° — стоит
    вертикально. Снаряд отрывается, когда ложка ударяется об упор (угол α₁).
    """
    k: float              # жёсткость пружины, Н/м
    dx_cocked: float      # растяжение пружины во взведённом состоянии Δx₀, м
    dx_release: float     # растяжение пружины в момент отрыва снаряда Δx₁, м
    m: float              # масса снаряда, кг
    m_arm: float          # масса ложки M, кг
    arm_length: float     # расстояние от оси до центра снаряда L, м
    pivot_height: float   # высота оси над поверхностью H, м
    angle_cocked: float   # угол ложки во взведённом положении α₀, °
    angle_release: float  # угол ложки у упора α₁, °

    @property
    def pivot(self) -> tuple[float, float]:
        """Координаты оси вращения ложки."""
        return self.arm_length * math.cos(math.radians(self.angle_release)), self.pivot_height

    def tip(self, angle: float) -> tuple[float, float]:
        """Координаты снаряда на конце ложки при угле ложки angle."""
        px, py = self.pivot
        a = math.radians(angle)
        return px - self.arm_length * math.cos(a), py + self.arm_length * math.sin(a)


@dataclass
class Launch:
    """Состояние снаряда в момент отрыва от ложки."""
    v0: float     # скорость, м/с
    angle: float  # угол вылета к горизонту θ, °
    h0: float     # высота над поверхностью, м
    # Баланс энергии (заполняется только при расчёте по параметрам катапульты), Дж
    e_spring: float = 0.0   # энергия деформированной пружины
    dh: float = 0.0         # подъём снаряда за время разгона, м
    e_pot: float = 0.0      # потенциальная энергия подъёма снаряда
    e_arm_pot: float = 0.0  # потенциальная энергия подъёма ложки
    e_arm_kin: float = 0.0  # кинетическая энергия ложки (теряется при ударе об упор)
    e_kin: float = 0.0      # кинетическая энергия снаряда


def launch(c: Catapult, g: float = G) -> Launch:
    """Скорость снаряда в момент отрыва из баланса энергии.

    k(Δx₀² − Δx₁²)/2 = m·g·Δh + M·g·Δh/2 + m·V₀²/2 + I·ω²/2,

    где Δh = L(sin α₁ − sin α₀) — подъём снаряда (центр масс ложки поднимается
    на Δh/2), I = M·L²/3 — момент инерции ложки относительно оси, ω = V₀/L.
    Кинетическая энергия ложки I·ω²/2 = M·V₀²/6 по теореме Карно теряется
    при неупругом ударе ложки об упор. Снаряд летит по касательной к
    окружности, которую описывает конец ложки, т. е. θ = 90° − α₁.
    """
    _require(c.k > 0, "Жёсткость пружины k должна быть положительной.")
    _require(c.dx_cocked > 0, "Растяжение пружины при взводе Δx₀ должно быть положительным.")
    _require(0 <= c.dx_release < c.dx_cocked,
             "Растяжение пружины при отрыве Δx₁ должно быть в пределах [0; Δx₀).")
    _require(c.m > 0, "Масса снаряда m должна быть положительной.")
    _require(c.m_arm >= 0, "Масса ложки M не может быть отрицательной.")
    _require(c.arm_length > 0, "Длина ложки L должна быть положительной.")
    _require(c.pivot_height >= 0, "Высота оси H не может быть отрицательной.")
    _require(0 < c.angle_release < 180, "Угол ложки у упора α₁ должен быть в пределах (0°; 180°).")
    _require(-90 <= c.angle_cocked < c.angle_release,
             "Угол ложки при взводе α₀ должен быть меньше α₁ (и не меньше −90°).")
    _require(g > 0, "Ускорение свободного падения g должно быть положительным.")

    a0, a1 = math.radians(c.angle_cocked), math.radians(c.angle_release)
    e_spring = c.k * (c.dx_cocked ** 2 - c.dx_release ** 2) / 2
    dh = c.arm_length * (math.sin(a1) - math.sin(a0))
    e_pot = c.m * g * dh
    e_arm_pot = c.m_arm * g * dh / 2
    e_free = e_spring - e_pot - e_arm_pot
    _require(e_free > 0, f"Энергии пружины ({e_spring:.4f} Дж) не хватает на подъём "
                         f"снаряда и ложки ({e_pot + e_arm_pot:.4f} Дж).")
    v0 = math.sqrt(2 * e_free / (c.m + c.m_arm / 3))
    return Launch(
        v0=v0, angle=90 - c.angle_release, h0=c.tip(c.angle_release)[1],
        e_spring=e_spring, dh=dh, e_pot=e_pot, e_arm_pot=e_arm_pot,
        e_arm_kin=c.m_arm * v0 ** 2 / 6, e_kin=c.m * v0 ** 2 / 2,
    )


def manual_launch(v0: float, angle: float, h0: float) -> Launch:
    """Параметры отрыва, заданные напрямую (например, измеренные на модели)."""
    _require(v0 > 0, "Скорость отрыва V₀ должна быть положительной.")
    _require(-90 < angle < 90, "Угол вылета θ должен быть в пределах (−90°; 90°).")
    _require(h0 >= 0, "Высота точки отрыва h₀ не может быть отрицательной.")
    _require(h0 > 0 or angle > 0, "При h₀ = 0 угол вылета θ должен быть больше нуля.")
    return Launch(v0, angle, h0)


# ---------------------------------------------------------------------------
# Раздел 5. Кинематический анализ: траектория полёта
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    t: np.ndarray        # моменты времени, с
    x: np.ndarray        # координаты снаряда, м
    y: np.ndarray
    flight_time: float   # время полёта, с
    distance: float      # дальность полёта от точки отрыва, м
    max_height: float    # наибольшая высота над поверхностью, м
    x_at_max: float      # координата x наивысшей точки, м
    impact_speed: float  # скорость в момент падения, м/с
    impact_angle: float  # угол падения к горизонту, °


def flight(start: Launch, g: float = G, n: int = 400) -> Trajectory:
    """Полёт без сопротивления воздуха — движение по параболе.

    x(t) = V₀·cosθ·t,   y(t) = h₀ + V₀·sinθ·t − g·t²/2.
    Время полёта — положительный корень уравнения y(t) = 0:
    t = (V₀·sinθ + √(V₀²·sin²θ + 2·g·h₀)) / g.
    """
    _require(g > 0, "Ускорение свободного падения g должно быть положительным.")
    th = math.radians(start.angle)
    vx, vy = start.v0 * math.cos(th), start.v0 * math.sin(th)
    t_flight = (vy + math.sqrt(vy ** 2 + 2 * g * start.h0)) / g
    t = np.linspace(0.0, t_flight, n)
    x = vx * t
    y = start.h0 + vy * t - g * t ** 2 / 2
    y[-1] = 0.0  # убираем погрешность округления в точке падения
    t_top = max(vy / g, 0.0)  # момент прохождения вершины (0, если бросок вниз)
    vy_end = vy - g * t_flight
    return Trajectory(
        t=t, x=x, y=y, flight_time=t_flight, distance=vx * t_flight,
        max_height=start.h0 + vy * t_top - g * t_top ** 2 / 2, x_at_max=vx * t_top,
        impact_speed=math.hypot(vx, vy_end),
        impact_angle=math.degrees(math.atan2(-vy_end, vx)),
    )


def parabola(start: Launch, g: float = G) -> tuple[float, float, float]:
    """Коэффициенты уравнения траектории y(x) = a + b·x − c·x²."""
    th = math.radians(start.angle)
    return start.h0, math.tan(th), g / (2 * start.v0 ** 2 * math.cos(th) ** 2)


def flight_with_drag(start: Launch, m: float, diameter: float, cd: float = 0.47,
                     rho: float = 1.2, g: float = G, dt: float = 1e-3) -> Trajectory:
    """Полёт с квадратичным сопротивлением воздуха F = ½·ρ·Cx·S·v², S = π·d²/4.

    Уравнения движения dvx/dt = −β·v·vx,  dvy/dt = −g − β·v·vy,  β = ρ·Cx·S/(2m)
    интегрируются численно методом Рунге — Кутты 4-го порядка с шагом dt.
    """
    _require(g > 0, "Ускорение свободного падения g должно быть положительным.")
    _require(m > 0, "Масса снаряда m должна быть положительной.")
    _require(diameter > 0, "Диаметр снаряда d должен быть положительным.")
    _require(cd >= 0, "Коэффициент сопротивления Cx не может быть отрицательным.")
    _require(rho >= 0, "Плотность воздуха ρ не может быть отрицательной.")
    beta = rho * cd * math.pi * diameter ** 2 / 4 / (2 * m)

    def deriv(s: tuple[float, ...]) -> tuple[float, ...]:
        _, _, vx, vy = s
        v = math.hypot(vx, vy)
        return vx, vy, -beta * v * vx, -g - beta * v * vy

    def shift(s: tuple[float, ...], d: tuple[float, ...], h: float) -> tuple[float, ...]:
        return tuple(si + h * di for si, di in zip(s, d))

    th = math.radians(start.angle)
    s = (0.0, start.h0, start.v0 * math.cos(th), start.v0 * math.sin(th))  # x, y, vx, vy
    t = 0.0
    ts, xs, ys = [t], [s[0]], [s[1]]
    max_height, x_at_max = s[1], s[0]
    while True:
        k1 = deriv(s)
        k2 = deriv(shift(s, k1, dt / 2))
        k3 = deriv(shift(s, k2, dt / 2))
        k4 = deriv(shift(s, k3, dt))
        new = tuple(si + dt / 6 * (a + 2 * b + 2 * c + d)
                    for si, a, b, c, d in zip(s, k1, k2, k3, k4))
        if new[1] <= 0:
            # Точку касания поверхности находим линейной интерполяцией внутри шага
            frac = s[1] / (s[1] - new[1])
            s = tuple(si + frac * (ni - si) for si, ni in zip(s, new))
            t += frac * dt
            ts.append(t)
            xs.append(s[0])
            ys.append(0.0)
            break
        s, t = new, t + dt
        ts.append(t)
        xs.append(s[0])
        ys.append(s[1])
        if s[1] > max_height:
            max_height, x_at_max = s[1], s[0]
        _require(t < 600, "Снаряд не приземлился за 10 минут полёта — проверьте исходные данные.")
    return Trajectory(
        t=np.array(ts), x=np.array(xs), y=np.array(ys), flight_time=t, distance=s[0],
        max_height=max_height, x_at_max=x_at_max,
        impact_speed=math.hypot(s[2], s[3]),
        impact_angle=math.degrees(math.atan2(-s[3], s[2])),
    )
