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


@dataclass
class Prototype:
    """Исходные данные для ложки команды с осью внутри стержня."""
    k: float
    l0: float
    l_cocked: float
    l_release: float
    m: float
    m_arm: float
    radius: float
    spoon_length: float
    axis_offset: float
    angle_cocked: float
    angle_release: float


def prototype_speed(p: Prototype, g: float = G) -> float:
    """Скорость вылета по балансу энергии из раздела 4 записки."""
    _require(p.k > 0 and p.l0 > 0, "Жёсткость и свободная длина резинки должны быть положительными.")
    _require(p.l0 <= p.l_release < p.l_cocked,
             "Длины резинки должны удовлетворять l₀ ≤ l₁ < l при взводе.")
    _require(p.m > 0 and p.m_arm > 0, "Массы снаряда и ложки должны быть положительными.")
    _require(0 < p.radius <= p.spoon_length, "Расстояние до снаряда должно быть больше нуля и не длиннее ложки.")
    _require(0 <= p.axis_offset <= p.spoon_length / 2, "Смещение оси должно быть в пределах половины длины ложки.")
    _require(-90 <= p.angle_cocked < p.angle_release <= 90,
             "Углы ложки должны удовлетворять −90° ≤ α₀ < α₁ ≤ 90°.")
    _require(g > 0, "Ускорение свободного падения g должно быть положительным.")

    dx_cocked = p.l_cocked - p.l0
    dx_release = p.l_release - p.l0
    energy = p.k * (dx_cocked**2 - dx_release**2) / 2
    delta_sine = math.sin(math.radians(p.angle_release)) - math.sin(math.radians(p.angle_cocked))
    lift = g * (p.m * p.radius + p.m_arm * p.axis_offset) * delta_sine
    inertia = p.m_arm * (p.spoon_length**2 / 12 + p.axis_offset**2)
    _require(energy > lift, "Энергии резинки не хватает на подъём снаряда и ложки.")
    return math.sqrt(2 * (energy - lift) / (p.m + inertia / p.radius**2))


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
