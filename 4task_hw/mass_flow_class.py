import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from iapws import IAPWS97
import numpy as np
import pandas as pd 



def r(x):
    a = (round(x))
    return a 

MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None

class mass_flow_class():
    """
        Класс для расчёта расхода в конденсатор / турбину
        Атрибуты:
            shum: class, - класс для расчёта параметров состояния
        Методы:
            condenser_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,
        mechanical_efficiency,generator_efficiency) - рассчитывает расход на входе в конденсатор 
            inlet_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,
        t_feed_water,electrical_power,mechanical_efficiency,generator_efficiency) - рассчитывает расход на входе в турбину 
            efficiency(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha) - рассчитывает КПД
    """
    def __init__(self,shum, p0,t0,p_middle,t_middle,pk,t_feed_water,p_feed_water,electrical_power,
                 internal_efficiency,mechanical_efficiency,generator_efficiency,alpha):
        self.shum = shum
        self.p0 = p0 
        self.t0 = t0 
        self.p_middle = p_middle
        self.t_middle = t_middle 
        self.pk = pk
        self.t_feed_water = t_feed_water
        self.p_feed_water = p_feed_water

        self.electrical_power = electrical_power
        self.internal_efficiency = internal_efficiency
        self.mechanical_efficiency = mechanical_efficiency
        self.generator_efficiency = generator_efficiency
        self.alpha = 0.83
        self.delta_p0 = delta_p0 = 0.05 * self.p0
        self.delta_p_middle = 0.1 * self.p_middle
        self.delta_p_1 = 0.03 * self.p_middle
        
    @property
    def point_0t(self):
        point_0t = self.shum.calc_point(p= self.p0 * unit, t =self.t0)
        return point_0t
        
    @property
    def point_0(self):
        real_p0 = self.p0 - self.delta_p0
        point_0 = self.shum.calc_point(p=real_p0 * unit, h=self.point_0t.h)
        return point_0
    
    @property
    def point_1t(self):
        point_1t = self.shum.calc_point(p=self.p_middle * unit, s=self.point_0t.s)
        return point_1t
    
    @property
    def point_1(self):
        real_p1t = self.p_middle + self.delta_p_middle
        h_1 = self.point_0.h - self.hp_heat_drop
        point_1 = self.shum.calc_point(p=real_p1t * unit, h=h_1)
        return point_1
    
    @property
    def hp_heat_drop(self):
        hp_heat_drop = (self.point_0t.h - self.point_1t.h) * self.internal_efficiency
        return hp_heat_drop
    
    @property
    def point_middle_t(self):
        point_middle_t = self.shum.calc_point(p=self.p_middle * unit, t=self.t_middle)
        return point_middle_t 
    
    @property
    def point_middle(self):
        real_p_middle = self.p_middle - self.delta_p_1
        point_middle = self.shum.calc_point(p=real_p_middle * unit, h=self.point_middle_t.h)
        return point_middle
    
    @property
    def point_2t(self):
        point_2t = self.shum.calc_point(p=self.pk * unit, s=self.point_middle_t.s)
        return point_2t
    
    @property
    def point_2(self):
        h_2 = self.point_middle.h - self.lp_heat_drop
        point_2 = self.shum.calc_point(p=self.pk * unit, h=h_2)
        return point_2
    
    @property
    def lp_heat_drop(self):
        lp_heat_drop = (self.point_middle_t.h - self.point_2t.h) * self.internal_efficiency
        return lp_heat_drop
    
    @property 
    def efficiency_hp(self):
        efficiency_hp = (self.point_0t.h - self.point_1.h) / (self.point_0t.h - self.point_1t.h)
        return efficiency_hp
    
    @property
    def efficiency_lp(self):
        efficiency_lp = (self.point_middle_t.h - self.point_2.h) / (self.point_middle_t.h - self.point_2t.h)
        return efficiency_lp
    
    @property
    def point_k_water(self):
        point_k_water = self.shum.calc_point(p=self.pk, x=0)
        return point_k_water
    
    @property
    def point_feed_water(self):
        point_feed_water = self.shum.calc_point(p=self.p_feed_water, t=self.t_feed_water)
        return point_feed_water
        
    @property
    def numenator_without(self):
        numenator_without = to_kelvin(self.point_2.T) * (self.point_middle_t.s - self.point_k_water.s)
        return numenator_without
    
    @property
    def denumenator_without(self):
        denumenator_without = (self.point_0.h - self.point_1t.h) + (self.point_middle.h - self.point_k_water.h)
        return denumenator_without
    
    @property
    def without_part(self):
        without_part = 1 - (self.numenator_without / self.denumenator_without)
        return without_part
    
    @property
    def numenator_infinity(self):
        numenator_infinity = to_kelvin(self.point_2.T) * (self.point_middle_t.s - self.point_feed_water.s)
        return numenator_infinity
    
    @property
    def denumenator_infinity(self):
        denumenator_infinity = (self.point_0.h - self.point_1t.h) + (self.point_middle.h - self.point_feed_water.h)
        return denumenator_infinity
    
    @property
    def infinity_part(self):
        infinity_part = 1 - (self.numenator_infinity / self.denumenator_infinity)
        return infinity_part
    
    @property
    def ksi_infinity(self):
        ksi_infinity = 1 - (self.without_part / self.infinity_part)
        return ksi_infinity
    
    @property
    def coeff(self):
        coeff = (self.point_feed_water.T - self.point_2.T) / (to_kelvin(374.2) - self.point_2.T)
        return coeff
    
    @property
    def ksi(self):
        ksi = self.alpha * self.ksi_infinity
        return ksi
    
    @property
    def efficiency(self):
        eff_num = self.hp_heat_drop + self.lp_heat_drop
        eff_denum = self.hp_heat_drop + (self.point_middle.h - self.point_k_water.h)
        efficiency = (eff_num / eff_denum) * (1 / (1 - self.ksi))
        return efficiency
    
    @property
    def estimated_heat_drop(self):
        estimated_heat_drop = self.efficiency * ((self.point_0.h - self.point_feed_water.h) + (self.point_middle.h - self.point_1.h))
        return estimated_heat_drop
    
    @property
    def inlet_mass_flow(self):
        inlet_mass_flow = self.electrical_power / (self.estimated_heat_drop * 1000 * self.mechanical_efficiency * self.generator_efficiency)
        return inlet_mass_flow
    
    @property
    def condenser_mass_flow(self):
        condenser_mass_flow = (
            self.electrical_power /
            ((self.point_2.h - self.point_k_water.h) * 1000 * self.mechanical_efficiency * self.generator_efficiency) * ((1 / self.efficiency) - 1)
        )
        return condenser_mass_flow
    
    def legend_without_duplicate_labels(self,ax: plt.Axes) -> None:
        """
        Убирает дубликаты из легенды графика
        :param plt.Axes ax: AxesSubplot с отрисованными графиками
        :return None:
        """
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
    
    
    def plot_process(self,ax: plt.Axes, points: List[IAPWS97], **kwargs) -> None:
        """
        Отрисовка процесса расширения по точкам
        :param plt.Axes ax: AxesSubplot с отрисованными графиками
        :param List[IAPWS97] points: Список инициализиованных точек процесса
        :param kwargs:
        :return None:
        """
        ax.plot([point.s for point in points], [point.h for point in points], **kwargs)
    
    
    def get_isobar(self,point: IAPWS97) -> Tuple[List[float], List[float]]:
        """
        Собрать координаты изобары в hs осях
        :param IAPWS97 point: Точка для изобары
        :return Tuple[List[float], List[float]]:
        """
        s = point.s
        s_values = np.arange(s * 0.9, s * 1.1, 0.2 * s / 1000)
        h_values = [self.shum.calc_point(p=point.P, s=_s).h for _s in s_values]
        return s_values, h_values
    
    
    def _get_isoterm_steam(self,point: IAPWS97) -> Tuple[List[float], List[float]]:
        """
        Собрать координаты изотермы для пара в hs осях
        :param IAPWS97 point: Точка для изотермы
        :return Tuple[List[float], List[float]]:
        """
        t = point.T
        p = point.P
        s = point.s
        s_max = s * 1.2
        s_min = s * 0.8
        p_values = np.arange(p * 0.8, p * 1.2, 0.4 * p / 1000)
        h_values = np.array([self.shum.calc_point(p=_p, t=t).h for _p in p_values])
        s_values = np.array([self.shum.calc_point(p=_p, t=t).s for _p in p_values])
        mask = (s_values >= s_min) & (s_values <= s_max)
        return s_values[mask], h_values[mask]
    
    
    def _get_isoterm_two_phases(self,point: IAPWS97) -> Tuple[List[float], List[float]]:
        """
        Собрать координаты изотермы для влажного пара в hs осях
        :param IAPWS97 point: Точка для изотермы
        :return Tuple[List[float], List[float]]:
        """
        x = point.x
        p = point.P * MPa
        x_values = np.arange(x * 0.9, min(x * 1.1, 1), (1 - x) / 1000)
        h_values = np.array([self.shum.calc_point(p=p, x=_x).h for _x in x_values])
        s_values = np.array([self.shum.calc_point(p=p, x=_x).s for _x in x_values])
        return s_values, h_values
    
    
    def get_isoterm(self,point) -> Tuple[List[float], List[float]]:
        """
        Собрать координаты изотермы в hs осях
        :param IAPWS97 point: Точка для изотермы
        :return Tuple[List[float], List[float]]:
        """
        if point.x < 1 and point.x > 0:
            return self._get_isoterm_two_phases(point)
        return self._get_isoterm_steam(point)
    
    
    def plot_isolines(self,ax: plt.Axes, point: IAPWS97) -> None:
        """
        Отрисовка изобары и изотермы
        :param plt.Axes ax: AxesSubplot на котором изобразить линии
        :param IAPWS97 point: Точка для изображения изолиний
        :return None:
        """
        s_isobar, h_isobar = self.get_isobar(point)
        s_isoterm, h_isoterm = self.get_isoterm(point)
        ax.plot(s_isobar, h_isobar, color='green', label='Изобара')
        ax.plot(s_isoterm, h_isoterm, color='blue', label='Изотерма')
    
    
    def plot_points(self,ax: plt.Axes, points: List[IAPWS97]) -> None:
        """
        Отрисовать точки на hs-диаграмме
        :param plt.Axes ax: AxesSubplot на котором изобразить точки
        :param List[IAPWS97] points: Точки для отображения
        return None
        """
        for point in points:
            ax.scatter(point.s, point.h, s=50, color="red")
            self.plot_isolines(ax, point)
    
    
    def get_humidity_constant_line(self,
            point: IAPWS97,
            max_p: float,
            min_p: float,
            x: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Собрать координаты линии с постоянной степенью сухости в hs осях
        :param IAPWS97 point: Точка для изолинии
        :param float max_p: Максимальное давление для линии
        :param float min_p: Минимальное давление для линии
        :param Optional[float] x: Степень сухости для отрисовки
        :return Tuple[List[float], List[float]]:
        """
 
        _x = x if x else point.x
        p_values = np.arange(min_p * MPa, max_p * MPa, (max_p*MPa - min_p*MPa) / 1000)
        h_values = np.array([self.shum.calc_point(p=_p, x=_x).h for _p in p_values])
        s_values = np.array([self.shum.calc_point(p=_p, x=_x).s for _p in p_values])
        return s_values, h_values
    
    
    def plot_humidity_lines(self,ax: plt.Axes, points: List[IAPWS97]) -> None:
        """
        Отрисовать изолинии для степеней сухости на hs-диаграмме
        :param plt.Axes ax: AxesSubplot на котором изобразить изолинии
        :param List[IAPWS97] points: Точки для отображения
        return None
        """
        pressures = [point.P for point in points]
        min_pressure = min(pressures) if min(pressures) > 700 / 1e6 else 700 / 1e6
        max_pressure = max(pressures) if max(pressures) < 22 else 22
        for point in points:
            if point.x < 1 or point.x > 0:
                s_values, h_values = self.get_humidity_constant_line(point, max_pressure, min_pressure, x=1)
                ax.plot(s_values, h_values, color="grey")
                s_values, h_values = self.get_humidity_constant_line(point, max_pressure, min_pressure)
                ax.plot(s_values, h_values, color="grey", label='Линия сухости')
                ax.text(s_values[10], h_values[10], f'x={round(point.x, 2)}')
    
    
    def plot_hs_diagram(self,ax: plt.Axes, points: List[IAPWS97]) -> None:
        """
        Построить изобары и изотермы для переданных точек. Если степень сухости у точки не равна 1, то построется
        дополнительно линия соответствующей степени сухости
        :param plt.Axes ax: AxesSubplot на котором изобразить изолинии
        :param List[IAPWS97] points: Точки для отображения
        return None
        """
        self.plot_points(ax, points)
        self.plot_humidity_lines(ax, points)
        ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
        ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
        ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
        ax.legend()
        ax.grid()
        self.legend_without_duplicate_labels(ax)
        
    @property    
    def all_hs_plot(self):
        fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
    
        self.plot_hs_diagram(
            ax,
            points=[self.point_0t, self.point_0, self.point_1t, self.point_1, self.point_middle_t, self.point_middle, self.point_2, self.point_2t]
        )
        self.plot_process(ax, points=[self.point_0t, self.point_0, self.point_1], color='black')
        self.plot_process(ax, points=[self.point_middle_t, self.point_middle, self.point_2], color='black')
        self.plot_process(ax, points=[self.point_0t, self.point_0, self.point_1t], alpha=0.5, color='grey')
        self.plot_process(ax, points=[self.point_middle_t, self.point_middle, self.point_2t], alpha=0.5, color='grey')
        plt.show()
    
  
        
        
        
        
        
        
        
        