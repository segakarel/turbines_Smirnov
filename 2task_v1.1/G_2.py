from typing import List, Tuple, Optional
from iapws import IAPWS97
import matplotlib.pyplot as plt
import numpy as np
from steam import Steam as sgas
from math import sqrt, pi, sin, cos, atan, degrees, radians, asin, acos
import numpy as np
from numpy import deg2rad, rad2deg
import math
import matplotlib.pyplot as plt

def r(x):
    a = (round(x))
    return a 

MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None

name = "steam"
R = 461
c_p = 287.05

input_class = sgas(name = name, R = R, c_p = c_p)

class calc_mass_flow():
    """
        Класс для расчёта расхода в конденсатор / турбину
        Атрибуты:
            input_class: class, - класс для расчёта параметров состояния
        Методы:
            condenser_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,
        mechanical_efficiency,generator_efficiency) - рассчитывает расход на входе в конденсатор 
            inlet_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,
        t_feed_water,electrical_power,mechanical_efficiency,generator_efficiency) - рассчитывает расход на входе в турбину 
            efficiency(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha) - рассчитывает КПД
    """
    def __init__(self,input_class, p0,t0,p_middle,t_middle,pk,t_feed_water,p_feed_water,electrical_power,
                 internal_efficiency,mechanical_efficiency,generator_efficiency,alpha):
        self.input_class = input_class
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
    
    @property
    def calc_zero_point(self):
        delta_p0 = 0.05 * self.p0
        real_p0 = self.p0 - delta_p0
        _point_0 = self.input_class.calc_point(p= self.p0 * unit, t =self.t0)
        point_0 = self.input_class.calc_point(p=real_p0 * unit, h=_point_0.h)
        return point_0, _point_0
    
    @property
    def calc_first_point(self):
        delta_p_middle = 0.1 * self.p_middle
        real_p1t = self.p_middle + delta_p_middle
        _point_0 = self.calc_zero_point[0]
        point_0 = self.calc_zero_point[1]
        point_1t = self.input_class.calc_point(p=real_p1t * unit, s=_point_0.s)
        hp_heat_drop = (_point_0.h - point_1t.h) * self.internal_efficiency
        h_1 = point_0.h - hp_heat_drop
        point_1 = self.input_class.calc_point(p=real_p1t * unit, h=h_1)
        return point_1, point_1t, hp_heat_drop
    
    @property   
    def calc_middle_point(self):
        delta_p_1 = 0.03 * self.p_middle
        real_p_middle = self.p_middle - delta_p_1
        _point_middle = self.input_class.calc_point(p=self.p_middle * unit, t=self.t_middle)
        point_middle = self.input_class.calc_point(p=real_p_middle * unit, h=_point_middle.h)
        point_2t = self.input_class.calc_point(p=self.pk * unit, s=_point_middle.s)
        return _point_middle, point_2t, point_middle
    
    @property
    def lp_heat_drop(self):
        _point_middle = self.calc_middle_point[0]
        point_2t = self.input_class.calc_point(p=self.pk * unit, s=_point_middle.s)
        lp_heat_drop = (_point_middle.h - point_2t.h) * self.internal_efficiency
        return lp_heat_drop
    
    @property
    def calc_second_point(self):
        point_middle = self.calc_middle_point[2]
        h_2 = point_middle.h - self.lp_heat_drop
        point_2 = self.input_class.calc_point(p=self.pk * unit, h=h_2)
        return point_2
    
    @property 
    def efficiency_hp(self):
        _point_0 = self.calc_zero_point[1]
        point_1 = self.calc_first_point[0]
        point_1t = self.calc_first_point[1]
        efficiency_hp = (_point_0.h - point_1.h) / (_point_0.h - point_1t.h)
        return efficiency_hp
    
    @property
    def efficiency_lp(self):
        _point_middle = self.calc_middle_point[0]
        point_2t = self.calc_middle_point[1]
        point_2 = self.calc_second_point
        efficiency_lp = (_point_middle.h - point_2.h) / (_point_middle.h - point_2t.h)
        return efficiency_lp

    @property
    def numenator_without(self):
        point_k_water = self.input_class.calc_point(p=self.pk, x=0)
        _point_middle = self.calc_middle_point[0]
        point_2 = self.calc_second_point
        numenator_without = to_kelvin(point_2.T) * (_point_middle.s - point_k_water.s)
        return numenator_without
    @property
    def denumenator_without(self):
        point_0 = self.calc_zero_point[0]
        point_1t = self.calc_first_point[1]
        point_middle = self.calc_middle_point[2]
        point_k_water = self.input_class.calc_point(p=self.pk, x=0)
        denumenator_without = (point_0.h - point_1t.h) + (point_middle.h - point_k_water.h)
        return denumenator_without
    
    @property
    def without_part(self):
        without_part = 1 - (self.numenator_without / self.denumenator_without)
        return without_part
    
    @property
    def numenator_infinity(self):
        point_2 = self.calc_second_point
        _point_middle = self.calc_middle_point[0]
        point_feed_water = self.input_class.calc_point(p=self.p_feed_water * unit, t=self.t_feed_water)
        numenator_infinity = to_kelvin(point_2.T) * (_point_middle.s - point_feed_water.s)
        return numenator_infinity
    
    @property
    def denumenator_infinity(self):
        point_0 = self.calc_zero_point[0]
        point_1t = self.calc_first_point[1]
        point_middle = self.calc_middle_point[2]
        point_feed_water = self.input_class.calc_point(p=self.p_feed_water * unit, t=self.t_feed_water)
        denumenator_infinity = (point_0.h - point_1t.h) + (point_middle.h - point_feed_water.h)
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
        point_2 = self.calc_second_point[0]
        point_feed_water = self.input_class.calc_point(p=self.p_feed_water * unit, t=self.t_feed_water)
        coeff = (point_feed_water.T - point_2.T) / (to_kelvin(374.2) - point_2.T)
        return coeff
    
    @property
    def ksi(self):
        ksi = self.alpha * self.ksi_infinity
        return ksi

    @property
    def efficiency(self):
        _point_0 = self.calc_zero_point[1]
        point_1t = self.calc_first_point[1]
        hp_heat_drop = (_point_0.h - point_1t.h) * self.internal_efficiency
        eff_num = hp_heat_drop + self.lp_heat_drop
        point_k_water = self.input_class.calc_point(p=self.pk, x=0)
        point_middle = self.calc_middle_point[2]
        eff_denum = hp_heat_drop + (point_middle.h - point_k_water.h)
        efficiency = (eff_num / eff_denum) * (1 / (1 - self.ksi))
        return efficiency
    
    @property
    def estimated_heat_drop(self):
        point_0 = self.calc_zero_point[0]
        point_middle = self.calc_middle_point[2]
        point_feed_water = self.input_class.calc_point(p=self.p_feed_water * unit, t=self.t_feed_water)
        point_1 = self.calc_first_point[0]
        estimated_heat_drop = self.efficiency * ((point_0.h - point_feed_water.h) + (point_middle.h - point_1.h))
        return estimated_heat_drop
    @property
    def inlet_mass_flow(self):
        inlet_mass_flow = self.electrical_power / (self.estimated_heat_drop * 1000 * self.mechanical_efficiency * self.generator_efficiency)
        return inlet_mass_flow
    
    @property
    def condenser_mass_flow(self):
        point_2 = self.calc_second_point
        point_k_water = self.input_class.calc_point(p=self.pk, x=0)
        condenser_mass_flow = (
            self.electrical_power /
            ((point_2.h - point_k_water.h) * 1000 * self.mechanical_efficiency * self.generator_efficiency) * ((1 / self.efficiency) - 1)
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
        h_values = [self.input_class.calc_point(p=point.P, s=_s).h for _s in s_values]
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
        h_values = np.array([self.input_class.calc_point(p=_p, t=t).h for _p in p_values])
        s_values = np.array([self.input_class.calc_point(p=_p, t=t).s for _p in p_values])
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
        h_values = np.array([self.input_class.calc_point(p=p, x=_x).h for _x in x_values])
        s_values = np.array([self.input_class.calc_point(p=p, x=_x).s for _x in x_values])
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
        h_values = np.array([self.input_class.calc_point(p=_p, x=_x).h for _p in p_values])
        s_values = np.array([self.input_class.calc_point(p=_p, x=_x).s for _p in p_values])
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
        _point_0 = self.calc_zero_point[0]
        point_0 = self.calc_zero_point[1]
        point_1t = self.calc_first_point[1]
        point_1 = self.calc_first_point[0]
        _point_middle = self.calc_middle_point[0]
        point_middle = self.calc_middle_point[2]
        point_2 = self.calc_second_point
        point_2t = self.calc_middle_point[1]
        self.plot_hs_diagram(
            ax,
            points=[_point_0, point_0, point_1t, point_1, _point_middle, point_middle, point_2, point_2t]
        )
        self.plot_process(ax, points=[_point_0, point_0, point_1], color='black')
        self.plot_process(ax, points=[_point_middle, point_middle, point_2], color='black')
        self.plot_process(ax, points=[_point_0, point_0, point_1t], alpha=0.5, color='grey')
        self.plot_process(ax, points=[_point_middle, point_middle, point_2t], alpha=0.5, color='grey')
        
    
    #def c_f_1_calc(self,p0,t0, 
                 # H_0, p_middle, n, inlet_mass_flow, b_1, t_rel, alpha_1_eff, outlet_area, 
                 # d_1, k, veernost_1, d_2, b_2, alpha_y, beta_1, t_2_opt):
#        G = inlet_mass_flow
#         point_0 = self.calc_zero_point(p0,t0)[0]
#         point_1 = self.calc_first_point(p0,t0,p_middle)[0]
#         c_f = sqrt(2000*(H_0))
#         u_1 = d_1*pi*n #rotation_speed
#         a_1t = sqrt(((1.31))*R*point_0.T)
# #        a_1t = input_class.get_w_pt(point_0.P,point_0.s)
#         degree_of_reaction = 0.08
#         rho = degree_of_reaction #point_1.rho
#         H_0_stator = (1 - rho)* H_0
#         H_0_rotor = H_0*rho #rotor_heat_drop
#         h_1_t = point_0.h - H_0_stator
        
#         point_1_t = input_class.calc_point(h = h_1_t, s = point_0.s)
#         phi = sqrt(1 - H_0_stator/(point_0.h - point_1_t.h))
#         discharge_coefficient_1  = inlet_mass_flow / (outlet_area  * c_f * point_0.rho) #mu https://github.com/input_classilinDmA/turbines_autumn_2024/blob/main/steam_turbines/lesson_3/Exercises.ipynb
#         print("discharge_coefficient_1 ", discharge_coefficient_1)
#         c_1t = sqrt(2000 * H_0_stator)
#         # a_1t = sqrt(k*point_1.P * point_1_t.v* MPa)
#         M_1t = c_1t/a_1t
#         F_1_calc = (inlet_mass_flow*point_1_t.v)/(discharge_coefficient_1*c_1t)
#         u_cf = u_1/c_f
        
#         root_reaction_degree = 0.05 #https://github.com/input_classilinDmA/turbines_autumn_2024/blob/main/steam_turbines/lesson_6/Exercises.ipynb
#         overlapping = 0.003
#         reactivity = self.get_reaction_degree(root_reaction_degree, veernost_1)
#         point_2 = input_class.calc_point(h = h_1_t, s = point_0.s)
#         #u_cf_1 = 
#         upper = inlet_mass_flow * point_2.v * u_cf
#         lower = discharge_coefficient_1 * np.sin(np.deg2rad(alpha)) * u_1 * (np.pi * d_1) ** 2 * (1 - reactivity) ** 0.5
#         blade_length_1 = lower / upper #l1
#         blade_length_2 = blade_length_1 + overlapping #l2
        
#         root_diameter = d_1 - blade_length_2 # avg_diam_1 = d
        
#         b1_l1 = b_1/blade_length_1
        
#         dzeta_prof_1 =  3.7708 * ((discharge_coefficient_1)**3) + 14.189 * ((discharge_coefficient_1)**2) - 13.478 * (discharge_coefficient_1) + 5.6125
#         dzeta_konical_1 = 2*b1_l1 + blade_length_2
#         dzeta_stator = dzeta_prof_1 + dzeta_konical_1
#         print("dzeta_stator ", dzeta_stator)
#         phi = sqrt(1 - (dzeta_stator)/100000)
#         c_1 = c_1t*phi
#         w_1 = (c_1 ** 2 + u_1 ** 2 - 2 * c_1 * u_1 * cos(np.deg2rad(alpha_1_eff))) ** 0.5
        
#         beta_1 = atan(sin(np.deg2rad(alpha_1_eff))) / (cos(np.deg2rad(alpha_1_eff) - u_1 / c_1))
#         delta_beta_deg = 5
#         beta_2_deg = degrees(beta_1) - delta_beta_deg
        
#         print("discharge_coefficient_1 ", discharge_coefficient_1, ", dzeta_stator ", dzeta_stator, ", inlet_mass_flow: ", inlet_mass_flow )
#         print("upper", upper, " ", "lower", lower)
#         print("phi: ", phi, ", blade_length: ", blade_length_1, ", beta_1: ", np.rad2deg(beta_1),", beta_2: ", beta_2_deg)
        
        
#         u_2 = d_2*pi*n
#         h_2t = point_1.h - H_0_rotor
#         point_2_t = input_class.calc_point(h=h_2t, s = point_1.s)
#         # w_2t = sqrt(2000*H_0_rotor + w_1**2)
#         # a_2t = a_1t
#         # c_2t = sqrt(2000*H_0_rotor)
#         # M_2t = c_2t/a_2t
#         # b2_l2 = b_2 / blade_length_2
#         # dzeta_prof_2 =  7.4173 * ((discharge_coefficient_1)**3) -10.511 * ((discharge_coefficient_1)**2) + 1.0066 * (discharge_coefficient_1) + 5.6125
#         # dzeta_konical_2 = 4.8786 * (b2_l2) + 4.9714
#         # dzeta_rotor = dzeta_prof_2 + dzeta_konical_2
#         # psi = sqrt(1 - dzeta_rotor/10000)
#         # w_2 = w_2t*psi
#         # c_2 = ((w_2**2) + (u_2**2) - 2 * w_2 * u_2 * np.cos(np.deg2rad(beta_2_deg))) ** 0.5
#         # # dzeta_prof_1 = 
#         # alpha_2 = np.arctan((np.sin(np.deg2rad(beta_2_deg)))/((np.cos(np.deg2rad(beta_2_deg))) - (u_2/w_2)))
#         # print("psi: ", psi, ", blade_length: ", blade_length_2, ", beta_2: ", beta_2_deg,", alpha_2: ", alpha_2)
        
#         u = u_1
#         beta_1 = math.atan(math.sin(np.deg2rad(alpha_1_eff))) / (math.cos(np.deg2rad(alpha_1_eff) - u / c_1))
#         delta_beta_deg = 5
#         beta_2_deg = math.degrees(beta_1) - delta_beta_deg
        
#         w_2t = (w_1 ** 2 + 2 * H_0_rotor * 1000) ** 0.5
#         psi = math.sqrt(1 - H_0_rotor/ (point_1.h - point_1_t.h))
#         w_2 = w_2t * psi
        
#         sin_beta_2 = math.sin(math.radians(beta_2_deg))
#         cos_beta_2 = math.cos(math.radians(beta_2_deg))
#         c_2 = (w_2 ** 2 + u ** 2 - 2 * w_2 * u * cos_beta_2) ** 0.5
#         alpha_2 = math.atan(sin_beta_2 / (cos_beta_2 - u / w_2))
#         alpha_2_deg = math.degrees(alpha_2)
#         sin_alpha_1 = math.sin(math.radians(alpha_1_eff))
#         cos_alpha_1 = math.cos(math.radians(alpha_1_eff))
        
#         c1_plot = [[0, -c_1 * cos_alpha_1], [0, -c_1 * sin_alpha_1]]
#         u1_plot = [[-c_1 * cos_alpha_1, -c_1 * cos_alpha_1 + u], [-c_1 * sin_alpha_1, -c_1 * sin_alpha_1]]
#         w1_plot = [[0, -c_1 * cos_alpha_1 + u], [0, -c_1 * sin_alpha_1]]

#         w2_plot = [[0, w_2 * cos_beta_2], [0, -w_2 * sin_beta_2]]
#         u2_plot = [[w_2 * cos_beta_2, w_2 * cos_beta_2 - u], [-w_2 * sin_beta_2, -w_2 * sin_beta_2]]
#         c2_plot = [[0, w_2 * cos_beta_2 - u], [0, -w_2 * sin_beta_2]]
        
#         fig, ax  = plt.subplots(1, 1, figsize=(15, 5))

#         ax.plot(c1_plot[0], c1_plot[1], label='C_1', c='red')
#         ax.plot(u1_plot[0], u1_plot[1], label='u_1', c='blue')
#         ax.plot(w1_plot[0], w1_plot[1], label='W_1', c='green')
        
#         ax.plot(w2_plot[0], w2_plot[1], label='W_2', c='green')
#         ax.plot(u2_plot[0], u2_plot[1], label='u_2', c='blue')
#         ax.plot(c2_plot[0], c2_plot[1], label='C_2', c='red')
        
#         ax.set_title("Треугольник скоростей")
#         plt.show()
#         ax.legend()
        
        
#         absolute_projection = c_1 * cos_alpha_1 + c_2 * math.cos(alpha_2)
#         relative_projection = w_1 * math.cos(beta_1) + w_2 * cos_beta_2
        
#         #assert round(absolute_projection, 5) == round(relative_projection, 5)
        
#         # print(absolute_projection, relative_projection)
        
#         outlet_speed_loss = 0.5 * c_2 ** 2
        
#         stator_speed_loss = 0.5 * ((c_1t ** 2) - (c_1 ** 2))
#         rotor_speed_loss = 0.5 * ((w_2t ** 2) - (w_2 ** 2))
        
#         def get_available_energy(kappa) -> float:
#             return h_1_t * 1000 - kappa * outlet_speed_loss
        
        
#         def turbine_efficiency(kappa: float) -> float:
#             constant_part = get_available_energy(kappa) - stator_speed_loss - rotor_speed_loss
#             useful_energy = constant_part - (1 - kappa) * outlet_speed_loss
#             return useful_energy / get_available_energy(kappa)
        
#         print("turbine_efficiency(1): ", turbine_efficiency(1))
        
#         chi_vs = (c_1/c_2)**2
        
        # def get_k_frictions(s_div_r, re):
        #     return 2.5 * 10 ** (-2) * s_div_r ** 0.1 * re **(-0.2)
            
        # def get_friction_loss_pu(k, d, u_div_dummy_speed, F):
        #     return k * d ** 2 * u_div_dummy_speed ** 3 / F 
            
        # def get_ventilation_loss_pu(m, k, sin, e, u_div_dummy_speed):
        #     first = k / sin
        #     second = (1 - e) / e
        #     third = u_div_dummy_speed ** 3
            
        #     return first * second * third * m
            
        # def get_segment_loss_pu(B, l, F, u_div_dummy_speed, blade_efficiency, segments):
        #     first = 0.25 * B * l / F
        #     second = u_div_dummy_speed * blade_efficiency * segments
        #     return first * second
            
        # def compute_equal_gap(z, delta_r, mu_r, delta_a, mu_a):
        #     first = 1 / (mu_a * delta_a) ** 2
        #     second = z / (mu_r * delta_r) ** 2
        #     return (first + second) ** (-0.5)
        
        # def get_bandage_leak_loss_pu(d_shroud, delta_eq, F, dor, l, efficiency):
        #     d_avg = d_shroud - l
            
        #     first = math.pi * d_shroud * delta_eq / F
        #     second = dor + 1.8 * (l / d_avg)
            
        #     return 0.1# first * (second) ** 0.5 * efficiency
            
        # def get_disk_leak_loss_pu(K, F, mu_r, mu_nozzle, F_nozzle, z, efficiency):
        #     upper = mu_r * K * F * efficiency
        #     lower = mu_nozzle * F_nozzle * z ** 0.5
        #     return upper / lower  
        
        # p = p0
        # t = t0
        # u = u_1
        # blade_efficiency = turbine_efficiency(1)
        # F1 = outlet_area
        # d_r = d_1
        # s_div_r = 0.2 
        
        # e = 0.8
        # sin_alpha_1 = sin_alpha_1
        # blade_width = b_1
        # blade_length = blade_length_1
        # segments = 4
        
        # degree__of_reaction = 0.1

        # z_bandage = 3
        # delta_r_bandage = 1.17 / 1000
        # delta_a_bandage = 4 / 1000
        
        # z_rotor = round(pi*d_1*0.85/(b_1*t_rel))
        # d_leak_rotor = 0.36
        # delta_leak_rotor = 0.4 / 1000
        # F_leak_rotor = math.pi * d_leak_rotor * delta_leak_rotor
        # mu_a = 0.5
        # mu_r = 0.8
        # delta_eq_bandage = compute_equal_gap(z_bandage, delta_r_bandage, mu_r, delta_a_bandage, mu_a)
        
        # dummy_speed = (2000 * H_0) ** 0.5
        # u_div_dummy_speed= u / dummy_speed
        
        # eff = []
        # d_r = []
        # for i in range (9000, 11001, 1):
        #     d_r.append(i*0.0001)
        # for i in range(len(d_r)):
        #     kinematic_viscosity = 0.128*10**6
        #     Re_number = u * d_r[i] * 0.5 / kinematic_viscosity
        #     k_frictions = get_k_frictions(s_div_r, Re_number)
            
        #     friction_loss_pu = get_friction_loss_pu(k_frictions, d_r[i], u_div_dummy_speed, F1)
        #     #friction_loss = friction_loss_pu * H_0
        
        #     ventilation_loss_pu = get_ventilation_loss_pu(
        #         m=1,
        #         k=0.065,
        #         e=e,
        #         u_div_dummy_speed=u_div_dummy_speed,
        #         sin=sin_alpha_1
        #     )
        
        #     segment_loss_pu = get_segment_loss_pu(
        #         B=blade_width,
        #         l=blade_length,
        #         F=F1,
        #         u_div_dummy_speed=u_div_dummy_speed,
        #         blade_efficiency=blade_efficiency,
        #         segments=segments
        #     )
        
        #     partial_losses_pu = segment_loss_pu + ventilation_loss_pu
        #     #partial_losses = H_0 * partial_losses_p
        
        
        #     d_shroud = delta_r_bandage / 0.001 
        #     bandage_leak_loss_pu = get_bandage_leak_loss_pu(
        #         d_shroud=d_shroud,
        #         delta_eq=delta_eq_bandage,
        #         F=F1,
        #         dor=degree__of_reaction,
        #         l=blade_length,
        #         efficiency=blade_efficiency
        #     )
        #     K_y = 1
        #     mu_r_rotor = 0.8
        #     mu_nozzle = 0.97
        #     disk_leak_loss_pu = get_disk_leak_loss_pu(
        #         K = K_y,
        #         F = F_leak_rotor,
        #         mu_r = mu_r_rotor,
        #         mu_nozzle=mu_nozzle,
        #         F_nozzle=F1,
        #         z=z_rotor,
        #         efficiency=blade_efficiency
        #     )
        #     leak_losses_pu = disk_leak_loss_pu + bandage_leak_loss_pu
        #     #leak_losses = H_0 * leak_losses_pu
        #     internal_efficiency = (blade_efficiency - friction_loss_pu - partial_losses_pu - leak_losses_pu) * 100
        #     eff.append(-internal_efficiency)        
        
        # # print(eff[-2:-1])
        # plt.xlabel(r"$d_1$, мм ")
        # plt.ylabel(r'$\eta_{oi}$, %')
        # # plt.plot(d_r,eff, "o", color="blue")
        # plt.plot(d_r,eff, color="blue")
        # plt.title("График зависимости $\eta_{oi} = f(d_1)$")
        # plt.show()
        
        ## начало расчёта по методике Попова, неприсланная презентация
        
#         u = pi*d_1 * n #                                                        №4
#         rho = 0.1 #степень реактивность 
#         H_0c = (1 - rho) * H_0
#         H_0p = H_0*rho
# #        point_1_t = input_class.calc_point(h = h_1t, s = point_0.s)
#         point_0 = self.calc_zero_point(p0,t0)[0]
#         h_1t = point_0.h - H_0c
#         p_1 = input_class.calc_point(h = h_1t, p = point_0.P).P
#         point_1t = input_class.calc_point(h = h_1t, p = point_0.s)
#         v_1t = point_1t.v
#         c_1t = sqrt(2000*H_0c)
#         k = 1.31
#         a_1t = sqrt(k*R*point_1t.T) #input_class.get_w_pt(point_1t.P, point_1t.s)#sqrt(k*point_1t.P*point_1t.v)
#         M_1t = c_1t/a_1t
#         mu_1_eq = 0.97
#         alpha_0 = alpha_1_eff
#         alpha_1_eff = 15# - эффективное альфа
#         F_1_sh = inlet_mass_flow*point_1t.v/(mu_1_eq*c_1t)
#         F_1 = outlet_area*0.0001
#         el_1 = F_1/(pi*d_1*sin(deg2rad(alpha_1_eff))) #                         №23
#         e_opt = (4+6)/2*sqrt(el_1)
#         if e_opt > 0.85: e_opt = 0.85
#         l_1 = el_1/e_opt
#         mu_1 = 0.982 - 0.005*(b_1/l_1)
#         t_1_opt = t_rel #                                                       №29
#         z_1 = round((pi*d_1*e_opt)/(b_1*t_1_opt))
#         if z_1 % 2 != 0: z_1 += 1
#         t_1_exact = (pi*d_1*e_opt)/(b_1*z_1)
#         alpha_y = alpha_y
#         dzeta_s = 2 # по атласу профилей                                        №34
#         phi = sqrt(1 - dzeta_s/100)
#         phi_eq = 0.98 - 0.008*(b_1/l_1)
#         print ((phi - phi_eq)/phi_eq * 100)#                                    №36
#         c_1 = c_1t * phi
#         alpha_1 = asin(mu_1/phi*sin(deg2rad(alpha_1_eff)))
#         w_1 = sqrt(c_1**2 + u**2 - 2* c_1 * u * cos(alpha_1))
#         tan_beta_1 = sin(alpha_1)/(cos(alpha_1) - u/c_1)#                       №42
        
#         Delta_H_c = c_1t**2 * (1 - phi**2)
#         h_1 = h_1t + Delta_H_c
#         s_1 = input_class.calc_point(h = h_1, s = point_0.s).s
#         h_2t = h_1 - H_0p
#         p_2 = input_class.calc_point(h = h_2t, s = s_1).P#                      №48
#         w_2t = sqrt(H_0p + w_1**2)
#         overlap = (0.003+0.004)/2# перекрыша, м
#         l_2 = l_1 + overlap
#         b_2 = b_2 
#         mu_2 = 0.965 - 0.01 * (b_2/l_2)#                                        №53
#         v_2t  = input_class.calc_point(h = h_2t, s = s_1).v#  
#         a_2t = sqrt(k*R*input_class.calc_point(h = h_2t, s = s_1).T)#input_class.get_w_pt(p_2, input_class.calc_point(h = h_2t, s = s_1).s)#sqrt(k*p_2*v_2t)
#         M_2t = w_2t/a_2t
#         F_2 = (inlet_mass_flow*v_2t)/(mu_2*w_2t)
#         arcsin_beta_2_eff = (F_2) / (e_opt*pi*d_1*l_2)
#         t_2_opt = t_2_opt
#         z_2 = round((pi*d_1)/(b_2*t_2_opt))
#         if z_2 % 2 != 0: z_2 += 1
#         t_2_exact = (pi*d_1)/(b_2*z_2)
#         print((t_2_opt - t_2_exact)/ t_2_exact *100)#                           №63
#         print("arcsin_beta_2_eff: ", arcsin_beta_2_eff)
#         beta_2_eff = asin(deg2rad(arcsin_beta_2_eff))
#         beta_2y = (np.rad2deg(beta_2_eff) - 3.25*(t_2_exact-0.7) + 63.8)
#         print("M_1t: ", M_1t, ", M_2t: ", M_2t)
#         dzeta_p = 6 
#         psi = sqrt(1 - dzeta_p/100)
#         psi_eq = 0.96-0.014*(b_2/l_2)
#         print("psi error: ", (psi - psi_eq)/ psi_eq *100, "; ", psi, "; psi_eq ", psi_eq, " b_2/l_2: ", b_2/l_2)#                      №68
#         w_2 = w_2t*psi
#         beta_2_grad = np.rad2deg(asin(mu_2/psi*sin(beta_2_eff)))
#         c_2 = sqrt(w_2**2 + u**2 - 2*w_2*cos(deg2rad(beta_2_grad)))
#         tan_alpha_2 = (sin(deg2rad(beta_2_grad)))/(cos(deg2rad(beta_2_grad)) - u/w_2)
#         alpha_2_deg = np.rad2deg(atan(tan_alpha_2))
#         Delta_H_p = w_2t**2/2*(1-psi**2)
#         Delta_H_vs = c_2**2/2
#         chi_vs = 0                                                          #   №77
#         E_0 = H_0 - chi_vs*Delta_H_vs
#         eta_ol_energy = (E_0 - Delta_H_c - Delta_H_p - (1 - chi_vs)*Delta_H_vs)/E_0
#         eta_ol_traing = (u*c_1*cos(alpha_1) + c_2* cos(deg2rad(alpha_2_deg)))/(E_0)
#         print("KPD: ", eta_ol_energy, " ", eta_ol_traing, "; delta: ", (eta_ol_energy - eta_ol_traing)/eta_ol_traing*100)
        
        
        ##Заново расчёт по Попову
        
        # inlet_mass_flow = 257.13
        # G = inlet_mass_flow
        # d = 0.9
        # n = 60
        # b_1 = 52.5/1000
        # t_1opt = (0.72+0.85)/2
        # b_2 = 25.9/1000
        
        # u = pi*d*n
        # p_0 = self.calc_zero_point(p0,t0)[0].P
        # t_0 = self.calc_zero_point(p0,t0)[0].T
        # s_0 = self.calc_zero_point(p0,t0)[0].s
        # h_0 = self.calc_zero_point(p0,t0)[0].h
        # # c_0 = &&????????????????????
        # rho = 0.1
        # H_0c = (1-rho)*H_0
        # H_0p = H_0*rho
        # h_1t = h_0 - H_0c
        # p_1 = input_class.calc_point(h = h_1t, p = p_0).P
        # v_1t = input_class.calc_point(h = h_1t, p = p_0).v
        # c_1t = sqrt(2000*H_0c)
        # k = 1.33
        # a_1t = sqrt(k*input_class.calc_point(h = h_1t, p = p_0).P*input_class.calc_point(h = h_1t, p = p_0).v*(10**6)) #gas(p = p_1*unit, s = s_0).w #sqrt(k*R*t_0)#p_1*v_1t)
        # # print (a_1t)
        # M_1t = c_1t/a_1t
        # print("M_1t: ", M_1t)
        # mu_1_eq = 0.97
        # F_1_eq = (G*v_1t)/(mu_1_eq*c_1t)
        # alpha_1eq = 15 #deg
        # el_1 = F_1_eq/(pi*d*sin(deg2rad(alpha_1eq)))
        # e_opt = (4+6)/2*sqrt(el_1)
        # l_1 = el_1/e_opt
        # b_1 = b_1
        # mu_1 = 0.982-0.005*(b_1/l_1)
        # t_1opt = t_1opt
        # z_1 = (pi*d*e_opt)/(b_1*t_1opt)
        # if z_1 % 2 != 0: z_1 += 1#                                              №31
        # t_1 = (pi*d*e_opt)/(b_1*z_1)
        # print("t_1: ", t_1)
        # alpha_y = alpha_1eq -10*(t_1 - 0.75) + 21.2
        # dzeta_c = 2
        # phi = sqrt(1 - dzeta_c/100)
        # phi_eq = 0.98 - 0.008 * (b_1/l_1)
        # print("phi error: ", (phi - phi_eq)/phi_eq*100)
        # c_1 = c_1t*phi
        # alpha_1 = rad2deg(asin(mu_1/phi*sin(alpha_1eq)))
        # w_1 = sqrt(c_1**2 + u**2 - 2*c_1*u*cos(deg2rad(alpha_1)))
        # beta_1 = rad2deg(atan(sin(deg2rad(alpha_1))/(cos(deg2rad(alpha_1) - u/c_1))))
        # Delta_H_c = c_1t**2/2*(1-phi**2)
        # h_1 = h_1t + Delta_H_c
        # s_1 = input_class.calc_point(h = h_1, p = p_1).s
        # # h_1 = input_class.calc_point(h = h_1, p = p_1).h
        # h_2t = h_1 - H_0p
        # p_2 = input_class.calc_point(h = h_2t, s = s_1).P
        # w_2t = sqrt(2000*H_0p + w_1**2)
        # Delta = (0.003+0.004)/2
        # l_2 = l_1 + Delta
        # b_2 = b_2
        # mu_2 = 0.965 - 0.01*(b_2/l_2)
        # v_2t = input_class.calc_point(h = h_2t, s = s_1).v
        # a_2t = sqrt(k*input_class.calc_point(h = h_2t, s = s_1).P * input_class.calc_point(h = h_2t, s = s_1).v * 10**6)#R*input_class.calc_point(h = h_2t, s = s_1).T)
        # M_2t = w_2t/a_2t
        # print('M_2t: ', M_2t)
        # F_2 = (G*v_2t)/(mu_2*w_2t)
        # beta_2e = rad2deg(asin(F_2/(e_opt*pi*d*l_2)))
        # t_opt2 = (0.6+0.75)/2
        # z_2 = (pi*d)/(b_2*t_opt2)
        # if z_2 % 2 != 0: z_2 += 1
        # t_2 = (pi*d)/(b_2*z_2)
        # print("t_2: ", t_2)
        # beta_y = beta_2e - 3.25* (t_2 - 0.7) + 63.8
        # dzeta_p = 7.8
        # psi = sqrt(1 - dzeta_p/100)
        # psi_eq = 0.96-0.014*(b_2/l_2)
        # print("psi error: ", (psi_eq - psi)/psi*100)
        # w_2 = w_2t*psi
        # beta_2 = rad2deg(asin(mu_2/psi*sin(deg2rad(beta_2e))))
        # c_2 = sqrt(w_2**2 + u**2 - 2*w_2*u*cos(beta_2))
        
        
        
        # alpha_2 = rad2deg(atan((sin(deg2rad(beta_2)))/(cos(deg2rad(beta_2))) - u/w_2))
        # Delta_H_p = w_2t**2/2*(1 - psi**2)
        # Delta_H_vc = c_2**2/2 
        # chi_vs = 0
        # E_0 = H_0 - chi_vs*Delta_H_vc
        # print("beta_2: ", beta_2, " ", "alpha_2: ", alpha_2)
        
        # eta_ol_energy = (E_0 - Delta_H_c/2000 - Delta_H_p/2000 - (1 - chi_vs)*Delta_H_vc/2000)/E_0
        # eta_ol_angles = (u*(c_1*cos(deg2rad(alpha_1)) + c_2 * cos(deg2rad(alpha_2))))/(E_0)
        # print("eta error: ", (eta_ol_energy - eta_ol_angles)/eta_ol_angles*100, eta_ol_energy, eta_ol_angles)



        #####___________________________________________________________________________________________________
        
        # расчёт ещё раз
        
        # inlet_mass_flow = 257.13
        # G = inlet_mass_flow
        # d = 0.9
        # n = 60
        # b_1 = 52.5/1000
        # t_1opt = (0.72+0.85)/2
        # b_2 = 25.9/1000
        # rho = 0.1
        # H_0 = 100
        
        # H_0c = (1 - rho)*H_0
        # H_0r = rho*H_0
        # k = 1.3 
        # h_1_t = self.calc_zero_point(p0,t0)[0].h - H_0c #input_class.calc_point(h = h_2t, s = s_1).v
        # point_1_t = input_class.calc_point(h = h_1_t, s = self.calc_zero_point(p0,t0)[0].s)
        # c_1_t = sqrt(2000*H_0c)
        # a_1_t = sqrt(k*point_1_t.P*point_1_t.v*10**6)
        # M_1_t = c_1_t/a_1_t
        # print("M_1_t: ", M_1_t)
        # mu_1_approx = 0.97
        # F_1_approx = (G*point_1_t.v)/(mu_1_approx*c_1_t)
        # u = d*pi*n
        # c_f = sqrt(2000*H_0)
        # u_c_f = u/c_f
        # b_1 = b_1
        # alpha_1_e = 15
        # el_1 = F_1_approx/(pi*d*sin(deg2rad(alpha_1_e)))
        # e_opt = 5*sqrt(el_1)
        # if e_opt > 0.85: e_opt = 0.85
        # l_1 = el_1/e_opt
        # z_1 = round((pi*d*e_opt)/(b_1*t_1opt))
        # if z_1 % 2 != 0: z_1 += 1
        # t_1 = (pi*d*e_opt)/(b_1*z_1)
        # mu_1 = 0.982 - 0.005 * (b_1/l_1)
        # b1_l1 = b_1 / l_1
        # dzeta_prof = - 3.7708 * ((mu_1_approx)**3) + 14.189 * ((mu_1_approx)**2) - 13.478 * (mu_1_approx) + 5.6125
        # dzeta_konc_l_d = 2*b1_l1+2
        # alpha_y = (alpha_1_e) - 16*(t_1opt - 0.75) + 23.1
        # dzeta_c = dzeta_prof + dzeta_konc_l_d
        # phi = sqrt(1 - dzeta_c/100)
        # c_1 = c_1_t + phi
        # alpha_1 = asin((mu_1/phi)*sin(deg2rad(alpha_1_e)))
        # w_1 = ((c_1**2)+(u**2) - 2*c_1*u*cos(alpha_1))
        # tan_beta_1 = sin((alpha_1))/(np.cos((alpha_1)) - (u/c_1))
        # overlap = 0.003
        # Delta_H_c = c_1_t ** 2 * ( 1 - phi ** 2 ) / 2000
        # h_1 = h_1_t + Delta_H_c
        # point_1 = input_class.calc_point(h = h_1, p = point_1_t.P)
        # h_2_t = point_1.h - H_0r
        # point_2_t = input_class.calc_point(h = h_2_t, s = point_1.s)
        # w_2_t = sqrt(2000*H_0r+ w_1**2)
        # l_2 = l_1 + overlap
        # a_2_t = (k*point_2_t.P*point_2_t.v*10**6)
        # M_2_t = w_2_t/a_2_t
        # print("M_2_t: ", M_2_t)
        
        # b_2 = b_2
        # t_opt_2 = 0.65
        # t_opt_2 = t_opt_2
        # b2_l2 = b_2/l_2
        # mu_2 = 0.965 - 0.01*(b2_l2)
        # F_2 = (G * point_2_t.v)/(mu_2 * w_2_t)
        # a_2_t = sqrt(k * point_2_t.P * point_2_t.v * (10 ** 6))
        # beta_2e = asin(F_2/(e_opt*pi*d*l_2))
        # z_2 = round((pi * d)/(b_2 * t_opt_2))
        # if z_2 % 2 != 0: z_2 += 1
        # t_2 = (pi * d)/(b_2 * z_2)
        # beta_y = rad2deg(beta_2e) - 19.3*(t_opt_2 - 0.6)
        # dzeta_prof_2 = 7.4173*(M_2_t**3) - 10.511 * (M_2_t**2) + 1.0066*(M_2_t)
        # dzeta_konc_2 = 4.8786*(b2_l2)+4.9714
        # dzeta_Sum = dzeta_prof_2 + dzeta_konc_2
        # psi = sqrt(1 -dzeta_Sum/100)
        # w_2 = w_2_t*psi
        # beta_2 = asin(mu_2/psi * sin(beta_2e))
        # c_2 = sqrt((w_2**2+u**2) - 2*w_2 *u*cos(beta_2))
        # alpha_2 = atan((sin(beta_2))/(cos(beta_2)) - (u/w_2))
        
        # chi_vc = 0
        # Delta_H_p = (w_2_t**2 * (1 - psi**2))/2000
        # Delta_H_vc = (c_2**2)/2000
        # E_0 = H_0 - chi_vc*Delta_H_vc
        # eff_oi_energy = (E_0 - Delta_H_c - Delta_H_p - (1 - chi_vc)*Delta_H_vc)/E_0
        # eff_oi_angles = u*(c_1*(cos(alpha_1)) + c_2*(cos(alpha_2))/(E_0*1000))
        # u_c_f_opt = (phi+cos(alpha_1))/(2*sqrt(1-rho))
        # h_2 = point_2_t.h+Delta_H_p
        # # point_2 = input_class.calc_point(p = point_2_t.P, h= h_2)
        # print("Main error: ", (eff_oi_energy - eff_oi_angles)/eff_oi_angles*100, eff_oi_angles, eff_oi_energy)
        
        
        
        
        ###_______________________________________________________________________________________________
        
        
    
    def efficiency_oi_ol(self, total_enthalpy_drop, p0, t0, adiabatic_index, rpm, degree_of_reaction, 
                        nozzle_blade_height_mm, nozzle_angle_deg, nozzle_pitch_ratio, 
                        rotor_pitch_ratio, rotor_blade_height_mm, mass_flow, overlap, 
                        exit_loss_coeff, diameter):
        """Calculate turbine efficiency and velocity parameters"""
        
        initial_point = self.calc_zero_point[0]
        
        nozzle_enthalpy_drop = (1 - degree_of_reaction) * total_enthalpy_drop 
        rotor_enthalpy_drop = total_enthalpy_drop * degree_of_reaction
        
        nozzle_exit_enthalpy_ideal = initial_point.h - nozzle_enthalpy_drop 
        nozzle_exit_point_ideal = input_class.calc_point(h=nozzle_exit_enthalpy_ideal, s=initial_point.s)
        nozzle_exit_velocity_ideal = (2000 * nozzle_enthalpy_drop) ** 0.5 
        nozzle_sound_speed = (adiabatic_index * nozzle_exit_point_ideal.P * nozzle_exit_point_ideal.v * 1e6) ** 0.5
        nozzle_mach = nozzle_exit_velocity_ideal / nozzle_sound_speed
        
        nozzle_flow_coeff = 0.97 
        nozzle_throat_area_estimated = (mass_flow * nozzle_exit_point_ideal.v) / (nozzle_flow_coeff * nozzle_exit_velocity_ideal)
        blade_speed = diameter * np.pi * rpm
        characteristic_velocity = (2000 * total_enthalpy_drop) ** 0.5
        speed_ratio = blade_speed / characteristic_velocity
        
        nozzle_blade_height = nozzle_blade_height_mm * 1e-3 
        rotor_blade_height = rotor_blade_height_mm * 1e-3
        
        nozzle_angle_rad = np.deg2rad(nozzle_angle_deg)
        relative_pitch = nozzle_throat_area_estimated / (np.pi * diameter * np.sin(nozzle_angle_rad))
        optimal_contraction = 5 * (relative_pitch ** 0.5)
        if optimal_contraction > 0.85: 
            optimal_contraction = 0.85
        
        nozzle_blade_length = relative_pitch / optimal_contraction
        nozzle_blade_count = round((pi * diameter * optimal_contraction) / (nozzle_blade_height * nozzle_pitch_ratio), 0)
        if nozzle_blade_count % 2 != 0: 
            nozzle_blade_count += 1
        
        actual_pitch = (pi * diameter * optimal_contraction) / (nozzle_blade_height * nozzle_blade_count)
        actual_flow_coeff = 0.982 - 0.005 * (nozzle_blade_height / nozzle_blade_length)
        
        height_to_length_ratio = nozzle_blade_height / nozzle_blade_length
        profile_loss_coeff = (-3.7708 * (nozzle_flow_coeff**3) + 14.189 * (nozzle_flow_coeff**2) - 
                             13.478 * nozzle_flow_coeff + 5.6125)
        end_loss_coeff = 2 * height_to_length_ratio + 2
        optimal_angle = rad2deg(nozzle_angle_rad) - 16 * (nozzle_pitch_ratio - 0.75) + 23.1
        total_loss_coeff = profile_loss_coeff + end_loss_coeff
        velocity_coeff = (1 - (total_loss_coeff)/100) ** 0.5
        
        actual_exit_velocity = nozzle_exit_velocity_ideal * velocity_coeff
        actual_exit_angle = asin((actual_flow_coeff/velocity_coeff) * sin(nozzle_angle_rad))
        relative_inlet_velocity = sqrt((actual_exit_velocity ** 2) + (blade_speed ** 2) - 
                                      2 * actual_exit_velocity * blade_speed * np.cos(actual_exit_angle))
        beta_1 = atan(np.sin(actual_exit_angle)/(cos(actual_exit_angle) - (blade_speed/actual_exit_velocity)))
        
        nozzle_energy_loss = nozzle_exit_velocity_ideal ** 2 * (1 - velocity_coeff ** 2) / 2000  
        nozzle_exit_enthalpy_actual = nozzle_exit_enthalpy_ideal + nozzle_energy_loss
        nozzle_exit_point_actual = input_class.calc_point(p=nozzle_exit_point_ideal.P, h=nozzle_exit_enthalpy_actual)
        
        rotor_exit_enthalpy_ideal = nozzle_exit_point_actual.h - rotor_enthalpy_drop
        rotor_exit_point_ideal = input_class.calc_point(h=rotor_exit_enthalpy_ideal, s=nozzle_exit_point_actual.s) 
        rotor_relative_velocity_ideal = (2000 * rotor_enthalpy_drop + relative_inlet_velocity ** 2) ** 0.5
        rotor_blade_length = nozzle_blade_length + overlap
        
        rotor_sound_speed = (adiabatic_index * rotor_exit_point_ideal.P * rotor_exit_point_ideal.v * 1e6) ** 0.5
        rotor_mach_estimated = rotor_relative_velocity_ideal / rotor_sound_speed
        rotor_height_to_length = rotor_blade_height / rotor_blade_length
        rotor_flow_coeff = 0.965 - 0.01 * rotor_height_to_length
        rotor_throat_area = (mass_flow * rotor_exit_point_ideal.v) / (rotor_flow_coeff * rotor_relative_velocity_ideal)
        
        rotor_exit_angle = np.arcsin(rotor_throat_area / (optimal_contraction * np.pi * diameter * rotor_blade_length))
        rotor_blade_count = round((pi * diameter) / (rotor_blade_height * rotor_pitch_ratio)) 
        if rotor_blade_count % 2 != 0: 
            rotor_blade_count += 1
        
        rotor_pitch = (np.pi * diameter) / (rotor_blade_height * rotor_blade_count)
        optimal_beta_2 = np.rad2deg(rotor_exit_angle) - 19.3 * (rotor_pitch_ratio - 0.6) + 60
        
        rotor_profile_loss = (7.4173 * (rotor_mach_estimated**3) - 10.511 * (rotor_mach_estimated**2) + 
                             1.0066 * rotor_mach_estimated + 6.4416)
        rotor_end_loss = 4.8786 * rotor_height_to_length + 4.9714
        rotor_total_loss = rotor_profile_loss + rotor_end_loss
        rotor_velocity_coeff = (1 - rotor_total_loss/100) ** 0.5 
        
        actual_rotor_exit_velocity = rotor_relative_velocity_ideal * rotor_velocity_coeff
        actual_beta_2 = np.arcsin((rotor_flow_coeff/rotor_velocity_coeff) * np.sin(rotor_exit_angle))
        absolute_exit_velocity = ((actual_rotor_exit_velocity**2) + (blade_speed**2) - 
                                 2 * actual_rotor_exit_velocity * blade_speed * np.cos(actual_beta_2)) ** 0.5
        absolute_exit_angle = np.arctan((np.sin(actual_beta_2))/((np.cos(actual_beta_2)) - (blade_speed/actual_rotor_exit_velocity)))
        
        rotor_energy_loss = rotor_relative_velocity_ideal ** 2 * (1 - (rotor_velocity_coeff ** 2)) / 2000
        exit_velocity_loss = (absolute_exit_velocity ** 2) / 2000
        available_energy = total_enthalpy_drop - exit_loss_coeff * exit_velocity_loss
        
        energy_loss_efficiency = (available_energy - nozzle_energy_loss - rotor_energy_loss - 
                                (1 - exit_loss_coeff) * exit_velocity_loss) / available_energy
        velocity_triangle_efficiency = (blade_speed * (actual_exit_velocity * np.cos(actual_exit_angle) + 
                                      absolute_exit_velocity * np.cos(absolute_exit_angle))) / (available_energy * 1000)
        optimal_speed_ratio = (velocity_coeff * np.cos(actual_exit_angle)) / (2 * (1 - degree_of_reaction) ** 0.5)
        
        rotor_exit_enthalpy_actual = rotor_exit_point_ideal.h + rotor_energy_loss
        rotor_exit_point_actual = input_class.calc_point(p=rotor_exit_point_ideal.P, h=rotor_exit_enthalpy_actual)
        
        seal_count = 3
        mean_diameter = diameter + rotor_blade_length
        axial_clearance_coeff = 0.5
        axial_clearance = 0.0025
        radial_clearance_coeff = 0.75
        radial_clearance = 0.001 * mean_diameter
        
        equivalent_clearance = ((1/(axial_clearance_coeff * axial_clearance)**2) + 
                              (seal_count/(radial_clearance_coeff * radial_clearance)**2)) ** (-0.5)
        
        leakage_loss_coeff = ((np.pi * mean_diameter * equivalent_clearance * energy_loss_efficiency) * 
                            ((degree_of_reaction + 1.8 * (rotor_blade_length/diameter)) ** 0.5) / nozzle_throat_area_estimated)
        
        leakage_energy_loss = leakage_loss_coeff * available_energy
        
        friction_coeff = 0.7e-3
        friction_loss_coeff = (friction_coeff * (diameter ** 2)) * (speed_ratio ** 3) / nozzle_throat_area_estimated
        friction_energy_loss = friction_loss_coeff * available_energy
        
        partial_admission_coeff = 0.065
        admission_segments = 1
        partial_loss_coeff = (partial_admission_coeff * (1 - optimal_contraction) * (speed_ratio ** 3) * admission_segments) / (
            (np.sin(np.rad2deg(nozzle_angle_rad))) * optimal_contraction)
        
        B_2 = rotor_blade_height * np.sin(np.deg2rad(optimal_beta_2))
        partition_count = 4
        segment_loss_coeff = 0.25 * (B_2 * rotor_blade_length * speed_ratio * partition_count * energy_loss_efficiency) / nozzle_throat_area_estimated
        total_partial_loss_coeff = partial_loss_coeff + segment_loss_coeff
        partial_energy_loss = total_partial_loss_coeff * available_energy
        
        internal_energy = (available_energy - nozzle_energy_loss - rotor_energy_loss - 
                          (1 - exit_loss_coeff) * exit_velocity_loss - 
                          leakage_energy_loss - friction_energy_loss - partial_energy_loss)
        internal_efficiency = internal_energy / available_energy
        
        if diameter == 1.1:
            self._plot_velocity_triangles(actual_exit_velocity, actual_exit_angle, blade_speed,
                                        actual_rotor_exit_velocity, rotor_exit_angle,
                                        absolute_exit_velocity, diameter)
        
        return energy_loss_efficiency, velocity_triangle_efficiency, internal_efficiency, optimal_speed_ratio
    
    def _plot_velocity_triangles(self, c1, alpha1, u, w2, beta2e, c2, diameter):
        """Helper method to plot velocity triangles"""
        sin_alpha1 = math.sin(alpha1)
        cos_alpha1 = math.cos(alpha1)
        sin_beta2 = math.sin(beta2e)
        cos_beta2 = math.cos(beta2e)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        
        ax.plot([0, -c1 * cos_alpha1], [0, -c1 * sin_alpha1], label='C_1', c='red')
        ax.plot([-c1 * cos_alpha1, -c1 * cos_alpha1 + u], 
                [-c1 * sin_alpha1, -c1 * sin_alpha1], label='u_1', c='blue')
        ax.plot([0, -c1 * cos_alpha1 + u], [0, -c1 * sin_alpha1], label='W_1', c='green')
        
        ax.plot([0, w2 * cos_beta2], [0, -w2 * sin_beta2], label='W_2', c='green')
        ax.plot([w2 * cos_beta2, w2 * cos_beta2 - u], 
                [-w2 * sin_beta2, -w2 * sin_beta2], label='u_2', c='blue')
        ax.plot([0, w2 * cos_beta2 - u], [0, -w2 * sin_beta2], label='C_2', c='red')
        
        ax.set_title(f"Velocity triangles for diameter with maximum blade efficiency: {diameter}")
        ax.legend()
        plt.show()