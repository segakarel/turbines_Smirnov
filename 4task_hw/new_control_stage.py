from steam import Steam as sgas
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from iapws import IAPWS97
import numpy as np
from mass_flow_class import mass_flow_class as mfc
import math 
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
MPa = 10**6
unit = 1 / MPa

class MyClass:
    def __init__(self,steam_class, mfc_class, d, n, ro, H_0, mass_flow,alpha_1e, b_1, t_1opt, b_2, t_2opt, xi_vs):
        self.gas = steam_class
        self.mfc = mfc_class
        self.d = d
        self.n = n
        self.ro = ro
        self.H_0 = H_0
        self.k = 1.33
        self.mu_t = 0.97
        self.G = mass_flow
        self.alpha_1e = np.deg2rad(alpha_1e)
        self.b_1 = b_1
        self.t_1opt = t_1opt
      
        self.mu_a = 0.5
        self.delt_a = 0.0025
        self.mu_r = 0.75
        self.z = 3
        self.b_2 = b_2
        self.t_2opt = t_2opt
        self.xi_vs = xi_vs
        self.update()
     
    def plot_distribution(self, values, n_stages,  ax_name):
        fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=300)
        ax.plot(range(1, n_stages+1), values,  marker='o')
        ax.set_xlabel("Номер ступени")
        ax.set_ylabel(ax_name)
        ax.grid()
    
    def blade_update(self, n_stages,my_blade_No):
        mass_flow = self.G
        rotation_speed = self.n
        h0 = self._point_2_t.h + self._delt_H_parc +self._delt_H_p +self._delt_H_y + self.delt_H_vc 
        speed_stage_diam = self.d
        delta_diam =  0.2 # Разница между диаметром первой нерегулируемой ступени и регулирующей ступени
        root_reaction_degree = 0.03  # Степень реактивности первой нерегулируемой ступени в корне
        speed_coefficient = 0.93    # Коэффициент скорости сопловой решетки
        discharge_coefficient = 0.95     #Коэффициент расхода сопловой решетки первой нерегулируемой ступени
        overlapping = 0.03 #Перекрыша между высотами лопаток первой нерегулируемой ступени
        efficiency = 0.88   #Внутренний КПД ЦВД
        p0 = self._point_2.P
        veernost_1 = 37
        while True:
            
            avg_diam_1 = speed_stage_diam - delta_diam
    
            point_0 = self.gas.calc_point(p=p0, h=h0)
            
            def get_reaction_degree(root_dor, veernost):
                return root_dor + (1.8 / (veernost + 1.8))
            
            def get_u_cf(dor):
                cos = np.cos((self._alpha_1))
                return speed_coefficient * cos / (2 * (1 - dor) ** 0.5)
            
            def get_heat_drop(diameter, u_cf):
                first = (diameter / u_cf) ** 2
                second = (rotation_speed / 50) ** 2
                return 12.3 * first * second
            
            avg_reaction_degree_1 = get_reaction_degree(root_reaction_degree, veernost_1)
            u_cf_1 = get_u_cf(avg_reaction_degree_1)
            heat_drop_1 = get_heat_drop(avg_diam_1, u_cf_1)
            
            h1 = point_0.h - heat_drop_1
            point_2 = self.gas.calc_point(h=h1, s=point_0.s)
            
            upper = mass_flow * point_2.v * u_cf_1
            lower = discharge_coefficient * np.sin(self._alpha_1) * rotation_speed * (np.pi * avg_diam_1) ** 2 * (1 - avg_reaction_degree_1) ** 0.5
            
            blade_length_1 = upper / lower
            blade_length_2 = blade_length_1 + overlapping
            print("Ошибка определения веерности: ", abs(avg_diam_1 / blade_length_1-veernost_1)/veernost_1*100, "%'")
            if abs(avg_diam_1 / blade_length_1-veernost_1)/veernost_1*100  <1 :
                break
            veernost_1 = avg_diam_1 / blade_length_1
        # while True:
              
        root_diameter = avg_diam_1 - blade_length_2
        pz = self.mfc.point_1.P * MPa
        point_zt = self.gas.calc_point(p=pz * unit, s=point_0.s)
        full_heat_drop = h0 - point_zt.h
        actual_heat_drop = full_heat_drop * efficiency
        hz = h0 - actual_heat_drop
        point_z = self.gas.calc_point(p=pz * unit, h=hz)
        
        def equation_to_solve(x):
            return x ** 2 + x * root_diameter - avg_diam_1 * blade_length_2 * point_z.v / point_2.v
        
        blade_length_z = fsolve(equation_to_solve, 0.01)[0]
        
        avg_diam_2 = root_diameter + blade_length_z
        
        def linear_distribution(left, right, x):
            return (right - left) * x + left
        
        x = np.cumsum(np.ones(n_stages) * 1 / (n_stages - 1)) - 1 / (n_stages - 1)
        diameters = linear_distribution(avg_diam_1, avg_diam_2 , x)

        blade_lengths = linear_distribution(blade_length_2, blade_length_z , x)
        veernosts = diameters / blade_lengths
        
        reaction_degrees = get_reaction_degree(root_dor=root_reaction_degree, veernost=veernosts)
        u_cf = get_u_cf(dor=reaction_degrees)
        
        heat_drops = get_heat_drop(diameters, u_cf)
        output_speed_coeff_loss = np.full_like(heat_drops, 0.95)
        output_speed_coeff_loss[0] = 1
        
        actual_heat_drops = output_speed_coeff_loss * heat_drops
        mean_heat_drop = np.mean(actual_heat_drops)
        reheat_factor = 4.8 * 10 ** (-4) * (1 - efficiency) * full_heat_drop * (n_stages - 1) / n_stages
        
        print("Новое число ступеней: ", (full_heat_drop * (1 + reheat_factor) / mean_heat_drop))
            # if round(full_heat_drop * (1 + reheat_factor) / mean_heat_drop)- n_stages == 0:
            #     break
            
        # n_stages = round(full_heat_drop * (1 + reheat_factor) / mean_heat_drop)
    
        bias = full_heat_drop * (1 + reheat_factor) - np.sum(actual_heat_drops)
        bias = bias / n_stages
        
        new_actual_heat_drop = actual_heat_drops + bias
        
        self.plot_distribution(diameters, n_stages, "d, m")
        self.plot_distribution(blade_lengths, n_stages, "l, m")
        self.plot_distribution(veernosts, n_stages, "Веерность")
        self.plot_distribution(reaction_degrees, n_stages, "Степень реактивности")
        self.plot_distribution(u_cf, n_stages, "U/Cф")
        self.plot_distribution(new_actual_heat_drop, n_stages, "Теплоперепады по ступеням")
        
        points = [self.gas.calc_point(p=p0 , h=h0)]  # Начальная точка
        pressures = [p0 ]  # Начальное давление
        temperatures = [points[0].T]  # Начальная температура
        
        for i in range(n_stages):
            # Рассчитываем параметры после текущей ступени
            h_next = points[i].h - new_actual_heat_drop[i]
            # Для давления можно использовать политропический процесс
            s_next = points[i].s  # Для изоэнтропического процесса
            point_next = self.gas.calc_point(h=h_next, s=s_next)
            points.append(point_next)
            pressures.append(point_next.P)
            temperatures.append(point_next.T)
        
        # Теперь для любой ступени (например, 8-й) можно получить параметры на входе:
        if my_blade_No <= n_stages:
            p_in = pressures[my_blade_No-1]
            T_in = temperatures[my_blade_No-1]
            h_in = points[my_blade_No-1].h
        else:
            raise ValueError("Запрошен номер ступени, превышающий общее количество ступеней")
                
        return (diameters[my_blade_No-1], blade_lengths[my_blade_No-1], veernosts[my_blade_No-1], 
                reaction_degrees[my_blade_No-1], u_cf[my_blade_No-1], new_actual_heat_drop[my_blade_No-1],
                p_in, T_in, h_in)  # Добавляем возвращаемые параметры
    
    def my_blade(self, p0, t0, adiabatic_index,
                       exit_loss_coeff, diameter,atlas_min_modulus, nozzle_blade_length):
       """Calculate turbine efficiency and velocity parameters"""
       
       h0 = self._point_2_t.h + self._delt_H_parc +self._delt_H_p +self._delt_H_y + self.delt_H_vc 
       point_0 = self.gas.calc_point(p=p0, h=h0)
       total_enthalpy_drop = self.H_0
       rpm = self.n
       degree_of_reaction = 0.05
       nozzle_blade_height_mm = nozzle_blade_length*1000#self.b_1
       nozzle_angle_deg = self.alpha_1e
       nozzle_pitch_ratio = self.t_1opt
       rotor_pitch_ratio = self.t_2opt
       rotor_blade_height_mm = self.b_2
       mass_flow = self.G
       overlap = 0.03
       diametr = self.d
       
       initial_point = point_0 
       
       nozzle_enthalpy_drop = (1 - degree_of_reaction) * total_enthalpy_drop 
       rotor_enthalpy_drop = total_enthalpy_drop * degree_of_reaction
       
       nozzle_exit_enthalpy_ideal = initial_point.h - nozzle_enthalpy_drop 
       nozzle_exit_point_ideal = self.gas.calc_point(h=nozzle_exit_enthalpy_ideal, s=initial_point.s)
       nozzle_exit_velocity_ideal = (2000 * nozzle_enthalpy_drop) ** 0.5 
       nozzle_sound_speed = (adiabatic_index * nozzle_exit_point_ideal.P * nozzle_exit_point_ideal.v * 1e6) ** 0.5
       
       nozzle_flow_coeff = 0.97 
       nozzle_throat_area_estimated = (mass_flow * nozzle_exit_point_ideal.v) / (nozzle_flow_coeff * nozzle_exit_velocity_ideal)
       blade_speed = diameter * np.pi * rpm #u
       characteristic_velocity = (2000 * total_enthalpy_drop) ** 0.5 #c_f
       speed_ratio = blade_speed / characteristic_velocity
       
       nozzle_blade_height = nozzle_blade_height_mm * 1e-3 
       rotor_blade_height = rotor_blade_height_mm * 1e-3
       
       nozzle_angle_rad = np.deg2rad(nozzle_angle_deg)
       relative_pitch = nozzle_throat_area_estimated / (np.pi * diameter * np.sin(nozzle_angle_rad))
       optimal_contraction = 5 * (relative_pitch ** 0.5)
       if optimal_contraction > 0.85: 
           optimal_contraction = 0.85
       
       # nozzle_blade_length = relative_pitch / optimal_contraction
       nozzle_blade_count = round((math.pi * diameter * optimal_contraction) / (nozzle_blade_height * nozzle_pitch_ratio), 0)
       if nozzle_blade_count % 2 != 0: 
           nozzle_blade_count += 1
       
       actual_flow_coeff = 0.982 - 0.005 * (nozzle_blade_height / nozzle_blade_length)
       
       height_to_length_ratio = nozzle_blade_height / nozzle_blade_length
       profile_loss_coeff = (-3.7708 * (nozzle_flow_coeff**3) + 14.189 * (nozzle_flow_coeff**2) - 
                             13.478 * nozzle_flow_coeff + 5.6125)
       end_loss_coeff = 2 * height_to_length_ratio + 2
       optimal_angle = np.rad2deg(nozzle_angle_rad) - 16 * (nozzle_pitch_ratio - 0.75) + 23.1
       total_loss_coeff = profile_loss_coeff + end_loss_coeff
       velocity_coeff = (1 - (total_loss_coeff)/100) ** 0.5
       
       actual_exit_velocity = nozzle_exit_velocity_ideal * velocity_coeff
       actual_exit_angle = math.asin((actual_flow_coeff/velocity_coeff) * math.sin(nozzle_angle_rad))
       relative_inlet_velocity = math.sqrt((actual_exit_velocity ** 2) + (blade_speed ** 2) - 
                                     2 * actual_exit_velocity * blade_speed * np.cos(actual_exit_angle))
       beta_1 = math.atan(np.sin(actual_exit_angle)/(math.cos(actual_exit_angle) - (blade_speed/actual_exit_velocity)))
       
       nozzle_energy_loss = nozzle_exit_velocity_ideal ** 2 * (1 - velocity_coeff ** 2) / 2000  
       nozzle_exit_enthalpy_actual = nozzle_exit_enthalpy_ideal + nozzle_energy_loss
       nozzle_exit_point_actual = self.gas.calc_point(p=nozzle_exit_point_ideal.P, h=nozzle_exit_enthalpy_actual)
       
       rotor_exit_enthalpy_ideal = nozzle_exit_point_actual.h - rotor_enthalpy_drop
       rotor_exit_point_ideal = self.gas.calc_point(h=rotor_exit_enthalpy_ideal, s=nozzle_exit_point_actual.s) 
       rotor_relative_velocity_ideal = (2000 * rotor_enthalpy_drop + relative_inlet_velocity ** 2) ** 0.5
       rotor_blade_length = nozzle_blade_length + overlap
       
       rotor_sound_speed = (adiabatic_index * rotor_exit_point_ideal.P * rotor_exit_point_ideal.v * 1e6) ** 0.5
       rotor_mach_estimated = rotor_relative_velocity_ideal / rotor_sound_speed
       rotor_height_to_length = rotor_blade_height / rotor_blade_length
       rotor_flow_coeff = 0.965 - 0.01 * rotor_height_to_length
       rotor_throat_area = (mass_flow * rotor_exit_point_ideal.v) / (rotor_flow_coeff * rotor_relative_velocity_ideal)
       
       rotor_exit_angle = np.arcsin(rotor_throat_area / (optimal_contraction * np.pi * diameter * rotor_blade_length))
       rotor_blade_count = round((math.pi * diameter) / (rotor_blade_height * rotor_pitch_ratio)) 
       if rotor_blade_count % 2 != 0: 
           rotor_blade_count += 1
       
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
       rotor_exit_point_actual = self.gas.calc_point(p=rotor_exit_point_ideal.P, h=rotor_exit_enthalpy_actual)
       
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
       
       leakage_energy_loss = 2*leakage_loss_coeff * available_energy
       
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
       
       # Power and strength calculations
       internal_power = mass_flow * internal_energy
       original_rotor_height = rotor_blade_height
       
       # Atlas minimum modulus (примерное значение, нужно уточнить)
       # atlas_min_modulus = 1e-6  # W_min_atlas - необходимо задать значение
       actual_min_modulus = atlas_min_modulus * (rotor_blade_height / original_rotor_height)**2
       
       # Material properties
       allowable_bending_stress = 20  # MPa
       material_density = 7800  # kg/m^3
       
       # Bending stress calculation
       bending_stress = (mass_flow * total_enthalpy_drop * velocity_triangle_efficiency * rotor_blade_length) / \
                       (2000 * blade_speed * rotor_blade_count * actual_min_modulus * optimal_contraction) 
       
       # Adjust rotor blade height if stress exceeds allowable

       if bending_stress > allowable_bending_stress:
           rotor_blade_height = original_rotor_height * math.sqrt(bending_stress / allowable_bending_stress)
           # Recalculate with new blade height
           bending_stress = (mass_flow * total_enthalpy_drop * velocity_triangle_efficiency * rotor_blade_length) / (2000 * blade_speed * rotor_blade_count * actual_min_modulus * optimal_contraction) 
           original_rotor_height = rotor_blade_height
       
       # Centrifugal stress calculation
       angular_velocity = 2 * math.pi * rpm
       centrifugal_stress = 0.5 * material_density * angular_velocity**2 * diameter * rotor_blade_length * 1e-6
       
       
       
       # Plot velocity triangles if needed
       # if diameter == 1.1:
       print(f"Centrifugal stress: {centrifugal_stress:.2f} MPa; "
             f"Bending stress: {bending_stress:.2f} MPa; "
             f"Final rotor height: {original_rotor_height*1000:.2f} m")
       
       self.both_plot_velocity_triangles
       ig, ax  = plt.subplots(1, 1, figsize=(15, 15))

   
       self.mfc.plot_points(ax, [initial_point])
       self.mfc.plot_points(ax, [nozzle_exit_point_actual])
       self.mfc.plot_points(ax, [rotor_exit_point_actual])
      
       ax.scatter(nozzle_exit_point_ideal.s, nozzle_exit_point_ideal.h, s=50, color="red")
       ax.scatter(rotor_exit_point_ideal.s, rotor_exit_point_ideal.h, s=50, color="red")
       self.mfc.plot_process(ax, points=[ initial_point, nozzle_exit_point_actual], color='black')
       self.mfc.plot_process(ax, points=[nozzle_exit_point_actual, rotor_exit_point_actual], color='black')
       self.mfc.plot_process(ax, points=[initial_point, nozzle_exit_point_ideal], alpha=0.5, color='grey')
       self.mfc.plot_process(ax, points=[nozzle_exit_point_actual, rotor_exit_point_ideal], alpha=0.5, color='grey')
       ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
       ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
       ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
       ax.legend()
       ax.grid()
       ax.set_ylim(rotor_exit_point_actual.h - 10, initial_point.h + 10)
       ax.set_xlim(initial_point.s - 0.005, rotor_exit_point_actual.s + 0.005)
       self.mfc.legend_without_duplicate_labels(ax)
       plt.show()

       return energy_loss_efficiency, velocity_triangle_efficiency, internal_efficiency,speed_ratio
   
    def plot_smoothed_profile(self,xu, yu, xl, yl, num_points=100):
        # Интерполяция для верхнего профиля (xu, yu)
        f_u = interp1d(xu, yu, kind='cubic')  # cubic даёт гладкую кривую
        x_u_smooth = np.linspace(min(xu), max(xu), num_points)
        y_u_smooth = f_u(x_u_smooth)
    
        # Интерполяция для нижнего профиля (xl, yl)
        f_l = interp1d(xl, yl, kind='cubic')
        x_l_smooth = np.linspace(min(xl), max(xl), num_points)
        y_l_smooth = f_l(x_l_smooth)
    
        # Отрисовка
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_u_smooth, y_u_smooth, label='Верхний профиль (сглаженный)')
        ax.plot(x_l_smooth, y_l_smooth, label='Нижний профиль (сглаженный)')
        
        # Отображение исходных точек (опционально)
        ax.scatter(xu, yu, color='blue', s=30, label='Исходные точки (верх)')
        ax.scatter(xl, yl, color='red', s=30, label='Исходные точки (низ)')
        
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.title("Профиль рабочей лопатки")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
    def update(self):
        self.delta = 0.0035
        self._u = math.pi * self.d * self.n
        self._H_0c = (1 - self.ro) * self.H_0
        self._H_0r = self.H_0 * self.ro
        self._h_1t = self.mfc.point_0.h - self.H_0c
        self._point_1_t = self.gas.calc_point(h = self.h_1t, s = self.mfc.point_0.s)
        self._c_1t = (2000 * self.H_0c)**0.5
        self._a_1t = (self.k * self._point_1_t.P * self._point_1_t.v * MPa) ** 0.5
        self._M_1t = self._c_1t / self._a_1t
        self._F_1 = (self.G * self._point_1_t.v) / (self.mu_t * self._c_1t)
        self._el_1 = self._F_1 / (math.pi* self.d* math.sin(self.alpha_1e))
        
        self._e_opt = 5 *self._el_1**0.5
        if self._e_opt > 0.85:
            self._e_opt = 0.85
        else:self._e_opt = self._e_opt
        
        
        self._l_1 = self._el_1 / self._e_opt
        self._mu_1 = 0.982 - 0.005*(self.b_1 / self._l_1)
        self._z_1 = math.ceil((math.pi * self.d * self._e_opt) / (self.b_1 * self.t_1opt))
        self._t_1 = (math.pi*self.d * self._e_opt) / (self.b_1 * self._z_1)
        self._alpha_ust = self.alpha_1e - 16*(self._t_1  - 0.75) + 23.1
        self._dzita_c = - 3.7708 * ((self._mu_1)**3) + 14.189 * ((self._mu_1)**2) - 13.478 * (self._mu_1) + 5.6125
        self._phi = (1- (self._dzita_c )/100)**0.5
        self._phi_t = 0.98 - 0.008*(self.b_1 / self._l_1)
        self._delta_phi = (self._phi - self._phi_t) / self._phi * 100
        
        self._c_1 = self._c_1t * self._phi
        self._alpha_1 = math.asin((self._mu_1/ self._phi)* math.sin(self.alpha_1e))
        self._w_1 = (self._c_1**2  + (self.u **2) - 2*self._c_1 * self.u* math.cos(self._alpha_1) )**0.5
        self._betta_1 = math.atan(math.sin(self._alpha_1) /( math.cos(self._alpha_1) - self.u / self._c_1))
        
        self._delta_Hc = self._c_1t**2*(1- self._phi**2) /2000  
        self._h_1 = self._h_1t + self._delta_Hc
        self._point_1 = self.gas.calc_point(p = self._point_1_t.P, h =self._h_1)
        self._h_2_t = self._point_1.h - self._H_0r
        self._point_2_t = self.gas.calc_point(h = self._h_2_t, s = self._point_1.s) 
        
        self._w_2_t =(2000 * self._H_0r + self._w_1 ** 2) ** 0.5
        self._l_2 = self._l_1 + self.delta
        
        self._a_2t = (1.4 * self._point_2_t.P * MPa * self._point_2_t.v)**0.5
        
        self._m_2_pred = self._w_2_t/ self._a_2t

        self._b_2_l_2 = self.b_2/self._l_2
        self._mu_2 = 0.965 - 0.01*(self._b_2_l_2)
        self._F_2 = (self.G * self._point_2_t.v)/(self._mu_2 * self._w_2_t)
        
        self._a_2_t = (self.k * self._point_2_t.P * self._point_2_t.v * (10 ** 6)) ** 0.5 
        
        self._betta_2e = np.arcsin(self._F_2/(self._e_opt * np.pi * self.d * self._l_2 ))
        self._z_2 = math.ceil((math.pi * self.d)/(self.b_2 * self.t_2opt)) 
        self._t_2 = (np.pi * self.d)/(self.b_2 * self._z_2)
        self._bet_yst = np.rad2deg(self._betta_2e) - 19.3 * (self.t_2opt - 0.6) + 60
        self._zit_prof = 7.4173 * ((self._m_2_pred)**3) -10.511 * ((self._m_2_pred)**2) + 1.0066 * (self._m_2_pred) + 6.4416
        self._zit_konc_l_d = 4.8786 * (self._b_2_l_2) + 4.9714
        self._zit_r = self._zit_prof + self._zit_konc_l_d 
        self._ksi = (1 - self._zit_r/100) ** 0.5
        self._w_2 = self._w_2_t * self._ksi
        self._bett_2 = np.arcsin((self._mu_2/self._ksi)* np.sin(self._betta_2e))
        self._c_2 = ((self._w_2**2) + (self._u**2) - 2 * self._w_2 * self._u * np.cos(self._bett_2)) ** 0.5
        self._alpha_2 = np.arctan((np.sin(self._bett_2))/((np.cos(self._bett_2)) - (self._u/self._w_2)))
        self._delt_H_p = self._w_2_t ** 2 * (1 - (self._ksi ** 2)) /2000
        self._delt_H_vc = (self._c_2 ** 2)/(2 * 1000)
        self._E_0 = self.H_0 - self.xi_vs * self._delt_H_vc
        self._eff_oi_loss = (self._E_0 - self._delta_Hc - self._delt_H_p - (1 - self.xi_vs) *  self._delt_H_vc) / self._E_0
        self._u_c_f_opt = (self._phi * np.cos(self._alpha_1)) /(2 * (1 - self.ro) ** 0.5)
        self._c_f = (2 * self.H_0*1000)**0.5
        self._u_c_f = self._u / self._c_f
        self._h_2 = self._point_2_t.h + self._delt_H_p
        self._point_2 = self.gas.calc_point( p = self._point_2_t.P, h = self._h_2)
        self._d_p = self.d + self._l_2

        self._delt_g = 0.001 * self._d_p
        
        self._delt_e = ((1/(self.mu_a * self.delt_a)**2) + (self.z/(self.mu_r * self._delt_g)**2)) ** (-0.5)
        
        self._ksi_b_y = (np.pi * self._d_p * self._delt_e * self._eff_oi_loss) * ((self.ro + 1.8 * (self._l_2/self.d)) ** 0.5) / self._F_1
        
        self._delt_H_y = self._ksi_b_y * self._E_0
        self._k_tr = 0.7 * (10 ** -3)
        self._ksi_tr = (self._k_tr  * ((self.d) ** 2)) * ((self._u_c_f) ** 3) / self._F_1
        self._delt_H_tr = self._ksi_tr * self._E_0
        k_v = 0.065
        m = 1
        self._zitta_v = (k_v * (1 - self._e_opt) * ((self._u_c_f) ** 3) * m) / ((np.sin(np.rad2deg(self.alpha_1e))) * self._e_opt)
        self.B_2 = self.b_2 * np.sin(np.deg2rad(self._bet_yst))
        i = 4
        self._ksi_segm = 0.25 * (self.B_2 * self._l_2 * self._u_c_f * i * self._eff_oi_loss) / self._F_1
        self._ksi_parc = self._zitta_v + self._ksi_segm
        self._delt_H_parc = self._ksi_parc * self._E_0
        self._H_i = self._E_0- self._delta_Hc - self._delt_H_p - (1 - self.xi_vs) *  self._delt_H_vc - self._delt_H_y - self._delt_H_tr - self._delt_H_parc
        self._kpd_oi = self._H_i/self._E_0
        self._eff_oi_velocity = self._u * (self._c_1 * (np.cos(self._alpha_1)) + self._c_2 * (np.cos(self._alpha_2))) / (self.E_0 * 1000)
    
    def Vibration_diagram(self, m, t, beta, density, E,z,d, l,f,J,delta,B, psi):
        mm = 1 / 1000
        i = (J / f) ** 0.5
        _lambda = l / i
        def static_frequency(i):
            _m = {
                1: 0.56,
                2: 3.51,
                3: 9.82
         
            }
            first = psi * _m[i] / (l ** 2)
            second = ((E * J) / (density * f)) ** 0.5
            return first * second
        H = 0.12
        J_b = B * (delta ** 3) / 12

        k = (12 * (m - 1) * H * E * J_b * l * np.sin(np.deg2rad(beta)) ** 2) / (m * t * J * E)
        nu = B * delta * t / (f * l)
        B_bandage = 0.5 * ((d/l) - 1) * ((nu+1/2)/(nu+1/3)) + np.sin(np.deg2rad(beta)) ** 2
        def to_dynamic_frequency(f, n=60):
            root = (1 + B_bandage * (n / f) ** 2) ** 0.5
            return f * root
        
        def min_max(f, delta=0.05):
            return f * (1-delta) , f * (1 + delta)
        f_a0 = static_frequency(1) * 0.8
        f_a1 = static_frequency(1) * 6
        f_b0 = static_frequency(1) * 4.2
        n_line = np.linspace(0, 60)
        min_line, max_line = min_max(to_dynamic_frequency(f_a0, n=n_line))
        def k_line(k, n=n_line):
            return k * n_line
        
        fig, ax = plt.subplots(1,1,figsize=(15,10))
        ax.plot(n_line, to_dynamic_frequency(f_a0, n=n_line), label='$f_{a0}$')
        ax.fill_between(n_line, y1=min_line, y2=max_line, alpha=0.5)
        
        ax.plot(n_line, k_line(1), label=f'k={1}')
        ax.plot(n_line, k_line(2), label=f'k={2}')
        ax.plot(n_line, k_line(3), label=f'k={3}')
        ax.plot(n_line, k_line(4), label=f'k={4}')
        ax.plot(n_line, k_line(5), label=f'k={5}')
        ax.plot(n_line, k_line(6), label=f'k={6}')
        ax.set_xlabel("n, rps")
        ax.set_xlabel("f, Hz")
        ax.grid()
        ax.legend()
        ax.set_title("Вибрационная диаграмма");
        pass
    
    
    
    def stress(self, b_2,b_2atl ,moment_of_resistance_min_atl, rho):
        moment_of_resistance_min = (b_2 / b_2atl)**3 * moment_of_resistance_min_atl
        sigma_isg_base = 20
        sigma_isg = (self.G * self.H_0 * self._eff_oi_loss * self._l_2 )/ (2 * self._u * self._z_2 * moment_of_resistance_min * self._e_opt * 10**6) 
        b_2 = self.b_2 * (sigma_isg / sigma_isg_base)**0.5
        w = 2 * math.pi * self.n
        sigma_rast = 0.5 * rho * w**2 * self.d * self._l_2 * 1e-6
        return sigma_isg, sigma_rast
    
    @property
    def hs(self):
        fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
        initial_point = self.mfc.point_0
        nozzle_exit_point_ideal  = self._point_1_t
        nozzle_exit_point_actual = self._point_1
        rotor_exit_point_ideal = self._point_2_t
        rotor_exit_point_actual = self._point_2
        points=[ initial_point, nozzle_exit_point_ideal, nozzle_exit_point_actual, rotor_exit_point_ideal, rotor_exit_point_actual]
    
        self.mfc.plot_points(ax, [initial_point])
        self.mfc.plot_points(ax, [nozzle_exit_point_actual])
        self.mfc.plot_points(ax, [rotor_exit_point_actual])
       
        ax.scatter(nozzle_exit_point_ideal.s, nozzle_exit_point_ideal.h, s=50, color="red")
        ax.scatter(rotor_exit_point_ideal.s, rotor_exit_point_ideal.h, s=50, color="red")
        self.mfc.plot_process(ax, points=[ initial_point, nozzle_exit_point_actual], color='black')
        self.mfc.plot_process(ax, points=[nozzle_exit_point_actual, rotor_exit_point_actual], color='black')
        self.mfc.plot_process(ax, points=[initial_point, nozzle_exit_point_ideal], alpha=0.5, color='grey')
        self.mfc.plot_process(ax, points=[nozzle_exit_point_actual, rotor_exit_point_ideal], alpha=0.5, color='grey')
        ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
        ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
        ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
        ax.legend()
        ax.grid()
        ax.set_ylim(rotor_exit_point_actual.h - 10, initial_point.h + 10)
        ax.set_xlim(initial_point.s - 0.005, rotor_exit_point_actual.s + 0.005)
        self.mfc.legend_without_duplicate_labels(ax)
        plt.show()
        
    @property
    def both_plot_velocity_triangles(self):
        alpha_1 = self._alpha_1 
        betta_2 = self._bett_2
        c_1 = self._c_1
        w_2 = self._w_2
        
        sin_alpha1 = math.sin(alpha_1)
        cos_alpha1 = math.cos(alpha_1)
        sin_beta2 = math.sin(betta_2)
        cos_beta2 = math.cos(betta_2)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot([0, -c_1 * cos_alpha1], [0, -c_1 * sin_alpha1], label='C_1', c='red')
        ax.plot([-c_1 * cos_alpha1, -c_1 * cos_alpha1 + self.u], 
                [-c_1 * sin_alpha1, -c_1 * sin_alpha1], label='u_1', c='blue')
        ax.plot([0, -c_1 * cos_alpha1 + self.u], [0, -c_1 * sin_alpha1], label='W_1', c='green')
        
        ax.plot([0, w_2 * cos_beta2], [0, -w_2 * sin_beta2], label='W_2', c='green')
        ax.plot([w_2 * cos_beta2, w_2 * cos_beta2 - self.u], 
                [-w_2 * sin_beta2, -w_2 * sin_beta2], label='u_2', c='blue')
        ax.plot([0, w_2 * cos_beta2 - self.u], [0, -w_2 * sin_beta2], label='C_2', c='red')
        
        ax.set_title("Треугольники скоростей при оптимальном диаметре")
        ax.legend()
        plt.grid(True)
        plt.show()
        
    @property
    def create_results_tables(self):
            """
            Создает три таблицы с результатами расчетов ступени турбины
            
            Args:
                c_1, c_2: абсолютные скорости [м/с]
                alpha_1, alpha_2: абсолютные углы [рад]
                w_1, w_2: относительные скорости [м/с]
                bett_1, bett_2: относительные углы [рад]
                delt_H_*: различные потери энергии [кДж/кг]
                E_o: располагаемая энергия [кДж/кг]
                point_*: объекты с термодинамическими параметрами
                
            Returns:
                tuple: (velocity_df, loss_df, thermodynamic_df)
            """
            geom_df = pd.DataFrame({
                "Длина, м": [
                    self._l_1,
                    self._l_2,
                    self.b_1,
                    self.b_2,
                ],
                "Геометрия": [
                    "Длина рабочей лопатки",
                    "Длина сопловой лопатки",
                    "Ширина рабочей лопатки",
                    "Ширина сопловой лопатки"
                ]
            }).set_index("Геометрия")
        
            velocity_df = pd.DataFrame({
                "Абсолютная скорость (с), м/с": [np.round(self._c_1, 3), np.round(self._c_2, 3)],
                "Абсолютный угол (α), °": [np.round(np.rad2deg(self._alpha_1), 3), 
                                          np.round(np.rad2deg(self._alpha_2), 3)],
                "Относительная скорость (w), м/с": [np.round(self._w_1, 3), np.round(self._w_2, 3)],
                "Относительный угол (β), °": [np.round(np.rad2deg(self._betta_1), 3), 
                                             np.round(np.rad2deg(self._bett_2), 3)],
                "Локация": ["За сопловой решеткой", "За рабочей решеткой"]
            }).set_index("Локация")
        
     
            loss_df = pd.DataFrame({
                "Потери, %": [
                    np.round(self._delta_Hc/self._E_0 * 100, 3),
                    np.round(self._delt_H_p/self._E_0 * 100, 3),
                    np.round(self._delt_H_vc/self._E_0 * 100, 3),
                    np.round(self._delt_H_y/self._E_0 * 100, 3),
                    np.round(self._delt_H_tr/self._E_0 * 100, 3),
                    np.round(self._delt_H_parc/self._E_0 * 100, 3)
                ],
                "Тип потерь": [
                    "В сопловой решетке",
                    "В рабочей решетке", 
                    "Энергия выходной скорости",
                    "Потери от утечек через уплотнение",
                    "Потери от трения диска",
                    "Потери от парциальности"
                ]
            }).set_index("Тип потерь")
        
         
            thermodynamic_df = pd.DataFrame({
                "Давление (P), МПа": [
                    np.round(self.mfc.point_0.P, 3),
                    np.round(self.mfc.point_1t.P, 3),
                    np.round(self.mfc.point_1.P, 3),
                    np.round(self.mfc.point_2t.P, 3),
                    np.round(self.mfc.point_2.P, 3)
                ],
                "Температура (T), K": [
                    np.round(self.mfc.point_0.T, 3),
                    np.round(self.mfc.point_1t.T, 3),
                    np.round(self.mfc.point_1.T, 3),
                    np.round(self.mfc.point_2t.T, 3),
                    np.round(self.mfc.point_2.T, 3)
                ],
                "Энтальпия (h), кДж/кг": [
                    np.round(self.mfc.point_0.h, 3),
                    np.round(self.mfc.point_1t.h, 3),
                    np.round(self.mfc.point_1.h, 3),
                    np.round(self.mfc.point_2t.h, 3),
                    np.round(self.mfc.point_2.h, 3)
                ],
                "Энтропия (s), кДж/(кг·K)": [
                    np.round(self.mfc.point_0.s, 3),
                    np.round(self.mfc.point_1t.s, 3),
                    np.round(self.mfc.point_1.s, 3),
                    np.round(self.mfc.point_2t.s, 3),
                    np.round(self.mfc.point_2.s, 3)
                ],
                "Состояние": [
                    "Параметры перед ступенью",
                    "Теоретические параметры за сопловой решеткой",
                    "Фактические параметры за сопловой решеткой",
                    "Теоретические параметры за ступенью",
                    "Фактические параметры за ступенью"
                ]
            }).set_index("Состояние")
        
            return geom_df, velocity_df, loss_df, thermodynamic_df
    @property
    def u(self):return self._u
    
    @property
    def H_0c(self):return self._H_0c
    
    @property
    def H_0r(self):return self._H_0r
    
    @property
    def h_1t(self):return self._h_1t
    
    @property 
    def point_1_t(self):return self._point_1_t
    
    @property 
    def c_1t(self):return self._c_1t
    
    @property 
    def a_1t(self):return self._a_1t
    
    @property 
    def w_1(self):return self._w_1
    
    @property 
    def c_1(self):return self._c_1
    
    @property 
    def M_1t(self):return self._M_1t
    
    @property 
    def F_1(self):return self._M_1t
    
    @property 
    def el_1(self):return self._el_1
    
    @property 
    def e_opt(self):return self._e_opt
    
    @property 
    def l_1(self):return self._l_1
    
    @property
    def mu_1(self):return self._mu_1
    
    @property
    def z_1(self):return self._z_1
    
    @property
    def t_1(self):return self._t_1
    
    @property
    def alpha_ust(self):return self._alpha_ust
    
    @property
    def dzita_c(self):return self._dzita_c
    
    @property
    def phi(self):return self._phi
    
    @property
    def e_opt(self):return self._e_opt
    
    @property
    def ksi(self):return self._ksi
    
    @property
    def w_2_t(self):return self._w_2_t
    
    @property
    def w_2(self):return self._w_2
    
    @property
    def c_2(self):return self._c_2

    @property
    def delt_H_vc(self):return self._delt_H_vc
    
    @property
    def eff_oi_loss(self):return self._eff_oi_loss
    
    
    @property
    def eff_oi_velocity(self):return self._eff_oi_velocity
    
    @property
    def u_c_f_opt(self):return  self._u_c_f_opt
    
    @property
    def u_c_f(self):return self._u_c_f
    
    @property
    def c_f(self):return self._c_f

    @property
    def zitta_v(self):return self._zitta_v
    
    @property
    def ksi_segm(self):return self._ksi_segm
    
    @property 
    def ksi_parc(self):return self._ksi_parc
    
    @property 
    def delt_H_parc(self):return self._delt_H_parc
    @property 
    def E_0(self):return self._E_0
    
    @property
    def H_i(self):return self._H_i
    
    @property
    def kpd_oi(self):return self._kpd_oi

# # Пример
# obj = MyClass(1, 2, 0.5, 100)
# print(obj.H_0c)  # 50.0
# obj.ro = 0.8
# obj.update()  # пересчёт
# print(obj.h_1t)  # 20.0