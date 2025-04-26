from steam import Steam as sgas
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from iapws import IAPWS97
import numpy as np
from mass_flow_class import mass_flow_class as mfc
from math import sqrt, pi, sin, cos, atan, degrees, radians, asin, acos,ceil
import math
import pandas as pd 
from scipy.optimize import fsolve
MPa = 10**6
unit = 10**(-6)

class control_stage():
    """
    Проведение расчета регулирующей ступени и определение зависимости ηол от 
    U/cф. Диапазон варьируемого параметра (диаметр или теплоперепад) дан в 
    задании.
    """
    def __init__(self,shum, mfc_class, H_0, d, n, b_1,t_1opt, alpha_1e, b_2, t_2opt):
        self.mfc = mfc_class
        self.gas = shum
        #ro  - степень реактивности 0.05 - 0.1
        self.ro = 0.075
        self. H_0 = H_0
        self.d = d
        self.n = n
        self.b_1 = b_1
        self.alpha_1e = np.deg2rad(alpha_1e)
        self.t_1opt = t_1opt
        self.mu_1t = 0.97
        self.delta = 0.0035 #перекрыша
        self.b_2 = b_2
        self.G = self.mfc.inlet_mass_flow
        self.t_2opt = t_2opt
        self.xi_vs = 0
        
        self.mu_a = 0.5
        self.delta_a = 0.0025
        self.mu_r = 0.75
        self.F_1 = 3.3 * 10**(-4)
        self.x = 0.0748541186078932183458
        point_0 = self.mfc.point_0
        self.k = 1.33
        
    @property
    def u(self):
        u = pi * self.d * self.n
        return u
    
    @property
    def H_0c(self):
        H_0c = (1- self.ro) * self.H_0
        return H_0c
    
    @property
    def H_0r(self):
        H_0r = self.H_0 * self.ro
        return H_0r
    
    @property
    def h_1t(self):
        h_1t = self.mfc.point_0.h - self.H_0c
        return h_1t
    
    @property 
    def point_1_t(self):
        point_1_t = self.gas.calc_point(h = self.h_1t, s = self.mfc.point_0.s)
        return point_1_t
    @property
    def M_1t(self):
        c_1t = (2000 * self.H_0c)**0.5
        a_1t = (self.k * self.point_1_t.P * self.point_1_t.v * MPa) ** 0.5
        M_1t = c_1t / a_1t
        return M_1t

    @property
    def create_results_tables(self):
            table = self.eff1[10]
            c_1 = table[0], 
            c_2= table[1]
            alpha_1 =  table[2]
            alpha_2 =  table[3]
            w_1 =  table[4]
            w_2 =  table[5]
            bett_1 =  table[6]
            bett_2 =  table[7]
            delt_H_c =  table[8]
            delt_H_p =  table[9]
            delt_H_vc =  table[10]
            delt_H_y =  table[11]
            delt_H_tr =  table[12]
                      
            delt_H_parc =  table[13]
            E_o=  table[14]
            point_0 =  table[15]
            point_1_t =  table[16]
            point_1 =  table[17]
            point_2_t =  table[19]
            point_2 =  table[19]
            
        
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
            
            velocity_df = pd.DataFrame({
                "Абсолютная скорость (с), м/с": [np.round(c_1, 3), np.round(c_2, 3)],
                "Абсолютный угол (α), °": [np.round(np.rad2deg(alpha_1), 3), 
                                          np.round(np.rad2deg(alpha_2), 3)],
                "Относительная скорость (w), м/с": [np.round(w_1, 3), np.round(w_2, 3)],
                "Относительный угол (β), °": [np.round(np.rad2deg(bett_1), 3), 
                                             np.round(np.rad2deg(bett_2), 3)],
                "Локация": ["За сопловой решеткой", "За рабочей решеткой"]
            }).set_index("Локация")
        
     
            loss_df = pd.DataFrame({
                "Потери, %": [
                    np.round(delt_H_c/E_o * 100, 3),
                    np.round(delt_H_p/E_o * 100, 3),
                    np.round(delt_H_vc/E_o * 100, 3),
                    np.round(delt_H_y/E_o * 100, 3),
                    np.round(delt_H_tr/E_o * 100, 3),
                    np.round(delt_H_parc/E_o * 100, 3)
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
                    np.round(point_0.P, 3),
                    np.round(point_1_t.P, 3),
                    np.round(point_1.P, 3),
                    np.round(point_2_t.P, 3),
                    np.round(point_2.P, 3)
                ],
                "Температура (T), K": [
                    np.round(point_0.T, 3),
                    np.round(point_1_t.T, 3),
                    np.round(point_1.T, 3),
                    np.round(point_2_t.T, 3),
                    np.round(point_2.T, 3)
                ],
                "Энтальпия (h), кДж/кг": [
                    np.round(point_0.h, 3),
                    np.round(point_1_t.h, 3),
                    np.round(point_1.h, 3),
                    np.round(point_2_t.h, 3),
                    np.round(point_2.h, 3)
                ],
                "Энтропия (s), кДж/(кг·K)": [
                    np.round(point_0.s, 3),
                    np.round(point_1_t.s, 3),
                    np.round(point_1.s, 3),
                    np.round(point_2_t.s, 3),
                    np.round(point_2.s, 3)
                ],
                "Состояние": [
                    "Параметры перед ступенью",
                    "Теоретические параметры за сопловой решеткой",
                    "Фактические параметры за сопловой решеткой",
                    "Теоретические параметры за ступенью",
                    "Фактические параметры за ступенью"
                ]
            }).set_index("Состояние")
        
            return velocity_df, loss_df, thermodynamic_df
    @property
    def eff1(self):
        u = self.u
        c_1t = (2000 * self.H_0c)**0.5
        a_1t = (self.k * self.point_1_t.P * self.point_1_t.v * MPa) ** 0.5
        M_1t = c_1t / a_1t
        _mu_1 = 0.97
        _F_1 = (self.G * self.point_1_t.v) / (_mu_1 * c_1t)
        
        el_1 = _F_1 / (pi* self.d* sin(self.alpha_1e))
        
        e_opt = 5 *el_1**0.5
        if e_opt > 0.85:
            e_opt = 0.85
        else:e_opt = e_opt
        
        l_1 = el_1 / e_opt
        
        mu_1 = 0.982 - 0.005*(self.b_1 / l_1)
        
        z_1 = (pi * self.d * e_opt) / (self.b_1 * self.t_1opt)
        z_1 = ceil(z_1)
        t_1 = (pi*self.d * e_opt) / (self.b_1 * z_1)
        
        alpha_ust = self.alpha_1e - 16*(t_1 - 0.75) + 23.1
        dzita_c = - 3.7708 * ((mu_1)**3) + 14.189 * ((mu_1)**2) - 13.478 * (mu_1) + 5.6125
        
        phi = (1- (dzita_c)/100)**0.5
        phi_t = 0.98 - 0.008*(self.b_1 / l_1)
        delta_phi = (phi - phi_t) / phi * 100
        
        c_1 = c_1t * phi
        alpha_1 = asin((mu_1/ phi)*sin(self.alpha_1e))
        w_1 = (c_1**2  + (self.u **2) - 2* c_1 * self.u* cos(alpha_1) )**0.5
        betta_1 =  atan(sin(alpha_1) /( cos(alpha_1) - self.u / c_1))
        
        delta_Hc =  c_1t**2*(1- phi**2) /2000  
        h_1 = self.h_1t + delta_Hc
        point_1 = self.gas.calc_point(p = self.point_1_t.P, h = h_1)
        h_2_t = point_1.h - self.H_0r
        point_2_t = self.gas.calc_point(h = h_2_t, s = point_1.s) 
        
        w_2_t =(2000 * self.H_0r + w_1 ** 2) ** 0.5
        l_2 = l_1 + self.delta
        
        a_2t = (1.4 * point_2_t.P * MPa * point_2_t.v)**0.5
        
        m_2_pred = w_2_t/ a_2t
        b_2 = self.b_2 
        b_2_l_2 = b_2/l_2
        mu_2 = 0.965 - 0.01*(b_2_l_2)
        F_2 = (self.G * point_2_t.v)/(mu_2 * w_2_t)
        
        a_2_t = (self.k * point_2_t.P * point_2_t.v * (10 ** 6)) ** 0.5 
        
        betta_2e = np.arcsin(F_2/(e_opt * np.pi * self.d * l_2 ))
        z_2 = ceil((pi * self.d)/(b_2 * self.t_2opt)) 
        t_2 = (np.pi * self.d)/(b_2 * z_2)
        
        bet_yst = np.rad2deg(betta_2e) - 19.3 * (self.t_2opt - 0.6) + 60
        zit_prof = 7.4173 * ((m_2_pred)**3) -10.511 * ((m_2_pred)**2) + 1.0066 * (m_2_pred) + 6.4416
        zit_konc_l_d = 4.8786 * (b_2_l_2) + 4.9714
        zit_r = zit_prof + zit_konc_l_d
        ksi = (1 - zit_r/100) ** 0.5
        w_2 = w_2_t * ksi
        bett_2 = np.arcsin((mu_2/ksi)* np.sin(betta_2e))
        c_2 = ((w_2**2) + (u**2) - 2 * w_2 * u * np.cos(bett_2)) ** 0.5
        alpha_2 = np.arctan((np.sin(bett_2))/((np.cos(bett_2)) - (u/w_2)))
        delt_H_p = w_2_t ** 2 * (1 - (ksi ** 2)) /2000
        delt_H_vc = (c_2 ** 2)/(2 * 1000)
        E_0 = self.H_0 - self.xi_vs * delt_H_vc
        eff_oi_loss = (E_0 - delta_Hc - delt_H_p - (1 - self.xi_vs) *  delt_H_vc) / E_0
        u_c_f_opt = (phi * np.cos(alpha_1)) /(2 * (1 - self.ro) ** 0.5)
        c_f = (2 * self.H_0*1000)**0.5
        u_c_f = self.u / c_f
        h_2 = point_2_t.h + delt_H_p
        point_2 = self.gas.calc_point( p = point_2_t.P, h = h_2)
        z = 3
        d_p = self.d + l_2
        mu_a = 0.5
        delt_a = 0.0025
    
        mu_r = 0.75
        delt_g = 0.001 * d_p
        
        delt_e = ((1/(mu_a * delt_a)**2) + (z/(mu_r * delt_g)**2)) ** (-0.5)
        
        ksi_b_y = (np.pi * d_p * delt_e * eff_oi_loss) * ((self.ro + 1.8 * (l_2/self.d)) ** 0.5) / _F_1
        
        delt_H_y = ksi_b_y * E_0
        k_tr = 0.7 * (10 ** -3)
        
        ksi_tr = (k_tr * ((self.d) ** 2)) * ((u_c_f) ** 3) / _F_1
        delt_H_tr = ksi_tr * E_0
        k_v = 0.065
        m = 1
        zitta_v = (k_v * (1 - e_opt) * ((u_c_f) ** 3) * m) / ((np.sin(np.rad2deg(self.alpha_1e))) * e_opt)
        B_2 = b_2 * np.sin(np.deg2rad(bet_yst))
        i = 4
        ksi_segm = 0.25 * (B_2 * l_2 * u_c_f * i * eff_oi_loss) / _F_1
        ksi_parc = zitta_v + ksi_segm
        delt_H_parc = ksi_parc * E_0
        H_i = E_0- delta_Hc - delt_H_p - (1 - self.xi_vs) *  delt_H_vc - delt_H_y - delt_H_tr - delt_H_parc
        kpd_oi = H_i/E_0
        
        
        eff_oi_velocity = self.u * (c_1 * (np.cos(alpha_1)) + c_2 * (np.cos(alpha_2))) / (E_0 * 1000)
        if (math.degrees(alpha_2)) < 0 :
            eff_oi_velocity = self.u * (c_1 * (np.cos(alpha_1)) - c_2 * (np.cos(alpha_2))) / (E_0 * 1000)
            
        table = [c_1, c_2,alpha_1,alpha_2,w_1,w_2,betta_1,betta_2e,delta_Hc,delt_H_p, delt_H_vc,delt_H_y,delt_H_tr,
                  delt_H_parc,E_0,self.mfc.point_0,self.point_1_t,point_1,point_2_t, point_2]
        
        initial_point = self.mfc.point_0
      
        nozzle_exit_point_ideal  = self.point_1_t
     
        nozzle_exit_point_actual = point_1
 
        rotor_exit_point_ideal = point_2_t
        rotor_exit_point_actual = point_2
        points = initial_point, nozzle_exit_point_ideal, nozzle_exit_point_actual, rotor_exit_point_ideal, rotor_exit_point_actual

        return eff_oi_loss, eff_oi_velocity, kpd_oi, u_c_f, alpha_1, betta_2e, c_1, w_2, alpha_2, points, table, E_0, l_2, z_2, e_opt, delt_H_p, delt_H_tr,delt_H_y ,delt_H_vc, c_1
    
    def my_blade(self, total_enthalpy_drop, p0, t0, adiabatic_index, rpm, degree_of_reaction, 
                       nozzle_blade_height_mm, nozzle_angle_deg, nozzle_pitch_ratio, 
                       rotor_pitch_ratio, rotor_blade_height_mm, mass_flow, overlap, 
                       exit_loss_coeff, diameter,atlas_min_modulus, nozzle_blade_length):
       """Calculate turbine efficiency and velocity parameters"""
       
       initial_point = self.mfc.point_0
       
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
       nozzle_blade_count = round((pi * diameter * optimal_contraction) / (nozzle_blade_height * nozzle_pitch_ratio), 0)
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
       actual_exit_angle = asin((actual_flow_coeff/velocity_coeff) * sin(nozzle_angle_rad))
       relative_inlet_velocity = sqrt((actual_exit_velocity ** 2) + (blade_speed ** 2) - 
                                     2 * actual_exit_velocity * blade_speed * np.cos(actual_exit_angle))
       beta_1 = atan(np.sin(actual_exit_angle)/(cos(actual_exit_angle) - (blade_speed/actual_exit_velocity)))
       
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
       rotor_blade_count = round((pi * diameter) / (rotor_blade_height * rotor_pitch_ratio)) 
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
                       (2 * blade_speed * rotor_blade_count * actual_min_modulus * optimal_contraction) / 1000
       
       # Adjust rotor blade height if stress exceeds allowable
       if bending_stress > allowable_bending_stress:
           rotor_blade_height = original_rotor_height * math.sqrt(bending_stress / allowable_bending_stress)
           # Recalculate with new blade height
           bending_stress = (mass_flow * total_enthalpy_drop * velocity_triangle_efficiency * rotor_blade_length) / \
                           (2 * blade_speed * rotor_blade_count * actual_min_modulus * optimal_contraction) / 1000
           original_rotor_height = rotor_blade_height
       
       # Centrifugal stress calculation
       angular_velocity = 2 * math.pi * rpm
       centrifugal_stress = 0.5 * material_density * angular_velocity**2 * diameter * rotor_blade_length * 1e-6
       
       
       
       # Plot velocity triangles if needed
       # if diameter == 1.1:
       print(f"Centrifugal stress: {centrifugal_stress:.2f} MPa; "
             f"Bending stress: {bending_stress*1e6:.2f} MPa; "
             f"Final rotor height: {original_rotor_height*1000:.2f} mm")
       
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

 
       points=[initial_point, nozzle_exit_point_ideal, nozzle_exit_point_actual, rotor_exit_point_ideal, rotor_exit_point_actual]
   
  
      
       return energy_loss_efficiency, velocity_triangle_efficiency, internal_efficiency,points, speed_ratio
 
    @property
    def get_point_z(self):
        delt_H_p = self.eff1[15]
        delt_H_tr = self.eff1[16]
        delt_H_y = self.eff1[17]
        delt_H_vc =self.eff1[18]
        h_0st = self.mfc.point_2t.h + delt_H_p  + delt_H_tr +delt_H_y+delt_H_vc
        return self.mfc.point_0.P, self.mfc.point_0.h, self.mfc.point_1.P
    
    def stress(self, b_2,b_2atl ,moment_of_resistance_min_atl):
        ro = 1000
        moment_of_resistance_min = (b_2 / b_2atl)**3 * moment_of_resistance_min_atl
        sigma_isg_base = 20
        l_2 = self.eff1[12]
        z_2 = self.eff1[13]
        e_opt = self.eff1[14]

        sigma_isg = (self.G * self.H_0*1000  * self.eff1[0] * l_2 )/ (2 * self.u * z_2 *moment_of_resistance_min * e_opt * MPa) 
        
        b_2 = self.b_2 * (sigma_isg / sigma_isg_base)**0.5

        w = 2 * math.pi * self.n
        sigma_rast = 0.5 * ro * w**2 * self.d * l_2 * 1e-6
        return sigma_isg, sigma_rast, b_2, l_2, z_2
    
   
        
  
    @property
    def hs1(self):
        points = self.eff1[9]
        fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
        initial_point = points[0]
        
        nozzle_exit_point_ideal  = points[1]
        nozzle_exit_point_actual = points[2]
        rotor_exit_point_ideal = points[3]
        rotor_exit_point_actual = points[4]
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
        alpha_1 = self.eff1[4]
        betta_2 = self.eff1[5]
        c_1 = self.eff1[6]
        w_2 = self.eff1[7]
        
        sin_alpha1 = sin(alpha_1)
        cos_alpha1 = cos(alpha_1)
        sin_beta2 = sin(betta_2)
        cos_beta2 = cos(betta_2)
        
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
    def _plot_velocity_triangles(self):
        """Helper method to plot velocity triangles"""
        alpha_1 = self.eff1[4]
        sin_alpha1 = sin(alpha_1)
        cos_alpha1 = cos(alpha_1)
        c_1 = self.eff1[19]
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        
        ax.plot([0, -c_1 * cos_alpha1], [0, -c_1 * sin_alpha1], label='C_1', c='red')
        ax.plot([-c_1 * cos_alpha1, -c_1 * cos_alpha1 + self.u], 
                [-c_1 * sin_alpha1, -c_1 * sin_alpha1], label='u_1', c='blue')
        ax.plot([0, -c_1 * cos_alpha1 + self.u], [0, -c_1 * sin_alpha1], label='W_1', c='green')
        
        ax.set_title("Входной треугольник скоростей")
        ax.legend()
        plt.grid(True)
        plt.show() 
        

    @property
    def out_plot_velocity_triangles(self):
        sin_beta2 = sin(self.beta_2)
        cos_beta2 = cos(self.beta_2)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        ax.plot([0, self.w_2 * cos_beta2], [0, -self.w_2 * sin_beta2], label='W_2', c='green')
        ax.plot([self.w_2 * cos_beta2, self.w_2 * cos_beta2 - self.u], 
                [-self.w_2 * sin_beta2, -self.w_2 * sin_beta2], label='u_2', c='blue')
        ax.plot([0, self.w_2 * cos_beta2 - self.u], [0, -self.w_2 * sin_beta2], label='C_2', c='red')
        
        ax.set_title("Выходной треугольник скоростей")
        ax.legend()
        plt.grid(True)
        plt.show()   
    

    
    def plot_distribution(self, values, n_stages,  ax_name):
        fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=300)
        ax.plot(range(1, n_stages+1), values,  marker='o')
        ax.set_xlabel("Номер ступени")
        ax.set_ylabel(ax_name)
        ax.grid()
          
    def each_blade(self, speed_stage_diam, rotation_speed, n_stages, mass_flow, p0, h0, 
                       pz, delta_diam, speed_coefficient, alpha_1, root_reaction_degree, discharge_coefficient, overlapping, efficiency, veernost_1, my_blade_No):
            while True:
                while True:
                    
                    avg_diam_1 = speed_stage_diam - delta_diam
            
                    point_0 = self.gas.calc_point(p=p0 * unit, h=h0)
                    
                    def get_reaction_degree(root_dor, veernost):
                        return root_dor + (1.8 / (veernost + 1.8))
                    
                    def get_u_cf(dor):
                        cos = np.cos(np.deg2rad(alpha_1))
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
                    lower = discharge_coefficient * np.sin(np.deg2rad(alpha_1)) * rotation_speed * (np.pi * avg_diam_1) ** 2 * (1 - avg_reaction_degree_1) ** 0.5
                    
                    blade_length_1 = upper / lower
                    blade_length_2 = blade_length_1 + overlapping
                    print("Ошибка определения веерности: ", abs(avg_diam_1 / blade_length_1-veernost_1)/veernost_1*100, "%'")
                    if np.isclose(avg_diam_1 / blade_length_1, veernost_1, rtol=0.01):
                        break
                    veernost_1 = avg_diam_1 / blade_length_1
                    
                # assert np.isclose(avg_diam_1 / blade_length_1, veernost_1, rtol=0.01)
                
                root_diameter = avg_diam_1 - blade_length_2
                
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
                
                # print("diff in veernost', %': ", (veernost_1-veernosts[0])/veernosts[0]*100)
                # veernosts = veernosts_new
                reaction_degrees = get_reaction_degree(root_dor=root_reaction_degree, veernost=veernosts)
                u_cf = get_u_cf(dor=reaction_degrees)
                
                heat_drops = get_heat_drop(diameters, u_cf)
                output_speed_coeff_loss = np.full_like(heat_drops, 0.95)
                output_speed_coeff_loss[0] = 1
                
                actual_heat_drops = output_speed_coeff_loss * heat_drops
                mean_heat_drop = np.mean(actual_heat_drops)
                reheat_factor = 4.8 * 10 ** (-4) * (1 - efficiency) * full_heat_drop * (n_stages - 1) / n_stages
                
                # full_heat_drop * (1 + reheat_factor) / mean_heat_drop
                print("Новое число ступеней: ", (full_heat_drop * (1 + reheat_factor) / mean_heat_drop))
                if round(full_heat_drop * (1 + reheat_factor) / mean_heat_drop) - n_stages == 0:
                    break
                n_stages = round(full_heat_drop * (1 + reheat_factor) / mean_heat_drop)
            
            bias = full_heat_drop * (1 + reheat_factor) - np.sum(actual_heat_drops)
            bias = bias / n_stages
            
            new_actual_heat_drop = actual_heat_drops + bias
            
            self.plot_distribution(diameters, n_stages, "d, m")
            self.plot_distribution(blade_lengths, n_stages, "l, m")
            self.plot_distribution(veernosts, n_stages, "Веерность")
            self.plot_distribution(reaction_degrees, n_stages, "Степень реактивности")
            self.plot_distribution(u_cf, n_stages, "U/Cф")
            self.plot_distribution(new_actual_heat_drop, n_stages, "Теплоперепады по ступеням")
            
            points = [self.gas.calc_point(p=p0 * unit, h=h0)]  # Начальная точка
            pressures = [p0 * unit]  # Начальное давление
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
            
        

