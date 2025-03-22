from steam import Steam as sgas
from ideal_gas import IdealGas as igas

def r(x):
    
    a = (round(x))
    return a 

MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None

p0 = 26 * MPa 
t0 = 545 
p_middle = 3.62 * MPa 
t_middle = 545 
pk = 3.5 * kPa 
t_feed_water = 263
p_feed_water = 1.35 * p0

electrical_power = 320*10**6
internal_efficiency = 0.85
mechanical_efficiency = 0.995
generator_efficiency = 0.99
z = 9
alpha = 0.83

name = "air"
R = 287
c_p = 1005
if name == "steam":
    input_class = sgas(name = name, R = R, c_p = c_p)
if name == "air":
    input_class = igas(name = name, R = R, c_p = c_p)


class calc_mass_flow():
    '''
        Mass flow condenser/turbine calculator
        Atributes:
            input_class: class, -- класс для расчёта параметров состояния
        
        Methods:
            condenser_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,
        mechanical_efficiency,generator_efficiency) -- inlet G in condenser calculator
            
            inlet_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,
        t_feed_water,electrical_power,mechanical_efficiency,generator_efficiency) -- inlet turbine calculator
            
            efficiency(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha) - coefficient calculator
    '''
    def __init__(self,input_class):
        self.input_class = input_class
    
    
    def calc_zero_point(self,p0,t0):
        delta_p0 = 0.05 * p0
        real_p0 = p0 - delta_p0
        _point_0 = input_class.calc_point(p= p0 * unit, t =t0)
        point_0 = input_class.calc_point(p=real_p0 * unit, h=_point_0.h)
        return point_0, _point_0
    
    def calc_first_point(self,p0,t0, p_middle):
        delta_p_middle = 0.1 * p_middle
        real_p1t = p_middle + delta_p_middle
        _point_0 = self.calc_zero_point(p0,t0)[0]
        point_0 = self.calc_zero_point(p0,t0)[1]
        point_1t = input_class.calc_point(p=real_p1t * unit, s=_point_0.s)
        hp_heat_drop = (_point_0.h - point_1t.h) * internal_efficiency
        h_1 = point_0.h - hp_heat_drop
        point_1 = input_class.calc_point(p=real_p1t * unit, h=h_1)
        return point_1, point_1t
        
    def calc_middle_point(self,p_middle, t_middle):
        delta_p_1 = 0.03 * p_middle
        real_p_middle = p_middle - delta_p_1
        _point_middle = input_class.calc_point(p=p_middle * unit, t=t_middle)
        point_middle = input_class.calc_point(p=real_p_middle * unit, h=_point_middle.h)
        point_2t = input_class.calc_point(p=pk * unit, s=_point_middle.s)
        return _point_middle, point_2t, point_middle
    
    def lp_heat_drop(self,p_middle,t_middle, internal_efficiency):
        _point_middle = self.calc_middle_point(p_middle, t_middle)[0]
        point_2t = input_class.calc_point(p=pk * unit, s=_point_middle.s)
        lp_heat_drop = (_point_middle.h - point_2t.h) * internal_efficiency
        return lp_heat_drop
    
    def calc_second_point(self,p_middle,t_middle,internal_efficiency):
        point_middle = self.calc_middle_point(p_middle,t_middle)[2]
        lp_heat_drop = self.lp_heat_drop(p_middle, t_middle, internal_efficiency)
        h_2 = point_middle.h - lp_heat_drop
        point_2 = input_class.calc_point(p=pk * unit, h=h_2)
        return point_2
    
    
    def efficiency_hp(self,p0,t0, p_middle):
        _point_0 = self.calc_zero_point(p0,t0)[1]
        point_1 = self.calc_first_point(p0,t0,p_middle)[0]
        point_1t = self.calc_first_point(p0,t0,p_middle)[1]
        efficiency_hp = (_point_0.h - point_1.h) / (_point_0.h - point_1t.h)
        return efficiency_hp
    
    def efficiency_lp(self,p_middle,t_middle,internal_efficiency):
        _point_middle = self.calc_middle_point(p_middle,t_middle)[0]
        point_2t = self.calc_middle_point(p_middle,t_middle)[1]
        point_2 = self.calc_second_point(p_middle,t_middle,internal_efficiency)
        efficiency_lp = (_point_middle.h - point_2.h) / (_point_middle.h - point_2t.h)
        return efficiency_lp

    def numenator_without(self,p_middle,t_middle, pk):
        point_k_water = input_class.calc_point(p=pk, x=0)
        _point_middle = self.calc_middle_point(p_middle,t_middle)[0]
        point_2 = self.calc_second_point(p_middle,t_middle,internal_efficiency)
        numenator_without = to_kelvin(point_2.T) * (_point_middle.s - point_k_water.s)
        return numenator_without
    
    def denumenator_without(self,p0,t0,p_middle,pk):
        point_0 = self.calc_zero_point(p0,t0)[0]
        point_1t = self.calc_first_point(p0,t0,p_middle)[1]
        point_middle = self.calc_middle_point(p_middle,t_middle)[2]
        point_k_water = input_class.calc_point(p=pk, x=0)
        denumenator_without = (point_0.h - point_1t.h) + (point_middle.h - point_k_water.h)
        return denumenator_without
    
    def without_part(self,p0,t0,p_middle,t_middle,pk):
        numenator_without = self.numenator_without(p_middle,t_middle,pk)
        denumenator_without = self.denumenator_without(p0,t0,p_middle,pk)
        without_part = 1 - (numenator_without / denumenator_without)
        return without_part
    
    def numenator_infinity(self,p_middle,t_middle,p_feed_water,t_feed_water):
        point_2 = self.calc_second_point(p_middle,t_middle,internal_efficiency)
        _point_middle = self.calc_middle_point(p_middle,t_middle)[0]
        point_feed_water = input_class.calc_point(p=p_feed_water * unit, t=t_feed_water)
        numenator_infinity = to_kelvin(point_2.T) * (_point_middle.s - point_feed_water.s)
        return numenator_infinity
    
    def denumenator_infinity(self,p0,t0,p_middle,t_middle,p_feed_water,t_feed_water):
        point_0 = self.calc_zero_point(p0,t0)[0]
        point_1t = self.calc_first_point(p0,t0,p_middle)[1]
        point_middle = self.calc_middle_point(p_middle,t_middle)[2]
        point_feed_water = input_class.calc_point(p=p_feed_water * unit, t=t_feed_water)
        denumenator_infinity = (point_0.h - point_1t.h) + (point_middle.h - point_feed_water.h)
        return denumenator_infinity
    
    def infinity_part(self,p0,t0,p_middle,t_middle,p_feed_water,t_feed_water):
        numenator_infinity = self.numenator_infinity(p_middle,t_middle,p_feed_water,t_feed_water)
        denumenator_infinity = self.denumenator_infinity(p0,t0,p_middle,t_middle,p_feed_water,t_feed_water)
        infinity_part = 1 - (numenator_infinity / denumenator_infinity)
        return infinity_part
    
    def ksi_infinity(self,p0,t0,p_middle,t_middle,pk):
        without_part = self.without_part(p0,t0,p_middle,t_middle,pk)
        infinity_part = self.infinity_part(p0,t0,p_middle,t_middle,p_feed_water,t_feed_water)
        ksi_infinity = 1 - (without_part / infinity_part)
        return ksi_infinity
    
    def coeff(self,p_middle,t_middle,internal_efficiency,p_feed_water,t_feed_water):
        point_2 = self.calc_second_point(p_middle,t_middle,internal_efficiency)[0]
        point_feed_water = input_class.calc_point(p=p_feed_water * unit, t=t_feed_water)
        coeff = (point_feed_water.T - point_2.T) / (to_kelvin(374.2) - point_2.T)
        return coeff
    
    def ksi(self,alpha,p0, t0, p_middle, t_middle, pk):
        ksi_infinity = self.ksi_infinity(p0, t0, p_middle, t_middle, pk)
        ksi = alpha * ksi_infinity
        return ksi

    def efficiency(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha):
        lp_heat_drop = self.lp_heat_drop(p_middle,t_middle, internal_efficiency)
        _point_0 = self.calc_zero_point(p0,t0)[1]
        point_1t = self.calc_first_point(p0,t0,p_middle)[1]
        hp_heat_drop = (_point_0.h - point_1t.h) * internal_efficiency
        eff_num = hp_heat_drop + lp_heat_drop
        point_k_water = input_class.calc_point(p=pk, x=0)
        point_middle = self.calc_middle_point(p_middle, t_middle)[2]
        eff_denum = hp_heat_drop + (point_middle.h - point_k_water.h)
        ksi = self.ksi(alpha,p0, t0, p_middle, t_middle, pk)
        efficiency = (eff_num / eff_denum) * (1 / (1 - ksi))
        return efficiency
    
    def estimated_heat_drop(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,t_feed_water):
        efficiency = self.efficiency(p0, t0, p_middle, t_middle, internal_efficiency, pk, alpha)
        point_0 = self.calc_zero_point(p0,t0)[0]
        point_middle = self.calc_middle_point(p_middle, t_middle)[2]
        point_feed_water = input_class.calc_point(p=p_feed_water * unit, t=t_feed_water)
        point_1 = self.calc_first_point(p0,t0,p_middle)[0]
        estimated_heat_drop = efficiency * ((point_0.h - point_feed_water.h) + (point_middle.h - point_1.h))
        return estimated_heat_drop
    
    def inlet_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,t_feed_water,electrical_power,mechanical_efficiency,generator_efficiency):
        estimated_heat_drop = self.estimated_heat_drop(p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,p_feed_water,t_feed_water)
        inlet_mass_flow = electrical_power / (estimated_heat_drop * 1000 * mechanical_efficiency * generator_efficiency)
        return inlet_mass_flow
        
    def condenser_mass_flow(self,p0,t0,p_middle,t_middle,internal_efficiency,pk,alpha,mechanical_efficiency,generator_efficiency):
        point_2 = self.calc_second_point(p_middle,t_middle,internal_efficiency)
        point_k_water = input_class.calc_point(p=pk, x=0)
        efficiency = efficiency = self.efficiency(p0, t0, p_middle, t_middle, internal_efficiency, pk, alpha)
        condenser_mass_flow = (
            electrical_power /
            ((point_2.h - point_k_water.h) * 1000 * mechanical_efficiency * generator_efficiency) * ((1 / efficiency) - 1)
        )
        return condenser_mass_flow
