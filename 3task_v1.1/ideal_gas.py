from dataclasses import dataclass
from Point import Point as Point
from math import exp, log

def is_not_None(*args, **kwargs):
    j = 0
    for i in args:
        if i != None:
            j += 1
    if j != len(args):
        return False
    else:
        return True


to_kelvin = lambda x: x + 273.15 if x else None
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
    
@dataclass    
class IdealGas():
    
    """ IdealGas parametra calculation class 
        Atrubutes:
            name: str, — gas name (steam, air applied)
            R: float,— the universal gas constant
            с_p: float,—  isobarical thermocapacity value
            
        Method calc_point(p,v,t,h,s): calcs all over the   
        uninitial values; the *Point* object as output.
        
        Method calc_isobar(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in isobaric properies; the *Point* object as output.
        
        Method calc_isohor(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in the isochoric process; the *Point* object as output.
        
        Method calc_isoterm(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in the isotermical process; the *Point* object as output.
        
        Method calc_isoentrope(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values  
        in isoentropic process; the *Point* object as output.
        
        standart entropy at standart physical conditions, can be obtained by s_0 = -c_p \ln T_0 + R \ln p_0; T_0 = 298.15, p_0 = 1 atm
            
        """

        
    name: str = "air"
    c_p: float = 1200
    R: float = 400
#    s_0: float = -2329.94
    c_v = c_p - R
    k = c_p / c_v   
    s_0 = -2329.94 #standart entropy at standart physical conditions, can be obtained by s_0 = -c_p \ln T_0 + R \ln p_0; T_0 = 298.15, p_0 = 1 atm
    omega = 0.01 # массовая доля водяного пара
    R_w = 461 #\text{Дж/(кг·К)} \) (водяной пар),
    c_pw = 1880 #c_{p,w} = 1880 \, \text{Дж/(кг·К)} \) (водяной пар),
    h_fg = 2501 #h_{fg} = 2501 \, \text{кДж/кг} \) (теплота парообразования воды).

    def calc_point(self, p = None, v = None, t = None, h = None, s = None, x = None, *args, **kwargs):
        if is_not_None(p, t):
            return self.calc_point_pt(p, t)
        if is_not_None(v, t):
            return self.calc_point_vt(v, t)
        if is_not_None(p, v):
            return self.calc_point_pv(p, v)
        if is_not_None(h, p):
            return self.calc_point_hp(h, p)
        if is_not_None(h, v):
            return self.calc_point_hv(h, v)
        if is_not_None(p, s):
            return self.calc_point_ps(p, s)
        if is_not_None(p,x):
            return self.calc_point_px(p, x)
        else:
            raise Exception("Values is not enough")
            
    def calc_point_pv(self, p, v):
        t =  p * v / self.R
        h = self.c_p * t
        s = self.c_v*log(v)+self.s_0
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    
    def calc_point_pt(self, p, t):
        v = self.R*t/p
        h = self.c_p * t
        s = self.c_v * log(t) - self.R*log(p) + self.s_0
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    
    def calc_point_vt(self, v, t):
        p = self.R * t / v
        h = self.c_p * t
        s = self.c_v * log(t) + self.R*log(v) +self.s_0
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    
    def calc_point_hp(self, h, p):
        t = h / self.c_p
        v = self.R*t/p
        s = self.c_p*log(h/self.c_p) - self.R * log(p) + self.s_0
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    
    def calc_point_hv(self, h, v):
        t = h / self.c_p
        p = self.R * t / v
        s = self.c_v*log(h/self.c_p)+self.R*log(v)+self.s_0
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    
    def calc_point_ps(self, p, s):
        t = exp((s - self.s_0 + self.R * log(p))/self.c_p)
        h = self.c_p * exp((s - self.s_0 + self.R*log(p))/self.c_p)
        v = self.R*p**(self.R/self.c_p - 1)*exp((s-self.s_0)/self.c_p)
        p_w = self.omega*p
        p_sat = 611*exp((11.27*(t - 273.15))/(t - 35.85))
        x = p_w/p_sat
        point = Point(p,v,t,h,s,x)
        return point
    def calc_point_px(self, p, x):
        p_w = self.omega*461/287*p
        v = 0.01
        R = (1 - self.omega)*self.R -self.omega*self.R_w
        t = p*v/R
        v = R*t/p
        h = (1-self.omega)*self.c_p*t +self.omega * (self.c_pw*t+ self.h_fg)
        p_w = self.omega*p*self.R_w/self.R
        p_a = p - p_w
        s = (1 - self.omega)*(self.c_p*log(t) -self.R*log(p_a)) + self.omega*(self.c_pw*log(t) - self.R_w*(p_w))
        point = Point(p,v,t,h,s,x)
        return point
       
    # Вычисление изобары
    def calc_isobar(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end= None,*args, **kwargs):
        point_start = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init)
        p_end = point_start.P
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end)
        return point_start, point_end
    
    def calc_isochor(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,*args, **kwargs):
        point_start = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init)
        v_end = point_start.v
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end)
        return point_start, point_end
    
    def calc_isoterm(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,*args, **kwargs):
        point_start = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init)
        t_end = point_start.T
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end)
        return point_start, point_end
        
    # Изоэнтропа

    def calc_isoentrope_tvv(self, t_init, v_init, v_end):  #проверено
        t_end = t_init * (v_init/v_end)**(self.k - 1)
        point_init =  self.calc_point(v = v_init, t = t_init)
        point_end =  self.calc_point(v = v_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_tvt(self, t_init, v_init, t_end):
        v_end = (t_init/t_end)**(1/(self.k-1)) * v_init
        point_init =  self.calc_point(v = v_init, t = t_init)
        point_end =  self.calc_point(v = v_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_pvp(self, p_init, v_init, p_end):
        v_end = (p_init/p_end * v_init**self.k)**(1/self.k)
        point_init =  self.calc_point(p = p_init, v = v_init)
        point_end =  self.calc_point(p = p_end, v = v_end)
        return point_init, point_end
    
    def calc_isoentrope_vpv(self, v_init, p_init, v_end):
        p_end = p_init * (v_init /v_end) ** self.k
        point_init =  self.calc_point(p = p_init, v = v_init)
        point_end =  self.calc_point(p = p_end, v = v_end)
        return point_init, point_end
    
    def calc_isoentrope_ptp(self, p_init, t_init, p_end):  #
        t_end = t_init * (p_init /p_end)**((1-self.k) / self.k)
        point_init =  self.calc_point(p = p_init, t = t_init)
        point_end =  self.calc_point(p = p_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_tpt(self, t_init, p_init, t_end):#
        p_end = (t_init /t_end)**(self.k/(1-self.k))* p_init
        point_init =  self.calc_point( p = p_init,t = t_init)
        point_end =  self.calc_point(p = p_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_php(self, p_init, h_init, p_end):#
        point_init = self.calc_point(p = p_init, h = h_init)
        t_init =point_init.T
        t_end = t_init * (p_init /p_end)**((1-self.k) / self.k)
        point_end =  self.calc_point(p = p_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_hph(self, h_init, p_init, h_end):
        point_init = self.calc_point(p = p_init, h = h_init)
        t_init =point_init.T
        t_end = h_end /self.c_p
        p_end = (t_init /t_end)**(self.k/(1-self.k))* p_init
        point_end =  self.calc_point(p = p_end, t = t_end)
        return point_init, point_end
    
    def calc_isoentrope_hvh(self, h_init, v_init, h_end):
        point_init = self.calc_point(v = v_init, h = h_init)
        t_init = point_init.T 
        p_init = point_init.P
        t_end = h_end /self.c_p
        p_end = (t_init /t_end)**(self.k/(1-self.k))* p_init
        point_end =  self.calc_point(p = p_end, h = h_end)
        return point_init, point_end
    
    def calc_isoentrope_vhv(self, v_init, h_init, v_end):
        point_init = self.calc_point(v = v_init, h = h_init)
        t_init = point_init.T
        t_end = t_init * (v_init/v_end)**(self.k - 1)
        point_end = self.calc_point(v = v_end, t = t_end)
        return point_init, point_end
    
    
    def calc_isoentrope(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):

        #tvv
        if is_not_None(t_init,v_init,v_end): 
            return self.calc_isoentrope_tvv(t_init, v_init, v_end)        
        #tvt
        if is_not_None(t_init,v_init,t_end): 
            return self.calc_isoentrope_tvt(t_init, v_init, t_end)        
        #pvp
        if is_not_None(p_init,v_init,p_end): 
            return self.calc_isoentrope_pvp(p_init, v_init, p_end)
        #vpv
        if is_not_None(v_init,p_init,v_end): 
            return self.calc_isoentrope_vpv(v_init, p_init, v_end)        
        #ptp
        if is_not_None(p_init,t_init,p_end): 
            return self.calc_isoentrope_ptp(p_init, t_init, p_end)        
        #tpt
        if is_not_None(t_init,p_init,t_end): 
            return self.calc_isoentrope_tpt(t_init, p_init, t_end)        
        #php
        if is_not_None(p_init,h_init,p_end): 
            return self.calc_isoentrope_php(p_init, h_init, p_end)        
        #hph
        if is_not_None(h_init,p_init,h_end): 
            return self.calc_isoentrope_hph(h_init, p_init, h_end)        
        #hvh
        if is_not_None(h_init,v_init,h_end): 
            return self.calc_isoentrope_hvh(h_init, v_init, h_end)        
        #vhv
        if is_not_None(v_init,h_init,v_end): 
            return self.calc_isoentrope_vhv(v_init, h_init, v_end)
