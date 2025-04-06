from iapws import IAPWS97 as gas
from scipy.optimize import minimize
from Point import Point as Point

def r(x):
    
    a = (round(x, 5))
    return a 

def is_not_None(*args, **kwargs):
    a = 0
    for i in range(len(args)):
        if args[i] is not None:
            a+=1
    if a == len(args):return True
    else: return False

to_kelvin = lambda x: x + 273.15 if x else None
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa

    
class Steam():
    """ Steam parametra calculation class
        Atrubutes:
            name: str, — gas name (steam, air applied)
            R: float,— the universal gas constant
            с_p: float,—  isobarical thermocapacity value
            
        Method calc_point(p,v,t,h,s,x): calcs all over the   
        uninitial values; the *Point* object as output.
        
        Method calc_isobar(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in isobaric properies; the *Point* object as output.
        
        Method calc_isochor(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in the isochoric process; the *Point* object as output.
        
        Method calc_isoterm(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values 
        in the isotermical process; the *Point* object as output.
        
        Method calc_isoentrope(p_1,p_2,v_1,v_2,t_1,t_2,h_1,h_2,s_2,s_2): finding values  
        in isoentropic process; the *Point* object as output.
    """
    
    def __init__(self, name, R, c_p):
        self.name = name
        self.c_p= c_p
        self.R = R
        self.c_v = self.c_p - self.R
        self.k = self.c_p / self.c_v   
        
    # Вычисление точки
    def calc_point(self,p = None,v = None,t = None,h = None,s = None, x = None): # +
        
        if is_not_None(p,x): return self.calc_point_px(p,x)
        if is_not_None(t,x): return self.calc_point_tx(t,x)
        if is_not_None(s,x): return self.calc_point_sx(s,x)
        if is_not_None(h,x): return self.calc_point_hx(h,x)
        if is_not_None(v,x): return self.calc_point_vx(v,x)
    
        if is_not_None(p,h): return self.calc_point_ph(p,h)  # +
        if is_not_None(p,s): return self.calc_point_ps(p,s)  # +
        if is_not_None(v,s):return self.calc_point_vs(v,s)   # +
        if is_not_None(v,h):return self.calc_point_vh(v,h)   # +
        if is_not_None(h,s):return self.calc_point_hs(h,s)   # +
        if is_not_None(h,t):return self.calc_point_ht(h,t)   # +
        if is_not_None(s,t):return self.calc_point_st(s,t)   # +
        
        if is_not_None(v,t,p):return self.calc_point_vtp(v,t,p) # +
        
        if is_not_None(p,v):return self.calc_point_pv(p,v)   # +
        if is_not_None(p,t):return self.calc_point_pt(p,t)   # +
        if is_not_None(v,t):return self.calc_point_vt(v,t)   # +
      
                                             
        else:raise Exception("This values is not enought")
        
    def calc_point_px(self,p,x):
        point = gas(P = p*unit, x=x)
        pnt = Point(r(point.P),r(point.v),r(point.T),r(point.h),r(point.s),point.x)
        return pnt
    
    def calc_point_tx(self,t,x):
        point = gas(T= t*unit, x=x)
        pnt = Point(r(point.P),r(point.v),r(point.T),r(point.h),r(point.s),point.x)
        return pnt
    
    def calc_point_sx(self,s,x):
        point = gas(s = s, x=x)
        pnt = Point(r(point.P),r(point.v),r(point.T),r(point.h),r(point.s),point.x)
        return pnt
    
    def calc_point_hx(self,h,x):
        point = gas(h = h, x=x)
        pnt = Point(r(point.P),r(point.v),r(point.T),r(point.h),r(point.s),point.x)
        return pnt
    
    def calc_point_vx(self,v,x):
        point = gas(v = v, x=x)
        pnt = Point(r(point.P),r(point.v),r(point.T),r(point.h),r(point.s),point.x)
        return pnt
        
    
    def calc_point_vtp(self,v1,t,p1): # 1
        
        point = gas(P = p1*unit, T = to_kelvin(t))

        if abs(point.v-v1) < 0.00015: 
            t = point.T
            h = point.h
            pnt = Point(r(p1*unit),r(point.v),r(t-273.15),r(h),r(point.s),point.x)
            return pnt
        
        else:
        
            point_streak =  gas(T = to_kelvin(t), x = 0)
            point_2streak = gas(T = to_kelvin(t), x = 1)
            v_str = point_streak.v
            v_2str = point_2streak.v
            x_ = (v1 - v_str) / (v_2str - v_str)
            point_fact = gas(P = p1 * unit, x = x_)
            v = point_fact.v
            t = point_fact.T
            h = point_fact.h
            s = point_fact.s
            pnt = Point(r(p1*unit),r(v),r(t-273.15),r(h),r(s),point.x)
            return pnt
    

        
    def calc_point_pv(self, p,v_1): 
    
        def error(t):
            v = float(gas(P=float(p  * unit), T=float((t+273.15))).v)
            return (v-v_1) ** 2
        
        t0 = p * v_1 /self.R
        t = float(minimize(error, x0 = t0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, T = to_kelvin(t))
        if abs(point.v - v_1) < 0.0001:
            p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
            return pnt
        else:
            raise Exception("The humidity steam area, I need the p, h or s")
    
    
    def calc_point_pt(self, p,t): 
        point = gas(P = p, T = to_kelvin(t))
        
        if (t <373.946) and (p <22.064):
            point_ = gas(T = to_kelvin(t), x = 1)
            h1 = point_.h
            h = point.h
            x = point.x
          
            p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),x)
            return pnt
        else:
            p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
            return pnt
    
   
    def calc_point_ps(self, p,s): 
        point = gas(P = p, s = s) 
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
        return pnt
    
   
    def calc_point_ph(self, p,h): 
        point = gas(P = p, h = h) 
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
        return pnt
    
  
    def calc_point_vt(self,v_1, t):#8

        def error(p0):
            v = float(gas(P=float(p0  * unit), T=float(to_kelvin(t))).v)
            return (v-v_1) ** 2
        
        p0 = to_kelvin(t) * self.R / v_1
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, T = (t+273.15))
        x = point.x
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        if (t < 647):
            pnt_ = gas(T = t, x = 1)
            if abs(pnt_.h - h) <1 or (x == 0):
                raise Exception("The humidity steam area, I need the p, h or s")
            else:
                pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
                return pnt
        else:
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
            return pnt
        

    
    def calc_point_st(self,s, t_1):
        def error(p0):
    
            t = float(gas(P=float(float(p0  * unit)), s = s).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_ht(self,h, t_1):
        def error(p0):
    
            t = float(gas(P=float(float(p0  * unit)), h = h).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p0 = 10**6
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_vs(self,v_1, s): 
        def error(p0):
            v = float(gas(P=float(p0  * unit), s = s).v)
            return (v-v_1) ** 2
        
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_vh(self,v_1, h):
        def error(p0):
            v = float(gas(P=float(p0  * unit), h = h).v)
            return (v-v_1) ** 2
        
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_hs(self,h, s):
        point = gas(h = h, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
      
    def calc_isobar(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_start = self.calc_point(p = p_end,v = v_init,t = t_init ,h = h_init,s = s_end)
        p_end = point_start.P
        point_end = self.calc_point( p = p_end, v = v_end,t = t_end,h = h_end, s =s_end)
        return point_start, point_end
    
    def calc_isochor(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_start = self.calc_point(p = p_end,v = v_init,t = t_init ,h = h_init,s = s_end)
        v_end = point_start.v
        point_end = self.calc_point( p = p_end, v = v_end,t = t_end,h = h_end, s =s_end)
        return point_start, point_end
    
    def calc_isoterm(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_start = self.calc_point(p = p_end,v = v_init,t = t_init ,h = h_init,s = s_end)
        t_end = point_start.T
        point_end = self.calc_point( p = p_end, v = v_end,t = t_end,h = h_end, s =s_end)
        return point_start, point_end
   
    def calc_isoentrope(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_start = self.calc_point(p = p_end,v = v_init,t = t_init ,h = h_init,s = s_end)
        s_end = point_start.s
        point_end = self.calc_point( p = p_end, v = v_end,t = t_end,h = h_end, s =s_end)
        return point_start, point_end
    
        