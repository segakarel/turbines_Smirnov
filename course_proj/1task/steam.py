from dataclasses import dataclass
from iapws import IAPWS97 as gas
from scipy.optimize import minimize
from Point import Point as Point


def is_not_None(*args):
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
    
    name: str = "steam"
    c_p: float = 4185.5
    R: float = 461.0
    c_v = c_p - R
    k = c_p / c_v   
    
        
    # point values calculation
    def calc_point(self, p = None, v = None, t = None, h = None, s = None, x = None):
        
        if is_not_None(p,x): 
            return self.calc_point_px(p,x)
        if is_not_None(t,x): 
            return self.calc_point_tx(t,x)
        if is_not_None(s,x): 
            return self.calc_point_sx(s,x)
        if is_not_None(h,x): 
            return self.calc_point_hx(h,x)
        if is_not_None(v,x): 
            return self.calc_point_vx(v,x)        
        
        if is_not_None(p,h): 
            return self.calc_point_ph(p,h)
        if is_not_None(p,s): 
            return self.calc_point_ps(p,s)
        if is_not_None(v,s): 
            return self.calc_point_vs(v,s)
        if is_not_None(v,h): 
            return self.calc_point_vh(v,h)
        if is_not_None(h,s,x): 
            return self.calc_point_hs(h,s,x)
        if is_not_None(h,t): 
            return self.calc_point_ht(h,t)
        if is_not_None(s,t): 
            return self.calc_point_st(s,t)

        if is_not_None(v,t,p): 
            return self.calc_point_vtp(v,t,p)

        if is_not_None(p,v):
            return self.calc_point_pv(p,v)
        if is_not_None(p,t):
            return self.calc_point_pt(p,t)
        if is_not_None(v,t):
            return self.calc_point_vt(v,t)
      
                                             
        else: raise Exception("Not enough the input values")
    
    def calc_point_px(self,p,x):
        point = gas(P = p*unit, x=x)
        pnt = Point(point.P,point.v,point.T,point.h,point.s,point.x)
        return pnt
    
    def calc_point_tx(self,t,x):
        point = gas(T= t*unit, x=x)
        pnt = Point(point.P,point.v,point.T,point.h,point.s,point.x)
        return pnt
    
    def calc_point_sx(self,s,x):
        point = gas(s = s, x=x)
        pnt = Point(point.P,point.v,point.T,point.h,point.s,point.x)
        return pnt
    
    def calc_point_hx(self,h,x):
        point = gas(h = h, x=x)
        pnt = Point(point.P,point.v,point.T,point.h,point.s,point.x)
        return pnt
    
    def calc_point_vx(self,v,x):
        point = gas(v = v, x=x)
        pnt = Point(point.P,point.v,point.T,point.h,point.s,point.x)
        return pnt
    
    
    def calc_point_vtp(self,v_1, t, p_1):
        
        point = gas(P = p_1*unit, T = to_kelvin(t))

        if abs(point.v-v_1) < 0.00015: # is the steam overheated?
            t = point.T
            h = point.h
            pnt = Point(p_1*unit, point.v, t-273.15, h, point.s)
            return pnt
        
        else: #is not, extropalation via humidity
        
            point_init_extr = gas(T = to_kelvin(t), x = 0)
            point_init_extr_2 = gas(T = to_kelvin(t), x = 1)
            v_init = point_init_extr.v
            v_2str = point_init_extr_2.v
            x_ = (v_1 - v_init) / (v_2str - v_init)
            find_gas = gas(P = p_1 * unit, x = x_)
            v = find_gas.v
            t = find_gas.T
            h = find_gas.h
            s = find_gas.s
            pnt = Point(p_1*unit, v, t-273.15, h, s)
            return pnt
    

        
    def calc_point_pv(self, p,v_1): #4
    
        def error(t):
            v = float(gas(P=float(p  * unit), T=float((t+273.15))).v)
            return (v-v_1) ** 2
        
        t0 = p * v_1 /self.R
        t = float(minimize(error, x0 = t0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, T = to_kelvin(t))
        if abs(point.v - v_1) < 0.0005:
            p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
            pnt = Point(p,v,t-273.15,h,s,x)
            return pnt
        else:
            raise Exception("The humidity steam area, I need the t, h or s")
    
    
    def calc_point_pt(self, p,t): #5
        print("p, t = ", p, "; ", to_kelvin(t))
        point = gas(P = p, T = to_kelvin(t))
        if (t <373.946) and (p <22.064):
            point_ = gas(T = to_kelvin(t), x = 1)
            h1 = point_.h
            h = point.h
            x = point.x
            if x != 0 and ((abs(h1-h)) > 5):
                p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
                pnt = Point(p,v,t-273.15,h,s,x)
                return pnt
            else:
                raise Exception("The humidity steam area, I need the t, h or s")
        else:
            p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
            pnt = Point(p,v,t-273.15,h,s,x)
            return pnt   
   
    def calc_point_ps(self, p,s): #6
        point = gas(P = p, s = s) 
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
   
    def calc_point_ph(self, p,h): #7
        point = gas(P = p, h = h) 
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
  
    def calc_point_vt(self,v_1, t):#8

        def error(p_0):
            v = float(gas(P=float(p_0  * unit), T=float(to_kelvin(t))).v)
            return (v-v_1) ** 2
        
        p_0 = to_kelvin(t) * self.R / v_1
        p = float(minimize(error, x0 = p_0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, T = (t+273.15))
        x = point.x
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        if (t < 647):
            pnt_ = gas(T = t, x = 1)
            if abs(pnt_.h - h) <1 or (x == 0):
                raise Exception("The humidity steam area, I need the p, h or s")
            else:
                pnt = Point(p,v,t-273.15,h,s,x)
                return pnt
        else:
            pnt = Point(p,v,t-273.15,h,s,x)
            return pnt
        

    
    def calc_point_st(self,s, t_1):#10
        def error(p_0):
    
            t = float(gas(P=float(float(p_0  * unit)), s = s).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p_0 = 10**4
        p = float(minimize(error, x0 = p_0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
    
    def calc_point_ht(self,h, t_1):#10
        def error(p_0):
    
            t = float(gas(P=float(float(p_0  * unit)), h = h).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p_0 = 10**6
        p = float(minimize(error, x0 = p_0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
    
    def calc_point_vs(self,v_1, s): #11
        def error(p_0):
            v = float(gas(P=float(p_0  * unit), s = s).v)
            return (v-v_1) ** 2
        
        p_0 = 10**4
        p = float(minimize(error, x0 = p_0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
    
    def calc_point_vh(self,v_1, h): #12
        def error(p_0):
            v = float(gas(P=float(p_0  * unit), h = h).v)
            return (v-v_1) ** 2
        
        p_0 = 10**4
        p = float(minimize(error, x0 = p_0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
    
    
    def calc_point_hs(self,h, s):#13
        point = gas(h = h, s = s)
        p,v,t,h,s,x = point.P, point.v, point.T, point.h, point.s, point.x
        pnt = Point(p,v,t-273.15,h,s,x)
        return pnt
      
    def calc_isobar(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_init = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init, s = s_init)
        p_end = point_init.P
        point_end = self.calc_point(p_end,v_end,t_end,h_end,s_end)
        return point_init, point_end
    
    def calc_isochor(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_init = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init, s = s_init)
        v_end = point_init.v
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end, s = s_end)
        return point_init, point_end
    
    def calc_isoterm(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_init = self.calc_point(p = p_init,v = v_init, t = t_init, h = h_init, s = s_init)
        t_end = point_init.T
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end, s = s_end)
        return point_init, point_end
   
    def calc_isoentrope(self,p_init = None,p_end = None,v_init = None,v_end = None,t_init = None,t_end = None,h_init = None,h_end = None,s_init = None,s_end = None):
        point_init = self.calc_point(p = p_init, v = v_init, t = t_init, h = h_init, s = s_init)
        s_end = point_init.s
        point_end = self.calc_point(p = p_end, v = v_end, t = t_end, h = h_end, s = s_end)
        return point_init, point_end
    
        