from iapws import IAPWS97 as gas
from scipy.optimize import minimize

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

class Point():

    """ Класс хранения параметров состояния газа
        Атрибуты:
            p: float, — давление газа, Па
            v: float, — удельный объём газа, м^3/кг
            t: float, — температура газа, °C
            h: float, — энтпальпия газа, Дж/кг
            s: float, — энттропия газа, Дж/кг
        
        Метод some_gas.P - возвращает значение давления класса
        Метод some_gas.v - возвращает удельный объём газа
        Метод some_gas.T - возвращает температуру газа
        Метод some_gas.h - возвращает энтальпию газа
        Метод some_gas.s - возвращает энтропию газа
        Метод some_gas.as_dict - возвращает словарь 
        {"p": value_p, "v":value_v, "t": value_t,"h":value_h, "s":value_s}
    """
    
    def __init__(self,p = None,v = None,t = None,h = None,s =None,x = None):
        self.P = p
        self.v = v
        self.T = t
        self.h = h
        self.s = s
        self.x = x
        
    @property
    def as_dict(self):
        dct = {"p": self.P, "v":self.v, "t": self.T,"h":self.h, "s":self.s, "x":self.x}
        return dct
    
class Steam():
    """ Класс для расчёта пара
        Атрибуты:
            name: str, — наименование газа
            R: float,— газовая постоянная газа
            с_p: float,—  изобарная теплоёмкость газа
        
        Если значение требует поиска  - None
            
        Метод calc_point(p,v,t,h,s): рассчитывает недостающие  
        параметры состояния в точке. На выходе класс Point.
        
        Метод calc_isobar(p1,p2,v1,v2,t1,t2,h1,h2,s1,s2):рассчитывает недостающие 
        параметры состояния в изобарном процессе. На выходе элемент класс Point.
        
        Метод calc_isohor(p1,p2,v1,v2,t1,t2,h1,h2,s1,s2):рассчитывает недостающие 
        параметры состояния в изохорном процессе. На выходе элемент класса Point.
        
        Метод calc_isoterm(p1,p2,v1,v2,t1,t2,h1,h2,s1,s2):рассчитывает недостающие 
        параметры состояния в изотермическом процессе. На выходе элемент класса Point
        
        Метод calc_isentrope(p1,p2,v1,v2,t1,t2,h1,h2,s1,s2):рассчитывает недостающие 
        параметры состояния в изоэнтропном процессе. На выходе элемент класса Point.
            
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
      
                                             
        else:raise Exception("Заданных значений недостаточно")
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

        if abs(point.v-v1) < 0.00015: #поверяем перегретый или нет
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
    

        
    def calc_point_pv(self, p,v_1): #4
    
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
            raise Exception("Область влажного пара, введите: t или h или s")
    
    
    def calc_point_pt(self, p,t): #5
        # print("Ave Maria",t)
        point = gas(P = p, T = to_kelvin(t))
        #print(point.h)
        
        if (t <373.946) and (p <22.064):
            point_ = gas(T = to_kelvin(t), x = 1)
            h1 = point_.h
            h = point.h
            # print("h1 ",h1,"h ",h)
            x = point.x
          
            # if x != 0 and ((abs(h1-h)) > 5):
            p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),x)
            return pnt
            # else:
            #     raise Exception("Область влажного пара, введите: v или h или s")
        else:
            p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
            return pnt
    
   
    def calc_point_ps(self, p,s): #6
        point = gas(P = p, s = s) 
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s),point.x)
        return pnt
    
   
    def calc_point_ph(self, p,h): #7
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
                raise Exception("Область влажного пара, введите: p или h или s")
            else:
                pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
                return pnt
        else:
            pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
            return pnt
        

    
    def calc_point_st(self,s, t_1):#10
        def error(p0):
    
            t = float(gas(P=float(float(p0  * unit)), s = s).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_ht(self,h, t_1):#10
        def error(p0):
    
            t = float(gas(P=float(float(p0  * unit)), h = h).T)
            return (t-(to_kelvin(t_1))) ** 2
            
        p0 = 10**6
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_vs(self,v_1, s): #11
        def error(p0):
            v = float(gas(P=float(p0  * unit), s = s).v)
            return (v-v_1) ** 2
        
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_vh(self,v_1, h): #12
        def error(p0):
            v = float(gas(P=float(p0  * unit), h = h).v)
            return (v-v_1) ** 2
        
        p0 = 10**4
        p = float(minimize(error, x0 = p0, method = 'Nelder-Mead').x)
        point = gas(P = p * unit, h = h)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
    
    
    def calc_point_hs(self,h, s):#13
        point = gas(h = h, s = s)
        p,v,t,h,s = point.P, point.v, point.T, point.h, point.s
        pnt = Point(r(p),r(v),r(t-273.15),r(h),r(s))
        return pnt
      
    def calc_isobar(self,ps = None,pf = None,vs = None,vf = None,ts = None,tf = None,hs = None,hf =None,ss= None,sf = None):
       # print (ps,pf,vs,vf,ts,tf,hs,hf,ss,sf)
        point_start = self.calc_point(p = ps,v = vs,t = ts ,h = hs,s = ss)
        pf = point_start.P
        point_end = self.calc_point( p = pf, v = vf,t = tf,h = hf, s =sf)
        return point_start, point_end
    
    def calc_isohor(self,ps = None,pf = None,vs = None,vf = None,ts = None,tf = None,hs = None,hf = None,ss = None,sf = None):
        point_start = self.calc_point(p = ps,v = vs,t = ts ,h = hs,s = ss)
        vf = point_start.v
        point_end = self.calc_point( p = pf, v = vf,t = tf,h = hf, s =sf)
        return point_start, point_end
    
    def calc_isoterm(self,ps = None,pf = None,vs = None,vf = None,ts = None,tf = None,hs = None,hf = None,ss = None,sf = None):
        point_start = self.calc_point(p = ps,v = vs,t = ts ,h = hs,s = ss)
        tf = point_start.T
        point_end = self.calc_point( p = pf, v = vf,t = tf,h = hf, s =sf)
        return point_start, point_end
   
    def calc_isentrope(self,ps = None,pf = None,vs = None,vf = None,ts = None,tf = None,hs = None,hf = None,ss = None,sf = None):
        point_start = self.calc_point(p = ps,v = vs,t = ts ,h = hs,s = ss)
        sf = point_start.s
        point_end = self.calc_point( p = pf, v = vf,t = tf,h = hf, s =sf)
        return point_start, point_end
    
        