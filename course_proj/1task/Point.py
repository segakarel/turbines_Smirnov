from dataclasses import dataclass

@dataclass
class Point():

    """ The steam parametra initialization class
        Atrubutes:
            p: float, — steam pressure, Pa
            v: float, — volume, m$^3$/kg
            t: float, — temperature, $^\circ$ C
            h: float, — enthalpy, J/kg
            s: float, — entropy, J/kg
        
        Method find_gas.P — returns pressure
        Method find_gas.v — returns volume
        Method find_gas.T — returns tempreature
        Method find_gas.h — returns enthalpy
        Method find_gas.s — returns entropy
        Method find_gas.as_dict - returns dictionary 
        {"p": value_p, "v":value_v, "t": value_t,"h":value_h, "s":value_s}
    """
    
    P: float
    v: float
    T: float
    h: float
    s: float
    x: float
        
    @property
    def print_dict(self):
        dct = {"p": self.P, "v": self.v, "t": self.T, "h": self.h, "s": self.s}
        return dct
    