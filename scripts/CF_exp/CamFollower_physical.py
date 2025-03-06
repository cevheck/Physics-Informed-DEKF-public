"""
CamFollower class and equations for p-PDEKF
"""
import jax.numpy as jnp

#----------------------------------------------------------------------------------------------------------------------#
#                                       case-specific options                                                          #
#----------------------------------------------------------------------------------------------------------------------#
class CamFollower():
    def __init__(self, HB_trainable=False, m_trainable=False, V_trainable=False, include_friction_term=False):
        self.HB_trainable = HB_trainable
        self.m_trainable = m_trainable
        self.V_trainable = V_trainable
        self.include_friction_term = include_friction_term
        
    def config(self):
        xdim = 2
        udim = 4
        zdim = 0
        if self.HB_trainable:
            udim -= 2
        if self.m_trainable:
            udim -= 1
        if self.V_trainable:
            udim -= 1
        n = {'x': xdim, 'u': udim, 'z': zdim}
        return n, None, None
        
    def p_initial(self):
        '''
        define the physical parameter vector p, if not sure give initial guess:
        [initial guess, lower bound, upper bound]
        '''
        p = {
            'J' : 0.0130,
            'c0': 9.87,
            'c1' :9.04,
            'm': 0.75,
            'g'  : 9.81,
            'Kphi' : 0.0305,
            'GR' : 8,
            'GB_torque' : 0.9,
            'mu': 0.107,
            'Ra' : [0.6, 0.3, 5],
            'D' : [0.16, 0.0001, 1],
            }
        
        if self.HB_trainable:
            p['H'] = [0.025, 0.01, 0.05]
            p['beta'] = [jnp.pi/2, 0, jnp.pi]
        if self.m_trainable:
            p['m_extra'] = [0.8, 0.0, 2.0]
        if self.V_trainable:
            p['V'] = [5.0, 0.0, 10.0]
        return p
    
    def get_params(self, x, u, p):
        th = x[:,0]
        omega = x[:,1]
        if not(self.HB_trainable and self.V_trainable): assert(sum([self.HB_trainable, self.m_trainable, self.V_trainable]) <= 1), "Not implemented for other combinations than V + HB"
        if self.HB_trainable and self.V_trainable:
            H = p['H']*jnp.ones_like(th)
            beta = p['beta']*jnp.ones_like(th)
            V = p['V']*jnp.ones_like(th)
            m = u[:,0] + p['m']
        elif self.HB_trainable:
            H = p['H']*jnp.ones_like(th)
            beta = p['beta']*jnp.ones_like(th)
            V = u[:,0]
            m = u[:,1] + p['m']
        elif self.m_trainable:
            m = (p['m'] + p['m_extra'])*jnp.ones_like(th)
            V = u[:,0]
            H = u[:,1]
            beta = u[:,2]
        elif self.V_trainable:
            V = p['V']*jnp.ones_like(th)
            m = u[:,0] + p['m']
            H = u[:,1]
            beta = u[:,2]
        else:
            V = u[:,0]
            m = u[:,1] + p['m']
            H = u[:,2]
            beta = u[:,3]
        return th, omega, V, m, H, beta
    
    def f(self, x,u,z,p):
        th, omega, V, m, H, beta = self.get_params(x,u,p)
        return f_func(th,omega,V,m,H,beta,z,p, include_friction_term=self.include_friction_term)
    
    def g(self, x, u, p):
        return None
#----------------------------------------------------------------------------------------------------------------------#
#                                       physics-based equations                                                        #
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------derivative function------------------------------------------------------------------------
def f_func(th,omega,V,m,H,beta,z,p, include_friction_term=False):
    # state
    omega_motor = omega * p['GR']

    kphi = p['Kphi']
    Ra = p['Ra']

    # define cam properties
    h = h_fun(th, H=H, beta=beta)
    dh_dth = dh_dth_fun(th, H=H, beta=beta)
    ddh_dth = ddh_dth_fun(th, H=H, beta=beta)

    # motor torque
    Tm = V*kphi/Ra - (kphi)**2/Ra * omega_motor
    T_shaft = Tm*p['GR']*p['GB_torque']
    
    # friction torque in driveline
    v = dh_dth * omega
    #fric_Gearbox = p['D'] * omega          # already fitted in Kphi en Ra constants!
    #fric_motor = p['D2'] * omega_motor     # already fitted in Kphi en Ra constants!
    fric_rest = p['D'] * omega              # rest of friction or similar losses (bearing-shaft, coupling, etc)
    
    # cam-follower contact dynamics
    r0 = 0.02
    R = (r0+h)
    dydth = dh_dth
    dxdth = r0 + h
    dydx = dydth/dxdth          
    delta = jnp.arctan(dydx)
    R = (r0+h)
    #r = R * jnp.sin(delta)
    #r = r/jnp.cos(delta)    # comes from conversion from force in y-direction of the force balance  F --> F_vertical in y-direction F_vert = F*cos(delta)
                                # could also say Fx = F*sin(delta), Fz = F*cos(delta) --> Fx = Fz*sin(delta)/cos(delta) --> T = Fx*R = Fz*R*sin(delta)/cos(delta)
    delta = jnp.where(delta==0, 1e-10, delta)                   # avoid division by zero
    r = R / ((jnp.cos(delta) / jnp.sin(delta)) - p['mu'])       # comes from Fy = ma+mg+fric with fric = mu*Fx + Fstribeck
                                                                                # --> mu*Fx = mu*Fy*sin/cos and bring to other side --> factor (1-mu*sin/cos) --> bring back as denominator
                                                                                # --> then going to torque we get r = R *sin/cos
                                                                                    # --> [.../(1-mu*sin/cos)]*sin/cos = [.../(1-mu*sin/cos)]/[cos/sin] = [.../(1-mu*sin/cos)*cos/sin] = [.../((cos/sin)-mu)]
    J_eq = p['J'] + m * dh_dth * r

    include_stribeck = False
    if include_stribeck: 
        fric = (p['c0'] * jnp.tanh(v) + p['c1'] * v)
        Tl_eq = (m * ddh_dth * omega**2 + m*p['g'] + fric)* r
    else:
        Tl_eq = (m * ddh_dth * omega**2 + m*p['g'])* r

    if include_friction_term:
        omega_dot =  (T_shaft-fric_rest-Tl_eq)/J_eq
    else:
        omega_dot =  (T_shaft-Tl_eq)/J_eq

    zero = jnp.array(0.0, dtype=omega_dot.dtype)
    condition = jnp.logical_and(omega <= zero, omega_dot < zero)
    omega_dot = jnp.where(condition, zero, omega_dot)  # don't let omega go under zero
    
    # xdot
    xdot = [omega, omega_dot]
    return jnp.stack(xdot, axis=1)

#----------------------------input q of neural network------------------------------------------------------------------
def get_g_func(nq):
    if nq == 6:
        def g_func(th,omega,V,m,H,beta):
            return jnp.stack([jnp.sin(th), jnp.cos(th), omega, V, H, beta], axis=1)
    elif nq == 4:
        def g_func(th,omega,V,m,H,beta):
            return jnp.stack([jnp.sin(th), jnp.cos(th), omega, V], axis=1)
    elif nq == 3:
        def g_func(th,omega,V,m,H,beta):
            return jnp.stack([jnp.sin(th), jnp.cos(th), omega], axis=1)
    return g_func

#----------------------------additional functions (optional) -----------------------------------------------------------
def h_fun(th, H, beta):
    th = th % (jnp.pi * 2)
    out = jnp.where(th < beta,
                    H / beta * th - H / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * th / beta),
                    H * (1 + 1 / (2 * jnp.pi - beta) * (beta - th) - 1 / (2 * jnp.pi) * jnp.sin(
                        2 * jnp.pi * (beta - th) / (2 * jnp.pi - beta)))
                )
    return out

def dh_dth_fun(th, H, beta):
    # first derivative of displacement with respect to angle
    th = th % (jnp.pi * 2)
    out = jnp.where(th < beta,
                    H / beta * (1 - jnp.cos(2 * jnp.pi * th / beta)),
                    H / (2 * jnp.pi - beta) * (-1 + jnp.cos(2 * jnp.pi * (beta - th) / (2 * jnp.pi - beta)))
                )
    return out

def ddh_dth_fun(th, H, beta):
    # second derivative of displacement with respect to angle
    th = th % (jnp.pi * 2)
    out = jnp.where(th < beta,
                    2 * jnp.pi * H / (beta ** 2) * jnp.sin(2 * jnp.pi * th / beta),
                    2 * jnp.pi * H / ((2 * jnp.pi - beta) ** 2) * jnp.sin(2 * jnp.pi * (beta - th) / (2 * jnp.pi - beta))
                )
    return out