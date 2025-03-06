"""
CamFollower class and equations for n-PDEKF
"""
import jax.numpy as jnp

#----------------------------------------------------------------------------------------------------------------------#
#                                       case-specific options                                                          #
#----------------------------------------------------------------------------------------------------------------------#
class CamFollower():
    def __init__(self, HB_trainable, m_trainable, V_trainable, include_friction_term, nq=None):
        self.g_func = get_g_func(nq)
        self.qdim = nq
        self.HB_trainable = HB_trainable
        self.m_trainable = m_trainable
        self.V_trainable = V_trainable
        assert(include_friction_term == False), "friction term always false for blackbox"

    def config(self):
        xdim = 2
        qdim = self.qdim
        zdim = 1
        udim = 4
        if self.HB_trainable:
            udim -= 2
        if self.m_trainable:
            udim -= 1
        if self.V_trainable:
            udim -= 1
        n = {'x': xdim, 'u': udim, 'q': qdim, 'z': zdim}
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
        return f_func(omega,z)
    
    def g(self, x, u, p):
        th, omega, V, m, H, beta = self.get_params(x,u,p)
        return self.g_func(th,omega,V,m,H,beta)
#----------------------------------------------------------------------------------------------------------------------#
#                                       physics-based equations                                                        #
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------derivative function------------------------------------------------------------------------
def f_func(omega,z):
    NN = z[:,0]
    
    # xdot
    xdot = [omega, NN]
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
    else:
        raise Exception(f"nq = {nq} not implemented")
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