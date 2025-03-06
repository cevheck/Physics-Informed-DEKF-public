"""
Main class used for DEKF filtering with adaptive Q and R matrices
Used by calling
- init
- simulate
"""
import jax
import jax.numpy as jnp
import timeit
from time import time as timeit
from functools import partial
import numpy as np

try:
    import wandb
except Exception as e:
    print(f"Warning: {e}")

class E_dekf():
    def __init__(self, Qx=0.01, R=0.01, state_dim=None, input_dim=None, models=None, xk_ini=None, Pxk_ini=None, wk_ini=None, Pw_ini=None, Qw=None, C=None, dt=None, update=True, p_true=None, alpha=0.0, regularized=False, trainable_names=False, MC=None, **kwargs):
        self.window = MC.window
        self.MC = MC
        self.models = models
        self.n_ensemble = len(self.models)
        self.dtype = self.models[0].HMdtype
        
        ## Kalman filter settings
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt
        self.p_true = p_true

        ## uncertainty updates
        self.Qx = jnp.array(Qx, dtype=self.dtype)       # state uncertainty
        self.R = jnp.array(R, dtype=self.dtype)         # measurement uncertainty
        
        ## network weights
        wk_ini_ensemble, Qw_ensemble, Pw_ini_ensemble = self.initialize_weights_and_uncertainties(Pw_ini, Qw, trainable_names)

        self.safety_checks_on_models()        # some safety check on how the different models compare -- ensures that all models have the same output, given (NN,p,scaler)
    
        wk_ini = jnp.array(wk_ini_ensemble, dtype=self.dtype)                       # initial weights
        if Qw is not None: self.Qw = jnp.array(Qw_ensemble, dtype=self.dtype)       # initial weight process uncertainty
        Pw_ini = jnp.array(Pw_ini_ensemble, dtype=self.dtype)                       # initial weight uncertainty
        
        self.Qw = jnp.array(self.Qw)                                                
        
        assert(Pw_ini.shape[1] == self.Qw.shape[1] == wk_ini.shape[1] == self.n_var)    # check on shape, should all match the # param
        ## measurement matrices
        if C is None:
            self.C = jnp.zeros(shape=(1, self.state_dim))
            self.C.at[0,0].set(1)   # only measure first state
            self.B = self.C.T   #model variance is only on first state
        else:
            self.C = jnp.array(C, dtype=self.dtype)

        ## initial guesses
        self.initial_condition = {'xk_ini':xk_ini, 'Pxk_ini':Pxk_ini, 'wk_ini':wk_ini, 'Pwk_ini':Pw_ini}
        self.initial_condition_numpy = {}
        for key, value in self.initial_condition.items():  # set initial conditions if not given to default values
            if value is None:   # choose standard
                if key == 'xk_ini':
                    tensorvalue = jnp.zeros(shape=(self.state_dim,1), dtype=self.dtype)
                elif key == 'Pxk_ini':
                    tensorvalue = 100*jnp.eye(self.state_dim, self.state_dim, dtype=self.dtype)  # arbitrarely high, model very insecure at the start
                elif key == 'wk_ini':
                    prngkey = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX
                    wkini = 0.01*jax.random.uniform(key=prngkey, shape=(self.n_var,1))
                    tensorvalue = jnp.array(wkini, dtype=self.dtype)
                elif key == 'Pwk_ini':
                    weight_uncertainty = 0.00005*jnp.ones(self.n_p)
                    physical_uncertainty = 0.001*jnp.ones(self.total_sinapses)
                    Pwkini = jnp.diag(jnp.hstack((weight_uncertainty,physical_uncertainty)))
                    tensorvalue = jnp.array(Pwkini, dtype=self.dtype)
                else:
                    raise Exception(f"Key {key} not found")
            else:
                tensorvalue = jnp.array(value, dtype=self.dtype)  
            if len(tensorvalue.shape) != 3:
                tensorvalue = jnp.tile(tensorvalue, reps=(self.n_ensemble,1,1)) # repeat for each ensemble member
            setattr(self, key, tensorvalue)
            self.initial_condition[key] = tensorvalue
            self.initial_condition_numpy[key] = np.array(tensorvalue)    # numpy to not return JAX objects in result_dict

    def __call__(self, *args, **kwargs):
        try:
            kwargs['p'] = kwargs['p'] | self.non_trainable_p_dict   # add non-trainable parameters to p
        except:
            pass
        model = kwargs['model']
        kwargs = {key:val for key,val in kwargs.items() if key != 'model'}
        return model(*args, **kwargs)   # call the model with the given arguments -- ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
    
    def simulate(self, filter_y, filter_u=None, history_y=None, history_u=None, future_y=None, future_u=None, cf=None, T=None, verbose=0):                
        # initialize time vector and length of simulation(parts)
        self.T = jnp.array(T)
        self.cf = cf

        # convert data to tensors
        history_y = jnp.array(history_y, dtype=self.dtype)
        history_u = jnp.array(history_u, dtype=self.dtype)
        filter_y = jnp.array(filter_y, dtype=self.dtype)
        filter_u = jnp.array(filter_u, dtype=self.dtype)
        future_y = jnp.array(future_y, dtype=self.dtype)
        future_u = jnp.array(future_u, dtype=self.dtype)
        
        # initialize trainable param bounds 
        self.max_bounds = jnp.array([p.max_val for p in self.models[0].p.values() if hasattr(p, 'max_val')])
        self.min_bounds = jnp.array([p.min_val for p in self.models[0].p.values() if hasattr(p, 'min_val')])
        
        # initialize uncertainty matrices
        R = jnp.repeat(jnp.expand_dims(jnp.array(self.R, dtype=self.dtype), axis=0), self.n_ensemble, axis=0)
        Qx = jnp.repeat(jnp.expand_dims(jnp.array(self.Qx, dtype=self.dtype), axis=0), self.n_ensemble, axis=0)
        
        # data for adaptive matrices
        xk_error_mw = jnp.zeros(shape=(self.n_ensemble, self.window, self.R.shape[0], 1), dtype=self.dtype)     # moving average of innovation error (xk=yk)
        A_prev = jnp.zeros(shape=(self.n_ensemble, self.state_dim, self.state_dim), dtype=self.dtype)           # previous A matrix required in Q,R update

        # load initial conditions
        xk = self.initial_condition['xk_ini']
        Pxk = self.initial_condition['Pxk_ini']
        wk = self.initial_condition['wk_ini']
        Pwk = self.initial_condition['Pwk_ini']
        deriv = jnp.zeros(shape=(self.n_ensemble, self.state_dim, self.n_var), dtype=self.dtype)                # recursive deriv needs a starting value
        Kx_prev = jnp.zeros(shape=(self.n_ensemble, self.state_dim, self.R.shape[0]), dtype=self.dtype)         # previous Kx matrix required in recursive deriv calculation
                
        ### KALMAN FILTER  (=filter) ###
        # go from [k,n['y']] to [k,n['y'], 1] such that at each time instant yk is a n['y']x1 vector
        filter_u = jnp.expand_dims(filter_u, -1) if filter_u is not None else [None for i in filter_y]
        filter_y = jnp.expand_dims(filter_y, -1)  
        
        # only select measured states
        if self.C.shape[1] != filter_y.shape[1]:
            assert self.C.shape[1] >= filter_y.shape[1], "C matrix should have more (or same) columns than filter_y"
            filter_y = jnp.concatenate((filter_y, jnp.zeros((filter_y.shape[0],1,1))), axis=1)
        filter_y = self.C @ filter_y   # real measurements
        
        # use modulo on measurement yes/no
        if self.C.shape[0] == 1:    # single measurement
            if self.C[0,0] == 1: 
                self.measurement = 'only_theta'      # single theta measurement
            else:
                self.measurement = 'only_omega'
        else:
            self.measurement = 'theta_and_omega'      # [theta, omega] measurement

        idx = 0
        self.len_history = len(history_y)
        
        tstart = timeit()
            
        ### FILTERING LOOP ###
        carry, y = jax.lax.scan(f = self.ensemble_step, 
                     init = (idx, xk, Pxk, wk, Pwk, deriv, Kx_prev, A_prev, Qx, R, xk_error_mw),
                     xs = (filter_y, filter_u))

        ### UNPACK FILTERING RESULTS ###
        xk, Pxk, wk, yk_error, Qx, xk_pred = y        
        
        xk_end, wk_end = carry[1], carry[3]
        simulation_filtering = {}
        simulation_filtering['Px'] = np.array(Pxk)
        simulation_filtering['x'] = np.array(xk)
        simulation_filtering['x_error'] = np.array(yk_error)
        simulation_filtering['w'] = np.array(wk)
        simulation_filtering['Qx'] = np.array(Qx)
        simulation_filtering['xk_pred'] = np.array(xk_pred)

        try:
            #simulation_filtering['Pw'] = np.array(Pwk)   # full array of Pw (memory issues)
            simulation_filtering['Pw'] = np.array(Pwk)[:,:,-self.n_p:,-self.n_p:]   # only physical parameters for now (memory issues)
        except:
            pass

        tend = timeit()
        if verbose: print(f'Filtering {len(filter_y)} steps took {tend-tstart} seconds')
        
        ## update each NN to their final converged value (for future prediction)
        for i in range(self.n_ensemble):
            endNNweights, endp_weights = self.wk_array_to_NNinput(wk_end[i])
            self.models[i].NNmodel.weights = endNNweights
            for key, val in endp_weights.items():
                self.models[i].p[key] = val
                    
        ### FUTURE PREDICTION (=future) ###
        if (future_y is not None and len(future_y) > 0) or (future_u is not None and len(future_u) > 0):
            tstart = timeit()
            xk_predictions = []
            for i in range(self.n_ensemble):
                endNNweights, endp_weights = self.wk_array_to_NNinput(wk_end[i])
                idx = 0
                
                carry, y = jax.lax.scan(f = self.future_step_func, 
                        init = (xk_end[i], endNNweights, endp_weights),
                        xs = (future_u))
                xk_predictions.append(y)
            simulation_future = {}
            simulation_future['x'] = np.array(xk_predictions)
            tend = timeit()
            if verbose: print(f'Predicting {len(future_y)} steps took {tend-tstart} seconds')
        else:
            simulation_future = []
        self.simulation = {'filtering': simulation_filtering, 'future': simulation_future}
        self.simulation['initial_condition'] = self.initial_condition_numpy
        self.simulation['dekf_param'] = self.total_sinapses, self.n_p

    @partial(jax.jit, static_argnums=(0,))
    def ensemble_step(self, carry, x):
        """
        Single timestep of the ensemble of Kalman filters
        Iterates (with jax.lax.scan) over the ensemble members and performs the Kalman filter update step for each member
        """
        idx, xk, Pxk, wk, Pwk, deriv, Kx_prev, A_prev, Qx, R, xk_error_mw = carry
        yk, uk = x
        carry_inner, y_inner = jax.lax.scan(f = self.inner_step_func, 
                     init = (yk, uk, idx),
                     xs = (xk, Pxk, wk, Pwk, deriv, Kx_prev, A_prev, Qx, R, xk_error_mw, self.Qw, self.scalervals))

        ## update variable values for next iteration
        yk, uk, idx = carry_inner 
        xk, Pxk, wk, Pwk, deriv, Kx, Kw, yk_error, xk_pred, A, Qx, R, xk_error_mw, condition = y_inner
            
        ##update counters
        idx += 1
    
        ##update carry for next iteration (with updated values for x,P,w, ..)
        carry = idx, xk, Pxk, wk, Pwk, deriv, Kx, A, Qx, R, xk_error_mw
        y_return = xk, Pxk, wk, yk_error, Qx, xk_pred
        return carry, y_return
    
    @partial(jax.jit, static_argnums=(0,))
    def inner_step_func(self, carry, x):    
        """
        Preparation for single timestep for single ensemble member (shaping of weights, inputs, ...)
        Perform single timestep for single ensemble member
        Post-filter update adjustments (clipping, avoid non-symmetric matrix)
        """
        yk, uk, idx = carry            
        xk, Pxk, wk, Pwk, deriv, Kx_prev, A_prev, Qx, R, xk_error_mw, Qw, scalervals = x

        prev = A_prev, Pxk, Kx_prev
        
        NNweights, p_weights = self.wk_array_to_NNinput(wk) # convert weight array to hybrid model format
        scaler = self.scaler_vals_to_scaler(scalervals)     # convert scaler values to scaler object
          
        if uk is not None:          # ----- add control input to input vector -----
            xu = jnp.concatenate((xk, uk), axis=0)
        else:
            xu = xk
            
        # perform Kalman Filter step
        xk, wk, Pxk, Pwk, Kx, Kw, deriv, yk_error, xk_pred, A, Qx, R, xk_error_mw, condition = self.step(xu, yk, wk, Pxk, Pwk, deriv, NNweights, p_weights, scaler, Qx, R, Qw, prev, xk_error_mw, idx)   # JAX calculations for the Kalman Filter
        
        wk = self.clip(wk)  # clip physical parameters to their bounds

        # avoid non-symmetric matrix
        Pxk, Pwk = self.transpose_cov(Pxk, Pwk)
        
        # return carry
        carry = yk, uk, idx
        y =     xk, Pxk, wk, Pwk, deriv, Kx, Kw, yk_error, xk_pred, A, Qx, R, xk_error_mw, condition
        return carry, y

    @partial(jax.jit, static_argnums=(0,))
    def future_step_func(self, carry, x):
        xk, endNNweights, endp_weights = carry
        uk = jnp.expand_dims(x, axis=1)
        
        NNin = jnp.concatenate((xk, uk), axis=0)
        xk = self.modelpredict_modulo_with_weights(NNin, endNNweights, endp_weights)        
        xk = jnp.expand_dims(xk, axis=1)
        
        # return carry
        carry = xk, endNNweights, endp_weights
        y = xk
        return carry, y
    
    #@partial(jax.jit, static_argnums=(0,))
    def transpose_cov(self, Pxk, Pwk):
        Pxk = (Pxk + jnp.transpose(Pxk)) / 2
        Pwk = (Pwk + jnp.transpose(Pwk)) / 2
        return Pxk, Pwk
    
    #@partial(jax.jit, static_argnums=(0,))
    def clip(self, wk):
        if self.n_p > 0:    # keep physical parameters inside of their constraints            
            p_param = wk[-self.n_p:]
            carry, y = jax.lax.scan(f = self.clipit, init = (None), xs = (p_param, self.min_bounds, self.max_bounds))
            wk = jnp.concatenate((wk[:-self.n_p], y), axis=0)
        return wk
    
    def clipit(self, carry, x):
        par, minbound, maxbound = x
        return None, jnp.clip(par, minbound, maxbound)

    #@partial(jax.jit, static_argnums=(0,))    
    def step(self, xu, yk, wk, Pxk, Pwk, deriv, NNweights, p_weights, scaler, Qx, R, Qw, prev, xk_error_mw, idx):
        """
        Single timestep of each ensemble member
        """
        A, B, J = self.model_derivatives(xu, NNweights, p_weights, scaler)
        wk, Pwk = self.weight_update(wk, Pwk, Qw)     # prediction step on weight-kalman filter                              
        
        # ----- state update -----
        xk, Pxk = self.state_update(xu, Pxk, A, Qx, NNweights, p_weights, scaler)  # prediction step on state-kalman filter 
        xk_pred = xk
        
        yk_error = self.calculate_error(xk, yk)     # innovation error
        
        # right after a switch in dataset, a few  no update to prevent big parameter updates for the initial big state errors
        condition = (idx > 0 + 4) 
        
        # ----- measurement weight update -----
        wk, deriv, Pwk, Kw, Kx = self.measurement_weight_update(xk, xu, yk, wk, deriv, A, J, Pxk, Pwk, prev, yk_error, R, condition, idx)

        # ----- measurement state update  &  Qx and R adaptive update -----
        args = xk, yk, A, Pxk, Pwk, prev, yk_error, Qx, R, xk_error_mw, Kx, idx, deriv
        xk, Pxk, Pwk, Qx, R, xk_error_mw, A, deriv, Kx = jax.lax.cond(condition,
                                                lambda args: self.perform_updates(*args), 
                                                lambda args: self.perform_switch(*args), 
                                                args)
                
        return xk, wk, Pxk, Pwk, Kx, Kw, deriv, yk_error, xk_pred, A, Qx, R, xk_error_mw, condition

    def perform_updates(self, xk, yk, A, Pxk, Pwk, prev, yk_error, Qx, R, xk_error_mw, Kx, idx, deriv):
        # update moving window
        xk_error_mw = jnp.concatenate((xk_error_mw[1:], jnp.expand_dims(yk_error, axis=0)), axis=0) 

        # ----- measurement state update -----
        xk, Pxk = self.measurement_state_update(xk, yk, Pxk, Kx, yk_error, R)
        
        # ----- update Q, R matrices -----
        Qx, R = self.adaptive_update(xk_error_mw, Qx, R, A, Kx, Pxk, prev, idx)
        return xk, Pxk, Pwk, Qx, R, xk_error_mw, A, deriv, Kx

    def perform_switch(self, xk, yk, A, Pxk, Pwk, prev, yk_error, Qx, R, xk_error_mw, Kx, idx, deriv):
        # don't add the big error to the moving average -- switch performed
        xk_error_mw = xk_error_mw

        # accept first measurement(s) completely
        if xk.shape == yk.shape:
            xk = yk
        else:
            xk = jnp.concatenate((yk, xk[-1:]), axis=0)
        # No change in uncertainty (might actually be better to arbitrarily increase it)
        Pxk = Pxk
        
        # No update yet in Q and R
        Qx, R = Qx, R

        # reset other variables to initial conditions
        # A = jnp.zeros(shape=(self.state_dim, self.state_dim), dtype=self.dtype)
        Pwk = self.initial_condition['Pwk_ini'][0]
        # deriv = jnp.zeros(shape=(self.state_dim, self.n_var), dtype=self.dtype)
        # Kx = jnp.zeros(shape=(self.state_dim, self.R.shape[0]), dtype=self.dtype)
        return xk, Pxk, Pwk, Qx, R, xk_error_mw, A, deriv, Kx
        
    def adaptive_update(self, xk_error_mw, Qx, R, A, Kx, Pxk, prev, idx):
        Qx, R = jax.lax.cond(idx >= self.window + 4, # wait till moving average is full
                                                lambda args: self.adaptiveQ_update_func(*args), 
                                                lambda args: self.leaveQ(*args), 
                                                (xk_error_mw, Qx, R, A, Kx, Pxk, prev))
        return Qx, R

    def crossprod(self, carry, x):
        x1, x2 = x
        y = x1 @ jnp.transpose(x2)
        return carry, y
    
    def adaptiveQ_update_func(self, xk_error_memory, Qx, R, A, Kx, Pxk, prev):
        ### T.Berry & T. Sauer: Adaptive ensemble Kalman filtering of non-linear systems
        init = []   # no carry is required

        if self.window <= 1:
            C0 = xk_error_memory[-2] @ jnp.transpose(xk_error_memory[-2])
            C1 = xk_error_memory[-1] @ jnp.transpose(xk_error_memory[-2])
        else:
            _, C00 = jax.lax.scan(self.crossprod, init, (xk_error_memory[:-1], xk_error_memory[:-1]))       # eps @ eps
            _, C11 = jax.lax.scan(self.crossprod, init, (xk_error_memory[1:], xk_error_memory[:-1]))        # eps @ eps_prev
            C00_ordened = jnp.sort(C00, axis=0)                                                             # Sort the array along axis 0
            median_of_middle_points = C00_ordened[int(self.window*1/4):int(self.window*3/4)]                # remove outliers
            C0 = jnp.mean(median_of_middle_points, axis=0)                                                  # take the mean
            C1 = jnp.mean(C11, axis=0)                                                                      # take the mean
        
        C = self.C
        A_prev, Pxk_prev, _ = prev
        
        Pk = jnp.linalg.pinv(C @ A) @ (C1 + C @ A @ Kx @ C0) @ jnp.transpose(jnp.linalg.pinv(C))
        Qk = Pk - A_prev @ Pxk_prev @ jnp.transpose(A_prev)
        Rk = C0 - C @ Pxk @ jnp.transpose(C)

        # ensure positive semidefinite
        Q_diag = jnp.diagonal(Qk)
        Q_offdiag = (1 - jnp.eye(Qk.shape[0])) * Qk
        Q_diag_semidefinite = jnp.max(jnp.array([Q_diag, jnp.zeros_like(Q_diag)]), axis=0)        
        Qk = jnp.diag(Q_diag_semidefinite) + Q_offdiag
        
        # update step
        Q_next = Qx + self.MC.delta * (Qk - Qx)
        R_next = R + self.MC.delta * (Rk - R)
        return Q_next, R_next

    def leaveQ(self, xk_error_memory, Qx, R, A, Kx, Pxk, prev):
        return Qx, R
    
    def weight_update(self, wk, Pwk, Qw):
        # ----- weight update -----
        wk = wk
        #option 1: Pwk = Pwk                                  # no decay -- will converge to true parameters (but slower than RLS and also difficulties with changing systems)
        #option 2: Pwk2 = 1 / self.lamda * Pwk                # this is recursive least squares variant of the algorithm (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.678.858&rep=rep1&type=pdf)
        #option 3: Estimate this addition as "how much uncertainty is added to the weights at each step
                    # will converge to estimates of the weight with a remaining uncertainty

        Pwk = Pwk + Qw
        return wk, Pwk

    def state_update(self, xu, Pxk, A, Qx, NNweights, p_weights, scaler):
        # ----- state update -----
        Pxk = A @ Pxk @ jnp.transpose(A) + Qx
        xk = jnp.reshape(self.modelpredict_modulo_with_weights(xu, NNweights=NNweights, p_weights=p_weights, scaler=scaler), newshape=(-1,1))
        return xk, Pxk

    def measurement_weight_update(self, xk, xu, yk, wk, deriv, A, J, Pxk, Pwk, prev, yk_error, R, condition, idx):
        C_T = jnp.transpose(self.C)
        Kx = Pxk @ C_T @ jnp.linalg.inv(self.C @ Pxk @ C_T + R)
        
        use_K_prev = True
        if use_K_prev:  # more correct way to calculate the derivative
            Kx_prev = prev[2]
            dxdw = (jnp.eye(self.state_dim, dtype=self.dtype) - Kx_prev @ self.C) @ deriv     # eq 5.36 in book "Kalman Filtering and Neural Networks"
            #dxdw = (jnp.eye(self.state_dim, dtype=self.dtype) - Kx_prev @ self.C) @ deriv + dK_prev/dw(y-Cx)    # eq 5.36 in book - requires Kx_prev to be written as a function with derivative
        else:
            dxdw = (jnp.eye(self.state_dim, dtype=self.dtype) - Kx      @ self.C) @ deriv

        ### dFdw = dFdw + dFdx * dxdw + dFdu * dudw 
        deriv = J + A @ dxdw  # recursive expression of gradient [+dFdU * dudw ; but dudw = 0]   # eq 5.35 in book "Kalman Filtering and Neural Networks"
        Cw = self.C @ deriv
        Cw_T = jnp.transpose(Cw)

        Kw = (Pwk @ Cw_T) @ jnp.linalg.inv((Cw @ Pwk @ Cw_T + R))        
        Kw_regularized = Kw    

        
        args = wk, Pwk, Kw, Cw, yk_error, R, Kw_regularized

        wk, Pwk = jax.lax.cond(condition,
                                    lambda args: self.perform_weight_update(*args), 
                                    lambda args: self.dont_perform_weight_update(*args), 
                                    args)
        return wk, deriv, Pwk, Kw, Kx
               
    
    def perform_weight_update(self, wk, Pwk, Kw, Cw, yk_error, R, Kw_regularized):
        wknew = wk + Kw @ (yk_error)
        
        if self.MC.model != 'physics':
            if self.MC.transfer_learning:  
                assert(self.MC.transfer_learning == 'method1') # only implemented method so far
                # method1: only update the last layer of the neural network
                NN_fixedpart = wk[:self.idx_last_layer[0]]
                updatedpart = wknew[self.idx_last_layer[0]:]
                wk = jnp.concatenate((NN_fixedpart, updatedpart))
            else:
                wk = wknew
        else:
            wk = wknew
        ''' Most computational load in the matrix multiplications below!'''
        Pwk = (jnp.eye(self.n_var) - Kw_regularized @ Cw) @ Pwk @ jnp.transpose(jnp.eye(self.n_var) - Kw_regularized @ Cw) + Kw_regularized @ R @ jnp.transpose(Kw_regularized)
        #Pwk = (jnp.eye(self.n_var) - Kw @ Cw) @ Pwk           # only valid with optimal Kalman gain
        return wk, Pwk
    
    def dont_perform_weight_update(self, wk, Pwk, Kw, Cw, yk_error, R, Kw_regularized):
        # host_callback.id_tap(self._print_consumer, (idx), result=None)    # print the idx of the step that is not updated
        return wk, Pwk

    
    def _print_consumer(self, idx, transforms):
        print(f"Not performing update for step {idx}")
        return None
    
    def measurement_state_update(self, xk, yk, Pxk, Kx, yk_error, R):
        # ----- measurement state update -----
        xk = xk + Kx @ yk_error
        Pxk = (jnp.eye(self.state_dim) - Kx @ self.C) @ Pxk @ jnp.transpose(jnp.eye(self.state_dim) - Kx @ self.C) + Kx @ R @ jnp.transpose(Kx)     # Joseph form
        #Pxk = (jnp.eye(self.state_dim) - Kx @ self.C) @ Pxk             # only valid with optimal Kalman gain
        return xk, Pxk

    def modelpredict_with_weights(self, NNin, NNweights, p_weights, scaler):
        x_in = jnp.reshape(NNin[:self.state_dim], newshape=(-1, self.state_dim))
        u_in = jnp.reshape(NNin[self.state_dim:], newshape=(-1, self.input_dim))
        NNout = self(x=x_in, u=u_in, model=self.models[0], NNweights=NNweights, p=p_weights, scaler=scaler)     # returns batch_sizexprediction_stepsxstate_dim tensor
        NNout = NNout[0,0,:]       # remove batch_size and prediction_step dimension (both = 1): grab single prediction
        return NNout
    
    def modelpredict_modulo_with_weights(self, NNin, NNweights, p_weights, scaler):
        NNout = self.modelpredict_with_weights(NNin, NNweights, p_weights, scaler)
        ThetaMod = jnp.mod(NNout[0], 2 * jnp.pi)      # modulo on theta
        modelOut = jnp.concatenate((jnp.expand_dims(ThetaMod, axis=-1), NNout[1:]))             # combine modulo theta + original omega
        return modelOut
    
    def calculate_error(self, xk, yk):
        # calculate the error, keeping in mind that (2pi + theta) - theta = 0
        yk_error1 = (yk - self.C @ xk)
        if self.measurement == "only_omega": 
            return yk_error1    
        yk_error2 = (yk + jnp.array([[2 * jnp.pi], [0]]) - self.C @ xk) # add 2pi to theta
        yk_error3 = (yk - jnp.array([[2 * jnp.pi], [0]]) - self.C @ xk) # subtract 2pi from theta
        yk_errors = jnp.array([yk_error1,yk_error2,yk_error3])          # 3 x "statedim" x 1 tensor
        min_theta_idx = jnp.argmin(jnp.abs(yk_errors), axis=0)[0][0]
        return yk_errors[min_theta_idx]
    
    def wk_array_to_NNinput(self, wk):
        NNweights, p_weights = self.wk_array_to_weights(wk)
        pweights = {key: p_weights[i] for i,key in enumerate(self.trainable_keys)}
        return NNweights, pweights

    def scaler_vals_to_scaler(self, scalervals):
        if scalervals is None: return None
        scalermean = scalervals[0]
        scalerstd = scalervals[1]
        scalerval = {'mean': scalermean, 'std': scalerstd}
        scaler = {'type': self.scalertype, 'func': self.scalerfunc, 'vals': scalerval}
        return scaler
    
    def wk_array_to_weights(self, wk):
        NNweights = []
        n_p = 0     # amount of weights in previous iteration
        for i in range(len(self.nodes)-1):
            n_weights = self.nodes[i]*self.nodes[i+1]
            w_i = jnp.reshape(wk[n_p:n_p+n_weights], newshape=(self.nodes[i], self.nodes[i+1]))
            b_i = jnp.reshape(wk[n_p+n_weights:n_p+n_weights+self.nodes[i+1]], newshape=(-1))
            NNweights.append(())            # empty tuple for layer 0,2,... (structure of jax neural network stax)
            NNweights.append((w_i, b_i))    # tuple (w,b) for layer 1,3,...
            n_p = n_p + n_weights + self.nodes[i+1]
        p_param = jnp.reshape(wk[self.total_sinapses:], newshape=(-1))
        return NNweights, p_param

    def model_derivatives(self, xu, NNweights, p_weights, scaler):
        modelgrad = jax.jacfwd(self.modelpredict_with_weights, argnums=(0,1,2))     # jacobian of model prediction function
        dy_dxu, dy_dNN, dy_dp = modelgrad(xu[:,0], NNweights, p_weights, scaler)    # evaluate jacobian at current state and weights

        A,B = self.dFdx_jax(dy_dxu)
        J = self.dFdw_jax(dy_dNN, dy_dp)
        return A,B,J
    
    def dFdx_jax(self, dy_dxu):
        A = dy_dxu[:,:self.state_dim]
        B = dy_dxu[:, self.state_dim:]
        return A, B

    def dFdw_jax(self, dy_dNN, dy_dp):
        # convert seperate dy_dNN and dy_dp to one jacobian matrix dy/dw
        if self.total_sinapses > 0:
            weights_total = jnp.zeros(shape=(self.state_dim, self.total_sinapses))
            var_idx = 0
            for layer in dy_dNN:
                if len(layer) != 0:
                    dweights, dbiases = layer
                    flatweights = jnp.reshape(dweights, newshape=(dweights.shape[0], -1))
                    flat_w_and_b = jnp.concatenate((flatweights, dbiases), axis=1)
                    weights_total = weights_total.at[:,var_idx:var_idx+flat_w_and_b.shape[1]].set(flat_w_and_b) 
                    var_idx += flat_w_and_b.shape[1]
        if self.n_p > 0:
            ps = jnp.zeros(shape=(self.state_dim, self.n_p))
            for idx, key in enumerate(self.trainable_keys):
                ps = ps.at[:,idx].set(dy_dp[key])
        if self.total_sinapses > 0 and self.n_p > 0: J = jnp.concatenate((weights_total, ps), axis=1)
        if self.total_sinapses > 0 and self.n_p == 0: J = weights_total
        if self.total_sinapses == 0 and self.n_p > 0: J = ps
        return J

    def safety_checks_on_models(self):
        self.check_p()
        self.check_NN()

    def check_p(self):
        dicts = [model.p for model in self.models]
        all_keys = [p.keys() for p in dicts]
        all_same_keys = np.all([key == all_keys[0] for key in all_keys])
        if not(all_same_keys): raise Exception('Not all dictionaries have the same keys')
        
        # check if non-trainable physical parameters are the same for all ensemble members
        self.non_trainable_p_dict = {}
        for key in all_keys[0]:
            all_values = [p[key] for p in dicts]
            all_same_values = np.all([value == all_values[0] for value in all_values])
            if key in self.trainable_keys:
                pass    # trainable keys can be different over ensemble members
            else:
                if not(all_same_values): raise Exception(f'Not all dictionaries have the same values for key {key}')
                self.non_trainable_p_dict[key] = all_values[0]
        non_trainable_p_dicts_check = [self.non_trainable_p_dict == {key: val for key, val in modeli.p.items() if "trainable" not in str(type(val))} for modeli in self.models]
        if not np.all(non_trainable_p_dicts_check): raise Exception('Not all non-trainable dictionaries are the same')
    
    def check_NN(self):
        if self.MC.model == 'physics':  # physical model has no NN
            self.scalervals = None
            return
        n = [model.n for model in self.models]
        assert(np.all([n_i == n[0] for n_i in n]))  # all NNs should have the same structure and input-output dimensions
        n = n[0]
        
        scalers = [model.NNmodel.scaler for model in self.models]
        assert(np.all([scaler_i['func'] == scalers[0]['func'] for scaler_i in scalers]))    # all scalers should have the same scaler function
        assert(np.all([scaler_i['type'] == scalers[0]['type'] for scaler_i in scalers]))    # all scalers should have the same scaler type
        
        # create a dummy scaler (mean) and input to just check if all ensemble models indeed give same result when given same scaler and input
        mean_scaler_vals = {}
        for key, val in scalers[0]['vals'].items():
            meanval = np.mean([scaler_i['vals'][key] for scaler_i in scalers], axis=0)
            mean_scaler_vals[key] = meanval
        self.scalertype = scalers[0]['type']
        self.scalerfunc = scalers[0]['func']
        if self.scalertype == 'StandardScaler':
            scalermeans = np.expand_dims(np.array([model.NNmodel.scaler['vals']['mean'] for model in self.models]),axis=1)
            scalerstds = np.expand_dims(np.array([model.NNmodel.scaler['vals']['std'] for model in self.models]), axis=1)
            scalervals = jnp.concatenate((scalermeans, scalerstds), axis=1)
            self.scalervals = scalervals
        else:
            raise NotImplementedError(f"Scaler type {self.scalertype} not implemented")
        rngseed = jax.random.PRNGKey(51435)
        
        # dummy inputs and weights
        toy_x = jax.random.uniform(key=rngseed, shape=(1, n['x']))
        toy_u = jax.random.uniform(key=rngseed, shape=(1, n['u']))
        toy_NNweigths = self.models[0].NNmodel.weights  # just take weights of the first NN model
        toy_p = self.models[0].p                        # just take physical parameters of the first model
        toy_p = {key: toy_p[key]() if "trainable_param" in str(type(toy_p[key])) else toy_p[key] for key in toy_p.keys()}  # get current value of trainable parameters
        toy_p = {key: jax.random.uniform(key=rngseed, shape=(val.shape)) for key,val in toy_p.items()}  # get current value of trainable parameters
        toy_scaler = {'func': scalers[0]['func'], 'type': scalers[0]['type'], 'vals': mean_scaler_vals}

        modelout = [model(toy_x, toy_u, toy_NNweigths, toy_p, toy_scaler) for model in self.models]
        assert(np.all([modelout_i == modelout[0] for modelout_i in modelout]))
    
    def initialize_weights_and_uncertainties(self, Pw_ini, Qw, trainable_names):
        wk_ini_ensemble = []
        Qw_ensemble = []
        Pw_ini_ensemble = []
        for model in self.models:
            if model is None: raise Exception
            ## extract weights from (pretrained) network as initial KF guess
            weights = []
            p_param = []
            self.trainable_keys = []
            param_idx = 0
            if len(model.NNmodel.weights) > 0:  # if there are weights in the NNmodel
                for layer_idx, layer in enumerate(model.NNmodel.weights):     # layer has structure of jax.stax ([], [w,b], [], [w,b], ...)
                    if len(layer) != 0:
                        if layer_idx == len(model.NNmodel.weights) - 1:
                            start_idx_last_layer = param_idx
                        weightlayer = jnp.reshape(layer[0], newshape=(-1))
                        biaslayer = jnp.reshape(layer[1], newshape=(-1))
                        weights.extend(weightlayer)
                        weights.extend(biaslayer)
                        param_idx += len(weightlayer) + len(biaslayer)
                        if layer_idx == len(model.NNmodel.weights) - 1:
                            end_idx_last_layer = param_idx
                self.idx_last_layer = [start_idx_last_layer, end_idx_last_layer]
            trainable_p_dict = {key: val for key, val in model.p.items() if "trainable" in str(type(val))}
            for key,p_val in trainable_p_dict.items():
                p_param.append(p_val())     # extract current value of physical parameters
                self.trainable_keys.append(key)
            if self.trainable_keys != trainable_names: raise Exception(f"Trainable names {trainable_names} do not match (order of) keys {self.trainable_keys} in model {model}")
            if len(weights) != 0:
                weight_concat = jnp.array(weights)
                self.total_sinapses = len(weight_concat)
                input_N = model.n['q']
                hidden_N = [i for i in model.cfg_eta['n_hidden']]
                output_N = model.n['z']
                self.nodes = [input_N] + hidden_N + [output_N]
                weight_uncertainty_added = Qw[0] * jnp.ones(self.total_sinapses)
                weight_uncertainty = Pw_ini[0] * jnp.ones(self.total_sinapses)
            else:
                self.total_sinapses = 0
                self.nodes = []

            if len(p_param)!=0:
                p_param_concat = jnp.array(p_param)
                self.n_p = len(p_param)
                physical_uncertainty_added = jnp.array(Qw[-self.n_p:])
                physical_uncertainty = jnp.array(Pw_ini[-self.n_p:])
            else:       # if there are only weights
                self.n_p = 0
            
            # combine 
            self.n_var = self.total_sinapses + self.n_p
            try:    # if there are weights and physical parameters
                wk_ini_i = jnp.reshape(jnp.concatenate((weight_concat, p_param_concat)), newshape=(-1, 1))
                Qw_i = jnp.diag(jnp.hstack((weight_uncertainty_added, physical_uncertainty_added)))
                Pw_ini_i = jnp.diag(jnp.hstack((weight_uncertainty, physical_uncertainty)))
            except: # if there are only physical parameters
                try:
                    wk_ini_i = jnp.reshape(p_param_concat, newshape=(-1, 1))
                    Qw_i = jnp.diag(physical_uncertainty_added)
                    Pw_ini_i = jnp.diag(physical_uncertainty)
                except: # if there are only weights
                    wk_ini_i = jnp.reshape(weight_concat, newshape=(-1, 1))
                    Qw_i = jnp.diag(weight_uncertainty_added)
                    Pw_ini_i = jnp.diag(weight_uncertainty)
            if Qw is not None: Qw_i = jnp.array(Qw_i, dtype=self.dtype)
            wk_ini_i = jnp.array(wk_ini_i, dtype=self.dtype)
            Pw_ini_i = jnp.array(Pw_ini_i, dtype=self.dtype)
            wk_ini_ensemble.append(wk_ini_i)
            Qw_ensemble.append(Qw_i)
            Pw_ini_ensemble.append(Pw_ini_i)
        return wk_ini_ensemble, Qw_ensemble, Pw_ini_ensemble