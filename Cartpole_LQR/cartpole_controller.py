import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr
from numpy import linalg as LA

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps 
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        #TODO
        ns = s_star.shape[0]
        na = a_star.shape[0]
        
        def f_s(s):
            return self.f(s,a_star) # a is fixed
        def f_a(a): 
            return self.f(s_star,a) # s is fixed
 
        def c_s(s):
            return self.c(s,a_star) #a is fixed
        def c_a(a):
            return self.c(s_star,a) #s is fixed
        
        def m_sa(s_a):
#             return self.c(a_s[1:],np.array(a_s[0]).reshape(1,))
            return self.c(s_a[0:ns], np.array(s_a[ns:]).reshape(1,))

        A = jacobian(f_s,s_star)
        B = jacobian(f_a,a_star)
        Q = hessian(c_s,s_star)
        R = hessian(c_a,a_star)
        
        q = gradient(c_s,s_star).reshape(-1,1)
        r = gradient(c_a,a_star).reshape(-1,1)
        
        s_a = np.append(s_star,a_star)    
        M = hessian(m_sa,s_a)
        
        
        M = M[0:ns,ns:]


        
        # eigenvalue trick
        H = np.vstack((np.hstack((Q, M)), np.hstack((M.T, R))))
        sigma,v = LA.eig(H)
        sigma = [0 if x < 0 else x for x in sigma]
        sigma = np.diag(sigma)
        H = v@sigma@np.linalg.inv(v)
        H = H + (0.0000001*np.identity(ns+na))
        
        Q = H[0:ns,0:ns]
        M = H[0:ns,ns:]
        R = H[ns:,ns:]

        
        Q = Q/2
        R = R/2
        q = (q.T - s_star.T@Q - a_star.T@M.T).T
        r = (r.T - a_star.T@R - s_star.T@M).T
        
        b = (self.c(s_star,a_star)+(0.5*s_star.T@(Q/2)@s_star)+(0.5*a_star.T@R@a_star)+(s_star.T@M@a_star)-(q.T@s_star)-(r.T@a_star)).reshape(1,)
        m = (self.f(s_star,a_star)-(A@s_star)-(B@a_star)).reshape(-1,1)
        
        return lqr(A, B, m, Q, R, M, q, r, b, T)
        
class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a



