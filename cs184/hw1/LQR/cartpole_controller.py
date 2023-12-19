import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, x, u):
        """
        Cost function of the env.
        It sets the state of environment to `x` and then execute the action `u`, and
        then return the cost. 
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        observation, cost, done, info = env.step(u)
        return cost

    def f(self, x, u):
        """
        State transition function of the environment.
        Return the next state by executing action `u` at the state `x`
        Parameter:
            x (1D numpy array) with shape (4,)
            u (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        next_observation, cost, done, info = env.step(u)
        return next_observation


    def compute_local_policy(self, x_star, u_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (x_star, u_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            x_star (numpy array) with shape (4,)
            u_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        # Compute the Jacobians of the dynamics f
        A = jacobian(lambda x: self.f(x, u_star), x_star)
        B = jacobian(lambda u: self.f(x_star, u), u_star)
        # Compute the gradients of the cost function
        q = gradient(lambda x: self.c(x, u_star), x_star)
        r = gradient(lambda u: self.c(x_star, u), u_star)
        # Compute the Hessians of the cost function
        Q = hessian(lambda x: self.c(x, u_star), x_star)
        R = hessian(lambda u: self.c(x_star, u), u_star)
        # The matrix M is a matrix of mixed partial derivatives
        nx, nu = x_star.shape[0], u_star.shape[0]
        M = np.zeros((nx, nu), dtype=np.float64)
        # Iterate over elements of x, i.e. the rows of M
        for i in range(nx):
            # Create copies so as to not modify the original point x
            x_plus, x_minus = np.copy(x_star), np.copy(x_star)
            # Use delta = 1e-5
            delta = 1e-5
            x_plus[i] += delta
            x_minus[i] -= delta
            # Functions for computing the gradient along u
            c_plus = lambda u: self.c(x_plus, u)
            c_minus = lambda u: self.c(x_minus, u)
            # Apply finite-difference method along x to the gradient along u
            M[i, :] = (gradient(c_plus, u_star) - gradient(c_minus, u_star)) / (2 * delta)
        # Initialize W matrix with Hessians computed
        W_top, W_bottom = np.hstack((Q, M)), np.hstack((M.T, R))
        W = np.vstack((W_top, W_bottom))
        # Make sure that W is positive definite
        W_PD = np.zeros_like(W)
        e_vals, e_vecs = np.linalg.eigh(W)
        for e_val, e_vec in zip(e_vals, e_vecs):
            if e_val > 0:
                W_PD += e_val * np.outer(e_vec, e_vec)
        # Use lambda = 2e-5 and extract Q, R, M
        W_PD += 2e-5 * np.identity(nx + nu)
        Q = W_PD[:nx, :nx]
        R = W_PD[nx:, nx:]
        M = W_PD[:nx, nx:]
        # Compute the remaining parameters to be used in LQR optimization
        q2 = (q - (Q.T @ x_star) - (M @ u_star)).reshape((-1, 1))
        r2 = (r - (R.T @ u_star) - (M.T @ x_star)).reshape((-1, 1))
        b = np.array([self.c(x_star, u_star) + (x_star.T @ Q @ x_star) / 2 + (u_star.T @ R @ u_star) / 2 \
            + (x_star.T @ M @ u_star) - (q.T @ x_star) - (r.T @ u_star)])
        m = (self.f(x_star, u_star) - (A @ x_star) - (B @ u_star)).reshape((-1, 1))
        # Call LQR method on parameters computed
        return lqr(A, B, m, Q / 2, R / 2, M, q2, r2, b, T)


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
