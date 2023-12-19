import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method
    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n = x.shape[0]
    gradient = np.zeros(n, dtype=np.float64)
    # For every element of x, consider small deviations only in that component
    for i in range(n):
        # Create copies so as to not modify the original point x
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += delta
        x_minus[i] -= delta
        # Approximate the ith element with the difference along the ith component
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * delta)
    return gradient


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method
    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    # Get array shapes and ensure x is type np.float64
    m, n = f(x).shape[0], x.shape[0]
    x = x.astype('float64')
    # Initialize Jacobian to all zeros
    J = np.zeros((m, n), dtype=np.float64)
    # Iterate over the columns of the Jacobian
    for col in range(n):
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[col] += delta
        x_minus[col] -= delta
        # The jth column is the derivative of each f component w.r.t. x_j
        J[:, col] = (f(x_plus) - f(x_minus)) / (2 * delta)
    return J


def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    n = x.shape[0]
    H = np.zeros((n, n), dtype=np.float64)
    # For every element of x, consider deviations only along that direction
    for i in range(n):
        # Create copies so as to not modify the original point x
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += delta
        x_minus[i] -= delta
        # Same finite difference along ith direction, using the gradient as the function
        H[i, :] = (gradient(f, x_plus) - gradient(f, x_minus)) / (2 * delta)
    return H
