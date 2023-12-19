import numpy as np


def compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p):
    """
    Compute the Q function parameters for time step t.
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
        Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        Q (2d numpy array): A numpy array with shape (n_x, n_x). Q is PD
        R (2d numpy array): A numpy array with shape (n_u, n_u). R is PD.
        M (2d numpy array): A numpy array with shape (n_x, n_u)
        q (2d numpy array): A numpy array with shape (n_x, 1)
        r (2d numpy array): A numpy array with shape (n_u, 1)
        b (1d numpy array): A numpy array with shape (1,)
        P (2d numpy array): A numpy array with shape (n_x, n_x). This is the quadratic term of the
            value function equation from time step t+1. P is PSD.
        y (2d numpy array): A numpy array with shape (n_x, 1).  This is the linear term
            of the value function equation from time step t+1
        p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
            value function equation from time step t+1
    Returns:
        C (2d numpy array): A numpy array with shape (n_x, n_x)
        D (2d numpy array): A numpy array with shape (n_u, n_u)
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x,1)
        g (2d numpy array): A numpy array with shape (n_u,1)
        h (1d numpy array): A numpy array with shape (1,)

        where the following equation should hold
        Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a + h

    """
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert Q.shape == (n_x, n_x)
    assert R.shape == (n_u, n_u)
    assert M.shape == (n_x, n_u)
    assert q.shape == (n_x, 1)
    assert r.shape == (n_u, 1)
    assert b.shape == (1,)
    assert P.shape == (n_x, n_x)
    assert y.shape == (n_x, 1)
    assert p.shape == (1,)

    # Apply the formulas given for C_t, D_t, E_t, f_t, g_t, and h_t
    C = Q + A.T @ P @ A
    D = R + B.T @ P @ B
    E = M + (2 * A.T @ P @ B)
    f = q + (A.T @ y) + 2 * (A.T @ P @ m)
    g = r + (B.T @ y) + 2 * (B.T @ P @ m)
    h = b + (m.T @ P @ m)[0] + (m.T @ y)[0] + p
    return C, D, E, f, g, h


def compute_policy(A, B, m, C, D, E, f, g, h):
    """
    Compute the optimal policy at the current time step t
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
    Let Q_t^*(x) = x^T C x + u^T D u + x^T E u + f^T x  + g^T u  + h
    Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        C (2d numpy array): A numpy array with shape (n_x, n_x). C is PD.
        D (2d numpy array): A numpy array with shape (n_u, n_u). D is PD.
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x, 1)
        g (2d numpy array): A numpy array with shape (n_u, 1)
        h (1d numpy array): A numpy array with shape (1,)
    Returns:
        K_t (2d numpy array): A numpy array with shape (n_u, n_x)
        k_t (2d numpy array): A numpy array with shape (n_u, 1)
        where the following holds
        \pi*_t(x) = K_t x + k_t
    """
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert C.shape == (n_x, n_x)
    assert D.shape == (n_u, n_u)
    assert E.shape == (n_x, n_u)
    assert f.shape == (n_x, 1)
    assert g.shape == (n_u, 1)
    assert h.shape == (1,)

    # Apply the formulas given for K_t and k_t
    half_Dinv = np.linalg.inv(D) / 2
    K = -half_Dinv @ E.T
    k = -half_Dinv @ g
    return K, k


def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    """
    Compute the V function parameters for the next time step
    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
    Let V_t^*(x) = x^TP_tx + y_t^Tx + p_t
    Parameters:
        A (2d numpy array): A numpy array with shape (n_x, n_x)
        B (2d numpy array): A numpy array with shape (n_x, n_u)
        m (2d numpy array): A numpy array with shape (n_x, 1)
        C (2d numpy array): A numpy array with shape (n_x, n_x). C is PD.
        D (2d numpy array): A numpy array with shape (n_u, n_u). D is PD.
        E (2d numpy array): A numpy array with shape (n_x, n_u)
        f (2d numpy array): A numpy array with shape (n_x, 1)
        g (2d numpy array): A numpy array with shape (n_u, 1)
        h (1d numpy array): A numpy array with shape (1,)
        K (2d numpy array): A numpy array with shape (n_u, n_x)
        k (2d numpy array): A numpy array with shape (n_u, 1)

    Returns:
        P_h (2d numpy array): A numpy array with shape (n_x, n_x)
        y_h (2d numpy array): A numpy array with shape (n_x, 1)
        p_h (1d numpy array): A numpy array with shape (1,)
    """
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert C.shape == (n_x, n_x)
    assert D.shape == (n_u, n_u)
    assert E.shape == (n_x, n_u)
    assert f.shape == (n_x, 1)
    assert g.shape == (n_u, 1)
    assert h.shape == (1,)
    assert K.shape == (n_u, n_x)
    assert k.shape == (n_u, 1)

    # Apply the formulas given for P_t, y_t, and  p_t
    P = C + (K.T @ D + E) @ K
    y = (2 * K.T @ D.T @ k) + (E @ k) + (K.T @ g) + f
    p = (k.T @ D.T @ k)[0] + (k.T @ g)[0] + h
    return P, y, p


def lqr(A, B, m, Q, R, M, q, r, b, T):
    """
    Compute optimal policies by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} x_t^T Q x_t + u_t^T R u_t + x_t^T M u_t + q^T x_t + r^T u_t
    subject to x_{t+1} = A x_t + B u_t + m, u_t = \pi_t(x_t)

    Let the shape of x_t be (n_x,), the shape of u_t be (n_u,)
    Let optimal \pi*_t(x) = K_t x + k_t
    Parameters:
    A (2d numpy array): A numpy array with shape (n_x, n_x)
    B (2d numpy array): A numpy array with shape (n_x, n_u)
    m (2d numpy array): A numpy array with shape (n_x, 1)
    Q (2d numpy array): A numpy array with shape (n_x, n_x). Q is PD.
    R (2d numpy array): A numpy array with shape (n_u, n_u). R is PD.
    M (2d numpy array): A numpy array with shape (n_x, n_u)
    q (2d numpy array): A numpy array with shape (n_x, 1)
    r (2d numpy array): A numpy array with shape (n_u, 1)
    b (1d numpy array): A numpy array with shape (1,)
    T (int): The number of total steps in finite horizon settings
    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (n_u, n_x), the shape of k_t is (n_u,)
    """
    n_x, n_u = B.shape
    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)
    assert m.shape == (n_x, 1)
    assert Q.shape == (n_x, n_x)
    assert R.shape == (n_u, n_u)
    assert M.shape == (n_x, n_u)
    assert q.shape == (n_x, 1)
    assert r.shape == (n_u, 1)
    assert b.shape == (1,)

    result = []
    # Start with base case at t = T
    Pt, yt, pt = np.zeros_like(Q), np.zeros_like(q), np.zeros_like(b)
    # Iterate over time steps
    for t in range(T):
        # First compute Q-function params, then policy, and then update the V-function params
        Ct, Dt, Et, ft, gt, ht = compute_Q_params(A, B, m, Q, R, M, q, r, b, Pt, yt, pt)
        Kt, kt = compute_policy(A, B, m, Ct, Dt, Et, ft, gt, ht)
        Pt, yt, pt = compute_V_params(A, B, m, Ct, Dt, Et, ft, gt, ht, Kt, kt)
        # Save the policy at this time step
        result.append((Kt, kt[0]))
    # The first iteration above is the last time step of the system, so we revert the result
    return result[::-1]
