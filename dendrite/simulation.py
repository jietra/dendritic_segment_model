import numpy as np
from numba import njit, prange

@njit
def xi_s(t, spike_times, tau_rise, tau_decay):
    """
    Compute bi-exponential synaptic activation trace at time t.

    Parameters
    ----------
    t : float
        Current time (ms).
    spike_times : list of float
        Spike times (ms).
    tau_rise : float
        Rise time constant (ms).
    tau_decay : float
        Decay time constant (ms).

    Returns
    -------
    float
        Synaptic activation value at time t.
    """
    total = 0.0
    for tf in spike_times:
        if t >= tf:
            total += (np.exp(-(t - tf) / tau_decay) - np.exp(-(t - tf) / tau_rise))
    norm = 1 / ((tau_decay / tau_rise) ** (-tau_rise / (tau_decay - tau_rise))
                - (tau_decay / tau_rise) ** (-tau_decay / (tau_decay - tau_rise)))
    return total * norm

@njit(fastmath=True)
def bi_exp_norm(tau_rise, tau_decay):
    """
    Normalization factor for bi-exponential synaptic kernel.
    """
    return 1.0 / ((tau_decay / tau_rise) ** (-tau_rise / (tau_decay - tau_rise))
                  - (tau_decay / tau_rise) ** (-tau_decay / (tau_decay - tau_rise)))

@njit(fastmath=True)
def precompute_xiu(t_half, spike_times, tau_rise, tau_decay):
    """
    Precompute synaptic activation trace for efficiency.

    Parameters
    ----------
    t_half : np.ndarray
        Array of midpoints (n+0.5)*dt.
    spike_times : np.ndarray
        Array of spike times.
    tau_rise : float
        Rise time constant.
    tau_decay : float
        Decay time constant.

    Returns
    -------
    np.ndarray
        Precomputed activation values.
    """
    xiu = np.zeros(t_half.shape[0], dtype=np.float32)
    norm = bi_exp_norm(tau_rise, tau_decay)
    for s in range(spike_times.shape[0]):
        tf = spike_times[s]
        n0 = int(np.ceil((tf / (t_half[1] - t_half[0])) - 0.5))
        if n0 < 0:
            n0 = 0
        for n in range(n0, t_half.shape[0]):
            dtf = t_half[n] - tf
            xiu[n] += (np.exp(-dtf / tau_decay) - np.exp(-dtf / tau_rise))
    for n in range(t_half.shape[0]):
        xiu[n] *= norm
    return xiu.astype(np.float32)

@njit(fastmath=True)
def G(D_L, tau, x, t):
    """
    Green's function for diffusion equation.

    Parameters
    ----------
    D_L : float
        Normalized diffusion coefficient.
    tau : float
        Membrane time constant.
    x : float
        Spatial distance.
    t : float
        Time difference.

    Returns
    -------
    float
        Green's function value.
    """
    return np.exp(-x**2/(4*D_L*t) - t/tau) / np.sqrt(4*np.pi*D_L*t)

@njit(parallel=True, fastmath=True)
def psiu(psi, n, dt, spike_times, tau_rise, tau_decay, D_L, tau, x, x_i, e_i):
    """
    Compute synaptic contribution to membrane potential at step n.

    Parameters
    ----------
    psi : np.ndarray
        Membrane potential array.
    n : int
        Current time index.
    dt : float
        Time step.
    spike_times : list of float
        Spike times.
    tau_rise : float
        Rise time constant.
    tau_decay : float
        Decay time constant.
    D_L : float
        Normalized diffusion coefficient.
    tau : float
        Membrane time constant.
    x : np.ndarray
        Spatial discretization points.
    x_i : float
        Synapse position.
    e_i : float
        Normalized reversal potential.

    Returns
    -------
    np.ndarray
        Contribution to psi at time step n+1.
    """
    psiu = np.zeros_like(psi)
    idx_i = np.searchsorted(x, x_i)
    for u in prange(n+1):
        xiu = xi_s((u+0.5)*dt, spike_times, tau_rise, tau_decay)
        psiu[:, u] = dt * xiu * G(D_L, tau, x - x_i, (n+1)*dt - (u+0.5)*dt) * (e_i - psi[idx_i, u])
    return np.sum(psiu, axis=1)
