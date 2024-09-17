"""Functions for simulating competitive inhibition in Wilson-Cowan model."""

# Author: Ryan Thorpe <ryvthorpe@gmail.com>

import numpy as np
from scipy.signal import find_peaks


def sigmoid(x, thresh, steepness):
    return 1 / (1 + np.exp(-steepness * (x - thresh)))


def dxdt_v2(x, w, inj_excite, tau=20.0, tau_2=None, thresh=0.5, thresh_2=None,
            steepness=8.0, steepness_2=None):
    """Wilson-Cowan ODEs w/ distinct activation functions for two subgroups."""
    if tau_2 is None:
        tau_2 = tau
    if thresh_2 is None:
        thresh_2 = thresh
    if steepness_2 is None:
        steepness_2 = steepness

    inputs = (x @ w) + inj_excite
    n_dim = len(x)
    x_ = np.zeros((n_dim,))
    x_[:n_dim // 2] = (-x[:n_dim // 2] + sigmoid(inputs[:n_dim // 2], thresh,
                                                 steepness)) / tau
    x_[n_dim // 2:] = (-x[n_dim // 2:] + sigmoid(inputs[n_dim // 2:], thresh_2,
                                                 steepness_2)) / tau_2
    return x_


def jacobian(x, w, tau, tau_2, thresh, thresh_2, steepness, steepness_2,
             inj_excite=0):
    """Jacobian of Wilson-Cowan ODEs (dxdt_v2)."""
    # NB: here, n_dim represents full rank of system (i.e., total number of
    # neural mass units across layers)
    n_dim = len(x)
    x = np.array(x)  # convert to array if not one already
    g = (x @ w) + inj_excite
    J = np.zeros((n_dim, n_dim))
    for i_idx in range(n_dim // 2):
        # upper layer (units 0-1)
        sig = sigmoid(g[i_idx], thresh, steepness)
        for j_idx in range(n_dim):
            dfdx = w[i_idx, j_idx] * steepness * (1 / sig - 1) * sig ** 2
            if i_idx == j_idx:
                J[i_idx, j_idx] = (-1 + dfdx) / tau
            else:
                J[i_idx, j_idx] = dfdx / tau

    for i_idx in range(n_dim // 2, n_dim):
        # lower layer (units 2-3)
        sig = sigmoid(g[i_idx], thresh_2, steepness_2)
        for j_idx in range(n_dim):
            dfdx = w[i_idx, j_idx] * steepness_2 * (1 / sig - 1) * sig ** 2
            if i_idx == j_idx:
                J[i_idx, j_idx] = (-1 + dfdx) / tau_2
            else:
                J[i_idx, j_idx] = dfdx / tau_2
    return J


def rk4(t, x, dt, dxdt):

    # Calculate slopes
    k1, t_input_current = dxdt(t, x)
    k2, _ = dxdt(t + (dt / 2.), x + (k1 / 2.))
    k3, _ = dxdt(t + (dt / 2.), x + (k2 / 2.))
    k4, _ = dxdt(t + dt, x + k3)

    # Calculate new x and y
    x_next = x + (dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
    t_next = t + dt

    return t_next, x_next, t_input_current


def plot_sim_dev(times, x, ax_1):
    colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    # ax_1.plot(times, np.mean(x, axis=1), color=[0.2, 0.2, 0.2], alpha=1.0)
    # plot time course of unit spike rates
    for unit_idx in range(x.shape[1]):
        color = colors[unit_idx]
        ls = '-'
        if unit_idx >= 2:
            ls = ':'
        x_traj = x[:, unit_idx]
        ax_1.plot(times, x_traj, color=color, ls=ls, alpha=1.0)
    # plot markers at avg event peaks
    x_avg = np.mean(x[:, :2], axis=1)
    peak_idxs, _ = find_peaks(x_avg)
    ax_1.scatter(times[peak_idxs], x_avg[peak_idxs], marker='_', color='k')


def sim_dev(dev, w_ii, w_ij, w_ii_l2, w_ij_l2, w_fb, tau, tau_2,
            thresh, thresh_2, steepness, steepness_2, ipsirepr_inhib=0.75,
            rep_interval=100):
    """Simulate competitive inhibitory network under deviant drive.

    Parameters
    ----------
    dev : float

    w_ii : float
        Recurrent connection weight for diagonal elements (ith -> ith unit)
        of the connectivity matrix. This value should be >=0 since this
        neural mass shouldn't inhibit itself.
    w_ij : float
        Recurrent connection weight for interacting off-diagonal elements
        (ith -> jth unit s.t. i != j) of the connectivity matrix. This value
        should be <=0 to provide competive inhbition.
    fb : float
        Feedback scaling factor for cross-laminar w_ij terms.

    Returns
    -------
    times :
    x :
    inj_excite :
    w :
    """

    # integration params
    dt = 0.1  # ms
    burn_in = 100  # ms
    # sim time: try 600 and fix below for one extra trial after dev
    # tstop = burn_in + 12 * rep_interval  # ms
    tstop = burn_in + 4 * rep_interval  # ms
    times = np.arange(0, tstop + dt, dt)

    # network params
    # number of competing neural mass units per subgroup (i.e., representations
    # per layer)
    n_dim = 2
    # total units: multiply by two subgroups (e.g., layers)
    n_units = n_dim * 2
    x = np.zeros((len(times), n_units))

    # drive params
    baseline = 0.01
    # baseline = np.array([0.01, 0.01, 0.01, 0.01])
    inj_delta = 0.10  # difference in drive magnitude between representations

    # set initial state
    x[0, :] = 0.0

    # repetative injected excitation (exogenous drive): half-period of square
    # wave at 20 ms, lasting 20 ms
    inj_excite = np.zeros_like(x)
    # calculate decendng offset values for each representation (dimension) that
    # are inj_delta apart, zero-centered
    inj_offsets = np.linspace(inj_delta, 0, n_dim)
    inj_offsets -= inj_offsets.mean()

    # define network connectivity weight matrix
    # setup to accomodate a network of arbitrary # of representations
    # (dimensions), but should look something like this for n_dim=2:
    # w = np.array([[w_ii, w_ij, 0, 0],
    #               [w_ij, w_ii, 0, 0],
    #               [w_fb, w_fb / 3, w_ii_l2, w_ij_l2],
    #               [w_fb / 3, w_fb, w_ij_l2, w_ii_l2]])
    w = np.zeros((n_units, n_units))

    # 1st subgroup (layer)
    # diagnal excitatory weights
    w[:n_dim, :n_dim] += w_ii * np.eye(n_dim)
    # off-diagnal inhibitory weights
    w[:n_dim, :n_dim] += w_ij * (np.ones((n_dim, n_dim)) - np.eye(n_dim))

    # 2nd subgroup (layer)
    # diagnal excitatory weights
    w[n_dim:, n_dim:] += w_ii_l2 * np.eye(n_dim)
    # off-diagnal inhibitory weights
    w[n_dim:, n_dim:] += w_ij_l2 * (np.ones((n_dim, n_dim)) -
                                    np.eye(n_dim))

    # inhibition from 2nd subgroup -> 1st subgroup
    # diagnal ipsi-representation inhibitory weights
    contrarepr_inhib = 1 - ipsirepr_inhib  # try 1/2 or 1/3
    w[n_dim:, :n_dim] += (w_fb * ipsirepr_inhib) * np.eye(n_dim)
    # off-diagnal contra-representation inhibitory weights
    w[n_dim:, :n_dim] += ((w_fb * contrarepr_inhib) *
                          (np.ones((n_dim, n_dim)) - np.eye(n_dim)))

    def dxdt_v2_w_injection(t, x,
                            w=w, tau=tau, tau_2=tau_2, thresh=thresh,
                            thresh_2=thresh_2, steepness=steepness,
                            steepness_2=steepness_2, burn_in=burn_in,
                            rep_interval=rep_interval, tstop=tstop,
                            n_units=n_units, n_dim=n_dim,
                            inj_offsets=inj_offsets, baseline=baseline,
                            dev=dev):
        """Complete ODE system including time-dependent injected excitation.

        Note that this is needed to allow the RK4 method to evalute the system
        at times in-between time steps at which the injected current
        hasn't yet been explicitly defined.
        """
        # determine current 'stimulus' repetition onset time
        rep_tstart = burn_in + (np.abs(t - burn_in) // rep_interval
                                * rep_interval)
        # if within the time bounds of afferent drive for the evoked response,
        # apply injected excitation that surpasses baseline drive
        inj_excite = np.full((n_units,), baseline, dtype=float)
        if t >= rep_tstart + 20.0 and t < rep_tstart + 40.0:
            for unit_idx in range(n_dim):
                inj_excite[unit_idx::n_dim] = 0.5 + inj_offsets[unit_idx]

            # on final trial, reduce injected excitation (exogneous drive)
            if rep_tstart == tstop - rep_interval:
            # if rep_tstart >= burn_in + (2 + dev) * rep_interval:  # noqa
            # if rep_tstart >= burn_in + (3 * rep_interval) and rep_tstart < burn_in + (6 * rep_interval):  # noqa
                inj_excite *= (1 + dev)
                # for unit_idx in range(n_dim):
                #     inj_excite[unit_idx::n_dim] = 0.5 - inj_offsets[unit_idx]
            # else:
            #     inj_excite *= (1 - dev)

        return (dxdt_v2(x, w, inj_excite,
                        tau=tau, tau_2=tau_2,
                        thresh=thresh, thresh_2=thresh_2,
                        steepness=steepness, steepness_2=steepness_2),
                inj_excite)

    # forward-Euler
    # for t_idx, time in enumerate(times):
    #     if t_idx > 0:
            # dx = dxdt_v2(x[t_idx - 1, :], w, inj_excite[t_idx - 1, :],
            #              tau=tau, tau_2=tau_2,
            #              thresh=thresh, thresh_2=thresh_2,
            #              steepness=steepness, steepness_2=steepness_2) * dt
            # x[t_idx, :] = x[t_idx - 1, :] + dx

    # Runge-Kutta (RK) 4
    # NB: be sure to use pre-computed time values as time calculated
    # recursively contains too much rounding error
    for t_idx, time in enumerate(times[:-1]):
        _time, x[t_idx + 1, :], inj_excite[t_idx, :] = rk4(time,
                                                           x[t_idx, :].copy(),
                                                           dt,
                                                           dxdt_v2_w_injection)

    return times, x, inj_excite, w
