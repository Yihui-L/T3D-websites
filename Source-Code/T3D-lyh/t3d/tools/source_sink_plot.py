import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from t3d.tools.profile_plot import plotter


def profile_time_derivative(values, times, accepted, row, order=None):
    """Derivative at a saved accepted profile, not a Newton trial."""
    values, times = np.asarray(values), np.asarray(times)
    accepted = np.asarray(accepted, dtype=int)
    row = int(np.arange(len(times))[row])
    if row not in accepted:
        return np.full_like(values[row], np.nan, dtype=float)
    previous = accepted[times[accepted] < times[row]]
    if not previous.size or order == 0:
        return np.full_like(values[row], np.nan, dtype=float)
    old = previous[-1]
    dt = times[row] - times[old]
    diff = values[row] - values[old]
    if previous.size < 2 or order == 1:
        return diff/dt
    older = previous[-2]
    ratio = dt / (times[old] - times[older])
    return ((1+2*ratio)*diff - ratio**2*(values[old]-values[older])) / ((1+ratio)*dt)


def source_sink_bar_plot(infile,
                         t_idx=-1,
                         plot_dens=True,
                         SI=True):
    data = plotter()
    data.read_output(infile)
    t = int(np.arange(len(data.time))[t_idx])
    if not data.t_diagnostics_valid[t]:
        raise ValueError("Selected row has no synchronized transport diagnostics")

    # Parse time and step data
    t_trin = data.time / data.t_ref
    tc = data.t_step_idx_converged

    fig = plt.figure(figsize=(14,8.5))
    if plot_dens:
        gs = gridspec.GridSpec(6, 3)
        ax02 = fig.add_subplot(gs[1, 2])
        ax12 = fig.add_subplot(gs[2:-1, 2])
    else:
        gs = gridspec.GridSpec(6, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax00 = fig.add_subplot(gs[1, 0])
    ax10 = fig.add_subplot(gs[2:-1, 0])
    ax01 = fig.add_subplot(gs[1, 1])
    ax11 = fig.add_subplot(gs[2:-1, 1])
    ax2 = fig.add_subplot(gs[-1, :])

    # fitler iterations
    t_time = data.time  # plotter converts this from trin to SI
    t_rms = data.t_rms
    T,N = data.n.shape  # (number of steps, number of radial points including boundary)
    step_idx = np.arange(T)
    ax1.plot(step_idx,t_rms, '--')
    ax1.plot(step_idx[tc],t_rms[tc], 'C1', label=f'rms error = {t_rms[t]:.2e}')
    ax1.set_ylabel("trinity rms")
    if t > -1:
        ax1.axvline(t,ls='--',color='k')
    else:
        tm = T + t
        ax1.axvline(tm,ls='--',color='k')

    if np.any(np.isfinite(t_rms) & (t_rms > 0)):
        ax1.set_yscale('log')
    ax1.legend(loc=1, fontsize=8)
    ax1.set_xlabel("t_idx")

    drho = data.drho
    axis = data.rho
    cax = axis[:-1]
    G = data.t_G_fac[t,:-1]
    r_axis = np.arange(N-1)

    # compute div flux
    if SI:
        P_ref = data.P_ref_MWm3
        S_ref = data.Sn_ref_SI20
    else:
        P_ref = 1
        S_ref = 1

    def div_flux(F, G, drho, a_minor):
        dF = np.empty_like(F)
        dF[0] = F[0]
        dF[1:] = np.diff(F)
        return G * dF / drho / a_minor**3

    Fn = data.Gam_SI20_e['tot'][t] / (1 if SI else data.Sn_ref_SI20)
    divGe = div_flux(Fn, G, drho, data.a_minor)

    Fpi = data.Q_MW_i['tot'][t] / (1 if SI else data.P_ref_MWm3)
    divQi = div_flux(Fpi, G, drho, data.a_minor)

    Fpe = data.Q_MW_e['tot'][t] / (1 if SI else data.P_ref_MWm3)
    divQe = div_flux(Fpe, G, drho, data.a_minor)

    def cut(f, norm='power'):
        '''
        selects time t
        cuts last point
        applies unit conversion from trinity power to MW/m3
        '''

        func = f[t][:-1]

        if norm == 'power':
            scale = P_ref
        elif norm == 'particle':
            scale = S_ref

        return func * scale

    labels = {"aux":"Aux", "alpha":"Alpha", "rad":"Rad",
              "heating":"Turb Heat", "coll":"Collisions"}

    def source_bars(sources, divergence):
        return {"div Q": -divergence} | {
            labels.get(label,label):cut(profile)
            for label,profile in sources.items() if label != "tot"
        }

    ion_dict = source_bars(data.Sp_i,divQi)
    elc_dict = source_bars(data.Sp_e,divQe)

    Sp_i = pd.DataFrame(ion_dict, index=cax)
    Sp_e = pd.DataFrame(elc_dict, index=cax)

    Sp_i.plot(kind='bar', stacked=True, ax=ax10)
    Sp_e.plot(kind='bar', stacked=True, ax=ax11)

    ax10.plot(r_axis, cut(data.Sp_i["tot"]) - divQi, 'ko-', label='Total')
    ax11.plot(r_axis, cut(data.Sp_e["tot"]) - divQe, 'ko-', label='Total')

    # Saved snapshots from legacy files do not identify order. Treat the
    # derivative there as unavailable rather than guess a BDF2 stencil.
    order = None if data.t_arrival_order is None else data.t_arrival_order[t]

    def getDfDt(values):
        if order is None or data.t_iter_idx[t] != 0:
            return np.full(len(cax), np.nan)
        return profile_time_derivative(values, t_trin, tc, t, order)[:-1]

    Wi = getDfDt(data.pi) * 1.5 * P_ref
    We = getDfDt(data.pe) * 1.5 * P_ref

    ax10.plot(r_axis,Wi,'x--', color='gray', label=r'$\Delta p_i / \Delta t$')
    ax11.plot(r_axis,We,'x--', color='gray', label=r'$\Delta p_e / \Delta t$')

    ax00.set_title('Ti [keV]')
    ax01.set_title('Te [keV]')
    ax00.set_ylabel('Profiles',fontsize=10)
    if SI:
        ax10.set_ylabel('Heating Power Density ' + r'[MW / m$^3$]')
    else:
        ax10.set_ylabel('Heating Power Density [trin]')

    if plot_dens:
        ne_dict = {}
        ne_dict['div Gam'] = -divGe

        Sn = cut(data.Sn_e.get("tot", data.Sn_e["aux"]), norm='particle')
        ne_dict['Total source'] = Sn
        Sn_e = pd.DataFrame(ne_dict, index=r_axis)
        Sn_e.plot(kind='bar',stacked=True, ax=ax12)
        ax12.plot(r_axis, Sn - divGe, 'ko-', label='Total')

        ax02.set_title(r'Ne [$10^{20} / m^3$]')
        ax12.set_ylabel(r"Particle Source Density [$10^{20} / m^3 /s$]",fontsize=10)
        ax02.plot(axis, data.n[t],'go-', fillstyle='none', label=f"t = {t_time[t]:.2f} s")
        ax02.plot(axis, data.n[0],'k--', label="t = 0 s")
        ax02.set_ylim(bottom=0)
        ax02.legend(fontsize=7)

        Wn = getDfDt(data.n) * S_ref

        ax12.plot(r_axis,Wn,'x--', color='gray', label=r'$\Delta n_e / \Delta t$')
        ax12.legend(loc=3, fontsize=8)
        ax12.set_xlabel('radius centers')
        ax12.set_xticklabels([f'{x:.2f}' for x in cax], rotation=0)

    # plot profiles
    ax00.plot(axis, data.Ti[t],'ro-', fillstyle='none', label=f"t = {t_time[t]:.2f} s")
    ax01.plot(axis, data.Te[t],'bo-', fillstyle='none', label=f"t = {t_time[t]:.2f} s")
    ax00.plot(axis, data.Ti[0],'k--', label="t = 0 s")
    ax01.plot(axis, data.Te[0],'k--', label="t = 0 s")

    for a in [ax00, ax01]:
        a.set_ylim(bottom=0)
        a.legend(fontsize=7)

    for a in [ax10, ax11]:
        a.legend(loc=3, fontsize=8)
        a.set_xlabel('radius centers')
        a.set_xticklabels([f'{x:.2f}' for x in cax], rotation=0)

    # plot y err
    if plot_dens:
        M = 3  # number of profiles (Ti, Te, n)
    else:
        M = 2
    y = np.concatenate([data.Ti, data.Te, data.n], axis=1)
    prior = tc[t_time[tc] < t_time[t]]
    if len(prior):
        window = np.r_[prior[-(M-1):],t]
        yerr = np.std(y[window],axis=0)/y[t]
        ax2.plot(yerr, 'o', label=rf"$\Delta t$ = {t_time[t]-t_time[prior[-1]]:.3g} s")
    else:
        ax2.plot([], [], label="No earlier accepted profile")
    ax2.axvline(N - 0.5, ls='--',color='k')
    ax2.axvline(2*N - 0.5, ls='--',color='k')
    ax2.set_ylabel("y_err")
    ax2.set_xlabel("index y=[Ti, Te, n]")
    ax2.legend(loc=1, fontsize=8)

    ax1.set_title(f"{data.filename} : t_idx = {t}")
    fig.tight_layout()

    if N > 10:
        # plot every other label to prevent rho-axis from being over-crowded
        def modify_axis(ax):
            # Get the current tick positions and labels
            ticks = ax.get_xticks()
            labels = ax.get_xticklabels()

            # Set every other tick and label
            ax.set_xticks(ticks[1::2])  # Set every other tick
            ax.set_xticklabels(labels[1::2])  # Set every other label

        modify_axis(ax10)
        modify_axis(ax11)
        if plot_dens:
            modify_axis(ax12)

    plt.show()


def main():

    import argparse

    description = "This program create a source/sink plot"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("infile",
                        help="Specify T3D output file")

    args = parser.parse_args()

    source_sink_bar_plot(infile=args.infile)


if __name__ == "__main__":
    main()
