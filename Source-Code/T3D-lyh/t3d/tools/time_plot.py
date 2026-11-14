import os
import numpy as np
import matplotlib.pyplot as plt

from t3d.tools.profile_plot import plotter


def time_trace(infile,
               skip_Newton_iterations=True):

    # load data
    data = plotter()
    data.read_output(infile)
    arr = data.t_step_idx_converged

    def plotTrace(profile, ax, label=''):

        profile = np.array(profile)
        tax = np.arange(len(profile))
        if not skip_Newton_iterations:
            ax.plot(profile, ls='--')
        ax.plot(tax[arr], profile[arr], label=label)

    P_aux_t = data.Sp_aux_int_MW_tot
    S_tot_t = np.sum(data.Sn_cumint_SI20_i['tot'] + data.Sn_cumint_SI20_e['tot'], axis=-1) * 1e20

    W_tot_t = data.Wtot_MJ
    N_tot_t = data.N_tot_SI20

    tauE_t = W_tot_t / P_aux_t
    tauP_t = N_tot_t / S_tot_t

    P_alpha_t = np.array(data.Palpha_int_MW)
    Q_power_t = 5.03 * P_alpha_t / P_aux_t

    n0 = data.n[:,0] * 1e20
    Ti0 = data.Ti[:,0]  # keV
    nTtau_t = n0*Ti0*tauE_t
    ntau_t = n0*tauE_t

    t_rms = data.t_rms
    # calculate sudo density limit
    P_ext_total = P_alpha_t + P_aux_t
    # checked that (R,B) includes scaling
    B0 = data.Btor
    R0 = data.R_major
    a0 = data.a_minor

    V0 = (np.pi * a0 * a0) * (2*np.pi*R0)
    n_sudo_t = np.sqrt(P_ext_total * B0 / V0)

    P_rad_t = data.Sp_int_MW_e['rad']

    # calculate beta
    p_avg = W_tot_t*1e6 / V0
    mu0 = 4*np.pi/1e7
    beta_t = p_avg / B0**2 * (2 * mu0)

    fig,axs = plt.subplots(5,2,figsize=(12,8))
    # these are NOT filtered to remove Newton iteration
    # actually these have no time axis at all
    axs[0,0].plot(P_alpha_t, label=r'$P_\alpha$ [MW]')
    axs[0,0].plot(P_aux_t, label=r'$P_\mathrm{aux}$ [MW]')
    axs[0,0].plot(P_ext_total, label=r'$P_\alpha + P_\mathrm{aux}$ [MW]')
    axs[0,0].plot(-P_rad_t, label=r'$-P_\mathrm{rad}$ [MW]')
    plotTrace(t_rms, axs[0,1], label='rms error')
    plotTrace(W_tot_t, axs[1,0], label='W_stored [MJ]')
    axs[1,1].plot(N_tot_t, label='total particles')
    axs[2,0].plot(Q_power_t, label='Q')
    plotTrace(tauE_t, axs[2,1], label='tau_E')
    axs[2,1].plot(tauP_t, label='tau_P')
    axs[3,0].plot(ntau_t, label=r'$n_0 \tau_E [s/m^3]$')
    axs[3,1].plot(nTtau_t, label=r'$n_0 T_{i0} \tau_E [s \cdot keV / m^3]$')
    plotTrace(beta_t, axs[4,0], label=r'$\beta_{avg}$')
    axs[4,1].plot(n_sudo_t, label=r'$n_{sudo}$')
    axs[4,1].plot(n0/1e20, label=r'$n_{0}$')  # core or avg?

    axs[0,0].set_yscale('log')
    axs[0,1].set_yscale('log')
    axs[2,0].set_yscale('log')
    axs[2,1].set_yscale('log')
    axs[3,0].set_yscale('log')
    axs[3,1].set_yscale('log')
    # axs[4,1].set_yscale('log')

    axs[3,0].axhline(1.5e20, ls='--', color='r', label='lawson')
    axs[3,1].axhline(3e21, ls='--', color='r', label='lawson')
    axs[2,0].axhline(1, ls='--', color='r', label='breakeven')

    [a.legend(loc=1) for a in np.ndarray.flatten(axs)]
    axs[-1,0].set_xlabel('integer step index')
    axs[-1,1].set_xlabel('integer step index')

    fig.suptitle(infile)
    plt.tight_layout()

    # if profile_plot:
    #     data.tidx = np.arange(len(data.time))
    #     data.create_color_maps()

    #     fig, axs = plt.subplots(4,3,figsize=(12,9))

    #     t_idx = -1
    #     data.plot_state_profiles(axs[0,0], profile='Ti', t_stop=t_idx)
    #     data.plot_state_profiles(axs[0,1], profile='Te', t_stop=t_idx)
    #     data.plot_state_profiles(axs[0,2], profile='ne', t_stop=t_idx)

    #     data.plot_source_total(axs[1,0], profile='Ti', t=t_idx)
    #     data.plot_source_total(axs[1,1], profile='Te', t=t_idx)
    #     data.plot_source_total(axs[1,2], profile='ne', t=t_idx)

    #     data.plot_sink_terms(axs[2,0], profile='Ti', t=t_idx)
    #     data.plot_sink_terms(axs[2,1], profile='Te', t=t_idx)
    #     data.plot_sink_terms(axs[2,2], profile='ne', t=t_idx)

    #     data.plot_flux(axs[-1,0], profile='Ti', t_stop=t_idx)
    #     data.plot_flux(axs[-1,1], profile='Te', t_stop=t_idx)
    #     data.plot_flux(axs[-1,2], profile='ne', t_stop=t_idx)

    #     fig.suptitle(infile)
    #     plt.tight_layout()


def main():

    import argparse

    description = "This program creates time trace plots of various parameters of interest"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("infiles",
                        nargs="+",
                        help="Specify T3D output file(s)")
    parser.add_argument("-s", "--save",
                        help="Save plots to file rather than interactive display",
                        action="store_true")
    parser.add_argument("-o", "--outdir",
                        help="Specifies an output directory",
                        default="plot-t")
    parser.add_argument("-n", "--skip-newton-iterations",
                        help="Skip Newton iterations",
                        action="store_true")

    args = parser.parse_args()

    for fin in args.infiles:
        time_trace(fin, args.skip_newton_iterations)

        if args.save:
            if not os.path.exists(args.outdir):
                os.mkdir(args.outdir)

            filename = os.path.basename(fin).split('.')[0]
            out = os.path.join(args.outdir, f"plot-t_{filename}.png")
            plt.savefig(out)
            print(f"  saved {out}")
            plt.close()

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
