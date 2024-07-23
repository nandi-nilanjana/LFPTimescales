""" Create a code that gets the already computed PSD and fits the FOOOF model to it. It
estimates the aperiodic part, including the knee, which will be used to calculate the tau."""

import os
import h5py
import numpy as np
import re
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from fooof import FOOOF
from scipy.signal import savgol_filter


def plot_laminar_psd(psd, area_label, session_label):
    """
    Plot the PSD of the laminar data. PSD is a xarray with all the info needed.
    :param psd: xarray with the PSD data
    :param area_label: str, the area label
    :param session_label: str, the session label
    ----
    :return: fig
    """
    # Specify the number of colors you want
    num_colors = len(psd.channels.values)

    # Get the colormap from the name
    colormap = plt.get_cmap('viridis')

    # Generate a list of colors
    colors = [colormap(i / (num_colors - 1)) for i in range(num_colors)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    for i_ch, channel_name in enumerate(psd.channels.values):
        ax.loglog(psd.freqs.values,
                  psd.mean(dim='trials')[i_ch].values,
                  label=channel_name, color=colors[i_ch])
    ax.legend(loc='best')
    ax.set_xlabel('Frequency (Hz)', weight='bold')
    ax.set_ylabel('Power', weight='bold')
    fig.suptitle(f'{session_label} - {area_label}', weight='bold')
    sns.despine(fig)
    return fig


def compute_and_plot_fooof(psd, area_label, session_label):
    """
    Compute the fooof model based on the single-channel psd. Compute the knee and tau. Plot
    the model fitting and the aperiodic part in two different figures, each subplot one channel.
    :param psd: xarray with the PSD data
    :param area_label: str, the area label
    :param session_label: str, the session label
    ----
    :return: fig, fig2
    """
    # fit the FOOOF model, plot both the fitting and the aperiodic part (w knee and tau)
    fig, ax = plt.subplots(8, 4, figsize=(20, 20))
    fig2, ax2 = plt.subplots(8, 4, figsize=(20, 20))
    # get the fooof method with knee
    fm = FOOOF(peak_threshold=0.1, aperiodic_mode='knee')
    # freq_range = [1, 152]
    all_knees = []
    for i_ch, ch_name in enumerate(psd.channels.values):
        fm.fit(psd.freqs.values,
               psd.mean(dim='trials')[i_ch].values)
        lw1 = 1.0
        fm.plot(plot_aperiodic=True, plt_log=False,
                ax=ax.flatten()[i_ch],
                add_legend=False, aperiodic_kwargs={'lw': lw1, 'zorder': 10},
                peak_kwargs={'lw': lw1}, data_kwargs={'lw': lw1},
                model_kwargs={'lw': lw1})

        knee_value = fm.aperiodic_params_[1]
        all_knees.append(knee_value)
        tau = 1 / (knee_value * 2 * np.pi)

        ax.flatten()[i_ch].axvline(knee_value, color='red', linestyle='--')
        ax.flatten()[i_ch].set_title(f'{ch_name} - knee: {knee_value:.2f} Hz')

        ax2.flatten()[i_ch].axvline(knee_value, color='red', linestyle='--')
        ax2.flatten()[i_ch].loglog(fm.freqs, 10**(fm._ap_fit), label=ch_name)
        ax2.flatten()[i_ch].loglog(fm.freqs, psd.mean(dim='trials')[i_ch].values, color='k')
        ax2.flatten()[i_ch].set_title(f'{ch_name} - tau: {tau:.3f} - knee: {knee_value:.2f} Hz')
        ax2.flatten()[i_ch].set_xlabel('log(Frequency (Hz))', weight='bold')
        ax2.flatten()[i_ch].set_ylabel('Power', weight='bold')

    fig2.suptitle(f'Aperiodic fitting - {session_label} - {area_label}',
                  weight='bold', y=0.99)
    fig2.tight_layout()
    sns.despine(fig2)

    fig.suptitle(f'FOOOF fit with knee - {session_label} - {area_label}',
                 weight='bold', y=0.99)
    fig.tight_layout()
    sns.despine(fig)

    return fig, fig2, all_knees


if __name__ == '__main__':

    # session
    SESSIONS = ['Mo180626003', 'Mo180627003']

    for session in SESSIONS:
        # check where are we running the code
        current_path = os.getcwd()

        if current_path.startswith('C:'):
            server = 'W:'  # local w VPN
        elif current_path.startswith('/home/'):
            server = '/envau/work/'  # niolon
        elif current_path.startswith('/hpc/'):
            server = '/envau/work/'  # niolon

        # set the path
        path = server + '/comco/nandi.n/LFP_timescales/Results/PSDs/'
        path_plots = server + '/comco/nandi.n/LFP_timescales/Results/PSDs/Plots/'
        path_fooof = server + '/comco/nandi.n/LFP_timescales/Results/FOOOF/'

        # get the filename of the data
        filenames = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and
                     f'{session}' in i and '.nc' in i and 'PSD' in i and 'bipolar' in i]

        # iterate in the filenames (i.e. sites)
        for filename_site in filenames:
            # load the data array with the PSD of each site
            PSD_single_trial = xr.load_dataarray(os.path.join(path, filename_site))

            # get the area
            area = re.split('-', PSD_single_trial.channels.values[0])[0]

            # plot the data itself, average across trials, each channel one line.
            fig1 = plot_laminar_psd(PSD_single_trial, area_label=area, session_label=session)
            fig1.savefig(os.path.join(path_plots, f'{session}_{area}_PSD.png'), dpi=300)

            # compute and plot the FOOOF model
            fig_fooof, fig_aperiodic,\
                laminar_knees = compute_and_plot_fooof(PSD_single_trial, area_label=area,
                                                       session_label=session)

            # get the laminar tau
            laminar_tau_ms = (1 / (np.array(laminar_knees) * 2 * np.pi)) * 1000  # in ms

            # store the average psd with all the info about the laminar
            PSD_avg = PSD_single_trial.mean(dim='trials')

            # add the knees to the PSD_avg
            PSD_avg = PSD_avg.assign_coords(laminar_knees=('channels', laminar_knees),
                                            laminar_tau_ms=('channels', laminar_tau_ms))

            # save the PSD_avg
            PSD_avg.to_netcdf(os.path.join(path_fooof, f'{session}_{area}_PSD_avg_fooof.nc'))

            fig_fooof.savefig(os.path.join(path_plots,
                                           f'{session}_{area}_FOOOF_aperiodic_fit.png'), dpi=300)

            fig_aperiodic.savefig(os.path.join(path_plots,
                                               f'{session}_{area}_FOOOF_fit.png'), dpi=300)

            plt.close('all')
