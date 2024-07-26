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
import pandas as pd

#%%subfunctions to use in the main code

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


def compute_plot_foof_layerwise(psd, area_label, session_label):
    
    """
    Compute the fooof model based on averaging the channels first (layerwise). 
    That is one psd array for sup, L5 , L6 or mixed
    Then Compute the knee and tau with fooof
    the model fitting and the aperiodic part in two different figures, each subplot one channel.
    :param psd: xarray with the PSD data
    :param area_label: str, the area label
    :param session_label: str, the session label
    ----
    :return: fig, fig2, the dataframe with knee value for each layer , L23, L5 and L6 for this session
    """
    # fit the FOOOF model, plot both the fitting and the aperiodic part (w knee and tau)
    fig, ax = plt.subplots(1, 3, figsize=(10,3))
    fig2, ax2 = plt.subplots(1, 3, figsize=(10,3))
    
    # get the fooof method with knee
    fm = FOOOF(peak_threshold=0.1, aperiodic_mode='knee')
    # freq_range = [1, 152]
    all_knees = []
    
    #average psd across trials for each channel 
    # Average across trials (axis 0)
    mu_psd_across_trials = psd.mean(dim= 'trials') # Shape is now (channels, frequency)

    
    #layers
    
    #corrected_layer_info 
    layers_data = psd.corrected_layers.values
    unique_layer_data = np.unique(layers_data ) #check the layers present in the data 
    
    psd_layer = {}
    knee_layer ={}
    for i, l in enumerate(unique_layer_data):
        
        if l != 'mixed': #dont save the mixed channels
        
            #find the indices for L23 or L5 or L6 
            idx  = np.where(layers_data == l)
            
            #now average across the channels for each layer using these idx
            psd_layerwise = np.mean(mu_psd_across_trials[idx].values, axis = 0 ) #dimension should be just frequency now
            
            psd_layer[l] = psd_layerwise 
    
    #now computing fooof on averaged psd s for each layer 
    
            fm.fit(psd.freqs.values,psd_layerwise)
            lw1 = 1.0
            fm.plot(plot_aperiodic=True, plt_log=False,
                    ax=ax.flatten()[i],
                    add_legend=False, aperiodic_kwargs={'lw': lw1, 'zorder': 10},
                    peak_kwargs={'lw': lw1}, data_kwargs={'lw': lw1},
                    model_kwargs={'lw': lw1})
             
            knee_value = fm.aperiodic_params_[1]
            knee_layer[l] = (knee_value)
           
            tau = 1 / (knee_value * 2 * np.pi)
            
             
            ax.flatten()[i].axvline(knee_value, color='red', linestyle='--')
            ax.flatten()[i].set_title(f'{l} - knee: {knee_value:.2f} Hz')
             
            ax2.flatten()[i].axvline(knee_value, color='red', linestyle='--')
            ax2.flatten()[i].loglog(fm.freqs, 10**(fm._ap_fit), label=l)
            
            ax2.flatten()[i].loglog(fm.freqs, psd_layerwise, color='k')
            ax2.flatten()[i].set_title(f'{l} - tau: {tau:.3f} - knee: {knee_value:.2f} Hz')
            ax2.flatten()[i].set_xlabel('log(Frequency (Hz))', weight='bold')
            ax2.flatten()[i].set_ylabel('Power', weight='bold')

            fig2.suptitle(f'Aperiodic fitting - {session_label} - {area_label}',
                          weight='bold', y=0.99)
            fig2.tight_layout()
            sns.despine(fig2)
            
            fig.suptitle(f'FOOOF fit with knee - {session_label} - {area_label}',
                         weight='bold', y=0.99)
            fig.tight_layout()
            sns.despine(fig)
                
    return fig, fig2, knee_layer, psd_layer

    
#%% Main function    

if __name__ == '__main__':

    # session
    #SESSIONS = ['Mo180626003', 'Mo180627003']
    
    SESSIONS = ['Mo180328001', 'Mo180712006']
    

    for session in SESSIONS:
        # check where are we running the code
        current_path = os.getcwd()

        if current_path.startswith('/Users'):
            server = '/Volumes/work'  # local w VPN
        elif current_path.startswith('/home/'):
            server = '/envau/work/'  # niolon
        elif current_path.startswith('/envau'):
            server = '/envau/work/'  # niolon

        # set the path
        path = server + f'/comco/nandi.n/LFP_timescales/Results/PSDs/{session}'
        path_plots = server + f'/comco/nandi.n/LFP_timescales/Results/Plots/{session}'
        path_fooof = server + f'/comco/nandi.n/LFP_timescales/Results/FOOOF/{session}'


        if not os.path.exists(path_plots):
            os.mkdir(path_plots)
            
        if not os.path.exists(path_fooof):
            os.mkdir(path_fooof)
            
        # # get the filename of the data
        # filenames = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and
        #              f'{session}' in i and '.nc' in i and 'PSD' in i and 'bipolar' in i]
        
        filenames = []
        for i in os.listdir(path):
            if (os.path.isfile(os.path.join(path,i)) and f'{session}' in i and '.nc' in i and 
            'PSD' in i and 'bipolar' in i):
                filenames.append(i)
                
                

        # iterate in the filenames (i.e. sites)
        data_list = []
        for filename_site in filenames:
            # load the data array with the PSD of each site
            PSD_single_trial = xr.load_dataarray(os.path.join(path, filename_site))

            # get the area
            area = re.split('-', PSD_single_trial.channels.values[0])[0]

            # plot the data itself, average across trials, each channel one line.
            fig1 = plot_laminar_psd(PSD_single_trial, area_label=area, session_label=session)
            fig1.savefig(os.path.join(path_plots, f'{session}_{area}_PSD.png'), dpi=300)

            # # compute and plot the FOOOF model
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
            
            #compute and plot the fooof model after averaging across trials and then channels belonging to the same layer 
            
            fig_fooof_layerwise, fig_aperiodic_layerwise, knee_layer, psd_layer = compute_plot_foof_layerwise(PSD_single_trial, 
                                                    area_label=area, session_label=session)

            
            fig_fooof_layerwise.savefig(os.path.join(path_plots,
                                            f'{session}_{area}_FOOOF_aperiodic_Layerwise_fit.png'), dpi=300)

            fig_aperiodic_layerwise.savefig(os.path.join(path_plots,
                                                f'{session}_{area}_FOOOF_Layerwise_fit.png'), dpi=300)
            
            
            
            plt.close('all')
            
            #save as the list for dataframe 
            # Collect data for the dataframe
            
            for layer, psd in psd_layer.items():
                knee = knee_layer[layer]
                tau =( 1 / (knee * 2 * np.pi)) *1000 #ms
                data_list.append({
                    'psd': psd,
                    'knee': knee,
                    'tau': tau,
                    'layer': layer,
                    'area': area,
                    'session': session
                })
                # Create a dataframe
                
        df = pd.DataFrame(data_list)

        # Save the dataframe as a pickle file
        df.to_pickle(server + f'/comco/nandi.n/LFP_timescales/Results/FOOOF/{session}/layerwise_fooof_results.pkl')
        print (f'Completed session {session}')
        
        
        


            
      
