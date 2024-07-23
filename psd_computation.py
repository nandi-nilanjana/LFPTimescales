"""
Create the psd for each individual laminar site at every channel and each area.
"""

# import packages
import os
import mne
import re
import h5py
from mne.time_frequency import psd_array_welch
import numpy as np
import xarray as xr
from frites import io
import warnings

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


# session
SESSIONS = ['Mo180626003', 'Mo180627003']


def get_mixed_layers(layers_site, depths_site, spacing=400):
    """
    Get the channels which have info from mixed layers. Assuming we are doing a bipolar with spacing
    s, then at a layer change depth_i, we know that depth_i+s/2 and depth_i-s/2 are from different
    layers. This function will give back a vector of the same shape as layers in which the values of
    the mixed channels are marked as 'mixed'.
    :param layers_site: vector with the layers of each channel
    :param depths_site: vector with the depths of each channel
    :param spacing: spacing between the bipolar channels
    ----
    :return:
    mixed_layers: vector with the same shape as layers_site with the values of the mixed channels
    """
    # get the change in layers
    change_layers = (np.where(np.array(layers_site[1:]) != np.array(layers_site[:-1]))[0])
    depth_changes = (np.array(depths_site)[change_layers] + np.array(depths_site)[change_layers+1])/2

    # get the limits of mixed layers
    spacing_mm = spacing/(2*1000)  # in mm and half of the spacing

    # get the mixed layers
    mixed_layers = np.array(layers_site, dtype=object)

    # iterate over the depths and layers
    for i, (layer, depth) in enumerate(zip(layers_site, depths_site)):
        for depth_change in depth_changes:
            if (depth_change - spacing_mm <= depth) & (depth_change + spacing_mm >= depth):
                mixed_layers[i] = 'mixed'
                # in case it already belongs to the first mixed layer, then it should be mixed
                break

    return mixed_layers


def redefine_layers(layers_site):
    """ Redefine the labels of the layers to ignore the split between superficial L23 and deep.
    This way we only have L23, L5 and L6. We know that the 'DEEP' and 'SUP' comes after a '-'.
    :param layers_site: vector with the layers of each channel
    ----
    :return:
        new_layers: vector with the new layers
    """
    new_layers = []
    for layer_i in layers_site:
        new_layers.append(re.split('-', layer_i)[0])

    return new_layers


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
    path = server + '/comco/nandi.n/LFP_timescales/Results/Bipolar_sites/'
    path_output = server + '/comco/nandi.n/LFP_timescales/Results/PSDs/'

    # get the filename of the data
    file_name = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and
                 f'{session}' in i and '-epo.fif' in i and 'LFP' in i and 'bipolar' in i]

    # load the data
    LFP_epochs = mne.read_epochs(os.path.join(path, file_name[0]), preload=False)

    # compute the psd
    n_per_seg = int(LFP_epochs.info['sfreq'])
    psd, freqs = psd_array_welch(LFP_epochs.get_data(), sfreq=LFP_epochs.info['sfreq'],
                                 fmin=0.1, fmax=150, average='median',
                                 n_per_seg=n_per_seg,
                                 n_overlap=int(n_per_seg/2), n_fft=n_per_seg)

    # save the data
    # build an xarray with the power in each site
    power_xr = xr.DataArray(psd,
                            dims=('trials', 'channels', 'freqs'),
                            coords={'trials': range(LFP_epochs.get_data().shape[0]),
                                    'channels': LFP_epochs.ch_names,
                                    'freqs': freqs})

    # check if there are two areas or one
    areas = [LFP_epochs.metadata[area][0] for area in LFP_epochs.metadata.keys()
             if 'area' in area]

    # save the xarray per area (i.e. per site)
    for i_area, area in enumerate(areas):
        # find the masks of channels
        area_mask = [i_ch for i_ch, channel in enumerate(LFP_epochs.ch_names) if area in channel]

        # create the new array
        power_area = power_xr[:, area_mask, :]

        # add the layers and the depths to the xarray. Change the layers to not split L23
        layers = redefine_layers(LFP_epochs.metadata[f'layers_{+1}'][0])
        depths = LFP_epochs.metadata[f'depths_{+1}'][0]

        # get the mixed layers
        new_mix_layers = get_mixed_layers(layers, depths, spacing=400)

        power_area = power_area.assign_coords(layers=('channels', layers),
                                              depths=('channels', depths),
                                              corrected_layers=('channels', new_mix_layers))

        # save the xarray
        power_area.to_netcdf(os.path.join(path_output,
                                          f'{session}-bipolar_PSD-area_{area}-power.nc'))
