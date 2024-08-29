"""
This script contains basic functions for preprocessing the raw LFPs. Not fancy analysis, just
filtering and organizing functions to sort the information and allow the objects to be built.

"""
# import packages
import re
import os
import numpy as np
import mne
from scipy import signal
from scipy.io import loadmat
import mat73


def concatenate_probes(filename_probe1, filename_probe2, loc_1, loc_2):
    """
    Concatenate LFPs coming from different probes but recorded in the same SESSION.

    :param filename_probe1: name of the file containing LFP in probe 1
    :param filename_probe2: name of the file containing LFP in probe 2
    :param loc_1: location of probe 1
    :param loc_2: location of probe 2
    ----
    :return: channel_names: list including the names of the channels 'area-ch_number'
    :return: concat_lfp: concatenated LFP array with (n_channels both probes, n_times)
    """

    # get the LFPs from mat files
    probe1 = open_matlab_lfp(filename_probe1)
    probe2 = open_matlab_lfp(filename_probe2)
    concat_lfp = np.concatenate((probe1, probe2))  # concatenate in the channel dimension

    # keep track of the number of channels in each probe and label them
    channels_probe1 = [loc_1 + '-' + str(ch) for ch in range(1, probe1.shape[0] + 1)]
    channels_probe2 = [loc_2 + '-' + str(ch) for ch in range(1, probe2.shape[0] + 1)]

    channel_names = np.append(channels_probe1, channels_probe2)  # join together all channel names

    return channel_names, concat_lfp


def open_matlab_lfp(filename):
    """
    Open matlab files containing the output of SortChanLFP in one SESSION and PROBE.

    :param filename: name of the file containing LFP
    ----
    :return: array_lfp: numpy array with LFP in shape (n_channels, n_times)
    """

    # import the matlab file
    mat_lfp = loadmat(filename)
    data_str_name = list(mat_lfp.keys())[-1]  # sub-structure name

    # store the LFPs in an array
    array_lfp = mat_lfp[data_str_name]  # array containing channels x time-points

    return array_lfp


def open_matlab_behaviour(filename):
    """
    Open a matlab behaviour structure from OrganizeBehaviour and save it in a python dictionary. One
    SESSION corresponds to one behavioural file - independently of the number of probes.

    :param filename: name of the matlab structure containing the behavioural information
    ----
    :return: behaviour: dictionary containing the same fields as the behaviour structure from matlab
    """

    # import the matlab file
    mat_behaviour = loadmat(filename)

    # get the sub-structure name
    data_str_name = list(mat_behaviour.keys())[-1]

    # get the fields of the structure and use them as keys for the new dictionary
    fields = []
    for key in mat_behaviour[data_str_name].dtype.fields.keys():
        fields.append(key)

    data = mat_behaviour[data_str_name][0][0]  # get the data inside the structure
    behaviour = {field: data[i_field][0] for i_field, field in enumerate(fields)}

    return behaviour


def open_matlab_analog(filename):
    """
    Open a matlab structure containing eye and hand information. Store it as a python dictionary.

    :param filename: name of the matlab structure containing the analog signals
    ----
    :return: analog_dictionary: python dictionary containing the structure fields
    """
    # import the matlab structure
    mat_structure = mat73.loadmat(filename)

    # get the sub-structure name
    data_str_name = list(mat_structure.keys())[-1]

    # access the structure to get the dictionary
    analog_dictionary = mat_structure[data_str_name]

    return analog_dictionary


def filter_lfp(lfp, f_sampling=1000, fc_hp=None, fc_notch=None, order_hp=6, order_notch=4):
    """
    Apply a high pass and a notch butterworth filters to the continuous recordings of LFP. Remove
    very slow fluctuations - very low frequencies - and powerline noise.

    :param lfp: raw lfp signal coming form SortChanLFP, low-pass filtered at 250Hz
    :param f_sampling: sampling frequency of the LFP
    :param fc_hp:  cut-off frequency of the high-pass filter
    :param fc_notch: cut-off frequency range of the notch filter
    :param order_hp: order of the high-pass butterworth filter
    :param order_notch: order of the notch butterworth filter
    ----
    :return: lfp_filtered: filtered LFP
    """
    # perform a High pass filter - butterworth filter of order order_hp with cut-off frequency fc_hp
    sos = signal.butter(order_hp, fc_hp, 'hp', fs=f_sampling, output='sos')
    lfp_hp = signal.sosfilt(sos, lfp)

    # perform a Notch filter of order order_notch for powerline noise with frequency limits FC_NOTCH
    sos_notch = signal.butter(order_notch, fc_notch, 'bandstop', fs=f_sampling, output='sos')
    lfp_filtered = signal.sosfilt(sos_notch, lfp_hp)

    return lfp_filtered


def cut_in_epochs(lfp, t_markers=None, t_min=500, t_start=None, t_end=None):
    """
    Cut the continuous LFP filtered signal into epochs. By default, one epoch corresponds to one
    full trial, aligned in the SEL cue appearing. For better precision, we should use 'TimLED'
    times, stored in the behaviour dictionary. It is also possible to align in times from the
    analog behaviour dictionary, e.g. hand movement onset.

    :param lfp: continuous filtered LFP signal, array with (n_channels, n_times)
    :param t_markers: list of time markers determining the start of the event to align to.
    :param t_min: time to include before the start of the trial. Default 500ms before touch.
    :param t_start: time marker setting the start of the epoch [int]. If just a window inside the
    trial, this is the half-window size.
    :param t_end: time marker setting the end of the epoch [int]. If just a window inside the trial,
    this is the half-window size.
    ----
    :return: lfp_epochs: a 3D array with the filtered lfp cut in epochs and all trials with the same
    number of time-points (n_trials, n_channels, n_times)
    """
    # start an empty list
    lfp_epochs = []

    for t_marker in t_markers:
        # start t_min (500ms) before the t_start (touch) and aligned to t_marker (SEL) - default!
        t_1 = t_marker + t_start - t_min
        # finish at t_end (target) - default!
        t_2 = t_marker + t_end
        # append into a list
        lfp_epochs.append(lfp[:, t_1: t_2])

    lfp_epochs = np.array(lfp_epochs)  # convert list into np array

    return lfp_epochs


def get_layer_mask(df_layers, n_channels):
    """
    Get the layer mask for each probe using the information in the dataframe. Return a list with
    equal length to the n_channels which contains '-1' if the channel is not in the cortex or the
    string with the layer name otherwise.
    :param df_layers: dataframe entry with laminar information. String containing the layer names,
    separated by commas and the channel number of start and end of each layer.
    :param n_channels:  number of channels of that probe.
    ----
    :return: layers_mask: array with layer names of the same length as n_channels
    """
    layers_loc = re.split(', ', df_layers)  # split individual layers
    layers_mask = ['-1'] * n_channels  # create a list with all -1 of the shape of the output

    for layer in layers_loc:
        layer_name, layer_loc = re.split(':', layer)
        min_ch, max_ch = re.split('-', layer_loc)
        layers_mask[int(min_ch) - 1:int(max_ch)] = [layer_name] * (int(max_ch) - int(min_ch) + 1)

    return np.array(layers_mask)


def get_depth_from_layer(layer_mask, electrode_spacing):
    """
    Get the depth of each contact based on the layer mask. The first contact before the cortex will
    be depth 0 and then it will grow in steps of -s.
    :param layer_mask: array with layer names and -1 for contacts not belonging to the cortex. It is
    of size (n_channels,)
    :param electrode_spacing: electrode spacing [mm]
    ----
    :return: contact_depth: array with the depth of every contact, considering th electrode spacing
    and the 0 in the WM, i.e. last contact which is not in the cortex.
    """
    n_channels = len(layer_mask)  # get the number of channels
    contact_depth = [1] * n_channels  # initialize the array with the depths

    # check if there is any contact in the dura
    if '-1' in layer_mask:
        zero_idx = np.where(layer_mask == '-1')[0].max()  # check last contact on the dura mattter
        contact_depth[zero_idx] = 0

        # start the depths from the dura matter, i.e. DM = 0 and then grow in steps of -s
        depths = np.arange(len(contact_depth[zero_idx::])) * electrode_spacing
        contact_depth[zero_idx::] = -depths
    else:
        # all contacts are in the cortex
        depths = np.arange(len(contact_depth)) * electrode_spacing
        contact_depth = -depths

    return np.array(contact_depth)


def block_from_trial(trial_types):
    """
    Label the blocks of the trials in each session. Every time there's a change in trial type,
    there's a change in block. This function generates an array with the number of the block to
    which each trial belongs to.
    :param trial_types: list or array including the trial type for all trials in one session.
    ----
    :return: block: array of the same length as trial_types which represents the block instead of
    the trial type.

    """
    # define for each trial in which block it is
    idx_ttype_change = np.where(np.diff(trial_types))[0] + 1  # get when the trial type changes
    idx_ttype_change = np.insert(idx_ttype_change, 0, 0)  # add 0
    idx_ttype_change = np.append(idx_ttype_change, len(trial_types))  # add length of trial type

    # get the amount of trials in each block
    trial_per_block = np.diff(idx_ttype_change)

    block = []
    for value, idx in enumerate(trial_per_block):
        block += [value + 1] * idx  # start blocks at 1
    block = np.array(block).T

    return block


def interpolate_bad_channels(lfp, bad_channels, channel_names):
    """
    Take a list of bad channels and interpolate each one of them with the two neighbouring channels
    in the lfp matrix. If the list is empty, return the same array as input. The interpolated
    channel is just the average between ch+1 and ch-1.
    :param lfp: array of the shape (n_trials, n_channels, n_times)
    :param bad_channels: list of strings with the channel names indicating the bad channels, e.g.
    ['PMd-10']
    :param channel_names: array with all channel names to find the index of the bad channel(s)
    ----
    :return: lfp_clean: array of the same shape of lfp (n_trials, n_channels, n_times) with the
    replaced bad channel.
    """
    # define the new lfp as the previous one
    lfp_clean = np.copy(lfp)
    if not bad_channels:
        pass

    else:
        for ch_name in bad_channels:
            # get the channel index from the channel name
            idx = np.where(ch_name == channel_names)[0][0]
            if idx == 0 or idx == len(channel_names) - 1:
                # if the bad channel is the first or last, just ignore it. It will be replaced later
                pass
            else:
                # get the average of the two neighbouring channels
                lfp_ch_avg = np.mean((lfp[:, idx - 1, :], lfp[:, idx + 1, :]), axis=0)
                lfp_clean[:, idx, :] = lfp_ch_avg

    return lfp_clean


def remove_extreme_channels(bad_channels, channel_names, depths, layers):
    """
    Remove the extreme channels from the bad channels list. The extreme channels are the first and
    at the last of the list - considering ONLY the channels that are inside the cortex, i.e. layers
    different from -1. For those channels, we need to artificially set the depth to zero so they
    will be removed from the metadata and the lfp object.
    :param bad_channels: list of strings with the channel names indicating the bad channels, e.g.
    ['PMd-10']
    :param channel_names: array with all channel names to find the index of the bad channel(s). It
    contains only the channels of the probe we are checking.
    :param depths: array with the depth of each channel
    :param layers: array with the layer of each channel. We could modify a bit the code and NOT use
    this parameter!!!
    ----
    :return: new_depths: array with the depth of each channel, with the extreme channels set to 0
    """
    new_depths = np.copy(depths)
    for bad_channel in bad_channels:
        if bad_channel in channel_names:
            idx_bad_channel = np.where(bad_channel == channel_names)[0][0]

            # check if it's the first or last of the array, then set the depth to 0
            if idx_bad_channel == 0 or idx_bad_channel == len(channel_names) - 1:
                new_depths[idx_bad_channel] = 0
                print(f'Channel {bad_channel} will be removed because it is a bad channel and it is'
                      f'the first or last of the probe.')
            else:
                # check if the previous channel is in the cortex - if not, set the depth to 0
                # if new_depths[idx_bad_channel - 1] >= 0 or :
                if layers[idx_bad_channel - 1] == '-1' or layers[idx_bad_channel + 1] == '-1':
                    new_depths[idx_bad_channel] = 0
                    print(f'Channel {bad_channel} will be removed because it is a bad channel and '
                          f'it is not in the cortex - or it is the first or last channel in the '
                          f'cortex.')
        else:
            pass  # it may be in the other probe

    return new_depths


def get_times_from_markers(times, annotations, time_markers):
    """
    Get the times vector from the markers of the session.
    :param times: vector with all the times of the trial.
    :param time_markers: tuple of two strings with the markers to use to get the times.
    :param annotations: mne.Annotations object with the annotations of the session.
    ----
    :return:
    """
    # we only want to select the time on the trials inside the two time markers
    idx_start_marker = np.where(annotations.description == time_markers[0])[0][0]
    idx_end_marker = np.where(annotations.description == time_markers[1])[0][0]

    # find the index in times of the start and end of the trial (the closest one to the marker)
    idx_t_start = np.argmin(abs(times - annotations.onset[idx_start_marker]))
    idx_t_end = np.argmin(abs(times - annotations.onset[idx_end_marker]))

    # get the time vector between the markers
    new_times = times[idx_t_start:idx_t_end]

    return new_times


def get_time_idx_from_markers(times, annotations, time_markers):
    """
    Get the times vector from the markers of the session.
    :param times: vector with all the times of the trial.
    :param time_markers: tuple of two strings with the markers to use to get the times.
    :param annotations: mne.Annotations object with the annotations of the session.
    ----
    :return:
    """
    # we only want to select the time on the trials inside the two time markers
    idx_start_marker = np.where(annotations.description == time_markers[0])[0][0]
    idx_end_marker = np.where(annotations.description == time_markers[1])[0][0]

    # find the index in times of the start and end of the trial (the closest one to the marker)
    idx_t_start = np.argmin(abs(times - annotations.onset[idx_start_marker]))
    idx_t_end = np.argmin(abs(times - annotations.onset[idx_end_marker]))

    return idx_t_start, idx_t_end


def get_path_and_filenames(session, n_patterns=None, signal_type='LFP'):
    """
    Get the path and the filenames of the session.
    :param session: string with the session name, e.g. 'Mo180411001'
    :param n_patterns: int with the number of patterns to use for the NMF. If not specified, it
    will not give that path.
    :param signal_type: string with the signal type, either 'LFP' or 'MUA'.
    ----
    :return:
    """

    # check where are we running the code
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'L:'  # local w VPN
    else:
        # current_path.startswith('/home/') OR current_path.startswith('/hpc/')
        server = '/hpc'  # niolon

    # get the paths
    path_preprocessed = server + '/comco/lopez.l/ephy_laminar_LFP/Results/Preprocessed_data/' \
                               + session + '/'
    path_nmf = server + '/comco/lopez.l/ephy_laminar_LFP/Results/NMF/' + session + '/'
    path_nmf_bipolar = server + '/comco/lopez.l/ephy_laminar_LFP/Results/NMF_bipolar/' + session +\
        '/'
    path_power = server + '/comco/lopez.l/ephy_laminar_LFP/Results/Power/' + session + '/'
    path_figures_nmf = server + '/comco/lopez.l/ephy_laminar_LFP/Results/NMF/pattern_plots'
    path_mua_crackles = server + '/comco/lopez.l/ephy_laminar_LFP/Results/MUA_crackles/' \
                               + session + '/'
    path_mua_crackles_figures = server + '/comco/lopez.l/ephy_laminar_LFP/Results/MUA_crackles' \
                                         '/crackle_plots/'
    path_flip_power = server + '/comco/lopez.l/ephy_laminar_LFP/Results/FLIP/Power_full/' + \
        session + '/'
    path_flip_power_matlab = server + '/comco/lopez.l/ephy_laminar_LFP/Results/FLIP/Power_matlab/' \
                                    + session + '/'

    # if any of the paths do not exist, create it
    for path in [path_nmf, path_power, path_figures_nmf, path_mua_crackles]:
        if not os.path.exists(path):
            os.mkdir(path)

    # get the filenames of the files we need
    annotations_filename = [i for i in os.listdir(path_preprocessed) if
                            os.path.isfile(os.path.join(path_preprocessed, i)) and
                            f'{session}' in i and 'annot' in i and f'{signal_type}' in i][0]

    lfp_filename = [i for i in os.listdir(path_preprocessed) if
                    os.path.isfile(os.path.join(path_preprocessed, i)) and
                    f'{session}' in i and 'epo' in i and f'{signal_type}' in i
                    and 'bipolar' not in i][0]

    power_filename = [i for i in os.listdir(path_power) if
                      os.path.isfile(os.path.join(path_power, i)) and 'freqs' in i and
                      f'{session}' in i and '-tfr.h5' in i and 'gamma' not in i][0]

    mua_crackles_filenames = [i for i in os.listdir(path_mua_crackles) if
                              os.path.isfile(os.path.join(path_mua_crackles, i)) and f'{session}' in
                              i and 'nc' in i]

    power_flip_filename = [i for i in os.listdir(path_flip_power) if
                           os.path.isfile(os.path.join(path_flip_power, i)) and 'freqs' in i and
                           f'{session}' in i and '-tfr.h5' in i][0]

    if n_patterns:
        nmf_filenames = [i for i in os.listdir(path_nmf) if
                         os.path.isfile(os.path.join(path_nmf, i)) and f'{session}' in i and
                         f'k-{n_patterns}' in i]
    else:
        nmf_filenames = None

    # build a dictionary with all the paths and filenames
    paths_and_filenames = {'path_preprocessed': path_preprocessed, 'path_nmf': path_nmf,
                           'path_nmf_bipolar': path_nmf_bipolar,
                           'path_figures_nmf': path_figures_nmf,
                           'path_power': path_power, 'annotations_filename': annotations_filename,
                           'nmf_filenames': nmf_filenames, 'lfp_filename': lfp_filename,
                           'power_filename': power_filename, 'path_mua_crackles': path_mua_crackles,
                           'mua_crackles_filenames': mua_crackles_filenames,
                           'path_mua_crackles_figures': path_mua_crackles_figures,
                           'path_flip_power': path_flip_power,
                           'power_flip_filename': power_flip_filename,
                           'path_flip_power_matlab': path_flip_power_matlab}

    return paths_and_filenames


def get_area_from_coordinates(coordinate_ap, *coordinate_lat):
    """
    Get the area from the coordinates of the probe.
    :param coordinate_ap: integer with the coordinate in the AP axis.
    :param coordinate_lat: integer with the coordinate in the LAT axis.
    ----
    :return: string with the area.
    """
    # get the area
    if coordinate_ap <= -1:
        area = 'M1'
    else:
        area = 'PMd'

    return area


def get_n_times_from_markers(annotations, times, markers):
    """
    Gets the n_times of the NMF from the markers.
    :param annotations: mne.Annotations object with the markers of the trials.
    :param times: array with the times of the power in which the NMF was calculated.
    :param markers: tuple with the markers of the start and end of the trial.
    ----
    :return:
    """
    # get the times of the NMF
    idx_start_marker = np.where(annotations.description == markers[0])[0][0]
    idx_end_marker = np.where(annotations.description == markers[1])[0][0]

    # find the index in times of the start and end of the trial (the closest one to the marker)
    idx_t_start = np.argmin(abs(times - annotations.onset[idx_start_marker]))
    idx_t_end = np.argmin(abs(times - annotations.onset[idx_end_marker]))

    # get the number of time-points of the trial
    n_times = idx_t_end - idx_t_start

    return n_times
