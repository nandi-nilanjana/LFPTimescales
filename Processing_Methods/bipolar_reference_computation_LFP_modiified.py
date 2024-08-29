"""
This script takes the mne epochs object and calculates the bipolar reference for each channel with
an inter-space difference of 400um.
If s=200um, then we take the nearest neighbours, i.e. for channel 3 we take 2 and 4. If s=100um,
then we take the next-to-nearest neighbours, i.e. for channel 3 we take 1 and 5.
If s=150um, then we need to interpolate beforehand to have an approximation of the signal at every
100um, and then take the next-to-nearest neighbours.
"""

# import packages
import os
import re
import warnings
import mne
import h5py
import numpy as np
#from frites import io
from src.preproc_tools import get_path_and_filenames

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%

def interpolate_channels(lfp, new_spacing, area_name):
    """
    Interpolates the channels of the LFP epochs to have a new spacing between channels.
    :param lfp: mnep epochs object with the LFP data.
    :param new_spacing: new spacing between channels in um.
    :param area_name: string with the area of the probe, either 'M1' or 'PMd'.
    ----
    :return: lfp_new: mne epochs object with the interpolated channels.
             depths_new: list with the depths of the interpolated channels.

    """
    # get all the depths - spacing between channels in mm
    depths_probe = (np.array(lfp.metadata[f"depths_{probe}"][0]) * 1000).astype(int)  # um

    # get the indices of the channels that are in the area
    ch_idx = mne.pick_channels_regexp(lfp.ch_names, area_name + '*')  # select channels in area

    # get the new lfp epochs
    lfp_epochs_new = []
    depths_new = []

    # get the spacing between channels in mm
    spacing = int(lfp.metadata[f"s_{probe}"][0] * 1000)  # in um

    # get the number of channels
    n_channels = len(ch_idx)

    # get epochs belonging to this probe
    lfp_probe = lfp.get_data()[:, ch_idx, :]

    # iterate on the channels
    for channel in range(n_channels - 1):
        # get the number of channels
        idx_1 = channel
        idx_2 = channel + 1
        # get the position of channel 1
        pos_1 = depths_probe[idx_1]
        # get the position of channel 2
        pos_2 = depths_probe[idx_2]

        # get the new positions of channels between 1 and 2
        # we remove the first one because it is the same as channel 1
        new_pos = np.arange(pos_1, pos_2, -new_spacing)[1:]

        # get the lfp in the starting position
        lfp_epochs_new.append(lfp_probe[:, idx_1, :])

        # get the new depths
        depths_new.append(depths_probe[idx_1])

        # get the contribution of each of the channels to the new position, based on distance
        for new_pos_i in new_pos:
            # contribution of the first channel - value(0,1)
            contrib_1 = 1 - (abs(new_pos_i - pos_1) / spacing)

            # contribution of the second channel - value(0,1)
            contrib_2 = 1 - (abs(new_pos_i - pos_2) / spacing)

            # get the values for the new channel by summing contribution of the values of the
            # channels (weighted by their distance)
            new_channel_lfp = (contrib_1 * lfp_probe[:, idx_1, :] +
                               contrib_2 * lfp_probe[:, idx_2, :])

            # append the new channel to the lfp epochs
            lfp_epochs_new.append(new_channel_lfp)

            # append the new depths
            depths_new.append(new_pos_i)

    # get the last channel
    lfp_epochs_new.append(lfp_probe[:, -1, :])
    depths_new.append(depths_probe[-1])

    # create an x-array and store it
    depths_new = np.array(depths_new)
    lfp_epochs_new = np.array(lfp_epochs_new)

    return lfp_epochs_new, depths_new


def bipolar_derivation(lfp, depths, bipolar_spacing):
    """
    Computes the bipolar derivation of the LFP epochs.
    :param lfp: array with the LFP epochs, with shape (n_channels, n_trials, n_times).
    :param depths: array with the depths of the channels, with shape (n_channels,).
    :param bipolar_spacing: integer with the spacing between channels in um, for the bipolar
    computation.
    ----
    :return: lfp_bipolar: array with the bipolar derivation of the LFP epochs, with shape
    (n_channels_bipolar, n_trials, n_times).
                depths_bipolar: array with the depths of the bipolar channels, with shape
    (n_channels_bipolar,).
    """
    # get the channel step based on the depths vector and the bipolar spacing
    channel_idx_distance = abs(int(bipolar_spacing / np.mean(np.diff(depths))))

    # get the average between the two channels separated by the bipolar spacing
    lfp_epochs_bipolar = []
    depths_new_bipolar = []
    for channel in range(0, len(lfp) - channel_idx_distance):
        # get the subtraction of the two channels
        lfp_epochs_bipolar.append(lfp[channel, :, :] - lfp[channel + channel_idx_distance, :, :])
        # get the depth of the bipolar channel
        depths_new_bipolar.append(depths[channel] - int(bipolar_spacing / 2))

    return np.array(lfp_epochs_bipolar), np.array(depths_new_bipolar)




if __name__ == "__main__":
    # list all the sessions that we want to preprocess
   
   #Mourad
    #GOOD_LAMINAR_SESSIONS = ['Mo180405001','Mo180405004','Mo180411001','Mo180412002',
                    # 'Mo180418002','Mo180419003','Mo180426004','Mo180503002', 'Mo180523002','Mo180524003', 
                    # 'Mo180525003','Mo180531002','Mo180614002','Mo180614006',
                    # 'Mo180615002','Mo180615005', 'Mo180619002','Mo180620004','Mo180622002',
                    # 'Mo180626003', 'Mo180627003','Mo180629005', 'Mo180703003','Mo180704003', 'Mo180705002',
                    #   'Mo180706002', 'Mo180710002','Mo180711004']
    
    #Tomy
    # GOOD_LAMINAR_SESSIONS = ['t140924003','t140925001','t140926002','t140929003','t140930001','t141001001','t141008001','t141010003','t150122001',
    #                  't150123001','t150128001','t150204001','t150205004','t150212001','t150303002','t150319003','t150327002','t150327003','t150415002','t150416002','t150423002','t150430002','t150520003','t150716001']

    GOOD_LAMINAR_SESSIONS = ['t140930001','t141001001','t141008001','t141010003','t150122001',
                      't150123001','t150128001','t150204001','t150205004','t150212001','t150303002','t150319003',
                      't150327002','t150327003','t150415002','t150416002','t150423002','t150430002','t150520003',
                      't150716001']
    #add later t150218001
    # define the new spacing between channels in um
    BIPOLAR_SPACING = 400  # in um
    
    current_path = os.getcwd()
    if current_path.startswith('/Users'):
        server = 'Volumes' #local VPN
    elif current_path.startswith('/envau'):
        server = 'envau'

    for session in GOOD_LAMINAR_SESSIONS:
        
        path_preprocessed = f'/{server}/work/comco/nandi.n/LFP_timescales/Results/Unipolar_sites/{session}'
        path_output = f'/{server}/work/comco/nandi.n/LFP_timescales/Results/Bipolar_sites/{session}'
        
        if not os.path.exists(path_output) :
            os.mkdir(path_output)
        
        signal_type= 'LFP'

        # lfp_filename = [i for i in os.listdir(path_preprocessed) if
        #                 os.path.isfile(os.path.join(path_preprocessed, i)) and
        #                 f'{session}' in i and 'epo' in i and f'{signal_type}' in i
        #                 and 'bipolar' not in i][0]     
        
        
        if path_preprocessed is None:
            raise ValueError("path_preprocessed is None")
        
        # matching_file = [i for i in os.listdir(path_preprocessed)
        #                  if os.path.isfile(os.path.join(path_preprocessed, i)) and 
        #                  f'{session}' in i and 'epo' in i and f'{signal_type}' in i and 
        #                  'bipolar' not in i
                         
        #                  ]
    
    
        for i in os.listdir(path_preprocessed):
            print (i)
            if f'{session}' in i and 'epo' in i and f'{signal_type}' in i and 'bipolar' not in i:
                lfp_filename = i 
                
                
            
            
            
            
    #             # Check if any files match the criteria
    #     if not matching_file:
    #         raise FileNotFoundError("No files matching the criteria were found.")
    
    # # Get the first matching file
    #     lfp_filename = matching_file[0]


    
        # paths = get_path_and_filenames(session, signal_type='LFP')
        
        paths = { 'path_preprocessed' : path_preprocessed, 
                 'lfp_filename' : lfp_filename,
                  'path_output' : path_output
            }
        
        
        lfp_epochs = mne.read_epochs(os.path.join(paths['path_preprocessed'],
                                                  paths['lfp_filename']),
                                     preload=True)

        # get the fields of the excel with the areas (either one or two)
        areas_session = [key for key in lfp_epochs.metadata.keys() if 'area' in key]

        # get the metadata
        metadata = lfp_epochs.metadata

        # get the number of trials of the session
        n_trials = len(lfp_epochs.events)

        # initialize list with channel names
        ch_names = []

        # initialize list with all lfp epochs
        lfp_epochs_all = []

        for area_probe in areas_session:
            # get the area and probe of that site
            _, probe = re.split('_', area_probe)
            area = metadata[area_probe][0]

            # interpolate the channels to get a spacing of 50um in case the probe has a spacing of
            # 150um
            if lfp_epochs.metadata[f"s_{probe}"][0] == 0.15:
                INTERP_SPACING = 50  # in um

                # lfp_epochs_interp will be of the shape (n_channels, n_trials, n_times)
                lfp_epochs_interp, depths_interp = interpolate_channels(lfp_epochs,
                                                                        new_spacing=INTERP_SPACING,
                                                                        area_name=area)

                # get the lfp epochs and depths jumping 4 channels, i.e. 200um
                lfp_epochs_interp = lfp_epochs_interp[::4, :, :]
                depths_interp = depths_interp[::4]

            else:
                # get the indices of the channels that are in the area
                ch_idx_probe = mne.pick_channels_regexp(lfp_epochs.ch_names,
                                                        area + '*')  # select channels in area

                # lfp_epochs_interp will be of the shape (n_channels, n_trials, n_times)
                lfp_epochs_interp = np.moveaxis(lfp_epochs.get_data()[:, ch_idx_probe, :], 0, 1)
                depths_interp = (np.array(lfp_epochs.metadata[f"depths_{probe}"][0]) *
                                 1000).astype(int)

            # get the bipolar reference
            lfp_bipolar, depths_bipolar = bipolar_derivation(lfp=lfp_epochs_interp,
                                                             depths=depths_interp,
                                                             bipolar_spacing=BIPOLAR_SPACING)

            # find the indexes in which the layers change
            layers = metadata[f"layers_{probe}"][0]            
            change_layers = np.where(np.array(layers[1:]) != np.array(layers[:-1]))[0] + 1
            
            # get the depth of the bipolar channels in mm
            depths_bipolar_mm = np.round(depths_bipolar * 1e-3, 2)  # in mm
            
            if change_layers.size !=0:

                # get the depth corresponding to the change of the layers
                depths_layers = np.array(metadata[f"depths_{probe}"][0])[change_layers]
    
                # find the groups of channels in between the depth borders for the layers
                
                # if we don't change the sign, there is a problem with probes with only one change
                # in layers, because it computes 'smaller' than the cut, and it reverses the numbering
                layer_idx = np.digitize(-depths_bipolar_mm, -depths_layers)  # 0, 1,..3 for each channel


            else: #case where there is no layer change 
                layer_idx = np.zeros(len(depths_bipolar_mm),dtype=int) #since no matter what the unique layer for that channel be , 
                #it will only give one element in unique_layers so index would be 0
                #added dtype = int else it will throw indexing error , eg session t140929003
   
            # get the layers and assign to groups
            # get the layers and assign to groups - this way it always keeps the original order of
            # the layers
            unique_layers = np.array(list(dict.fromkeys(layers)))
            layers_bipolar = unique_layers[layer_idx]

            # need to change - layers depths and channel names from mne_epochs.
            metadata[f"depths_{probe}"] = [depths_bipolar_mm] * n_trials  # in mm
            metadata[f"layers_{probe}"] = [layers_bipolar] * n_trials
            metadata[f"s_{probe}"] = [BIPOLAR_SPACING * 1e-3] * n_trials  # in mm

            # get the channel names for the bipolar channels
            ch_names_bipolar = [f'{area}-bip-{i}' for i in range(1, len(depths_bipolar)+1)]

            # append the bipolar channels to the list of channel names
            ch_names.append(ch_names_bipolar)

            # append the bipolar channels to the list of lfp epochs
            lfp_epochs_all.append(lfp_bipolar)

        # concatenate the bipolar channels from both probes
        lfp_epochs_all = np.concatenate(lfp_epochs_all, axis=0)

        # reshape to have the shape (n_trials, n_channels, n_times)
        lfp_epochs_rshp = np.moveaxis(lfp_epochs_all, 0, 1)

        # create the channel names as a unique list
        ch_names_mne = [channel for element in ch_names for channel in element]

        lfp_info = mne.create_info(ch_names=ch_names_mne, sfreq=lfp_epochs.info['sfreq'],
                                   ch_types='seeg')

        # # set the zero time of the lfp as the negative time before the alignment - in seconds.
        lfp_epochs_mne = mne.EpochsArray(data=lfp_epochs_rshp, info=lfp_info,
                                         events=lfp_epochs.events,
                                         tmin=lfp_epochs.tmin, event_id=lfp_epochs.event_id,
                                         metadata=metadata)

        # save the epochs
        lfp_epochs_mne.save(os.path.join(paths['path_output'],
                                         f'bipolarLFP-{session}-{BIPOLAR_SPACING}um-epo.fif'),
                            overwrite=True)

 #       io.logger.info(f'Finished bipolar derivation for session {session}')
 
        print(f'Finished bipolar derivation for session {session}')
