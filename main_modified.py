"""
This scrip generates two mne objects, one with continuous electrophysiological signal (LFPs or MUA)
and the other one with annotations based on the matrices obtained from MATLAB. For LFP, it applies a
high-pass filter of 1Hz and a notch filter at 50Hz, for the power line interference. For MUA, it
does not include any filter. The data is cut in epochs depending on the alignment marker - either
outputing the full trial or a time window around the marker. The bad channels are interpolated using
the neighbours.
"""

# import packages
import os
import warnings
import ast
import h5py
import pickle
import numpy as np
import pandas as pd
import mne
#from frites import io
from src.preproc_tools import open_matlab_lfp, concatenate_probes, open_matlab_behaviour,\
    open_matlab_analog, filter_lfp, cut_in_epochs, get_layer_mask, block_from_trial, \
    get_depth_from_layer, interpolate_bad_channels, remove_extreme_channels
# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%




def preprocessing_pipeline(session, monkey, alignment='Sel', signal_type='LFP', complete_trial=True):
    """
    Build the mne epochs object and annotations with the correct trials, for one individual session.
    Use the .mat files to check the behaviour and the LFPs of each of the probes. Get the data in
    (n_trials, n_channels, n_times) with all the metadata associated, e.g. depths, electrode
    spacing, areas.
    :param session: string with the name of the session, e.g. 't150327002'
    :param alignment: string naming the marker inside the trial used for alignment. The options are
    'Sel', 'SC1', 'SC2', 'SC3', 'Go' or 'MVT'. Default is 'Sel'.
    :param signal_type: string with the type of signal to be loaded. The options are 'LFP' or 'MUA'.
    :param complete_trial: bool. If True, the epoch will contain the full trial. If False, just a
    time_window around the alignment marker will be kept.
    ----
    :return: saved 2 mne objects, one with annotations and the other with epochs: '-annot.fif',
    '-epo.fif'. The name includes the session, the alignment and the part of the trial - either full
    or t_window.
    """
    if complete_trial:
        print('Including full trial time.')

    # # check where are we running the code
    # current_path = os.getcwd()

    # if current_path.startswith('/Volumes'):
    #     server = 'L:'  # local w VPN
    # else:
    #     server = '/hpc'  # niolon

    path =  f'/Volumes/work/comco/nandi.n/LFP_timescales/RAWData/{monkey}/LFPmat'
    path_analog = '/Volumes/hpc/comco/kilavik.b/MatlabScripts/Behavior/Results/HandEyeMovements/Data'
   
    # path_output = server + '/comco/lopez.l/ephy_laminar_LFP/Results/Preprocessed_data/' + \
    #     session + '/'
    # path_output = server + '/comco/lopez.l/ephy_laminar_LFP/Results/FLIP/LFP_epochs_cut/' + \
    #     session + '/'
    
    path_output = f'/Volumes/work/comco/nandi.n/LFP_timescales/Results/Unipolar_sites/{session}'

    path_doc = '/Volumes/work/comco/nandi.n/LFP_timescales/docs' #path for the excel file with ephydataset
    # check if the preprocessing folder already exists - else create it
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # Check that the signal type is either 'LFP' or 'MUA'
    while signal_type not in ['LFP', 'MUA']:
        signal_type = input('Select the signal type: LFP or MUA.')

    # get the file names
    file_lfp_probe1 = os.path.join(path, f"{signal_type}-" + session + "-1.mat")
    file_behaviour = os.path.join(path, "behaviour-" + session + ".mat")
    file_electrodes_info = os.path.join(path_doc, "ephydataset_info_modified.xlsx")
    file_behaviour_analog = os.path.join(path_analog, session + "_HandEyeData.mat")

    # 0) ELECTRODES' INFO FILE. Get the information of the session
    df_electrodes = pd.read_excel(file_electrodes_info, sheet_name=0)  # read the excel with pandas
    df_electrodes.set_index('SESSION', inplace=True, drop=True)  # get session as the search index

    area_1 = df_electrodes.loc[session]['PROBE_1']  # get the brain area of  the first probe

    # 0.1) Get the bad channels from the excel file as long as the signal is 'LFP'
    bad_channels = []
    if signal_type == 'LFP':
        for i in (1, 2):
            if df_electrodes.loc[session][f'BAD_CHANNELS_{i}'] != '-':
                # check if the bad channels are a string (list of channels) or an int (one channel)
                if type(df_electrodes.loc[session][f'BAD_CHANNELS_{i}']) == str:
                    channels_number = ast.literal_eval(df_electrodes.loc[session][f'BAD_CHANNELS'
                                                                                  f'_{i}'])
                else:
                    # if it is one channel, make it a list
                    channels_number = [df_electrodes.loc[session][f'BAD_CHANNELS_{i}']]
                # add the bad channels to the list and attach the area of the probe
                for channel in channels_number:
                    bad_channels.append(df_electrodes.loc[session][f'PROBE_{i}'] + '-' +
                                        str(channel))
    else:
        bad_channels = None

    # print the details of the preprocessing
    print(f'Session {session}, alignment on {alignment}. Bad channels are {bad_channels}. \n')

    # 1) BEHAVIOUR. Get the behavioural information of the session coming from the matlab structure.
    # Build a dictionary with all the available fields
    behaviour = open_matlab_behaviour(file_behaviour)

    # 2) ANALOG SIGNAL. Get the hand and eye movement given by the external devices, as well as
    # their precise timing.
    behaviour_analog = open_matlab_analog(file_behaviour_analog)

    # If eliCorrect was used in OrganizeBehaviour [matlab] the number of trials between behaviour
    # and behaviour_analog will not match. For this we need a 'good_trials_mask'
    # idx of the good correct trials starting in 0
    good_trials_mask = behaviour['OrigCorrTrials'] - 1 

    # 3) LFP. Include the LFP coming from SortChanLFP matlab script. Check if there are two probes
    # for the same session or just one. If two probes, concatenate and name the channels
    # accordingly. One recording session will correspond to one LFP object.

    if df_electrodes.loc[session]['PROBE_2'] != '-':
        file_lfp_probe2 = os.path.join(path, f"{signal_type}-" + session + "-2.mat")
        area_2 = df_electrodes.loc[session]['PROBE_2']  # get the brain area of the first probe

        channel_names, all_lfps = concatenate_probes(filename_probe1=file_lfp_probe1,
                                                     filename_probe2=file_lfp_probe2,
                                                     loc_1=area_1, loc_2=area_2)
    else:
        all_lfps = open_matlab_lfp(file_lfp_probe1)
        channel_names = np.array([area_1 + '-' + str(ch) for ch in range(1, all_lfps.shape[0] + 1)])
        area_2 = []

    # 4) FILTERING. Basic filtering of the continuous LFP before cutting it into epochs. If the
    # signal is MUA, skip this step.
    if signal_type == 'LFP':
        fs_lfp = 1000  # sampling frequency of the LFP signal in Hz
        fc_hp = 1  # cut-off frequency of the high pass filter in Hz
        fc_notch = [49.5, 50.5]  # band of frequencies to be attenuated in the notch filter in Hz
        lfp_filtered = filter_lfp(all_lfps, f_sampling=fs_lfp, fc_hp=fc_hp, fc_notch=fc_notch)

    else:
        fs_lfp = 1000  # sampling frequency of the MUA signal in Hz
        lfp_filtered = all_lfps

    # 5) EPOCHS. Cut the continuous LFP in discrete epochs based on trial markers.
    # User should select the alignment marker. It is possible to align to any of the visual cues, or
    # to hand movement onset.
    alignments = ['Sel', 'SC1', 'SC2', 'SC3', 'Go', 'MVT']

    # average timing for the events: [touch SEL(0) SC1 SC2 SC3 GO MVT Target Rew]
    avg_event_times = behaviour['RelativeAverageTimes']

    # get the timing of the LED whenever it is possible. Not all sessions have LED argument.
    if 'SelTimLed' in behaviour:
        marker = 'TimLed'
    else:
        marker = 'Tim'
    event_markers = behaviour['Sel' + marker]  # pre-define the event markers as aligning on 'SEL'

    # ask until a valid alignment
    while alignment not in alignments:
        alignment = input('Select the alignment marker: Sel, SC1, SC2, SC3, Go or MVT.')

    if alignment == 'MVT':
        event_markers = behaviour_analog['HandMovementOnset'][good_trials_mask].astype(int)
    elif alignment == 'Sel':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC1':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC2':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC3':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'Go':
        event_markers = behaviour[alignment + marker]

    # decide if the epoch contains the full trial. If not full trial, determine the window around
    # the event marker
    if complete_trial:
        t_start = avg_event_times[0]  # negative touch time wrt SEL(0)
        t_end = avg_event_times[7]  # positive time when subjects reach the target wrt SEL(0)
        t_min = 500  # ms - default!

        event_labels = ['start', 'touch', 'SEL', 'SC1', 'SC2', 'SC3', 'GO', 'MVT']  # event labels
        avg_event_times = np.insert(avg_event_times, 0, t_start-500)  # average event times
        event_duration = np.diff(avg_event_times[:9])/1000  # set the event duration [s]
        event_duration[2:6] = .3  # use the cue appearance as the event duration (SEL and 3 SCs) [s]

        avg_event_times = avg_event_times[0:8]/1000  # get only the ones matching the labels [s]
        trial_part = 'full'

    else:
        t_min = 0
        t_window = int(input('Window size [in ms] around the event marker:'))
        t_start = -int(t_window/2)
        t_end = int(t_window/2)
        # build the event labels
        event_labels = [f'{alignment} - {t_end}', alignment, f'{alignment} + {t_end}']
        event_duration = [t_window/(2*1000)] * 3  # event duration in seconds
        avg_event_times = [t_start/1000, 0, t_end/1000]  # average event times in seconds
        trial_part = str(t_window) + 'ms'

    lfp_epochs = cut_in_epochs(lfp_filtered, t_markers=event_markers, t_start=t_start, t_end=t_end,
                               t_min=t_min)

    # 6) MNE OBJECT. Build an mne object with the epochs array.
    n_trials, _, _ = lfp_epochs.shape

    # Define the metadata to be included in the MNE object.
    # original trial number (before removing bad trials, both elitrials and other bad trials)
    trial_number = behaviour['OrigCorrTrials']

    # get the movement direction, w/ %10 we get the last digit of the code
    mvt_direction = np.array([direction % 10 for direction in behaviour['MVTdir']]).T

    # trial type: 1, 2, 3 (Blue, Green or Pink) and block to which trials belong
    trial_type = behaviour['ttype']
    trial_block = block_from_trial(trial_type)

    # trial events contains the alignment time of each trial - on the full session time - the
    # movement direction, and the trial type.
    trial_events = np.stack((event_markers, mvt_direction, trial_type)).transpose()
    # dictionary linking the trial_type number to the label of the trial
    trial_events_id = dict(zip(np.array(['blue', 'green', 'pink']), np.unique(trial_type)))

    # layer information and depth information for both probes - if exists
    layers_probe_1 = get_layer_mask(df_layers=df_electrodes.loc[session].LAYERS_1,
                                    n_channels=df_electrodes.loc[session].CHANNELS_1)
    depths_probe_1 = get_depth_from_layer(layer_mask=layers_probe_1,
                                          electrode_spacing=df_electrodes.loc[session].S1)
    electrode_spacing_1 = df_electrodes.loc[session].S1

    # remove the channels that cannot be interpolated - bc they are the first or last on the
    # array, meaning the first in the brain or the first before WM.
    if bad_channels:
        depths_probe_1 = remove_extreme_channels(bad_channels=bad_channels, depths=depths_probe_1,
                                                 channel_names=channel_names[0:len(depths_probe_1)],
                                                 layers=layers_probe_1)

    if df_electrodes.loc[session]['PROBE_2'] != '-':
        layers_probe_2 = get_layer_mask(df_layers=df_electrodes.loc[session].LAYERS_2,
                                        n_channels=df_electrodes.loc[session].CHANNELS_2)
        depths_probe_2 = get_depth_from_layer(layer_mask=layers_probe_2,
                                              electrode_spacing=df_electrodes.loc[session].S2)

        if bad_channels:
            # remove the depths of bad channels in the borders of the cortex
            depths_probe_2 = remove_extreme_channels(bad_channels=bad_channels,
                                                     depths=depths_probe_2,
                                                     channel_names=channel_names[len(depths_probe_1):],
                                                     layers=layers_probe_2)

        electrode_spacing_2 = df_electrodes.loc[session].S2

        # build the metadata dataframe - include only the depths and layers belonging to the cortex
        metadata = {'t_number': trial_number, 't_type': trial_type, 'MVT_dir': mvt_direction,
                    'SC1_dir': behaviour['dirSC1'], 'SC2_dir': behaviour['dirSC2'],
                    'SC3_dir': behaviour['dirSC3'], 'block': trial_block,
                    'layers_1': [layers_probe_1[depths_probe_1 < 0]]*n_trials,
                    'layers_2': [layers_probe_2[depths_probe_2 < 0]]*n_trials,
                    'depths_1': [depths_probe_1[depths_probe_1 < 0]]*n_trials,
                    'depths_2': [depths_probe_2[depths_probe_2 < 0]]*n_trials,
                    'area_1': area_1, 'area_2': area_2, 's_1': electrode_spacing_1,
                    's_2': electrode_spacing_2, 'bb_dominance_1': df_electrodes.loc[session].BB_1,
                    'bb_dominance_2': df_electrodes.loc[session].BB_2,
                    'AP_1': df_electrodes.loc[session].AP_1,
                    'AP_2': df_electrodes.loc[session].AP_2,
                    'LAT_1': df_electrodes.loc[session].LAT_1,
                    'LAT_2': df_electrodes.loc[session].LAT_2}
    else:
        depths_probe_2 = []

        # build the metadata dataframe - include only the depths and layers belonging to the cortex
        metadata = {'t_number': trial_number, 't_type': trial_type, 'MVT_dir': mvt_direction,
                    'SC1_dir': behaviour['dirSC1'], 'SC2_dir': behaviour['dirSC2'],
                    'SC3_dir': behaviour['dirSC3'], 'block': trial_block,
                    'layers_1': [layers_probe_1[depths_probe_1 < 0]]*n_trials,
                    'depths_1': [depths_probe_1[depths_probe_1 < 0]]*n_trials,
                    'area_1': area_1, 's_1': electrode_spacing_1,
                    'bb_dominance_1': df_electrodes.loc[session].BB_1,
                    'AP_1': df_electrodes.loc[session].AP_1,
                    'LAT_1': df_electrodes.loc[session].LAT_1}

    all_depths = np.concatenate((depths_probe_1, depths_probe_2))  # concat depths of both probes

    df_metadata = pd.DataFrame(metadata)

    # interpolate the bad channels w/ the neighbours
    lfp_epochs_clean = interpolate_bad_channels(lfp_epochs, bad_channels, channel_names)

    # create the three MNE objects: Annotations, info and EpochsArray
    lfp_info = mne.create_info(ch_names=list(channel_names), sfreq=fs_lfp, ch_types='seeg')
    annotations = mne.Annotations(onset=avg_event_times, duration=event_duration,
                                  description=event_labels)

    # set the zero time of the lfp as the negative time before the alignment - in seconds.
    lfp_epochs_mne = mne.EpochsArray(data=lfp_epochs_clean, info=lfp_info, events=trial_events,
                                     tmin=avg_event_times[0], event_id=trial_events_id,
                                     metadata=df_metadata)

    # drop the outside channels - the ones that don't belong to the cortex from mne object
    lfp_epochs_mne.drop_channels(np.array(channel_names)[all_depths >= 0])

    # 7) SAVE THE STRUCTURES
    annotations.save(os.path.join(path_output,
                                  f'{signal_type}-{session}-{alignment}-{trial_part}-annot.fif'),
                     overwrite=True)
    lfp_epochs_mne.save(os.path.join(path_output,
                                     f'{signal_type}-{session}-{alignment}-{trial_part}-epo.fif'),
                        overwrite=True)

    with open(os.path.join(path_output, f'{signal_type}-{session}-metadata.pkl'), 'wb') as file:
        pickle.dump(df_metadata, file)




if __name__ == "__main__":
    # list all the sessions that we want to preprocess
    SESSIONS = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                'Mo180704003', 'Mo180523002', 'Mo180705002', 'Mo180711004', 'Mo180712006',
                't150303002', 't150319003', 't150423002', 't150430002', 't150327002', 't150320002']

    GOOD_LAMINAR_SESSIONS = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003',
                             'Mo180523002', 'Mo180704003', 't150320002', 't150327002',
                             'Mo180712006', 'Mo180711004']

   # MORE_LAMINAR = ['Mo180619002', 'Mo180705002']
    MORE_LAMINAR = ['Mo180328001', 'Mo180712006']
    
    monkey = 'Mourad'

    for i_session in MORE_LAMINAR:
        ALIGNMENT = 'Sel'
        COMPLETE_TRIAL = True
        preprocessing_pipeline(session=i_session, monkey=monkey,alignment=ALIGNMENT, signal_type='LFP',
                               complete_trial=COMPLETE_TRIAL)
        
        print(f'Session {i_session} Completed!')
       # io.logger.info(f'Session {i_session} completed!')
