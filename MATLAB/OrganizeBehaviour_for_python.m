%Modified from Laura's script to only save the LFP and behaviour data 

%%% Organize behaviour for python in script %%%


% This is a script which aims to generate the LFPs in a matrix and plot them
% in an organized order. Use the data from Tomy and the existent functions
% to extract the necessary localizers for plotting the data.

%% Add to path
addpath(genpath('/hpc/comco/kilavik.b/MatlabScripts/Behavior/'));
addpath(genpath('/hpc/comco/kilavik.b/MatlabScripts/Preprocessing/'));
addpath(genpath('/hpc/comco/kilavik.b/TomyCerebus/RecordingSessions/')); %genpath includes also sub-folders
addpath(genpath('/hpc/comco/kilavik.b/MouradCerebus/RecordingSessions/')); %genpath includes also sub-folders

addpath(genpath('/hpc/comco/kilavik.b/MatlabScripts/CommonMatlab/NPMK-5.4.1.0/'));
addpath('/envau/work/comco/kilavik.b/Data_VisuoMotorLaminar/TomyCerebus/MUALFPmat/');
addpath('/envau/work/comco/kilavik.b/Data_VisuoMotorLaminar/MouradCerebus/MUALFPmat/');
addpath('/envau/work/comco/kilavik.b/Data_VisuoMotorLaminar/Lists&Documentation/');
addpath('/hpc/comco/kilavik.b/TomyAlmap/NEOstruct/');
addpath('/envau/work/comco/kilavik.b/Data_VisuoMotorLaminar/TomyAlmap/MUALFPmat/');

addpath('/hpc/comco/lopez.l/ephy_laminar_LFP/Raw_data/');

% this doesn't work maybe bc the space
%addpath('/hpc/comco/kilavik.b/MatlabScripts/CommonMatlab/Plexon Offline SDKs/Matlab OfflineFilesSDK/');

%% Open the data and extract the behaviour and LFPs
% define the session and the probe to use
%fname = 't150327002';
%fname = 'Mo180412002';
%fname = 'Mo180626003'
monkey_name = 'Mourad';
%fnames = ć
%fnames = {'Mo180411001', 'Mo180412002', 'Mo180523002'};
%fnames = {'Mo180712006', 'Mo180626003', 'Mo180627003', 'Mo180704003','Mo180411001', 'Mo180412002', 'Mo180523002'};

fnames = {'Mo18042703'};

probes = [1,2];
  % set where the data matrices are stored
  saveFolder = ['/envau/work/comco/nandi.n/LFP_timescales/RAWData/' monkey_name '/LFPmat'];
  
  %create directory if it does not exist
  direxist = isfolder(saveFolder);
  if ~direxist; mkdir(saveFolder); end
  
        

for i=1:length(fnames)
    fname = fnames{i};
    
    for j=1:length(probes)
        probe = probes(j);

        % get the table to find the elitrials associated to the session
        monkey_info = readtable([monkey_name '_NEWList2020_BetaBands.txt']);

        % find the indexes in the table that correspond to the session we're
        % checking
        selected_session = strcmp(monkey_info.x_Session, fname);
        idxs_session = find(selected_session == 1);
        

        elitrials = str2num(monkey_info(idxs_session(1), :).Elitrials_NEW{1});
        
%         % get the extra elitrials from the laminar inspection
%         monkey_info_laminar = readtable('ephydataset_laminar_info_MAT.txt', 'NumHeaderLines',0);
%         selected_session_laminar = strcmp(monkey_info_laminar.x_SESSION, fname);
%         idxs_session_laminar = find(selected_session_laminar == 1);
%         elitrials_laminar = str2num(monkey_info_laminar(idxs_session_laminar(1), :).ELITRIALS_LAMINAR{1});

        % combine all elitrials
        %elitrials_total = [elitrials elitrials_laminar];
        elitrials_total  = elitrials ; %modified by Nilu since I do not want to include the extra laminar elitrials included by Laura for her betaband analysis
        
        
     % get the eliCorrect and eliError from NEWLIST -- either Tomy or Mourad
        % /hpc/comco/kilavik.b/MatlabScripts/BetaBandAnalysis/

        % load only the behaviour of correct trials
        behaviour = OrganizeBehavior(fname, elitrials_total, 'led'); 
        behaviour = table2struct(behaviour);
        
        
        % elCorrect = []
        % behaviour = OrganizeBehavior(fname, eliCorrect, 'unamb','led','error',eliError)
        
        % load the LFPs
        LFP = SortChanLFP(fname, probe);
        
        %save LFPs and behaviour file 
        
        filePath_bhv = fullfile(saveFolder,['behaviour' '-' fname '.mat']);
        save(filePath_bhv, 'behaviour');
        
        filePath_LFP = fullfile(saveFolder, ['LFP-' fname '-' num2str(probe) '.mat']);
        save(filePath_LFP, 'LFP');
        
        
        %save(append('LFP', '-', fname, '-', string(probe), '.mat'), 'LFP')
        
        % load the MUA
       %MUA = SortChanMUA(fname,probe);

        % Save the two datasets: behaviour and LFP
        % Key step to read in PYTHON!!!!!

        % save behaviour file
%         behaviour = table2struct(behaviour)
%         save(append('behaviour','-',fname,'.mat'), 'behaviour')
% 
%         % save MUA
%         save(append('MUA', '-', fname, '-', string(probe), '.mat'), 'MUA')
        if length(idxs_session)==1
            disp(['Only one probe in session ' fname])
            break
        end
            
            
    end
end
% save MUA
%save(append('MUA','-',fname, probe, '.mat'), 'MUA')


% %% Open the file if there is NOT laminar info
% 
% monkey_name = 'Tomy';
% probe = -1;
% fnames = {'t140527002-07'};
% 
% channels = [2, 3, 4];
% 
% 
% for i=1:length(fnames)
%     fname = fnames{i};
%     
%     for j=1:length(channels)
%         channel = channels(j);
%         
%         % get the table to find the elitrials associated to the session
%         monkey_info = readtable([monkey_name '_NEWList2020_BetaBands.txt']);
% 
%         % find the indexes in the table that correspond to the session we're
%         % checking
%         selected_session = strcmp(monkey_info.x_Session, fname);
%         selected_channel = monkey_info.Channel == channel;
%         
%         % get the index for that session and channel
%         idx_session = find(selected_session == 1 & selected_channel ==1);
%         
%         % get the elitrials
%         elitrials = str2num(monkey_info(idx_session, :).Elitrials_NEW{1});
%         
%         % get the behavior for that session - w the elitrials for that
%         % probe and the LFP signal
%         behaviour = OrganizeBehavior(fname, elitrials, 'led'); 
%         LFP = SortChanLFP(fname, probe, channel);    
%         
%         save(append('LFP', '-', fname, '-', string(channel), '.mat'), 'LFP')
%         
%         % save behaviour file
%         behaviour = table2struct(behaviour)
%         save(append('behaviour','-',fname, '-', string(channel), '.mat'), 'behaviour')
%         
%     end    
% end
