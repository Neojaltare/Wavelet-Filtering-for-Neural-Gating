


%% Folder and Filename
clear; clc;
filename = 'GFAS_016_baseline_20220318_155010.mff';
% filename = 'GFAR_013_baseline_20211221_105453.mff';

addpath('Insert path to EEGLAB')
folder = 'Folder where data is stored';
mkdir('Folder where data is stored/Processed_Data')
writefolder = 'Folder where data is stored/Processed_Data/';
bad_occ = readtable('Path to/Pauline_GFAR_bad_occlusions.xlsx');
bad_trial = readtable('Path to/Pauline_GFAS_bad_trials.xlsx');


%% Initialize table to store preproc data
preprocvars = {'Dataset','Numbadchan','Chanremoved','Comps', 'TotalEp','Bad_Occ','Epochsremoved', 'FinalEpochs'};
d = "datasetname";
nchan = 1;
chanrem = {'test'};
co = {'test'};
ep = 1;
epr = 1;
fep = 1;
badocc = 0;
Preprocinfo  = table(d,nchan,chanrem,co,ep,badocc,epr,fep, 'VariableNames', preprocvars);
index = 1;

%% now loop over each folder

% Initialize EEGlab
eeglab nogui
EEG = pop_mffimport({strcat(folder,filename)},{'code'},0,0);
EEG = eeg_checkset( EEG );
tempEEG = EEG;
chanlocs = EEG.chanlocs;
channames = {EEG.chanlocs(:).labels};


% Create the 1Hz filtered dataset - Filter, re-reference, Bad Channels,
% Interpolate, ICA - For the 1Hz filtered dataset
tempEEG = pop_eegfiltnew(tempEEG, 'locutoff',1,'hicutoff',40,'plotfreqz',0);
tempEEG = eeg_checkset( tempEEG );
tempEEG = pop_reref( tempEEG, []);
tempEEG = pop_clean_rawdata(tempEEG, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
tempEEG = eeg_checkset( tempEEG );
tempEEG = pop_interp(tempEEG, chanlocs, 'spherical');
tempEEG = eeg_checkset( tempEEG );
tempEEG = pop_reref( tempEEG, []);
% Epoch the data
start = -2.5;
stop = 2.5;
if contains(filename,'GFAR')
    tempEEG = pop_epoch( tempEEG, {  'DIN8'  }, [start stop], 'newname', 'Epoched', 'epochinfo', 'yes');
    tempEEG = eeg_checkset( tempEEG );
elseif contains(filename,'GFAS')
    tempEEG = pop_epoch( tempEEG, {  'DIN4'  }, [start stop], 'newname', 'Epoched', 'epochinfo', 'yes');
    tempEEG = eeg_checkset( tempEEG );
end
pop_eegplot( tempEEG, 1, 1, 1); % script continues here but needs to stop before running the ICA
disp('Press a key to continue!' ) % Press a key here. ...
pause;
tempEEG = pop_runica(tempEEG, 'icatype', 'runica', 'extended',1,'rndreset','yes','PCA',30,'interrupt','on');
tempEEG = eeg_checkset( tempEEG );


% Initial processing for the original data
% pop_eegplot( EEG, 1, 1, 1); 
% disp('Press a key to continue!' )
% pause;
EEG = pop_eegfiltnew(EEG, 'locutoff',.3,'hicutoff',30,'plotfreqz',0);
EEG = pop_reref( EEG, []);
numchan = size(EEG.data,1);
EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
EEG = eeg_checkset( EEG );
numbadchan = numchan - size(EEG.data,1);
newchannames = {EEG.chanlocs(:).labels};
chanremoved = setdiff(channames,newchannames);
chanremoved = {string(chanremoved)};
EEG = pop_interp(EEG, chanlocs, 'spherical');
EEG = eeg_checkset( EEG );
EEG = pop_reref( EEG, []);


% Epoch the original data
start = -2.5;
stop = 2.5;
if contains(filename,'GFAR')
    EEG = pop_epoch( EEG, {  'DIN8'  }, [start stop], 'newname', 'Epoched', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
elseif contains(filename,'GFAS')
    EEG = pop_epoch( EEG, {  'DIN4'  }, [start stop], 'newname', 'Epoched', 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
end
TotalEP = size(EEG.data,3);



% Insert code for the exclusion of epochs with bad occlusions
if contains(filename,'GFAR')
    for sub = 1:height(bad_occ)
        tempsub = bad_occ.Participant_ID{sub};
        if contains(string(filename),string(tempsub))
            bad_occlusions = str2num(bad_occ.bad_occl{sub});
        end
    end

    if isempty(bad_occlusions)
        numbadocc = 0;
    else
        numbadocc = length(bad_occlusions);
        EEG = pop_select(EEG,'rmtrial',bad_occlusions);
    end
elseif contains(filename,'GFAS')
    for sub = 1:height(bad_trial)
        tempsub = bad_trial.Participant_ID{sub};
        if contains(string(filename),string(tempsub))
            bad_occlusions = str2num(bad_trial.bad_trials{sub});
        end
    end

    if isempty(bad_occlusions)
        numbadocc = 0;
    else
        numbadocc = length(bad_occlusions);
        EEG = pop_select(EEG,'rmtrial',bad_occlusions);
    end
end


% Transfer the ICA related matrices to the original dataset
EEG.icaweights = tempEEG.icaweights;
EEG.icasphere = tempEEG.icasphere;
EEG.icachansind = tempEEG.icachansind;
EEG = eeg_checkset(EEG);
% Plot and inspect ICs
pop_selectcomps(EEG, [1:20]);
EEG = eeg_checkset( EEG );
disp('Press a key to continue!' ) % Press a key here. ...
pause;
comps = find(EEG.reject.gcompreject == 1);
comps = comps(:)';
EEG = pop_subcomp( EEG, comps, 0);
EEG = pop_reref( EEG, []);
EEG = eeg_checkset( EEG );
pop_eegplot( EEG, 1, 1, 1); % script continues here but needs to stop before running the ICA
disp('Press a key to continue!' ) % Press a key here. ...
pause;

reducedEP = size(EEG.data,3);
removedEP = (TotalEP - reducedEP) - numbadocc;


% if contains(filename,'GFAR')
%     figure; pop_erpimage(EEG,1, [129],[[]],'E129',10,1,{ 'DIN8'},[],'latency' ,'yerplabel','\muV','erp','on','cbar','on','topo', { [129] EEG.chanlocs EEG.chaninfo } );
% elseif contains(filename,'GFAS')
%     figure; pop_erpimage(EEG,1, [129],[[]],'E129',10,1,{ 'DIN4'},[],'latency' ,'yerplabel','\muV','erp','on','cbar','on','topo', { [129] EEG.chanlocs EEG.chaninfo } );
% end
% 
% pop_topoplot(EEG, 1, [0:50:800] ,'ERPTopographies',[5 5] ,0,'electrodes','off');

newname = char(filename);
newname = string(newname(1:17));

Preprocinfo.Dataset(index) = newname;
Preprocinfo.Numbadchan(index) = numbadchan;
Preprocinfo.Chanremoved(index) = chanremoved;
Preprocinfo.Bad_Occ(index) = numbadocc;
Preprocinfo.Comps(index) = {comps};
Preprocinfo.TotalEp(index) = TotalEP;
Preprocinfo.Epochsremoved(index) = removedEP;
Preprocinfo.FinalEpochs(index) = reducedEP;

EEG = pop_saveset( EEG, 'filename',char(strcat('Processed',newname)),'filepath',char(writefolder));

writetable(Preprocinfo, strcat(writefolder,'Preprocinfo',filename,'.xlsx'))





