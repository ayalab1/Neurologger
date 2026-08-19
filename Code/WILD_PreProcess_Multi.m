function result = WILD_PreProcess_Multi(deviceFolders,varargin)
% WILD_PreProcess_Multi preprocesses and merges synchronized WILD loggers.
%
% This is an orchestration layer around WILD_PreProcess. Each device folder is
% preprocessed independently, sync events are aligned to the master logger time
% base, and the common recording interval is written as a channel-concatenated
% binary for downstream spike sorting.
%
% Example:
%   folders = {'D:\data\exp01\master','D:\data\exp01\slave01'};
%   result = WILD_PreProcess_Multi(folders, ...
%       'SyncEvent','device_event.d01.evt', ...
%       'OutputFolder','D:\data\exp01\merged');

defaultMasterIndex = 1;
selectedByGui = false;
if nargin < 1 || isempty(deviceFolders)
    deviceFolders = select_device_folders_gui();
    selectedByGui = true;
    if isempty(deviceFolders)
        error('No device folders were selected.');
    end
end

if ischar(deviceFolders) || isstring(deviceFolders)
    deviceFolders = cellstr(deviceFolders);
end
rawDeviceFolders = cellfun(@char,deviceFolders(:),'UniformOutput',false);
deviceFolders = cellfun(@char,deviceFolders(:),'UniformOutput',false);

p = inputParser;
addParameter(p,'MasterIndex',defaultMasterIndex,@(x) isnumeric(x) && isscalar(x));
addParameter(p,'SyncEvent','device_event.d01.evt',@(x) ischar(x) || isstring(x));
addParameter(p,'RunPreprocess',false,@islogical);
addParameter(p,'LfpGen',0,@(x) isnumeric(x) || islogical(x));
addParameter(p,'TrigGen',1,@(x) isnumeric(x) || islogical(x));
addParameter(p,'ProcessIMU',false,@islogical);
addParameter(p,'MergeEphys',true,@islogical);
addParameter(p,'MergeAnalog',true,@islogical);
addParameter(p,'OutputFolder','',@(x) ischar(x) || isstring(x));
addParameter(p,'OutputPrefix','multi_device',@(x) ischar(x) || isstring(x));
addParameter(p,'Overwrite',false,@islogical);
addParameter(p,'ChunkSeconds',5,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'PadValue',0,@(x) isnumeric(x) && isscalar(x));
addParameter(p,'AutoDiscoverDevices',true,@islogical);
addParameter(p,'MergeMethod','channelMerger',@(x) ischar(x) || isstring(x));
parse(p,varargin{:});
opts = p.Results;
opts.SyncEvent = char(opts.SyncEvent);
opts.OutputFolder = char(opts.OutputFolder);
opts.OutputPrefix = char(opts.OutputPrefix);
opts.MergeMethod = char(opts.MergeMethod);
opts.SelectedByGui = selectedByGui;

if opts.AutoDiscoverDevices
    [deviceFolders,parentFolders] = expand_device_folder_inputs(deviceFolders);
else
    parentFolders = {};
end
if isempty(deviceFolders)
    error('No WILD device folders were found.');
end

if selectedByGui && ~has_name_value_arg(varargin,'MasterIndex') && numel(deviceFolders) > 1
    opts.MasterIndex = select_master_device_gui(deviceFolders);
end

if opts.MasterIndex < 1 || opts.MasterIndex > numel(deviceFolders)
    error('MasterIndex is outside the device folder list.');
end

if strcmpi(opts.MergeMethod,'channelMerger') && opts.MasterIndex ~= 1
    order = [opts.MasterIndex, setdiff(1:numel(deviceFolders),opts.MasterIndex,'stable')];
    deviceFolders = deviceFolders(order);
    opts.MasterIndex = 1;
end

if isempty(opts.OutputFolder)
    if isscalar(rawDeviceFolders) && ~isempty(parentFolders)
        opts.OutputFolder = fullfile(rawDeviceFolders{1}, [opts.OutputPrefix '_merged']);
    else
        opts.OutputFolder = fullfile(deviceFolders{opts.MasterIndex}, [opts.OutputPrefix '_merged']);
    end
end
if ~exist(opts.OutputFolder,'dir')
    mkdir(opts.OutputFolder);
end

devices = discover_devices(deviceFolders);
fprintf('Discovered %d WILD device folder(s):\n',numel(devices));
for idx = 1:numel(devices)
    fprintf('  %d: %s\n',idx,devices(idx).folder);
end

if opts.RunPreprocess
    run_device_preprocess(devices,opts);
end

if strcmpi(opts.MergeMethod,'channelMerger')
    amplifierFiles = {devices.amplifierFile};
    if opts.SelectedByGui && ~opts.Overwrite
        opts.Overwrite = confirm_channel_merger_overwrite(amplifierFiles);
    end
    channelMergerResult = WILD_channelMerger(amplifierFiles,opts.Overwrite,0, ...
        'MasterIndex',opts.MasterIndex, ...
        'OutputFolder',opts.OutputFolder, ...
        'OutputPrefix',opts.OutputPrefix, ...
        'ChunkSeconds',opts.ChunkSeconds);
    result = struct;
    result.devices = devices;
    result.inputFolders = rawDeviceFolders;
    result.mergeMethod = opts.MergeMethod;
    result.channelMerger = channelMergerResult;
    result.outputFolder = opts.OutputFolder;
    save(fullfile(opts.OutputFolder,[opts.OutputPrefix '_channel_merger_result.mat']), ...
        'result','devices','channelMergerResult','opts');
    write_layout_tsv(fullfile(opts.OutputFolder,[opts.OutputPrefix '_channel_layout.tsv']),devices);
    return;
end

alignment = align_devices_from_events(devices,opts);
mergeInfo = struct([]);

if opts.MergeEphys
    mergeInfo = merge_stream_to_master_time(devices,alignment,opts,'ephys');
end

if opts.MergeAnalog
    mergeInfoAnalog = merge_stream_to_master_time(devices,alignment,opts,'analog');
    if isempty(mergeInfo)
        mergeInfo = mergeInfoAnalog;
    else
        mergeInfo.analog = mergeInfoAnalog;
    end
end

result = struct;
result.devices = devices;
result.inputFolders = rawDeviceFolders;
result.alignment = alignment;
result.mergeInfo = mergeInfo;
result.outputFolder = opts.OutputFolder;

save(fullfile(opts.OutputFolder,[opts.OutputPrefix '_sync_alignment.mat']), ...
    'result','devices','alignment','mergeInfo','opts');
write_layout_tsv(fullfile(opts.OutputFolder,[opts.OutputPrefix '_channel_layout.tsv']),devices);

end

function deviceFolders = select_device_folders_gui()
deviceFolders = {};

try
    chooser = javax.swing.JFileChooser(pwd);
    chooser.setDialogTitle('Select WILD subepoch, device folders, or a parent folder');
    chooser.setFileSelectionMode(javax.swing.JFileChooser.DIRECTORIES_ONLY);
    chooser.setMultiSelectionEnabled(true);
    status = chooser.showOpenDialog([]);
    if status == javax.swing.JFileChooser.APPROVE_OPTION
        selected = chooser.getSelectedFiles();
        nSelected = numel(selected);
        deviceFolders = cell(nSelected,1);
        for idx = 1:nSelected
            deviceFolders{idx} = char(selected(idx).getAbsolutePath());
        end
    end
catch
    deviceFolders = select_device_folders_repeated();
end

deviceFolders = deviceFolders(:);
end

function deviceFolders = select_device_folders_repeated()
deviceFolders = {};
while true
    folder = uigetdir(pwd,'Select a WILD subepoch, device folder, or parent folder');
    if isequal(folder,0)
        break;
    end
    deviceFolders{end+1,1} = folder; %#ok<AGROW>
    answer = questdlg('Add another device folder?', ...
        'WILD device folders', ...
        'Add another','Done','Done');
    if ~strcmp(answer,'Add another')
        break;
    end
end
end

function masterIndex = select_master_device_gui(deviceFolders)
masterIndex = 1;
labels = cellfun(@(x) sprintf('%s',x),deviceFolders,'UniformOutput',false);
[choice,ok] = listdlg( ...
    'PromptString','Select the master device folder:', ...
    'SelectionMode','single', ...
    'ListString',labels, ...
    'InitialValue',1, ...
    'Name','WILD master device');
if ok
    masterIndex = choice;
end
end

function tf = has_name_value_arg(args,name)
tf = false;
if isempty(args)
    return;
end
names = args(1:2:end);
for idx = 1:numel(names)
    if (ischar(names{idx}) || isstring(names{idx})) && strcmpi(char(names{idx}),name)
        tf = true;
        return;
    end
end
end

function overwrite = confirm_channel_merger_overwrite(amplifierFiles)
overwrite = false;
mergedFile = strrep(amplifierFiles{1},'.dat','_merged.dat');
if ~exist(mergedFile,'file')
    return;
end

answer = questdlg( ...
    sprintf('Merged output already exists:\n%s\n\nOverwrite it?',mergedFile), ...
    'WILD merged output exists', ...
    'Overwrite','Skip','Skip');
overwrite = strcmp(answer,'Overwrite');
end

function [deviceFolders,parentFolders] = expand_device_folder_inputs(inputFolders)
deviceFolders = {};
parentFolders = {};
for idx = 1:numel(inputFolders)
    folder = inputFolders{idx};
    if is_wild_device_folder(folder)
        deviceFolders{end+1,1} = folder; %#ok<AGROW>
    else
        found = find_wild_device_folders(folder);
        if isempty(found)
            error('No WILD device folders found below: %s',folder);
        end
        parentFolders{end+1,1} = folder; %#ok<AGROW>
        deviceFolders = [deviceFolders; found(:)]; %#ok<AGROW>
    end
end
deviceFolders = unique(deviceFolders,'stable');
end

function tf = is_wild_device_folder(folder)
tf = exist(folder,'dir') && ...
    exist(fullfile(folder,'amplifier.dat'),'file') && ...
    exist(fullfile(folder,'CE_params.bin'),'file');
end

function folders = find_wild_device_folders(parentFolder)
if ~exist(parentFolder,'dir')
    error('Folder does not exist: %s',parentFolder);
end
ampFiles = dir(fullfile(parentFolder,'**','amplifier.dat'));
folders = {};
for idx = 1:numel(ampFiles)
    folder = ampFiles(idx).folder;
    if exist(fullfile(folder,'CE_params.bin'),'file')
        folders{end+1,1} = folder; %#ok<AGROW>
    end
end
folders = sort(folders);
end

function devices = discover_devices(deviceFolders)
devices = repmat(struct,0,1);
for idx = 1:numel(deviceFolders)
    folder = deviceFolders{idx};
    ampFile = fullfile(folder,'amplifier.dat');
    analogFile = fullfile(folder,'analogin.dat');
    paramFile = fullfile(folder,'CE_params.bin');

    if ~exist(folder,'dir')
        error('Device folder does not exist: %s',folder);
    end
    if ~exist(ampFile,'file')
        error('Missing amplifier.dat in %s',folder);
    end
    if ~exist(paramFile,'file')
        error('Missing CE_params.bin in %s',folder);
    end

    sys = WILD_ReadHeader(paramFile);
    info = dir(ampFile);
    nSamples = floor(info.bytes / 2 / sys.Nch);

    devices(idx).index = idx;
    devices(idx).folder = folder;
    [devices(idx).deviceName,devices(idx).recordingName] = infer_device_labels(folder);
    devices(idx).amplifierFile = ampFile;
    devices(idx).analogFile = analogFile;
    devices(idx).paramFile = paramFile;
    devices(idx).sys = sys;
    devices(idx).fs = sys.fs;
    devices(idx).nChannels = sys.Nch;
    devices(idx).nSamples = nSamples;
    devices(idx).duration = (nSamples - 1) / sys.fs;
    devices(idx).analogChannels = analog_channel_count(sys);
end
end

function run_device_preprocess(devices,opts)
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir));
for idx = 1:numel(devices)
    fprintf('Preprocessing device %d/%d: %s\n',idx,numel(devices),devices(idx).folder);
    cd(devices(idx).folder);
    call_wild_preprocess(devices(idx),opts);
end
clear cleanupObj;
end

function call_wild_preprocess(device,opts)
preprocessPath = which('WILD_PreProcess');
if isempty(preprocessPath)
    error('WILD_PreProcess was not found on the MATLAB path.');
end

nInputs = nargin('WILD_PreProcess');
fprintf('  Using WILD_PreProcess: %s\n',preprocessPath);

if nInputs < 0 || nInputs >= 5
    WILD_PreProcess(device.amplifierFile,device.analogFile,opts.LfpGen,opts.TrigGen,opts.ProcessIMU);
elseif nInputs >= 4
    WILD_PreProcess(device.amplifierFile,device.analogFile,opts.LfpGen,opts.TrigGen);
elseif nInputs == 3
    WILD_PreProcess(device.amplifierFile,device.analogFile,opts.LfpGen);
elseif nInputs == 2
    WILD_PreProcess(device.amplifierFile,device.analogFile);
else
    WILD_PreProcess(device.amplifierFile);
end
end

function alignment = align_devices_from_events(devices,opts)
masterIdx = opts.MasterIndex;
masterEvents = load_sync_edges(fullfile(devices(masterIdx).folder,opts.SyncEvent));

alignment = struct;
alignment.masterIndex = masterIdx;
alignment.syncEvent = opts.SyncEvent;
alignment.commonStart = -inf;
alignment.commonEnd = inf;
alignment.device = repmat(struct,0,1);

for idx = 1:numel(devices)
    if idx == masterIdx
        tDevice = masterEvents;
        tMaster = masterEvents;
        fitCoeffs = [1 0];
    else
        tDeviceAll = load_sync_edges(fullfile(devices(idx).folder,opts.SyncEvent));
        n = min(numel(masterEvents),numel(tDeviceAll));
        if n < 2
            error('Need at least two matching sync edges for device %d.',idx);
        end
        tMaster = masterEvents(1:n);
        tDevice = tDeviceAll(1:n);
        fitCoeffs = polyfit(tDevice,tMaster,1);
    end

    residual = tMaster(:) - polyval(fitCoeffs,tDevice(:));
    mappedStart = polyval(fitCoeffs,0);
    mappedEnd = polyval(fitCoeffs,devices(idx).duration);
    if mappedEnd < mappedStart
        tmp = mappedStart;
        mappedStart = mappedEnd;
        mappedEnd = tmp;
    end

    alignment.device(idx).index = idx;
    alignment.device(idx).folder = devices(idx).folder;
    alignment.device(idx).a = fitCoeffs(1);
    alignment.device(idx).b = fitCoeffs(2);
    alignment.device(idx).nSyncEdges = numel(tDevice);
    alignment.device(idx).residual = residual;
    alignment.device(idx).residualStd = std(residual);
    alignment.device(idx).residualMaxAbs = max(abs(residual));
    alignment.device(idx).mappedStart = mappedStart;
    alignment.device(idx).mappedEnd = mappedEnd;

    alignment.commonStart = max(alignment.commonStart,mappedStart);
    alignment.commonEnd = min(alignment.commonEnd,mappedEnd);
end

if alignment.commonEnd <= alignment.commonStart
    error('No common time interval across devices after alignment.');
end
end

function edges = load_sync_edges(evtFile)
if ~exist(evtFile,'file')
    error('Missing sync event file: %s',evtFile);
end
events = LoadEvents(evtFile);
edges = events.time(:);
if numel(edges) >= 2
    edges = edges(1:2:end);
end
edges = edges(~isnan(edges));
if numel(edges) < 2
    error('Sync event file has fewer than two usable edges: %s',evtFile);
end
end

function mergeInfo = merge_stream_to_master_time(devices,alignment,opts,streamName)
master = devices(alignment.masterIndex);
switch streamName
    case 'ephys'
        outFs = master.fs;
        outFile = fullfile(opts.OutputFolder,'amplifier.dat');
        precision = 'int16';
        channelField = 'nChannels';
        inputField = 'amplifierFile';
    case 'analog'
        outFs = 1250;
        outFile = fullfile(opts.OutputFolder,'analogin.dat');
        precision = 'int16';
        channelField = 'analogChannels';
        inputField = 'analogFile';
    otherwise
        error('Unknown stream: %s',streamName);
end

if exist(outFile,'file') && ~opts.Overwrite
    error('Output exists. Set Overwrite=true to replace: %s',outFile);
end

for idx = 1:numel(devices)
    if ~exist(devices(idx).(inputField),'file')
        error('Missing %s for device %d: %s',streamName,idx,devices(idx).(inputField));
    end
end

startSample = ceil(alignment.commonStart * outFs);
endSample = floor(alignment.commonEnd * outFs);
nOut = endSample - startSample + 1;
if nOut <= 0
    error('No output samples for %s merge.',streamName);
end

totalChannels = sum([devices.(channelField)]);
chunkSamples = max(1,round(opts.ChunkSeconds * outFs));
fh = fopen(outFile,'w+');
if fh == -1
    error('Could not open output file: %s',outFile);
end
cleanupObj = onCleanup(@() fclose(fh));

ptr = 0;
while ptr < nOut
    curN = min(chunkSamples,nOut - ptr);
    sample0 = startSample + ptr + (0:(curN-1));
    tMaster = sample0(:) / outFs;
    chunkData = zeros(curN,totalChannels,'int16');
    chPtr = 1;

    for idx = 1:numel(devices)
        a = alignment.device(idx).a;
        b = alignment.device(idx).b;
        tDevice = (tMaster - b) ./ a;
        data = read_aligned_chunk(devices(idx),inputField,channelField,precision,tDevice,opts.PadValue,streamName);
        chN = devices(idx).(channelField);
        chunkData(:,chPtr:(chPtr+chN-1)) = data;
        chPtr = chPtr + chN;
    end

    fwrite(fh,chunkData',precision);
    ptr = ptr + curN;
    fprintf('Merged %s: %.1f%%\n',streamName,100 * ptr / nOut);
end

clear cleanupObj;

mergeInfo = struct;
mergeInfo.stream = streamName;
mergeInfo.file = outFile;
mergeInfo.fs = outFs;
mergeInfo.nSamples = nOut;
mergeInfo.nChannels = totalChannels;
mergeInfo.commonStart = alignment.commonStart;
mergeInfo.commonEnd = alignment.commonEnd;
mergeInfo.startSample = startSample;
mergeInfo.endSample = endSample;

if strcmp(streamName,'ephys')
    write_time_dat(fullfile(opts.OutputFolder,'time.dat'),nOut);
end
end

function dataOut = read_aligned_chunk(device,inputField,channelField,precision,tDevice,padValue,streamName)
switch streamName
    case 'ephys'
        fs = device.fs;
        nSamples = device.nSamples;
    case 'analog'
        fs = 1250;
        info = dir(device.(inputField));
        nSamples = floor(info.bytes / 2 / device.(channelField));
    otherwise
        error('Unknown stream: %s',streamName);
end

chN = device.(channelField);
sampleOneBased = tDevice(:) * fs + 1;
valid = sampleOneBased >= 1 & sampleOneBased <= nSamples;
dataOut = repmat(int16(padValue),numel(tDevice),chN);
if ~any(valid)
    return;
end

readStart0 = max(0,floor(min(sampleOneBased(valid))) - 2);
readEnd0 = min(nSamples - 1,ceil(max(sampleOneBased(valid))) + 1);
block = readmulti_frank(device.(inputField),chN,1:chN,readStart0,readEnd0,precision);
sourceSamples = (readStart0:readEnd0)' + 1;
interpData = interp1(sourceSamples,double(block),sampleOneBased(valid),'linear',double(padValue));
interpData = max(min(round(interpData),double(intmax('int16'))),double(intmin('int16')));
dataOut(valid,:) = int16(interpData);
end

function nAnalog = analog_channel_count(sys)
if sys.Nch == 64
    nAnalog = 16;
else
    nAnalog = 8;
end
end

function write_time_dat(filename,nSamples)
fh = fopen(filename,'w+');
if fh == -1
    error('Could not open time.dat for writing: %s',filename);
end
cleanupObj = onCleanup(@() fclose(fh));
chunk = 1e6;
ptr = 0;
while ptr < nSamples
    curN = min(chunk,nSamples - ptr);
    fwrite(fh,int32(ptr:(ptr+curN-1)),'int32');
    ptr = ptr + curN;
end
clear cleanupObj;
end

function write_layout_tsv(filename,devices)
fh = fopen(filename,'w+');
if fh == -1
    error('Could not write channel layout: %s',filename);
end
cleanupObj = onCleanup(@() fclose(fh));
fprintf(fh,'merged_channel\tdevice_index\tdevice_name\trecording_name\tdevice_channel\tdevice_folder\n');
mergedCh = 1;
for d = 1:numel(devices)
    for ch = 1:devices(d).nChannels
        fprintf(fh,'%d\t%d\t%s\t%s\t%d\t%s\n', ...
            mergedCh,d,devices(d).deviceName,devices(d).recordingName,ch,devices(d).folder);
        mergedCh = mergedCh + 1;
    end
end
clear cleanupObj;
end

function [deviceName,recordingName] = infer_device_labels(folder)
[parentFolder,recordingName] = fileparts(folder);
[~,deviceName] = fileparts(parentFolder);
if isempty(deviceName)
    deviceName = sprintf('device_%s',recordingName);
end
if isempty(recordingName)
    recordingName = 'recording';
end
end
