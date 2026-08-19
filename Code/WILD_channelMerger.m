function result = WILD_channelMerger(files,overwrite,use_cache,varargin)
% search for mergable files
%%
if(nargin<1 || isempty(files))
    folder = uigetdir(pwd,'Select a WILD subepoch folder or device recording folder');
    if isequal(folder,0)
        result = [];
        return;
    end
    files = folder;
end
if(nargin<2)
    overwrite=0;
end
if(nargin<3)
    use_cache=0;
end

opts = parse_channel_merger_options(varargin{:});
files = resolve_wild_merger_files(files);
if strcmpi(opts.Mode,'syncQC') || numel(files) > 2
    result = WILD_channelMerger_multiSyncQC(files,overwrite,opts);
    return;
end
if numel(files) ~= 2
    error('WILD_channelMerger currently expects exactly 2 amplifier.dat files. Found %d.',numel(files));
end

fname_new = strrep(files{1},'.dat','_merged.dat');

[p,f]=fileparts(files{1});
pths = cellfun(@fileparts,files,'uni',0);
misc_files = cellfun(@(x) fullfile(x,'analogin.dat'),pths,'uni',0);
fanalog1_new = fullfile(p,'analogin1.dat');
fanalog2_new = fullfile(p,'analogin2.dat');
fanalog_merged_new = fullfile(p,'analogin_merged.dat');

disp(['Merging:'])
cellfun(@(x) disp(x),files);
if(~isempty(dir(fname_new)))
    if(overwrite==0)
        disp(['File exists, skipping: ' fname_new]);
        disp('Set overwrite=1, or call WILD_PreProcess_Multi(..., "Overwrite", true) to regenerate it.');
        result = make_channel_merger_result(files,fname_new,fanalog_merged_new,fanalog2_new,[]);
        return;
    end
    disp(['Overwriting existing merged file: ' fname_new]);
end
fh=fopen(fname_new,'w+'); %start new file for parallel processing
fh_misc=fopen(fanalog_merged_new,'w+'); %start new file for parallel processing
fh_misc2=fopen(fanalog2_new,'w+'); %start new file for parallel processing

cache_dir = 'd:\temp\';
% files = {fileA,fileB};
[~,Nch,fs_values] = read_merger_headers(pths);
validate_merger_headers(files,Nch,fs_values);
fs = fs_values(1);
fs_misc = 1250;
Nch_misc = Nch./4;
f_info=cellfun(@(x) dir(x),files,'uni',0);
f_sizes=cellfun(@(x) x.bytes,f_info);
f_sizes = f_sizes(:);
Nch = Nch(:);
if any(mod(f_sizes,Nch*2)~=0)
    error('At least one amplifier.dat file size is not divisible by 2 bytes x channel count.');
end
Nsamples  = f_sizes./(Nch*2);
%                 [Nch,fs,Nsamples]=arrayfun(@(x) DAT_xmlread(files{x}),1:length(files));
Nsamples_new = max(Nsamples(:));
ch_order = arrayfun(@(x) 1:x,Nch,'uni',0);

% ch_order{2} = fliplr(ch_order{2}); %Customization:reverse ch group2
%                 ch_order{2} = Intan64Reversed(ch_order{2});

%% check alignment
res_rate = fs/fs_misc;
initialStart = round(opts.InitialStartSeconds * fs);
initialSamples = round(opts.InitialDurationSeconds * fs);
if initialSamples < 2
    error('InitialDurationSeconds is too short for alignment.');
end
initialEnd = initialStart + initialSamples - 1;
initialMaxLag = round(opts.InitialMaxLagSeconds * fs);
testdatas = arrayfun(@(x) readmulti_frank(files{x},Nch(x),1:Nch(x),initialStart,initialEnd,'int16'),1:length(files),'uni',0);
[b,a]=butter(2,200/fs*2,'high');
testdatas = cellfun(@(x) filtfilt(b,a,x),testdatas,'uni',0);
commonMode=cellfun(@(x) median(x,2),testdatas,'uni',0);
commonMode = cat(2,commonMode{:});
subplot(211);
plot(commonMode);
title('CommonMode');
legend(files,'Interpreter','none')
axis tight;
subplot(212);
[r,lags]=xcorr(commonMode(:,1),commonMode(:,2),initialMaxLag);
plot(lags,r);
[~,ix]=max(r);
if max(r)<2*median(r)
    disp(['Sync is bad, skipping...']);
    fclose(fh);
    fclose(fh_misc);
    fclose(fh_misc2);
    result = make_channel_merger_result(files,fname_new,fanalog_merged_new,fanalog2_new,[]);
    return;
end
fileB_Lag = - lags(ix) ;
legend(['B offset to A:' num2str(fileB_Lag) 'sps'],'Interpreter','none');
offsets = [0 fileB_Lag];
disp(['B offset to A:' num2str(fileB_Lag) 'sps']);
drawnow;
%% merge


if(use_cache==1)
    cached_files = cell(length(files),1);
    for x = 1:length(files)
        old_path = files{x};
        path_parts=fileparts(old_path);
        path_parts = strsplit(path_parts,'\');
        new_path = [cache_dir datestr(now,'YYmmDD_HHMMSS') '_' path_parts{end} '_amplifier.dat'];
%         system(['del "' new_path '"']);
        system(['copy "' old_path '" "' new_path '"']);
        files{x} = new_path;
        cached_files{x} = new_path;
    end
end

ptr=0;


while(ptr<Nsamples_new)
    chunk=round(opts.ChunkSeconds * fs);
    if(Nsamples_new-ptr<chunk)
        chunk = Nsamples_new-ptr;
    end
    
    data=arrayfun(@(x) read_chunk_padded(files{x},Nch(x),ch_order{x},offsets(x)+ptr,chunk,'int16'),1:length(files),'uni',0);
    
    ptr_misc = ptr/res_rate;
    offsets_misc = offsets/res_rate;
    chunk_misc = chunk/res_rate;
    misc = arrayfun(@(x) read_chunk_padded(misc_files{x},Nch_misc(x),1:Nch_misc(x),offsets_misc(x)+ptr_misc,chunk_misc,'int16'),1:length(files),'uni',0);
    %check alignments
    testdatas = cellfun(@(x) filtfilt(b,a,double(x)),data,'uni',0);
    commonMode=cellfun(@(x) median(x,2),testdatas,'uni',0);
    commonMode = cat(2,commonMode{:});
    maxChunkLag = min(round(opts.ChunkMaxLagSeconds * fs),max(chunk - 1,0));
    [r,lags]=xcorr(commonMode(:,1),commonMode(:,2),maxChunkLag);
%     subplot(313);
%     plot(lags,r);
%     drawnow
    [~,ix]=max(r);
    fileB_Lag = - lags(ix) ;
    offsets = bsxfun(@plus,offsets,[0 fileB_Lag]);
    
    data_cat = cat(2,data{:});
    fwrite(fh,data_cat','int16');
    misc2 = cat(2,misc{2});
    fwrite(fh_misc2,misc2','int16');
    misc_cat = cat(2,misc{:});
    fwrite(fh_misc,misc_cat','int16');
    ptr=ptr+chunk;
    progressPct = (ptr ./ Nsamples_new) * 100;
    disp(['Merging:'  num2str(progressPct) '% ' files{1} ' - ' files{2} ' Offsets:' num2str(offsets)]);
end
fclose(fh);
fclose(fh_misc);
fclose(fh_misc2);
if(use_cache==1)
    cellfun(@(x) system(['del "' x '"']),cached_files);
end

result = make_channel_merger_result(files,fname_new,fanalog_merged_new,fanalog2_new,offsets);
save(strrep(fname_new,'.dat','_mergeInfo.mat'),'result');
end

function files = resolve_wild_merger_files(files)
if ischar(files) || isstring(files)
    files = cellstr(files);
end
files = cellfun(@char,files(:),'UniformOutput',false);

resolved = {};
for idx = 1:numel(files)
    cur = files{idx};
    if exist(cur,'dir')
        directAmp = fullfile(cur,'amplifier.dat');
        if exist(directAmp,'file')
            resolved{end+1,1} = directAmp; %#ok<AGROW>
        else
            found = find_wild_amplifier_files(cur);
            resolved = [resolved; found(:)]; %#ok<AGROW>
        end
    elseif exist(cur,'file')
        [~,name,ext] = fileparts(cur);
        if strcmpi([name ext],'amplifier.dat')
            resolved{end+1,1} = cur; %#ok<AGROW>
        else
            error('Expected amplifier.dat file, got: %s',cur);
        end
    else
        error('Input path does not exist: %s',cur);
    end
end

files = unique(resolved,'stable');
end

function files = find_wild_amplifier_files(parentFolder)
ampFiles = dir(fullfile(parentFolder,'**','amplifier.dat'));
files = {};
for idx = 1:numel(ampFiles)
    folder = ampFiles(idx).folder;
    if exist(fullfile(folder,'CE_params.bin'),'file')
        files{end+1,1} = fullfile(folder,'amplifier.dat'); %#ok<AGROW>
    end
end
files = sort(files);
end

function [sysParams,Nch,fs_values] = read_merger_headers(recordingFolders)
sysParams = cell(numel(recordingFolders),1);
Nch = zeros(numel(recordingFolders),1);
fs_values = zeros(numel(recordingFolders),1);
for idx = 1:numel(recordingFolders)
    paramFile = fullfile(recordingFolders{idx},'CE_params.bin');
    if ~exist(paramFile,'file')
        error('Missing CE_params.bin for %s',recordingFolders{idx});
    end
    sysParams{idx} = WILD_ReadHeader(paramFile);
    Nch(idx) = sysParams{idx}.Nch;
    fs_values(idx) = sysParams{idx}.fs;
end
end

function validate_merger_headers(files,Nch,fs_values)
if any(Nch ~= 64)
    error('WILD_channelMerger currently supports only 64-channel recordings. Found channel counts: %s',num2str(Nch(:)'));
end
if any(fs_values ~= fs_values(1))
    error('All recordings must use the same sampling rate. Found sampling rates: %s',num2str(fs_values(:)'));
end
if fs_values(1) <= 0
    error('Invalid sampling rate in CE_params.bin: %g',fs_values(1));
end
if mod(fs_values(1),1250) ~= 0
    error('Sampling rate must be an integer multiple of 1250 Hz for analog merge. Found: %g',fs_values(1));
end
for idx = 1:numel(files)
    if ~exist(files{idx},'file')
        error('Missing amplifier.dat file: %s',files{idx});
    end
end
end

function data = read_chunk_padded(fname,numchannel,chselect,readStart,nSamples,precision)
if nargin < 6
    precision = 'int16';
end
readStart = round(readStart);
nSamples = round(nSamples);
if nSamples < 1
    data = zeros(0,numel(chselect),precision);
    return;
end

fileinfo = dir(fname);
if isempty(fileinfo)
    error('Missing input file: %s',fname);
end
bytesPerSample = get_format_bytes_local(precision);
totalSamples = floor(fileinfo.bytes / bytesPerSample / numchannel);
data = zeros(nSamples,numel(chselect),precision);

readEnd = readStart + nSamples - 1;
sourceStart = max(0,readStart);
sourceEnd = min(totalSamples - 1,readEnd);
if sourceEnd < sourceStart
    return;
end

destStart = sourceStart - readStart + 1;
temp = readmulti_frank(fname,numchannel,chselect,sourceStart,sourceEnd,precision);
nCopy = min(size(temp,1),nSamples - destStart + 1);
if nCopy > 0
    data(destStart:(destStart+nCopy-1),:) = temp(1:nCopy,:);
end
end

function bytes = get_format_bytes_local(format)
switch format
    case {'int8', 'uint8'}
        bytes = 1;
    case {'int16', 'uint16'}
        bytes = 2;
    case {'int32', 'uint32', 'single'}
        bytes = 4;
    case {'int64', 'uint64', 'double'}
        bytes = 8;
    otherwise
        error('Unknown format: %s',format);
end
end

function result = make_channel_merger_result(files,amplifierMerged,analogMerged,analog2,offsets)
result = struct;
result.files = files;
result.amplifierMerged = amplifierMerged;
result.analogMerged = analogMerged;
result.analog2 = analog2;
result.offsets = offsets;
result.createdAt = char(datetime('now'));
end

function opts = parse_channel_merger_options(varargin)
p = inputParser;
addParameter(p,'Mode','auto',@(x) ischar(x) || isstring(x));
addParameter(p,'MasterIndex',1,@(x) isnumeric(x) && isscalar(x));
addParameter(p,'OutputFolder','',@(x) ischar(x) || isstring(x));
addParameter(p,'OutputPrefix','wild_multilogger_sync',@(x) ischar(x) || isstring(x));
addParameter(p,'InitialStartSeconds',30,@(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p,'InitialDurationSeconds',120,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'InitialMaxLagSeconds',30,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'ChunkSeconds',5,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'ChunkMaxLagSeconds',5,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'MinPeakRatio',2,@(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p,'WarnLagStepSamples',100,@(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p,'WarnDriftSamples',1000,@(x) isnumeric(x) && isscalar(x) && x >= 0);
parse(p,varargin{:});
opts = p.Results;
opts.Mode = char(opts.Mode);
opts.OutputFolder = char(opts.OutputFolder);
opts.OutputPrefix = char(opts.OutputPrefix);
end

function result = WILD_channelMerger_multiSyncQC(files,overwrite,opts)
% Multi-logger path. It first checks every slave against one master logger
% using the same common-mode xcorr idea as the 2-logger merge path, then
% writes merged streams when Mode is not syncQC.
if numel(files) < 2
    error('WILD_channelMerger syncQC requires at least 2 amplifier.dat files.');
end
if opts.MasterIndex < 1 || opts.MasterIndex > numel(files)
    error('MasterIndex is outside the selected logger list.');
end
folders = cellfun(@fileparts,files,'UniformOutput',false);
if isempty(opts.OutputFolder)
    opts.OutputFolder = fileparts(folders{opts.MasterIndex});
end
if ~exist(opts.OutputFolder,'dir')
    mkdir(opts.OutputFolder);
end

[sysParams,Nch,fsValues,nSamples] = read_multi_qc_headers(folders,files);
validate_multi_qc_inputs(Nch,fsValues);
fs = fsValues(1);

matFile = fullfile(opts.OutputFolder,[opts.OutputPrefix '_qc.mat']);
tsvFile = fullfile(opts.OutputFolder,[opts.OutputPrefix '_qc.tsv']);
if (exist(matFile,'file') || exist(tsvFile,'file')) && ~overwrite
    error('QC output exists. Set overwrite=1 to regenerate: %s',opts.OutputFolder);
end

fprintf('WILD_channelMerger multi-logger sync QC\n');
fprintf('Master: %d %s\n',opts.MasterIndex,files{opts.MasterIndex});
fprintf('Output folder: %s\n',opts.OutputFolder);

[b,a] = butter(2,200/fs*2,'high');
pairResults = repmat(empty_multi_qc_pair_result(),0,1);
slaveOrder = reshape(setdiff(1:numel(files),opts.MasterIndex,'stable'),1,[]);
totalPairs = numel(slaveOrder);
pairCounter = 0;
for idx = slaveOrder
    pairCounter = pairCounter + 1;
    fprintf('Checking master %d vs slave %d\n',opts.MasterIndex,idx);
    fprintf('WILD_PROGRESS:sync_pair_%d_of_%d:%.3f\n',pairCounter,totalPairs,100 * (pairCounter - 1) / max(totalPairs,1));
    pairResult = run_multi_qc_pair(files{opts.MasterIndex},files{idx},folders{opts.MasterIndex},folders{idx}, ...
        opts.MasterIndex,idx,Nch(opts.MasterIndex),Nch(idx),fs,nSamples(opts.MasterIndex),nSamples(idx),b,a,opts,pairCounter,totalPairs);
    pairResults(end+1,1) = pairResult; %#ok<AGROW>
end
fprintf('WILD_PROGRESS:sync_qc_complete:100\n');

result = struct;
result.createdAt = char(datetime('now'));
result.mode = 'syncQC';
result.files = files;
result.folders = folders;
result.masterIndex = opts.MasterIndex;
result.sysParams = sysParams;
result.fs = fs;
result.nChannels = Nch;
result.nSamples = nSamples;
result.options = opts;
result.pairs = pairResults;
result.tsvFile = tsvFile;
result.matFile = matFile;

write_multi_qc_tsv(tsvFile,result);
fprintf('Saved QC MAT: %s\n',matFile);
fprintf('Saved QC TSV: %s\n',tsvFile);

if ~strcmpi(opts.Mode,'syncQC')
    if any(strcmp({pairResults.status},'FAIL'))
        save(matFile,'result','pairResults','opts','folders','files','sysParams');
        error('Sync QC has FAIL pair(s). Review QC output before writing merged dat files.');
    end
    mergeInfo = write_multi_logger_merged_streams(files,folders,Nch,nSamples,fs,pairResults,opts,overwrite);
    result.mode = 'multiMerge';
    result.mergeInfo = mergeInfo;
end

save(matFile,'result','pairResults','opts','folders','files','sysParams');
end

function pairResult = run_multi_qc_pair(masterFile,slaveFile,masterFolder,slaveFolder,masterIndex,slaveIndex,masterNch,slaveNch,fs,masterSamples,slaveSamples,b,a,opts,pairCounter,totalPairs)
initialStart = round(opts.InitialStartSeconds * fs);
initialN = round(opts.InitialDurationSeconds * fs);
initialMaxLag = round(opts.InitialMaxLagSeconds * fs);

masterCm = read_multi_qc_common_mode(masterFile,masterNch,initialStart,initialN,b,a);
slaveCm = read_multi_qc_common_mode(slaveFile,slaveNch,initialStart,initialN,b,a);
[initialLag,initialPeak,initialMedianAbs,initialPeakRatio,lags,r] = multi_qc_common_mode_lag(masterCm,slaveCm,initialMaxLag);
currentOffset = initialLag;

chunkSamples = round(opts.ChunkSeconds * fs);
chunkMaxLag = min(round(opts.ChunkMaxLagSeconds * fs),chunkSamples - 1);
maxSamples = min(masterSamples,slaveSamples);
ptr = 0;
chunkIndex = 0;
chunkRows = zeros(ceil(maxSamples / chunkSamples),8);
totalChunks = max(1,floor(maxSamples / chunkSamples));
while ptr + chunkSamples <= maxSamples
    chunkNumber = floor(ptr / chunkSamples) + 1;
    totalProgress = 100 * ((pairCounter - 1) + min(ptr / maxSamples,1)) / max(totalPairs,1);
    fprintf('WILD_PROGRESS:sync_pair_%d_of_%d_chunk_%d_of_%d:%.3f\n',pairCounter,totalPairs,chunkNumber,totalChunks,totalProgress);
    validMaster = valid_multi_qc_fraction(ptr,chunkSamples,masterSamples);
    validSlave = valid_multi_qc_fraction(ptr + currentOffset,chunkSamples,slaveSamples);
    if validMaster >= 0.80 && validSlave >= 0.80
        cmA = read_multi_qc_common_mode(masterFile,masterNch,ptr,chunkSamples,b,a);
        cmB = read_multi_qc_common_mode(slaveFile,slaveNch,ptr + currentOffset,chunkSamples,b,a);
        [lagStep,peak,medianAbs,peakRatio] = multi_qc_common_mode_lag(cmA,cmB,chunkMaxLag);
        currentOffset = currentOffset + lagStep;
        chunkIndex = chunkIndex + 1;
        chunkRows(chunkIndex,:) = [ptr / fs, currentOffset, lagStep, peakRatio, peak, medianAbs, validMaster, validSlave];
    end
    ptr = ptr + chunkSamples;
end
chunkRows = chunkRows(1:chunkIndex,:);

status = 'OK';
messages = {};
if initialPeakRatio < opts.MinPeakRatio
    status = 'FAIL';
    messages{end+1} = sprintf('initial peak ratio %.3g < %.3g',initialPeakRatio,opts.MinPeakRatio); %#ok<AGROW>
end
if isempty(chunkRows)
    status = 'FAIL';
    messages{end+1} = 'no valid chunks'; %#ok<AGROW>
else
    maxAbsStep = max(abs(chunkRows(:,3)));
    driftSamples = chunkRows(end,2) - chunkRows(1,2);
    minChunkPeakRatio = min(chunkRows(:,4));
    if minChunkPeakRatio < opts.MinPeakRatio && ~strcmp(status,'FAIL')
        status = 'WARN';
        messages{end+1} = sprintf('min chunk peak ratio %.3g < %.3g',minChunkPeakRatio,opts.MinPeakRatio); %#ok<AGROW>
    end
    if maxAbsStep > opts.WarnLagStepSamples && ~strcmp(status,'FAIL')
        status = 'WARN';
        messages{end+1} = sprintf('max lag step %.0f samples',maxAbsStep); %#ok<AGROW>
    end
    if abs(driftSamples) > opts.WarnDriftSamples && ~strcmp(status,'FAIL')
        status = 'WARN';
        messages{end+1} = sprintf('offset drift %.0f samples',driftSamples); %#ok<AGROW>
    end
end
if isempty(messages)
    messages = {'review QC figure before merge'};
end

pairResult = empty_multi_qc_pair_result();
pairResult.masterIndex = masterIndex;
pairResult.slaveIndex = slaveIndex;
pairResult.masterFolder = masterFolder;
pairResult.slaveFolder = slaveFolder;
pairResult.masterFile = masterFile;
pairResult.slaveFile = slaveFile;
pairResult.initialOffsetSamples = initialLag;
pairResult.initialPeak = initialPeak;
pairResult.initialMedianAbs = initialMedianAbs;
pairResult.initialPeakRatio = initialPeakRatio;
pairResult.finalOffsetSamples = currentOffset;
pairResult.status = status;
pairResult.message = strjoin(messages,'; ');
pairResult.chunkColumns = {'time_sec','offset_samples','lag_step_samples','peak_ratio','peak','median_abs_xcorr','valid_master','valid_slave'};
pairResult.chunks = chunkRows;
pairResult.figureFile = save_multi_qc_pair_figure(masterCm,slaveCm,lags,r,chunkRows,opts,masterFolder,slaveFolder,slaveIndex);
fprintf('  slave %d: %s, initial offset %.0f samples, final offset %.0f samples, %s\n', ...
    slaveIndex,status,initialLag,currentOffset,pairResult.message);
end

function cm = read_multi_qc_common_mode(file,nChannels,startSample,nSamples,b,a)
data = read_chunk_padded(file,nChannels,1:nChannels,startSample,nSamples,'int16');
if isempty(data)
    cm = zeros(0,1);
    return;
end
data = filtfilt(b,a,double(data));
cm = median(data,2);
cm = cm - mean(cm);
end

function [lagSamples,peak,medianAbs,peakRatio,lags,r] = multi_qc_common_mode_lag(cmA,cmB,maxLag)
n = min(numel(cmA),numel(cmB));
if n < 2
    error('Not enough samples to estimate common-mode lag.');
end
cmA = cmA(1:n);
cmB = cmB(1:n);
cmA = cmA - mean(cmA);
cmB = cmB - mean(cmB);
maxLag = min(maxLag,n - 1);
[r,lags] = xcorr(cmA,cmB,maxLag);
[peak,ix] = max(r);
lagSamples = -lags(ix);
medianAbs = median(abs(r));
peakRatio = peak / max(medianAbs,eps);
end

function figFile = save_multi_qc_pair_figure(masterCm,slaveCm,lags,r,chunkRows,opts,masterFolder,slaveFolder,slaveIndex)
masterLabel = multi_qc_folder_label(masterFolder);
slaveLabel = multi_qc_folder_label(slaveFolder);
figFile = fullfile(opts.OutputFolder,sprintf('%s_master_vs_%s_qc.png',opts.OutputPrefix,multi_qc_safe_filename(slaveLabel)));
fig = figure('Visible','off','Color','w','Position',[100 100 1100 780]);
subplot(3,1,1);
plot_multi_qc_decimated((0:(numel(masterCm)-1)),masterCm,20000);
hold on;
plot_multi_qc_decimated((0:(numel(slaveCm)-1)),slaveCm,20000);
title(sprintf('Initial common mode: master %s vs slave %s',masterLabel,slaveLabel),'Interpreter','none');
legend({'master','slave'},'Interpreter','none');
axis tight;
subplot(3,1,2);
plot(lags,r);
title('Initial xcorr');
xlabel('lag samples');
axis tight;
subplot(3,1,3);
if isempty(chunkRows)
    text(0.1,0.5,'No valid chunks','Units','normalized');
else
    yyaxis left;
    plot(chunkRows(:,1),chunkRows(:,2),'-o');
    ylabel('offset samples');
    yyaxis right;
    plot(chunkRows(:,1),chunkRows(:,4),'-');
    ylabel('peak ratio');
    xlabel('master time sec');
end
title(sprintf('Chunk-wise offset trajectory, slave %d',slaveIndex));
saveas(fig,figFile);
close(fig);
end

function plot_multi_qc_decimated(x,y,maxPoints)
if numel(y) > maxPoints
    step = ceil(numel(y) / maxPoints);
    x = x(1:step:end);
    y = y(1:step:end);
end
plot(x,y);
end

function frac = valid_multi_qc_fraction(startSample,nSamples,totalSamples)
readStart = round(startSample);
readEnd = readStart + round(nSamples) - 1;
sourceStart = max(0,readStart);
sourceEnd = min(totalSamples - 1,readEnd);
if sourceEnd < sourceStart
    frac = 0;
else
    frac = (sourceEnd - sourceStart + 1) / nSamples;
end
end

function [sysParams,Nch,fsValues,nSamples] = read_multi_qc_headers(folders,files)
sysParams = cell(numel(folders),1);
Nch = zeros(numel(folders),1);
fsValues = zeros(numel(folders),1);
nSamples = zeros(numel(folders),1);
for idx = 1:numel(folders)
    paramFile = fullfile(folders{idx},'CE_params.bin');
    if ~exist(paramFile,'file')
        error('Missing CE_params.bin for %s',folders{idx});
    end
    sysParams{idx} = WILD_ReadHeader(paramFile);
    Nch(idx) = sysParams{idx}.Nch;
    fsValues(idx) = sysParams{idx}.fs;
    info = dir(files{idx});
    nSamples(idx) = floor(info.bytes / 2 / Nch(idx));
end
end

function validate_multi_qc_inputs(Nch,fsValues)
if any(Nch ~= 64)
    error('WILD_channelMerger multi-logger sync QC currently supports 64-channel logger recordings. Found: %s',num2str(Nch(:)'));
end
if any(fsValues ~= fsValues(1))
    error('All loggers must use the same sampling rate. Found: %s',num2str(fsValues(:)'));
end
if fsValues(1) <= 0
    error('Invalid sampling rate: %g',fsValues(1));
end
end

function write_multi_qc_tsv(filename,result)
fh = fopen(filename,'w+');
if fh == -1
    error('Could not write QC TSV: %s',filename);
end
cleanupObj = onCleanup(@() fclose(fh));
fprintf(fh,'status\tmaster_index\tslave_index\tmaster_folder\tslave_folder\tinitial_offset_samples\tfinal_offset_samples\tinitial_peak_ratio\tchunk_count\tmin_chunk_peak_ratio\tmax_abs_lag_step_samples\toffset_drift_samples\tfigure_file\tmessage\n');
for idx = 1:numel(result.pairs)
    pair = result.pairs(idx);
    if isempty(pair.chunks)
        minChunkPeakRatio = NaN;
        maxAbsLagStep = NaN;
        driftSamples = NaN;
    else
        minChunkPeakRatio = min(pair.chunks(:,4));
        maxAbsLagStep = max(abs(pair.chunks(:,3)));
        driftSamples = pair.chunks(end,2) - pair.chunks(1,2);
    end
    fprintf(fh,'%s\t%d\t%d\t%s\t%s\t%.0f\t%.0f\t%.6g\t%d\t%.6g\t%.0f\t%.0f\t%s\t%s\n', ...
        pair.status,pair.masterIndex,pair.slaveIndex,pair.masterFolder,pair.slaveFolder, ...
        pair.initialOffsetSamples,pair.finalOffsetSamples,pair.initialPeakRatio,size(pair.chunks,1), ...
        minChunkPeakRatio,maxAbsLagStep,driftSamples,pair.figureFile,pair.message);
end
clear cleanupObj;
end

function mergeInfo = write_multi_logger_merged_streams(files,folders,Nch,nSamples,fs,pairResults,opts,overwrite)
outAmp = fullfile(opts.OutputFolder,'amplifier.dat');
outAnalog = fullfile(opts.OutputFolder,'analogin.dat');
outTime = fullfile(opts.OutputFolder,'time.dat');
outLayout = fullfile(opts.OutputFolder,'wild_preprocess_channel_layout.tsv');
outMergeInfo = fullfile(opts.OutputFolder,'wild_multilogger_mergeInfo.mat');
outIMU = fullfile(opts.OutputFolder,'IMU.mat');
outEventSummary = fullfile(opts.OutputFolder,'wild_multilogger_events.tsv');
if ~overwrite
    existing = {outAmp,outAnalog,outTime,outLayout,outMergeInfo,outIMU,outEventSummary};
    exists = existing(cellfun(@(x) exist(x,'file') ~= 0,existing));
    if ~isempty(exists)
        error('Merged output exists. Set overwrite=1 to regenerate: %s',strjoin(exists,', '));
    end
end

[commonStart,commonEnd] = multi_merge_common_interval(nSamples,pairResults,opts.MasterIndex);
nOut = commonEnd - commonStart;
if nOut <= 0
    error('No common valid sample interval for multi-logger merge.');
end

fprintf('Writing multi-logger amplifier.dat\n');
fprintf('  Output: %s\n',outAmp);
fprintf('  Common master samples: %d to %d (%d samples)\n',commonStart,commonEnd - 1,nOut);

chunkSamples = round(opts.ChunkSeconds * fs);
totalChannels = sum(Nch);
fh = fopen(outAmp,'w+');
if fh == -1
    error('Could not open output amplifier.dat: %s',outAmp);
end
cleanupAmp = onCleanup(@() fclose(fh));
ptr = 0;
while ptr < nOut
    curN = min(chunkSamples,nOut - ptr);
    masterSample0 = commonStart + ptr;
    chunkData = zeros(curN,totalChannels,'int16');
    chPtr = 1;
    for idx = 1:numel(files)
        offset = multi_merge_offset_for_device(idx,masterSample0,fs,pairResults,opts.MasterIndex);
        data = read_chunk_padded(files{idx},Nch(idx),1:Nch(idx),masterSample0 + offset,curN,'int16');
        chunkData(:,chPtr:(chPtr+Nch(idx)-1)) = data;
        chPtr = chPtr + Nch(idx);
    end
    fwrite(fh,chunkData','int16');
    ptr = ptr + curN;
    fprintf('WILD_PROGRESS:write_amplifier:%.3f\n',100 * ptr / nOut);
end
clear cleanupAmp;

analogInfo = write_multi_logger_analog(folders,Nch,fs,commonStart,nOut,pairResults,opts,overwrite,outAnalog);
write_multi_logger_time_dat(outTime,nOut,overwrite);
write_multi_logger_layout(outLayout,folders,Nch,overwrite);
eventInfo = write_multi_logger_events(outAnalog,folders,Nch,analogInfo,opts,overwrite,outEventSummary);
imuInfo = write_multi_logger_imu(outAnalog,folders,Nch,analogInfo,fs,commonStart,opts,overwrite,outIMU);

mergeInfo = struct;
mergeInfo.mode = 'multiMerge';
mergeInfo.files = files;
mergeInfo.folders = folders;
mergeInfo.masterIndex = opts.MasterIndex;
mergeInfo.fs = fs;
mergeInfo.nChannels = totalChannels;
mergeInfo.nSamples = nOut;
mergeInfo.commonStartSample = commonStart;
mergeInfo.commonEndSample = commonEnd - 1;
mergeInfo.amplifierFile = outAmp;
mergeInfo.analogFile = outAnalog;
mergeInfo.timeFile = outTime;
mergeInfo.layoutFile = outLayout;
mergeInfo.analog = analogInfo;
mergeInfo.events = eventInfo;
mergeInfo.imu = imuInfo;
mergeInfo.pairs = pairResults;
save(outMergeInfo,'mergeInfo');
fprintf('Saved merge info: %s\n',outMergeInfo);
fprintf('WILD_PROGRESS:multi_merge_complete:100\n');
end

function analogInfo = write_multi_logger_analog(folders,Nch,fs,commonStart,nOut,pairResults,opts,overwrite,outAnalog)
fsAnalog = 1250;
resRate = fs / fsAnalog;
nAnalog = Nch ./ 4;
analogFiles = cellfun(@(x) fullfile(x,'analogin.dat'),folders,'UniformOutput',false);
for idx = 1:numel(analogFiles)
    if ~exist(analogFiles{idx},'file')
        error('Missing analogin.dat for multi-logger merge: %s',analogFiles{idx});
    end
end
if exist(outAnalog,'file') && ~overwrite
    error('Merged analogin.dat exists. Set overwrite=1 to regenerate: %s',outAnalog);
end
nOutAnalog = floor(nOut / resRate);
commonStartAnalog = floor(commonStart / resRate);
chunkAnalog = max(1,round(opts.ChunkSeconds * fsAnalog));
totalAnalogChannels = sum(nAnalog);
fprintf('Writing multi-logger analogin.dat\n');
fprintf('  Output: %s\n',outAnalog);
fh = fopen(outAnalog,'w+');
if fh == -1
    error('Could not open output analogin.dat: %s',outAnalog);
end
cleanupAnalog = onCleanup(@() fclose(fh));
ptr = 0;
while ptr < nOutAnalog
    curN = min(chunkAnalog,nOutAnalog - ptr);
    masterAnalog0 = commonStartAnalog + ptr;
    masterSample0 = round(masterAnalog0 * resRate);
    chunkData = zeros(curN,totalAnalogChannels,'int16');
    chPtr = 1;
    for idx = 1:numel(analogFiles)
        offset = multi_merge_offset_for_device(idx,masterSample0,fs,pairResults,opts.MasterIndex);
        analogOffset = round(offset / resRate);
        data = read_chunk_padded(analogFiles{idx},nAnalog(idx),1:nAnalog(idx),masterAnalog0 + analogOffset,curN,'int16');
        chunkData(:,chPtr:(chPtr+nAnalog(idx)-1)) = data;
        chPtr = chPtr + nAnalog(idx);
    end
    fwrite(fh,chunkData','int16');
    ptr = ptr + curN;
    fprintf('WILD_PROGRESS:write_analog:%.3f\n',100 * ptr / max(nOutAnalog,1));
end
clear cleanupAnalog;
analogInfo = struct;
analogInfo.file = outAnalog;
analogInfo.fs = fsAnalog;
analogInfo.nSamples = nOutAnalog;
analogInfo.nChannels = totalAnalogChannels;
analogInfo.nChannelsPerDevice = nAnalog;
end

function eventInfo = write_multi_logger_events(analogFile,folders,Nch,analogInfo,opts,overwrite,summaryFile)
fprintf('Extracting multi-logger digital events\n');
fsAnalog = analogInfo.fs;
nAnalog = analogInfo.nChannelsPerDevice;
totalAnalogChannels = analogInfo.nChannels;
eventInfo = repmat(struct( ...
    'deviceIndex',[], ...
    'deviceName','', ...
    'recordingName','', ...
    'digitalChannel',[], ...
    'mergedAnalogChannel',[], ...
    'eventFiles',{{}}, ...
    'eventCounts',[]),numel(folders),1);

summaryRows = {};
for idx = 1:numel(folders)
    [deviceName,recordingName] = multi_logger_device_labels(folders{idx});
    blockStart = sum(nAnalog(1:(idx-1)));
    digitalCh = blockStart + 1;
    dig = readmulti_frank(analogFile,totalAnalogChannels,digitalCh,0,inf,'uint16');
    digExpanded = dec2digits(dig,16);
    eventFiles = {};
    eventCounts = zeros(16,1);
    for bitIdx = 1:size(digExpanded,2)
        digDiff = diff([0 digExpanded(:,bitIdx)' 0]);
        triggerStarts = find(digDiff == 1) / fsAnalog;
        triggerEnds = find(digDiff == -1) / fsAnalog;
        nEvents = min(numel(triggerStarts),numel(triggerEnds));
        if nEvents > 1
            triggerStarts = triggerStarts(1:nEvents);
            triggerEnds = triggerEnds(1:nEvents);
            evtName = fullfile(opts.OutputFolder,sprintf('device_event.dev%02d.d%02d.evt',idx,bitIdx));
            if exist(evtName,'file') && ~overwrite
                error('Event file exists. Set overwrite=1 to regenerate: %s',evtName);
            end
            events.description = cell(nEvents,2);
            for eventIdx = 1:nEvents
                events.description{eventIdx,1} = sprintf('%s DigitIn start %d',deviceName,bitIdx);
                events.description{eventIdx,2} = sprintf('%s DigitIn end %d',deviceName,bitIdx);
            end
            events.description = reshape(events.description',1,[]);
            events.time = reshape([triggerStarts(:)' ; triggerEnds(:)'],1,[]);
            if exist(evtName,'file')
                delete(evtName);
            end
            SaveEvents(evtName,events);
            eventFiles{end+1,1} = evtName; %#ok<AGROW>
            eventCounts(bitIdx) = nEvents;
            summaryRows(end+1,:) = {idx,deviceName,recordingName,bitIdx,digitalCh,nEvents,evtName}; %#ok<AGROW>
        else
            summaryRows(end+1,:) = {idx,deviceName,recordingName,bitIdx,digitalCh,nEvents,''}; %#ok<AGROW>
        end
    end
    eventInfo(idx).deviceIndex = idx;
    eventInfo(idx).deviceName = deviceName;
    eventInfo(idx).recordingName = recordingName;
    eventInfo(idx).digitalChannel = 1;
    eventInfo(idx).mergedAnalogChannel = digitalCh;
    eventInfo(idx).eventFiles = eventFiles;
    eventInfo(idx).eventCounts = eventCounts;
end
write_multi_logger_event_summary(summaryFile,summaryRows,overwrite);
fprintf('Saved event summary: %s\n',summaryFile);
end

function imuInfo = write_multi_logger_imu(analogFile,folders,Nch,analogInfo,ephysFs,commonStart,opts,overwrite,outIMU)
if exist(outIMU,'file') && ~overwrite
    error('IMU.mat exists. Set overwrite=1 to regenerate: %s',outIMU);
end
fprintf('Processing multi-logger IMU\n');
fsAnalog = analogInfo.fs;
resFs = 100;
nAnalog = analogInfo.nChannelsPerDevice;
totalAnalogChannels = analogInfo.nChannels;

IMU = struct;
IMU.createdAt = char(datetime('now'));
IMU.sourceAnalogFile = analogFile;
IMU.masterIndex = opts.MasterIndex;
IMU.fs_raw = fsAnalog;
IMU.fs = resFs;
IMU.masterStartSample = commonStart;
IMU.masterStartSec = commonStart / ephysFs;
IMU.time = [];
IMU.device = repmat(struct( ...
    'deviceIndex',[], ...
    'deviceName','', ...
    'recordingName','', ...
    'sourceFolder','', ...
    'analogChannelsOriginal',[], ...
    'analogChannelsMerged',[], ...
    'timestamp',[], ...
    'rawResampled',[], ...
    'imu',[], ...
    'fusionData',[], ...
    'status',''),numel(folders),1);

for idx = 1:numel(folders)
    [deviceName,recordingName] = multi_logger_device_labels(folders{idx});
    blockStart = sum(nAnalog(1:(idx-1)));
    imuChannelsMerged = blockStart + (2:10);
    dataRaw = readmulti_frank(analogFile,totalAnalogChannels,imuChannelsMerged,0,inf,'int16');
    dataResampled = resample(dataRaw,resFs,fsAnalog);
    timestamp = ((1:size(dataResampled,1)) - 1) / resFs;
    [scaledData,imu,fusionData] = WILD_scaleIMU(dataResampled,resFs,1,0);
    fusionData.timestamp = timestamp;
    fusionData.deviceIndex = idx;
    fusionData.deviceName = deviceName;
    fusionData.recordingName = recordingName;
    fusionData.sourceFolder = folders{idx};
    fusionData.analogChannelsOriginal = 2:10;
    fusionData.analogChannelsMerged = imuChannelsMerged;
    IMU.device(idx).deviceIndex = idx;
    IMU.device(idx).deviceName = deviceName;
    IMU.device(idx).recordingName = recordingName;
    IMU.device(idx).sourceFolder = folders{idx};
    IMU.device(idx).analogChannelsOriginal = 2:10;
    IMU.device(idx).analogChannelsMerged = imuChannelsMerged;
    IMU.device(idx).timestamp = timestamp;
    IMU.device(idx).rawResampled = scaledData;
    IMU.device(idx).imu = imu;
    IMU.device(idx).fusionData = fusionData;
    IMU.device(idx).status = 'processed_with_sensor_fusion';
    if isempty(IMU.time)
        IMU.time = timestamp;
    end
    fprintf('  IMU device %d %s: %d samples at %d Hz\n',idx,deviceName,numel(timestamp),resFs);
end
save(outIMU,'IMU','-v7.3');
fprintf('Saved multi-logger IMU: %s\n',outIMU);
imuInfo = struct;
imuInfo.file = outIMU;
imuInfo.fs = resFs;
imuInfo.nDevices = numel(folders);
imuInfo.sourceAnalogFile = analogFile;
end

function [commonStart,commonEnd] = multi_merge_common_interval(nSamples,pairResults,masterIndex)
offsetMin = zeros(numel(nSamples),1);
offsetMax = zeros(numel(nSamples),1);
for idx = 1:numel(pairResults)
    pair = pairResults(idx);
    offsets = [pair.initialOffsetSamples; pair.finalOffsetSamples];
    if ~isempty(pair.chunks)
        offsets = [offsets; pair.chunks(:,2)]; %#ok<AGROW>
    end
    offsetMin(pair.slaveIndex) = min(offsets);
    offsetMax(pair.slaveIndex) = max(offsets);
end
offsetMin(masterIndex) = 0;
offsetMax(masterIndex) = 0;
commonStart = max(0,ceil(max(-offsetMin)));
commonEnd = min(nSamples(:) - ceil(offsetMax(:)));
commonEnd = floor(commonEnd);
end

function offset = multi_merge_offset_for_device(deviceIndex,masterSample,fs,pairResults,masterIndex)
if deviceIndex == masterIndex
    offset = 0;
    return;
end
pairIdx = find([pairResults.slaveIndex] == deviceIndex,1,'first');
if isempty(pairIdx)
    error('Missing pair result for device index %d.',deviceIndex);
end
pair = pairResults(pairIdx);
if isempty(pair.chunks)
    offset = round(pair.initialOffsetSamples);
    return;
end
times = pair.chunks(:,1);
t = masterSample / fs;
idx = find(times <= t,1,'last');
if isempty(idx)
    offset = round(pair.initialOffsetSamples);
else
    offset = round(pair.chunks(idx,2));
end
end

function write_multi_logger_time_dat(filename,nSamples,overwrite)
if exist(filename,'file') && ~overwrite
    error('time.dat exists. Set overwrite=1 to regenerate: %s',filename);
end
fh = fopen(filename,'w+');
if fh == -1
    error('Could not write time.dat: %s',filename);
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
fprintf('Saved time.dat: %s\n',filename);
end

function write_multi_logger_layout(filename,folders,Nch,overwrite)
if exist(filename,'file') && ~overwrite
    error('Channel layout exists. Set overwrite=1 to regenerate: %s',filename);
end
fh = fopen(filename,'w+');
if fh == -1
    error('Could not write channel layout: %s',filename);
end
cleanupObj = onCleanup(@() fclose(fh));
fprintf(fh,'merged_channel\tdevice_index\tdevice_name\trecording_name\tdevice_channel\tdevice_folder\n');
mergedCh = 1;
for idx = 1:numel(folders)
    [parentFolder,recordingName] = fileparts(folders{idx});
    [~,deviceName] = fileparts(parentFolder);
    for ch = 1:Nch(idx)
        fprintf(fh,'%d\t%d\t%s\t%s\t%d\t%s\n',mergedCh,idx,deviceName,recordingName,ch,folders{idx});
        mergedCh = mergedCh + 1;
    end
end
clear cleanupObj;
fprintf('Saved channel layout: %s\n',filename);
end

function write_multi_logger_event_summary(filename,summaryRows,overwrite)
if exist(filename,'file') && ~overwrite
    error('Event summary exists. Set overwrite=1 to regenerate: %s',filename);
end
fh = fopen(filename,'w+');
if fh == -1
    error('Could not write event summary: %s',filename);
end
cleanupObj = onCleanup(@() fclose(fh));
fprintf(fh,'device_index\tdevice_name\trecording_name\tdigital_bit\tmerged_analog_channel\tn_events\tevent_file\n');
for idx = 1:size(summaryRows,1)
    fprintf(fh,'%d\t%s\t%s\t%d\t%d\t%d\t%s\n',summaryRows{idx,:});
end
clear cleanupObj;
end

function [deviceName,recordingName] = multi_logger_device_labels(folder)
[parentFolder,recordingName] = fileparts(folder);
[~,deviceName] = fileparts(parentFolder);
end

function pairResult = empty_multi_qc_pair_result()
pairResult = struct;
pairResult.masterIndex = [];
pairResult.slaveIndex = [];
pairResult.masterFolder = '';
pairResult.slaveFolder = '';
pairResult.masterFile = '';
pairResult.slaveFile = '';
pairResult.initialOffsetSamples = [];
pairResult.initialPeak = [];
pairResult.initialMedianAbs = [];
pairResult.initialPeakRatio = [];
pairResult.finalOffsetSamples = [];
pairResult.status = '';
pairResult.message = '';
pairResult.chunkColumns = {};
pairResult.chunks = [];
pairResult.figureFile = '';
end

function label = multi_qc_folder_label(folder)
[recordingParent,recordingName] = fileparts(folder);
[~,deviceName] = fileparts(recordingParent);
label = [deviceName '_' recordingName];
end

function value = multi_qc_safe_filename(value)
value = regexprep(value,'[^A-Za-z0-9_.-]','_');
end
