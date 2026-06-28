function result = WILD_channelMerger(files,overwrite,use_cache)
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

files = resolve_wild_merger_files(files);
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
testdatas=cellfun(@(x) readmulti_frank(x,64,1:64,30*20e3,(120+30)*20e3),files,'uni',0);
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
[r,lags]=xcorr(commonMode(:,1),commonMode(:,2),30*fs);
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
    chunk=fs*5;
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
    [r,lags]=xcorr(commonMode(:,1),commonMode(:,2),chunk);
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
    if exist(cur,'file')
        [~,name,ext] = fileparts(cur);
        if strcmpi([name ext],'amplifier.dat')
            resolved{end+1,1} = cur; %#ok<AGROW>
        else
            error('Expected amplifier.dat file, got: %s',cur);
        end
    elseif exist(cur,'dir')
        directAmp = fullfile(cur,'amplifier.dat');
        if exist(directAmp,'file')
            resolved{end+1,1} = directAmp; %#ok<AGROW>
        else
            found = find_wild_amplifier_files(cur);
            resolved = [resolved; found(:)]; %#ok<AGROW>
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
