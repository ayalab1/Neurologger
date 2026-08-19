function outputFile = WILD_generateIMUReference(analogFile,outputFile,totalDeviceCount,deviceIndices,varargin)
%WILD_GENERATEIMUREFERENCE Generate a legacy-MATLAB IMU reference product.
%
% outputFile = WILD_generateIMUReference(analogFile,outputFile, ...
%     totalDeviceCount,deviceIndices)
% outputFile = WILD_generateIMUReference(...,'Overwrite',true)
%
% analogFile is a merged WILD analogin.dat whose device blocks are in input
% recording order, with 16 int16 channels per device. deviceIndices selects
% one-based device blocks. Each selected block is processed exactly through
% the legacy chain used by WILD_channelMerger: channels 2:10, resample from
% 1250 Hz to 100 Hz, WILD_scaleIMU calibration, and ahrsfilter fusion.
%
% The output contains only comparison-friendly numeric arrays and metadata.
% It must be outside the source analog directory tree. The source file is
% opened read-only and is never modified. Existing output is preserved unless
% 'Overwrite' is explicitly true.

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser,'Overwrite',false,@(value) islogical(value) && isscalar(value));
parse(parser,varargin{:});
overwrite = parser.Results.Overwrite;

analogFile = localCanonicalPath(analogFile);
outputFile = localCanonicalPath(outputFile);
if ~isfile(analogFile)
    error('WILD:ReferenceAnalogMissing','Merged analog input does not exist: %s',analogFile);
end
if ~endsWith(lower(analogFile),[filesep 'analogin.dat']) && ...
        ~strcmpi(localFileName(analogFile),'analogin.dat')
    warning('WILD:ReferenceAnalogName', ...
        'Input is not named analogin.dat; interpreting it as the merged analog stream: %s',analogFile);
end
if ~(isnumeric(totalDeviceCount) && isscalar(totalDeviceCount) && ...
        isfinite(totalDeviceCount) && totalDeviceCount == fix(totalDeviceCount) && totalDeviceCount > 0)
    error('WILD:ReferenceDeviceCount','totalDeviceCount must be a positive integer.');
end
if ~(isnumeric(deviceIndices) && isvector(deviceIndices) && ~isempty(deviceIndices) && ...
        all(isfinite(deviceIndices)) && all(deviceIndices == fix(deviceIndices)) && ...
        all(deviceIndices >= 1) && all(deviceIndices <= totalDeviceCount) && ...
        numel(unique(deviceIndices)) == numel(deviceIndices))
    error('WILD:ReferenceDeviceIndices', ...
        'deviceIndices must contain unique one-based indices within totalDeviceCount.');
end
deviceIndices = reshape(double(deviceIndices),1,[]);

sourceDirectory = localCanonicalPath(fileparts(analogFile));
if localIsWithin(outputFile,sourceDirectory)
    error('WILD:ReferenceSourceWrite', ...
        'Reference output must be outside the source analog directory tree: %s',outputFile);
end
if strcmpi(outputFile,analogFile)
    error('WILD:ReferenceSourceWrite','Reference output cannot replace the source analog file.');
end
if isfile(outputFile) && ~overwrite
    error('WILD:ReferenceOutputExists', ...
        'Reference output exists. Pass ''Overwrite'',true to replace it: %s',outputFile);
end
outputDirectory = fileparts(outputFile);
if ~isfolder(outputDirectory)
    [created,message] = mkdir(outputDirectory);
    if ~created
        error('WILD:ReferenceOutputDirectory', ...
            'Could not create output directory %s: %s',outputDirectory,message);
    end
end

requiredFunctions = {'readmulti_frank','resample','WILD_scaleIMU','ahrsfilter','quat2rotm'};
for index = 1:numel(requiredFunctions)
    if exist(requiredFunctions{index},'file') == 0
        error('WILD:ReferenceDependencyMissing', ...
            'Required legacy IMU dependency is unavailable: %s',requiredFunctions{index});
    end
end

channelsPerDevice = 16;
totalChannels = channelsPerDevice * totalDeviceCount;
fileInfo = dir(analogFile);
frameBytes = totalChannels * 2;
if fileInfo.bytes == 0 || mod(fileInfo.bytes,frameBytes) ~= 0
    error('WILD:ReferenceAnalogStructure', ...
        'Merged analog byte length is not divisible by %d-byte frames: %s',frameBytes,analogFile);
end
canonicalRows = fileInfo.bytes / frameBytes;

fsRaw = 1250;
fs = 100;
reference = struct;
reference.formatVersion = 1;
reference.sourceAnalogFile = analogFile;
reference.sourceBytes = fileInfo.bytes;
reference.sourceRows = canonicalRows;
reference.totalDeviceCount = totalDeviceCount;
reference.deviceIndices = deviceIndices;
reference.fs_raw = fsRaw;
reference.fs = fs;
reference.matlabVersion = version;
reference.matlabRelease = version('-release');
reference.processing = 'readmulti_frank -> resample -> WILD_scaleIMU(calibration=1) -> ahrsfilter';
[~,resampleFilter] = resample(zeros(1,1),fs,fsRaw);
reference.resample = struct( ...
    'numerator',2, ...
    'denominator',25, ...
    'filter',double(resampleFilter(:)));
filterDefaults = ahrsfilter('SampleRate',fs);
reference.ahrs = struct( ...
    'class','ahrsfilter', ...
    'ReferenceFrame','NED', ...
    'SampleRate',filterDefaults.SampleRate, ...
    'DecimationFactor',filterDefaults.DecimationFactor, ...
    'AccelerometerNoise',filterDefaults.AccelerometerNoise, ...
    'GyroscopeNoise',filterDefaults.GyroscopeNoise, ...
    'MagnetometerNoise',filterDefaults.MagnetometerNoise, ...
    'GyroscopeDriftNoise',filterDefaults.GyroscopeDriftNoise, ...
    'LinearAccelerationNoise',filterDefaults.LinearAccelerationNoise, ...
    'LinearAccelerationDecayFactor',filterDefaults.LinearAccelerationDecayFactor, ...
    'MagneticDisturbanceNoise',filterDefaults.MagneticDisturbanceNoise, ...
    'MagneticDisturbanceDecayFactor',filterDefaults.MagneticDisturbanceDecayFactor, ...
    'ExpectedMagneticFieldStrength',filterDefaults.ExpectedMagneticFieldStrength, ...
    'InitialProcessNoise',double(filterDefaults.InitialProcessNoise));
reference.device = repmat(struct( ...
    'deviceIndex',[], ...
    'analogChannelsOriginal',[], ...
    'analogChannelsMerged',[], ...
    'timestamp',[], ...
    'resampledAdc',[], ...
    'scaledData',[], ...
    'acc',[], ...
    'gyr',[], ...
    'mag',[], ...
    'quaternion',[], ...
    'orientation',[], ...
    'accel',[], ...
    'speed',[]),numel(deviceIndices),1);

for outputIndex = 1:numel(deviceIndices)
    deviceIndex = deviceIndices(outputIndex);
    blockStart = (deviceIndex - 1) * channelsPerDevice;
    mergedChannels = blockStart + (2:10);
    raw = readmulti_frank(analogFile,totalChannels,mergedChannels,0,inf,'int16');
    resampledAdc = resample(raw,fs,fsRaw);
    [scaledData,imu,fusionData] = WILD_scaleIMU(resampledAdc,fs,1,0);
    timestamp = ((1:size(resampledAdc,1)) - 1)' / fs;
    quaternionNumeric = localQuaternionNumeric(fusionData.quaternion);

    reference.device(outputIndex).deviceIndex = deviceIndex;
    reference.device(outputIndex).analogChannelsOriginal = 2:10;
    reference.device(outputIndex).analogChannelsMerged = mergedChannels;
    reference.device(outputIndex).timestamp = timestamp;
    reference.device(outputIndex).resampledAdc = double(resampledAdc);
    reference.device(outputIndex).scaledData = double(scaledData);
    reference.device(outputIndex).acc = double(imu.acc);
    reference.device(outputIndex).gyr = double(imu.gyr);
    reference.device(outputIndex).mag = double(imu.mag);
    reference.device(outputIndex).quaternion = quaternionNumeric;
    reference.device(outputIndex).orientation = double(fusionData.orientation);
    reference.device(outputIndex).accel = double(fusionData.accel);
    reference.device(outputIndex).speed = double(fusionData.speed);
end

temporaryFile = [tempname(outputDirectory) '.mat'];
cleanup = onCleanup(@() localDeleteIfPresent(temporaryFile));
save(temporaryFile,'reference','-v7.3');
if isfile(outputFile) && overwrite
    [moved,message] = movefile(temporaryFile,outputFile,'f');
else
    [moved,message] = movefile(temporaryFile,outputFile);
end
if ~moved
    error('WILD:ReferenceOutputWrite','Could not publish reference output: %s',message);
end
fprintf('Saved MATLAB IMU reference: %s\n',outputFile);
end

function numeric = localQuaternionNumeric(value)
if isnumeric(value)
    numeric = double(value);
elseif exist('compact','file') ~= 0 || ismethod(value,'compact')
    numeric = double(compact(value));
else
    error('WILD:ReferenceQuaternion', ...
        'Could not convert ahrsfilter quaternion output to a numeric N-by-4 array.');
end
end

function result = localCanonicalPath(pathValue)
if ~(ischar(pathValue) || (isstring(pathValue) && isscalar(pathValue)))
    error('WILD:ReferencePath','Input and output paths must be text scalars.');
end
result = char(java.io.File(char(pathValue)).getCanonicalPath());
end

function name = localFileName(pathValue)
[~,base,extension] = fileparts(pathValue);
name = [base extension];
end

function inside = localIsWithin(candidate,parent)
candidate = lower(char(candidate));
parent = lower(char(parent));
if ~endsWith(parent,filesep)
    parent = [parent filesep];
end
inside = startsWith(candidate,parent);
end

function localDeleteIfPresent(pathValue)
if isfile(pathValue)
    delete(pathValue);
end
end
