classdef WILDPreProcessMultiDelegationTest < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function addCodePath(testCase)
            import matlab.unittest.fixtures.PathFixture

            testDir = fileparts(mfilename('fullpath'));
            repoRoot = fileparts(testDir);
            testCase.applyFixture(PathFixture(fullfile(repoRoot,'Code')));
        end
    end

    methods (Test)
        function channelMergerReceivesOutputOptions(testCase)
            import matlab.unittest.fixtures.TemporaryFolderFixture

            folderFixture = testCase.applyFixture(TemporaryFolderFixture);
            rootFolder = folderFixture.Folder;
            spyFolder = fullfile(rootFolder,'spy');
            mkdir(spyFolder);
            writeSpyChannelMerger(spyFolder);

            addpath(spyFolder,'-begin');
            cleanupPath = onCleanup(@() rmpath(spyFolder));
            clear WILD_channelMerger

            recA = fullfile(rootFolder,'deviceA','0_20260719_120000.000');
            recB = fullfile(rootFolder,'deviceB','0_20260719_120000.000');
            createMinimalRecording(recA);
            createMinimalRecording(recB);

            outputFolder = fullfile(rootFolder,'merged_output');
            result = WILD_PreProcess_Multi( ...
                {recA,recB}, ...
                'MasterIndex',1, ...
                'OutputFolder',outputFolder, ...
                'OutputPrefix','delegated', ...
                'Overwrite',true, ...
                'AutoDiscoverDevices',false, ...
                'ChunkSeconds',2.5);

            spyFile = fullfile(outputFolder,'spy_channel_merger_args.mat');
            testCase.verifyTrue(isfile(spyFile));
            spy = load(spyFile,'result');

            testCase.verifyEqual(result.outputFolder,outputFolder);
            testCase.verifyEqual(spy.result.options.MasterIndex,1);
            testCase.verifyEqual(spy.result.options.OutputFolder,outputFolder);
            testCase.verifyEqual(spy.result.options.OutputPrefix,'delegated');
            testCase.verifyEqual(spy.result.options.ChunkSeconds,2.5);

            clear cleanupPath
        end
    end
end

function writeSpyChannelMerger(folder)
lines = {
    'function result = WILD_channelMerger(files,overwrite,use_cache,varargin)'
    'p = inputParser;'
    'addParameter(p,''MasterIndex'',1);'
    'addParameter(p,''OutputFolder'','''');'
    'addParameter(p,''OutputPrefix'','''');'
    'addParameter(p,''ChunkSeconds'',5);'
    'parse(p,varargin{:});'
    'result = struct;'
    'result.files = files;'
    'result.overwrite = overwrite;'
    'result.use_cache = use_cache;'
    'result.options = p.Results;'
    'save(fullfile(p.Results.OutputFolder,''spy_channel_merger_args.mat''),''result'',''varargin'');'
    'end'
    };

filename = fullfile(folder,'WILD_channelMerger.m');
fh = fopen(filename,'w+');
if fh == -1
    error('Could not create spy WILD_channelMerger: %s',filename);
end
cleanup = onCleanup(@() fclose(fh));
fprintf(fh,'%s\n',lines{:});
clear cleanup
end

function createMinimalRecording(folder)
if ~exist(folder,'dir')
    mkdir(folder);
end
writeCeParams(fullfile(folder,'CE_params.bin'),1250,64);
writeInterleavedInt16(fullfile(folder,'amplifier.dat'),zeros(16,64,'int16'));
end

function writeCeParams(filename,fs,nChannels)
fh = fopen(filename,'w+');
if fh == -1
    error('Could not create CE_params.bin: %s',filename);
end
cleanup = onCleanup(@() fclose(fh));
fwrite(fh,zeros(512,1,'uint8'),'uint8');
fseek(fh,0,'bof');
fwrite(fh,uint32(fs),'uint32');
fseek(fh,8,'bof');
fwrite(fh,uint32(nChannels),'uint32');
fseek(fh,336,'bof');
fwrite(fh,uint8([1 1 19 26]),'uint8');
fseek(fh,356,'bof');
fwrite(fh,uint8([12 0 0 0]),'uint8');
clear cleanup
end

function writeInterleavedInt16(filename,data)
fh = fopen(filename,'w+');
if fh == -1
    error('Could not create dat file: %s',filename);
end
cleanup = onCleanup(@() fclose(fh));
fwrite(fh,data','int16');
clear cleanup
end
