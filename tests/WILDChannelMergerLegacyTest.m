classdef WILDChannelMergerLegacyTest < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function addCodePath(testCase)
            import matlab.unittest.fixtures.PathFixture

            testDir = fileparts(mfilename('fullpath'));
            repoRoot = fileparts(testDir);
            testCase.applyFixture(PathFixture(fullfile(repoRoot,'Code')));
        end
    end

    methods (Test)
        function twoFileMergeStillUsesLegacyOutputs(testCase)
            import matlab.unittest.fixtures.TemporaryFolderFixture

            folderFixture = testCase.applyFixture(TemporaryFolderFixture);
            rootFolder = folderFixture.Folder;
            fs = 1250;
            nChannels = 64;
            nAnalogChannels = 16;
            nSamples = 1500;

            recA = fullfile(rootFolder,'deviceA','0_20260719_120000.000');
            recB = fullfile(rootFolder,'deviceB','0_20260719_120000.000');
            createSyntheticRecording(recA,fs,nChannels,nAnalogChannels,nSamples,1);
            createSyntheticRecording(recB,fs,nChannels,nAnalogChannels,nSamples,1);

            previousFigureVisibility = get(0,'DefaultFigureVisible');
            cleanup = onCleanup(@() set(0,'DefaultFigureVisible',previousFigureVisibility));
            set(0,'DefaultFigureVisible','off');

            result = WILD_channelMerger( ...
                {fullfile(recA,'amplifier.dat'),fullfile(recB,'amplifier.dat')}, ...
                1, ...
                0, ...
                'InitialStartSeconds',0, ...
                'InitialDurationSeconds',0.5, ...
                'InitialMaxLagSeconds',0.05, ...
                'ChunkSeconds',0.25, ...
                'ChunkMaxLagSeconds',0.05);

            clear cleanup

            expectedAmplifier = fullfile(recA,'amplifier_merged.dat');
            expectedAnalogMerged = fullfile(recA,'analogin_merged.dat');
            expectedAnalog2 = fullfile(recA,'analogin2.dat');
            expectedMergeInfo = fullfile(recA,'amplifier_merged_mergeInfo.mat');

            testCase.verifyEqual(result.amplifierMerged,expectedAmplifier);
            testCase.verifyEqual(result.analogMerged,expectedAnalogMerged);
            testCase.verifyEqual(result.analog2,expectedAnalog2);
            testCase.verifyEqual(numel(result.files),2);
            testCase.verifyEqual(numel(result.offsets),2);
            testCase.verifyLessThanOrEqual(abs(result.offsets(2)),1);

            assertFileSize(testCase,expectedAmplifier,nSamples * nChannels * 2 * 2);
            assertFileSize(testCase,expectedAnalogMerged,nSamples * nAnalogChannels * 2 * 2);
            assertFileSize(testCase,expectedAnalog2,nSamples * nAnalogChannels * 2);
            testCase.verifyTrue(isfile(expectedMergeInfo));
        end
    end
end

function createSyntheticRecording(folder,fs,nChannels,nAnalogChannels,nSamples,seed)
if ~exist(folder,'dir')
    mkdir(folder);
end
writeCeParams(fullfile(folder,'CE_params.bin'),fs,nChannels);

rng(seed);
commonMode = round(300 * randn(nSamples,1));
channelOffsets = repmat(mod(1:nChannels,17),nSamples,1);
amplifierData = int16(repmat(commonMode,1,nChannels) + channelOffsets);
writeInterleavedInt16(fullfile(folder,'amplifier.dat'),amplifierData);

analogData = zeros(nSamples,nAnalogChannels,'int16');
analogData(:,1) = int16(mod((0:nSamples-1)',128));
writeInterleavedInt16(fullfile(folder,'analogin.dat'),analogData);
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

function assertFileSize(testCase,filename,expectedBytes)
testCase.verifyTrue(isfile(filename),sprintf('Missing expected file: %s',filename));
info = dir(filename);
testCase.verifyEqual(info.bytes,expectedBytes);
end
