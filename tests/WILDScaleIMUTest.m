classdef WILDScaleIMUTest < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function addCodePath(testCase)
            import matlab.unittest.fixtures.PathFixture

            testDir = fileparts(mfilename('fullpath'));
            repoRoot = fileparts(testDir);
            testCase.applyFixture(PathFixture(fullfile(repoRoot,'Code')));
        end
    end

    methods (Test)
        function requestedSensorFusionDoesNotUsePartialFallback(testCase)
            rawImu = zeros(32,9,'double');
            if exist('ahrsfilter','file') == 0
                testCase.verifyError(@() requestFusion(rawImu), ...
                    'WILD:MissingSensorFusionToolbox');
            else
                fusionData = requestFusion(rawImu);
                testCase.verifyTrue(isfield(fusionData,'quaternion'));
                testCase.verifyTrue(isfield(fusionData,'orientation'));
            end
        end
    end
end

function fusionData = requestFusion(rawImu)
[~,~,fusionData] = WILD_scaleIMU(rawImu,100,0,0);
end
