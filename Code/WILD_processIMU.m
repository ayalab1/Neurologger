function WILD_processIMU(acc_file,res_fs,overwrite)

if (nargin<1 || isempty(acc_file))
    acc_file=[pwd '\analogin.dat'];
end
if nargin<2
    res_fs=100;
end
if nargin<3
    overwrite=0;
end
[pth,~] = fileparts(acc_file);
imu_file = fullfile(pth,'IMU.mat');

fs=1250;
data_raw = readmulti_frank(acc_file,16,2:10,0,inf);
data_raw = resample(data_raw,res_fs,fs);
fs=res_fs;
timestamp = ((1:length(data_raw))-1)/fs;
if(isempty(dir(imu_file))||(overwrite~=0))
    [~,~,fusionData] = WILD_scaleIMU(data_raw,fs,1,1);
    fusionData.timestamp = timestamp;
    save(imu_file,'fusionData');
else
    load(imu_file);
    disp("Found processed IMU data");
end
disp("IMU data loaded, samples:"+length(data_raw));

end
