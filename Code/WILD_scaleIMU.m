function [data,imu, fusionData] = WILD_scaleIMU(data,fs,calibration,disp_on)
imu=[];
fusionData=[];
if(size(data,2)==3)
    type='3axis';
else
    type='9axis';
end
if strcmp(type,'9axis')
    data(:,1:3) = data(:,1:3)/32768*8*9.8; %convert unit in m/s^2
    data(:,4:6) = data(:,4:6)/32768*2000/180*pi; %convert unit in rad/s
    %      data(:,7:9)=data(:,7:9)/16;
    data(:,7:9) = bsxfun(@times,data(:,7:9)/32768,[1150 1150 2500]); %unit in uT
    imu.acc=data(:,1:3);
    imu.gyr=data(:,4:6);
    imu.mag = data(:,7:9);
    %disp data info
    acc_mean = median(imu.acc,1);
    disp("Acceleration average(m/S^2):"+num2str(acc_mean));
    
    if(calibration)
        disp('Running sensor calibration...')
        disp("Raw Magnetometer Data (mean ± std):");
        disp(mean(imu.mag) + " ± " + std(imu.mag));

        acc_1d = sqrt(sum(imu.acc.^2,2));
        imu.acc = imu.acc./median(acc_1d)*9.81;
        gyroBias = median(imu.gyr);
        imu.gyr = imu.gyr - gyroBias;
       % [A, b,expmfs ] = magcal(imu.mag );
        %disp("Extected magnetic field"+num2str(expmfs));
%         imu.mag  = (imu.mag -b)*A; % calibrated data
        disp("Calibrated Magnetometer Data (mean ± std):");
        disp(mean(imu.mag) + " ± " + std(imu.mag));

    end
    if nargout>2 %perform  sensor fusion
        if exist('ahrsfilter','file') == 0
            error('WILD:MissingSensorFusionToolbox', ...
                ['ahrsfilter is unavailable. IMU sensor fusion cannot be processed. ', ...
                'Install Sensor Fusion and Tracking Toolbox or Navigation Toolbox to compute orientation.']);
        end
        fusionData.imu = imu;
        fusionData.fs = fs;
        disp('Running sensor fusion...')
        % kF=imufilter('SampleRate',fs);
        % quaternionData  = kF(accelData,gyroData);
        kF=ahrsfilter('SampleRate',fs);
        fusionData.quaternion  = kF(imu.acc,imu.gyr,imu.mag);
        fusionData.orientation= quat2rotm(fusionData.quaternion);
        numFrames = size(fusionData.orientation, 3);
        accel_world=zeros(numFrames,3);
        for i = 1:numFrames
            accel_world(i, :) = (fusionData.orientation(:,:,i) * imu.acc(i, :)')'; % Transform to world frame
        end
        accel_world=accel_world-median(accel_world,1);
        speed_world=cumsum(accel_world./fs);
        [b,a]=butter(2,0.1*2/fs,'high');
        speed_world = filtfilt(b,a,speed_world);
        fusionData.accel = accel_world;
        fusionData.speed = speed_world;
        fusionData.fs = fs;
    end
    M=9;N=1;
    if(disp_on)
        figure(1)
        
        
        for idx=1:3
            subplot(M,N,1+(idx-1)*N)
            histogram(data(:,idx));hold on;
            histogram(imu.acc(:,idx));hold off;
            
        end
        
        for idx=1:3
            subplot(M,N,4+(idx-1)*N)
            histogram(data(:,idx+3));hold on;
            histogram(imu.gyr(:,idx));hold off;
        end
        
        for idx=1:3
            subplot(M,N,7+(idx-1)*N)
            histogram(data(:,idx+6));hold on;
            histogram(imu.mag(:,idx));hold off;
        end
        %             subplot(M,N,3)
        %             plot3(data(:,7),data(:,8),data(:,9),'LineStyle','none','Marker','X','MarkerSize',8)
        %             hold on
        %             grid(gca,'on')
        %             plot3(imu.mag(:,1),imu.mag(:,2),imu.mag(:,3),'LineStyle','none','Marker', 'o','MarkerSize',8,'MarkerFaceColor','r')
        %             xlabel('uT')
        %             ylabel('uT')
        %             zlabel('uT')
        %             legend('Uncalibrated Samples', 'Calibrated Samples','Location', 'southoutside')
        %             title('Magnetometer');
        
    end
end

if strcmp(type,'3axis')
    data(:,1:3)= data(:,1:3)/65535*2;
    imu.acc=data(:,1:3);
    %disp data info
    acc_mean = median(imu.acc,1);
    disp("Acceleration average(m/S^2):"+num2str(acc_mean));
    
end


