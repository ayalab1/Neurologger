function WILD_PreProcess(filename,analogFile,lfp_gen,trig_gen,imu_gen)
if(nargin<1)
    [filename, path] = uigetfile({'amplifier.dat','amplifier.dat'}, 'Select amplifier.dat', 'C:\Users\adaml\Documents\data\wild');
    filename = fullfile(path,"amplifier.dat");
end

[path,f,e]=fileparts(filename);

if nargin<2
    analogFile = 'analogin.dat';
end
if isempty(analogFile)
    analogFile = 'analogin.dat';
end
if nargin<3
    lfp_gen = 1;
end
if nargin<4
    trig_gen = 1;
end
if nargin<5
    imu_gen = 1;
end
[p_analog,fanalog,e]=fileparts(analogFile);
if isempty(p_analog)
    analogFile = fullfile(path,[fanalog,e]);
end
  

analogfile = strrep(filename,'amplifier.dat','analogin.dat');
audiofile = strrep(filename,'amplifier.dat','adc.dat');
rec_info_file=strrep(filename,'amplifier.dat','CE_params.bin');

sysparam = WILD_ReadHeader(fullfile(path,'CE_params.bin'));
% fh=fopen(rec_info_file);
% rec_info=fread(fh,512/4,'unsigned long');
fs_raw=sysparam.fs;
Nch=sysparam.Nch;
if(fs_raw==0)
    fs=1250;
    Nch=2;
else
    fs = fs_raw;
end
miscNch = 8;
fs_res=1250;
if Nch==64
    miscNch = 16;
end
info=dir(filename);
fileBytes=info.bytes;
Nsamples=fileBytes/2/Nch;

%generate time.dat
time_file=fullfile(path,'time.dat');
fh_time=fopen(time_file,'w+');
fwrite(fh_time,0:(Nsamples-1),'int32');
fclose all;

%generate info.rhd
WILD_genIntanHeader(rec_info_file);

%generate amplifier.lfp

%generate concatenated lfp file
if(lfp_gen~=0)
    if fs_raw>0
    
        [b,a]=butter(2,0.5*2/fs,'high');
        lfp_file = fullfile(path,'amplifier.lfp');
        if isempty(dir(lfp_file))
            dat2lfp_frank(filename,b,a);
        end
        ephys = readmulti_frank(lfp_file,Nch,1:Nch,0,inf);
    
    else
        [b,a]=butter(2,0.5*2/fs,'high');
        ephys = readmulti_frank(analogFile,miscNch,5:6,0,inf);
        ephys = filtfilt(b,a,ephys);
    end
    
    if length(ephys)>0
    
        %     ephys = resample(ephys,fs_res,fs);
        misc =  readmulti_frank(analogfile,miscNch,1:miscNch,0,inf,'int16');
    
        len  = min([length(ephys),length(misc)]);
        cat_data = zeros(len,Nch+miscNch);
        cat_data(:,1:Nch) = ephys(1:len,:);
        cat_data(:,Nch+1:end) = misc(1:len,:);
        sav2dat('amplifier_cat.lfp',cat_data);
    end
end

%% convert digitalIn to triggers
if(trig_gen~=0)
    dig = readmulti_frank(analogfile,miscNch,1,0,inf,'uint16');
    
    dig_expanded=dec2digits(dig,16);
    Eventname = 'DigitIn';
    for dch = 1:size(dig_expanded,2)
        dig_diff = diff([0 dig_expanded(:,dch)' 0]);
        trigger_starts = find(dig_diff==1)/fs_res;
        trigger_ends= find(dig_diff==-1)/fs_res;
        
        if(length(trigger_starts)>1)
            evtname = char(strrep(filename,'amplifier.dat',['device_event.d' , sprintf('%02d',dch) , '.evt']));
            events.description = cell(length(trigger_starts),1);
            events.time = reshape([trigger_starts ; trigger_ends],1,[]);
    
            for i = 1:length(trigger_starts)
                events.description{i,1} = [Eventname ' start ' num2str(dch)];
                events.description{i,2} = [Eventname ' end ' num2str(dch)];
            end
            events.description =  reshape(events.description',1,[]);
            if ~isempty(evtname)
                if ~ispc
                    system(['rm ''' evtname '''']);
                else
                    system(['del "' evtname '"']);
                end
            end
    
            SaveEvents(evtname,events);
        end
    end
    
    %% auto correct stim flags
    [sys,dsp]=WILD_ReadHeader(strrep(filename,'amplifier.dat','CE_params.bin'));
    for dch=[1 2 3 4]
        evt_file = strrep(filename,'amplifier.dat',['device_event.d' sprintf('%02d',dch) '.evt']);
        if(~isempty(dir(evt_file)))
            triggers = LoadEvents(evt_file);
            triggers = triggers.time(1:2:end);
    %         stim_on = LoadEvents(strrep(filename,'amplifier.dat','device_event.d14.evt'));
    %         stim_on = stim_on.time(1:2:end);
            % only use triggers after CL ready(last)
            stim_triggers = [];
    %         triggers(triggers<stim_on(end))=[];
            interval = diff([0;triggers]);
            stim_triggers=triggers(interval>(sys.stim_interval(1)/1000));
    %         for stim_idx = stim_on'
    %             triggers(triggers<=stim_idx)=[];
    %             cur_time = 0;
    %             
    %             stim_interval = sys.stim_interval(1)/1000;
    %             for idx=1:length(triggers)
    %                 if(triggers(idx)>=stim_interval+cur_time)
    %                     stim_triggers=[stim_triggers triggers(idx)];
    %                     cur_time = triggers(idx);
    %                 end
    %             end
    %             
    %         end
            Eventname = 'StimOn';
            evtname = strrep(filename,'amplifier.dat',['device_event.s' sprintf('%02d',dch) '.evt']);
            events.description = cell(length(stim_triggers),1);
            events.time = stim_triggers;
            for i = 1:length(stim_triggers)
                events.description{i,1} = [Eventname ' start ' num2str(dch)];
            end
            if(~isempty(dir(evtname)))
                system(['del "' evtname '"'])
            end
            SaveEvents(evtname,events);
        end
    end
end
%% generate a detection only,(no stim) file
try
    det_on = LoadEvents(strrep(filename,'amplifier.dat','device_event.s01.evt'));
    det_on = det_on.time(1:2:end);
    stim_on = LoadEvents(strrep(filename,'amplifier.dat','device_event.s02.evt'));
    stim_on = stim_on.time(1:2:end);
    stim_triggers = det_on(~ismember(det_on,stim_on));
    Eventname = 'Detect';
    evtname = strrep(filename,'amplifier.dat',['device_event.dns.evt']);%detect no stim
    events.description = cell(length(stim_triggers),1);
    events.time = stim_triggers;
    for i = 1:length(stim_triggers)
        events.description{i,1} = [Eventname ' start ' num2str(dch)];
    end
     if(~isempty(dir(evtname)))
            system(['del "' evtname '"'])
        end
    SaveEvents(evtname,events);
end

%% IMU tracking
if(imu_gen~=0)
    WILD_processIMU(analogFile,100);
end
