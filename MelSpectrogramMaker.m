close all,clear all

%Required Files
%Matlab audio toolbox
%Signal Processing Toolbox
%Perceptually uniform colormaps Version 1.3.2 by Ander Biguri

%This section muse be changed to your project Directory
currdir='C:\Users\wingn\PycharmProjects\FinalProject';

%Following this UI promt, navigate to the directory where aboave where each
%genres folder is
coordir=uigetdir;
cd(coordir)

cdir=dir(coordir);

%Derives Genre Name from folder, and uses this to rename image file as is
%done it GTZAN


namecell={cdir.name};

for i=3:numel(namecell)
    %Similalry, this code is designed to work under the filestucure as
    %described in the github
    garr=[currdir filesep 'scrape_data' filesep 'downloads' filesep namecell{i}];
    arr=[currdir '\' 'images_original' filesep namecell{i}];
    mkdir(arr)
    cd(garr)
    gdir=dir(garr);
    gnamecell={gdir.name};
    gnamecell=gnamecell(endsWith(gnamecell,'.mp3'));
    
    for p=1:numel(gnamecell)
        cd(garr)
        
        gname=gnamecell{p};
        try
            [audio, fs] = audioread(gname);
            % Convert to mono
            if size(audio,2) > 1
                audio = mean(audio, 2);
            end
            % Parameters for the mel spectrogram
            nfft       = 2048;
            hop        = 512;
            numBands   = 512;
            fmin       = 20;
            targetFs=22050;
            
            audio = resample(audio, targetFs, fs);
            fs = targetFs;
            fmax = fs/2;
            targetSamples = 90 * fs;
            if length(audio) > targetSamples
                audio = audio(1:targetSamples);
            elseif length(audio) < targetSamples
                audio(end+1:targetSamples) = 0;
            end
            % Compute mel spectrogram 
            [S, ~] = melSpectrogram(audio, fs, ...
                "Window", hann(nfft, "periodic"), ...
                "OverlapLength", nfft - hop, ...
                "FFTLength", nfft, ...
                "NumBands", numBands, ...
                "FrequencyRange", [fmin fmax]);
            
            
            % Convert to log scale (dB) 10log10
            Sdb = pow2db(S);
            Sdb=max(min(Sdb,0),-120);


            
            simg=imresize(Sdb,[565,721]); %Resize Image to specs.
            %Current Model architecture takes a [565x725]im which these
            %specs will create

            % Plot result
            %The extra code is is to assist in data being printed to spec.
            %Matlab struggles to print exactyl MxN images without also
            %having error bars
            f=figure('Units','pixels','Position',[100 100 721 565],'Color','none','Visible','on');
            ax=axes('Parent', f, 'Position', [0 0 1 1], ...
          'Color','none');
            imagesc(simg);
            colormap('magma');
            axis(ax,'off')
            ax.Visible='off';
            map=magma(256);
            cd(arr)
            exportgraphics(f,[namecell{i} num2str(p) '.png'],'Resolution',96,'BackgroundColor','none')
            close (f)
        catch
            warning("Skipping corrupted file: %s", gname);
            continue;   %Break Loop to continue through crash on corrupted audio
        end


    end
        

end


% 
% [audio, fs] = audioread(uigetfile('*.mp3'));
% 
% % Convert to mono
% if size(audio,2) > 1
%     audio = mean(audio, 2);
% end
% 
% % Parameters matching your Python example
% nfft       = 2048;
% hop        = 512;
% numBands   = 512;
% fmin       = 20;
% targetFs=22050;
% 
% audio = resample(audio, targetFs, fs);
% fs = targetFs;
% fmax = fs/2;
% 
% 
% 
% 
% 
% targetSamples = 90 * fs;
% if length(audio) > targetSamples
%     audio = audio(1:targetSamples);
% elseif length(audio) < targetSamples
%     audio(end+1:targetSamples) = 0;
% end
% 
% % Compute mel spectrogram (power)
% [S, ~] = melSpectrogram(audio, fs, ...
%     "Window", hann(nfft, "periodic"), ...
%     "OverlapLength", nfft - hop, ...
%     "FFTLength", nfft, ...
%     "NumBands", numBands, ...
%     "FrequencyRange", [fmin fmax]);
% 
% 
% % Convert to log scale (dB)
% Sdb = pow2db(S);
% 
% % Plot
% figure;
% imagesc(Sdb);
% axis xy;
% colormap('magma'); % requires MATLAB's color map or substitute 'parula'
% colorbar;
% xlabel("Time (frames)");
% ylabel("Mel Bands");
% title("Log-Mel Spectrogram");
% 
% 
% simg=imresize(Sdb,[512,512]);
% 
% 
% % Plot result
% figure;
% imagesc(simg);
% axis xy;
% colormap('magma');
% colorbar;
% title("512×512 Log-Mel Spectrogram");
% ylabel("Mel Bands");