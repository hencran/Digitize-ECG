%% Script to digitize signals from a scanned patient monitor strip report
% HTC 
% 05/15/24
clear;clc;close all;
%% Hyperparameters
params.nSignals = 7;
params.zero_pad_size = 10;
params.merge_size = 1;
params.th = 120;
params.mean_window = 5;
%% V2
close all;
% Load an image
image_path = '\\chips.eng.utah.edu\sanchez\Cuffless_BP\LVAD\rawData\11\pre_op_catheter\scans\LVAD_13_pre_op_all_last_page.jpg';
image = imread(image_path);

%Display the image
f = figure;
imshow(image);

% Have user specify the scales for the signal we are working on
% Get X-scale before we define signal's ROI
% ASSUMING WE ONLY NEED TO DO THIS ONCE
disp("Draw line on the horizontal scale of your signal");
title("Draw line on the horizontal scale of your signal");
xscale = drawline();
x1_pixel = xscale.Position(1,1);
x2_pixel = xscale.Position(2,1);

% add annotations for the points we drew
text(x1_pixel, xscale.Position(1,2), "x1", 'FontSize', 16, 'color', 'r')
text(x2_pixel, xscale.Position(2,2), "x2", 'FontSize', 16, 'color', 'r')
x1 = input("Enter the value for the lower x-value (x1): \n");
x2 = input("Enter the value for the upper x-value (x2): \n");

for k = 1:params.nSignals
    % Find all figure handles
    figHandles = findall(0, 'Type', 'figure');
    
    % Sort the figure handles by their figure number
    [~, sortIdx] = sort([figHandles.Number]);
    sortedFigures = figHandles(sortIdx);
    
    % Close all figures except for the first one
    if length(sortedFigures) > 1
        for i = 2:length(sortedFigures)
            close(sortedFigures(i));
        end
    end
    figure(f); 
    % Y-scale
    disp("Draw line on the vertical scale of your signal");
    title("Draw line on the vertical scale of your signal");
    yscale = drawline();
    y1_pixel = yscale.Position(1,2);
    y2_pixel = yscale.Position(2,2);
    
    % add annotations for the points we drew
    text(yscale.Position(1,1), y1_pixel, "y1", 'FontSize', 16, 'color', 'r')
    text(yscale.Position(2,1), y2_pixel, "y2", 'FontSize', 16, 'color', 'r')
    y1 = input("Enter the value for the lower y-value (y1): \n");
    y2 = input("Enter the value for the upper y-value (y2): \n");
    
    % define relationship between pixels and scaled values
    y_scale_factor = (y2-y1)/(y2_pixel -y1_pixel);
    x_scale_factor = (x2-x1)/(x2_pixel - x1_pixel);
    
    % Get the user-defined region of interest (This should exclude an
    % annotations or marks that may interfere with the edge detection.
    disp("Select the region of interest by drawing a rectangle");
    title('Select the region of interest by drawing a rectangle');
    roi = round(getPosition(imrect));
    
    % Extract the region of interest from the image
    roi_image = image(roi(2):roi(2)+roi(4), roi(1):roi(1)+roi(3), :);
    % Results are very sensitive to this region. If it includes annotations or
    % signals from the next row, things won't work. 
    
    % Display the region of interest
    figure;
    imshow(roi_image);
    title('Region of interest');
    drawnow();
    
    % Detect edges within the region of interest
    bw_image = im2gray(roi_image);
    edge_image = edge(bw_image, 'canny');
    
    % Display the edge image
    figure;
    imshow(edge_image);
    title('Edge detected image');
    drawnow();
    
    % % magnify features
    % background_dots_size = 3;
    % dilated_image = imdilate(edge_image, strel('disk', background_dots_size));
    % figure;
    % imshow(dilated_image);
    % title('Dilated image');
    
    % Remove grid points
    % zero pad the dilated image
    disp("padding image")
    padded_image = padarray(edge_image, [params.zero_pad_size, params.zero_pad_size], 0, 'both');
    
    % remove dots
    disp("removing grid points")
    cleaned_image = removeEnclosed(padded_image, 5, 5, 1);
    cleaned_image = removeEnclosed(cleaned_image, 10, 10, 1);
    cleaned_image = removeEnclosed(cleaned_image, 15, 15, 1);
    cleaned_image = removeEnclosed(cleaned_image, 20, 20, 1);
    
    % Remove pad to preserve dimensions
    cleaned_image = cleaned_image(params.zero_pad_size+1:end-params.zero_pad_size, params.zero_pad_size+1:end-params.zero_pad_size);
    
    % try to merge neighboring data points to make our life easier
    disp("removed grid points")
    merged_image = imclose(cleaned_image, strel('disk', params.merge_size));
    figure; imshow(merged_image); title("Merged cleaned image")
    
    % Convert image to signal by taking row numbers where we have ones
    nCols = size(merged_image, 2);
    mins_for_image = zeros(nCols, 1);
    maxs_for_image = zeros(nCols, 1);
    mins_for_scaling = zeros(nCols, 1);
    maxs_for_scaling = zeros(nCols, 1);
    time_vect = ((1:nCols) + roi(1))';
    % means = zeros(nCols, 1);
    % medians = zeros(nCols, 1);
    
    for i = 1:nCols
        % do this to catch any empty vectors
        if isempty(find(merged_image(:, i), 1 ))
            mins_for_image(i) =  NaN;
            mins_for_scaling(i) =  NaN;
        else
            mins_for_image(i) =  find(merged_image(:, i), 1 );
            mins_for_scaling(i) =  find(merged_image(:, i), 1 ) + roi(2);
        end
    
        if isempty(find(merged_image(:, i), 1, 'last'))
            maxs_for_image(i) =  NaN;
            maxs_for_scaling(i) =  NaN;
        else
            maxs_for_image(i) =  find(merged_image(:, i), 1, 'last');
            maxs_for_scaling(i) =  find(merged_image(:, i), 1, 'last') + roi(2);
        end
        % add ROI offset, so the row numbers match original image (need for y-scaling)
        % roi(2) corresponds to the top left point of the ROI. 
    end
    
    figure;
    imshow(merged_image); hold on;  plot(mins_for_image, 'r'); hold on; plot(maxs_for_image, 'g'); title("Cleaned Image with Envelope"); set(gca, 'YDir', 'reverse') 
    % subplot(6,3,7:9); plot(means); title("Mean of column"); set(gca, 'YDir', 'reverse') 
    % subplot(6,3,10:12); plot(medians); title("Median of column"); set(gca, 'YDir', 'reverse') 
    % subplot(6,3,13:15); plot(mins); title("Min of column"); set(gca, 'YDir', 'reverse') 
    % subplot(6,3,16:18); plot(maxs); title("Max of column"); set(gca, 'YDir', 'reverse') 
    
    % convert pixels to signal values
    mins_scaled = (mins_for_scaling-y1_pixel)*y_scale_factor + y1;
    maxs_scaled = (maxs_for_scaling-y1_pixel)*y_scale_factor + y1;
    time_scaled = (time_vect - x1_pixel)*x_scale_factor + x1;
    % means = means.*y_scale_factor;
    % medians = medians.*y_scale_factor;
    
    % Use the envelopes to generate a signal
    signal_for_image = (mins_for_image+maxs_for_image)./2;
    signal_scaled = (mins_scaled+maxs_scaled)./2;
    
    % if the difference between our envelopes is large, use one of the envelope values
    for j = 1:length(signal_for_image)
        if (maxs_for_image(j) - mins_for_image(j)) > params.th
            if signal_for_image(j) < mean(signal_for_image, 'omitnan')
                signal_for_image(j) = mins_for_image(j);
                signal_scaled(j) = mins_scaled(j);
            else
                signal_for_image(j) = maxs_for_image(j);
                signal_scaled(j) = maxs_scaled(j);
            end
        end
    end
    figure; subplot(2,1,1);
    imshow(merged_image); hold on; plot(signal_for_image, 'y'); title("Cleaned image with identified signal")
    subplot(2,1,2); plot(time_scaled, signal_scaled, 'b')
    drawnow();

    % pause here to let user check results
    s = input("look good? ('' = yes, anything else = no): \n", 's');
    if isempty(s) 
        signal_label = input("type name of signal ('ECG'): \n", 's');
        saved_signals.(signal_label) = table(time_scaled, signal_scaled);
    end
end

%% Save each signal
fn = fieldnames(saved_signals);
signal_tables = struct;
for i = 1:length(fn)
    tab = saved_signals.(fn{i});
    writetable(tab, fn{i}+".csv")
end
save("digitized_data.mat", "saved_signals")

%% process signals a bit

% get R
R = readtable('\\chips.eng.utah.edu\sanchez\Cuffless_BP\LVAD\rawData\11\pre_op_catheter\LVAD_11_pre_op_bioz.csv');
t = (R{:, 1}-R{1,1})*1e-6;
msk = t>(t(end)-30);
saved_signals.R = table(t(msk), R{msk, 2});
saved_signals.R.Properties.VariableNames = ["time_scaled", "signal_scaled"];

fn = fieldnames(saved_signals);
processed_signals = struct;
for i = 1:length(fn)
    % movmean
     filtered_signal = movmean(saved_signals.(fn{i}).signal_scaled, params.mean_window);
     processed_signals.(fn{i}) = table(saved_signals.(fn{i}).time_scaled, filtered_signal);
     processed_signals.(fn{i}).Properties.VariableNames = ["time", "signal"];
    %
end

% figure
figure;
for j = 1:length(fn)
    ax(j) = subplot(length(fn)+1, 1, j);
%     plot(saved_signals.(fn{j}).time_scaled, saved_signals.(fn{j}).signal_scaled, 'color', [0.1133, 0.1602, 0.3164])
%     hold on;
    plot(processed_signals.(fn{j}).time, processed_signals.(fn{j}).signal, 'color', [0.5, 0.2, 0.5])
    title(fn{j})
    %legend({'Original', 'MovMean'})
end

% linkaxes
%linkaxes(ax, 'x')
%% synchronize R with Pleth signal

[locs.Pleth] = findPeaks(processed_signals.Pleth.signal, 140, 0.5, 'max', 0, 1);
[locs.R] = findPeaks(processed_signals.R.signal, 50, 1e-3, 'min', 0, 1);
ts.Pleth = processed_signals.Pleth.time;
ts.R = processed_signals.R.time;
[ibi_times, ibi, d_ibi_times, d_ibi] = IBI(ts, locs, 0, 1, 1);

params.low_var_len = 5;
params.method = 'mad';
params.fixed_signal = "R";
params.lam = 0.6;
params.plot_switch = 1;
align_signals.Pleth = processed_signals.Pleth.signal;
align_signals.R = processed_signals.R.signal;
[ts_aligned, shift] = Align(align_signals, ibi_times, ibi, ts, locs, params);
%% FUNCTIONS

% this function will move a rect around the image and remove any points
% that are entirely encolsed by the rect (should get rid of grid lines with
% the appropriate hyperparameters)
function [im_out] = removeEnclosed(im_in, width, height, plot_switch)
    im_out = im_in;
    % Iterate over the image and remove enclosed points
    for x = 1:size(im_out, 2) - width
        for y = 1:size(im_out, 1) - height
            try
               
                % Extract the enclosed region
                enclosed_region = im_out(y:y+height, x:x+width, :);
                
                % Check if any of the edges are equal to 1
                edges_are_one = any(enclosed_region(1,:) | enclosed_region(end,:) | enclosed_region(:,1)' | enclosed_region(:,end)');
        
                if ~edges_are_one
                    if plot_switch
    
                    end
                    % If the circularity is above the threshold, remove the enclosed region
                    im_out(y:y+height, x:x+width, :) = 0;
                end
            catch
                warning("problem with encolsed region on: " + x + ", " + y)
            end
        end
    end
    
    if plot_switch
        figure;
        subplot(2,1,1); imshow(im_in); title("Original")
        subplot(2,1,2); imshow(im_out); title("Filtered")
    end
end

function [ibi_times, ibi, d_ibi_times, d_ibi] = IBI(ts, locs, clip, use_diff, plot_switch)
    % sort structs
    ts = orderfields(ts);
    locs = orderfields(locs);

    % unpack params
    fn_ts = fieldnames(ts); 
    fn_locs = fieldnames(locs);

    % check to make sure we have compatible structs
    if length(fn_locs) ~= length(fn_ts)
        error("time and locs structs have different number of fields...")
    end

     % Find IBI of both signals
    for i = 1:length(fn_locs)
        if fn_locs{i} == fn_ts{i}
            ibi.(fn_locs{i}) = diff(ts.(fn_locs{i})(locs.(fn_locs{i})));
            ibi_times.(fn_locs{i}) = cumsum(ibi.(fn_locs{i}));
            
            % calculate differential of IBI
            if use_diff
                d_ibi.(fn_locs{i}) = sign(diff(ibi.(fn_locs{i})));
                d_ibi_times.(fn_locs{i}) = ibi_times.(fn_locs{i})(2:end);
            end
        else
            error("loc and time fieldnames don't match: "+ i)
        end
    end

    % Option to clip IBI data
    if isstring(clip)
        for i = 1:length(fn_locs)
            if clip == "hrv"
                % Use HRV to clip
                rc = abs(diff(ibi.(fn_locs{i}))./ibi.(fn_locs{i})(1:end-1))*100;
                ibi.(fn_locs{i})(rc > 40) = 0.8;
            
            elseif clip == "th"
                % use absolute HR valuse to clip
                ibi.(fn_locs{i})(ibi.(fn_locs{i}) > 1.2) = 0.8;
                ibi.(fn_locs{i})(ibi.(fn_locs{i}) > 0.4) = 0.6;

            elseif clip == "stats"
                % use stats to remove outlier points
                avg = mean(ibi.(fn_locs{i}));
                SD = std(ibi.(fn_locs{i}));
                ibi.(fn_locs{i})(ibi.(fn_locs{i}) > avg + 2*SD) = 0.8;
                ibi.(fn_locs{i})(ibi.(fn_locs{i}) < avg - 2*SD) = 0.6;
            elseif clip == 0
                 % do nothing
            else
                error('IBI clip option not recognized. Specif hrv, th, stats, or 0')
            end
        end
    end

    % plot
    if plot_switch
        figure; hold on;
        for i = 1:length(fn_locs)
            subplot(2,1,1); hold on
            plot(ibi_times.(fn_locs{i}), ibi.(fn_locs{i}),'marker', '.', 'markersize', 10)
            title("IBI")
            subplot(2,1,2); hold on;
            plot(d_ibi_times.(fn_locs{i}), d_ibi.(fn_locs{i}),'marker', '.', 'markersize', 10) 
            title("dIBI")
        end
        legend(fn_locs)
    end
end

function [loc_extrema] = findPeaks(signal, distance, prominence, extrema_type, smart_prom, plot_switch)
    % Find maxima
    [maxima, maxLocs] = findpeaks(signal, 'MinPeakDistance', distance, 'MinPeakProminence', prominence);

    % Find minima
    [minima, minLocs] = findpeaks(-signal, 'MinPeakDistance', distance, 'MinPeakProminence', prominence);
    minima = -minima;

    % Find deltas, but only if the max/min values are within 100 points of
    % each other
    deltas=[];
    if smart_prom
        for i = 1:numel(maxLocs)
            max_loc = maxLocs(i);
            
            % Find nearest minima within 50 points
            min_loc = find(abs(minLocs - max_loc) <= 50, 1);
            
            % If a nearby minima is found, calculate the difference
            if ~isempty(min_loc)
                delta = signal(max_loc) - signal(min_loc);
                deltas = [deltas, delta];
            end
        end

        prominence = 0.5*median(deltas);

        % Find maxima
        [maxima, maxLocs] = findpeaks(signal, 'MinPeakDistance', distance, 'MinPeakProminence', prominence);
    
        % Find minima
        [minima, minLocs] = findpeaks(-signal, 'MinPeakDistance', distance, 'MinPeakProminence', prominence);
        minima = -minima;
    end

    % Plot the results (optional)
    if plot_switch
        figure;
        plot(signal);
        hold on;
        plot(maxLocs, maxima, 'r*', 'MarkerSize', 10);
        plot(minLocs, minima, 'bo', 'MarkerSize', 10);
        hold off;
        legend('Signal', 'Maxima', 'Minima');
        xlabel('Sample Index');
        ylabel('Amplitude');
    end

    % set the extrema locs
    if extrema_type == "max"
        loc_extrema = maxLocs;
    elseif extrema_type == "min"
        loc_extrema = minLocs;
    else
        error("Didn't recognize extrema type. Specify max or min")
    end
end

function [ts_aligned, shift] = Align(signals, ibi_times, ibi, ts, locs, params)
% Aligns the signal using a sliding operation (MedAD, MAE, or corr)
% parms needs to include: P_ibi_time, P_ibi, R_ibi_time, R_ibi, low_var_len, method, tiemR, timeP, R_peaks, P_peaks plot_switch
% returns aligned time vectors for P and R (same length as signal vectors)    
    
    % sort structs to be sure order of fields is identical
    signals = orderfields(signals);
    ibi_times = orderfields(ibi_times);
    ibi = orderfields(ibi);
    ts = orderfields(ts);
    locs = orderfields(locs);

    % unpack structs
    fn_sig = fieldnames(signals);
    fn_ibi_ts = fieldnames(ibi_times);
    fn_ibi = fieldnames(ibi);
    fn_ts = fieldnames(ts);
    fn_locs = fieldnames(locs);

    % check to make sure our structs are the same legnths
    if ~isequal(numel(fn_sig), numel(fn_ibi_ts), numel(fn_ibi), numel(fn_locs), numel(fn_ts))
        error("@Align function. lengths of struct not equal.")
    end
    
    % Find a clean chunk of our fixed signal
    [startIdx, endIdx] = findLowVarSeg(ibi.(params.fixed_signal), params.low_var_len, params.lam, 1);
    ibi_low_noise = ibi.(params.fixed_signal)(startIdx:endIdx);
    
    % Do some sliding operations to find best shift
    for i = 1:length(fn_sig)
        if fn_sig{i} == params.fixed_signal
            shift.(fn_sig{i}).time_zero = ts.(fn_sig{i})(locs.(fn_sig{i})(startIdx+1));
            shift.len_low_var = params.low_var_len;
            shift.startIdx = startIdx; 
            shift.endIdx = endIdx;
            ibi_times_shifted.(fn_sig{i}) = ibi_times.(fn_sig{i}) - ibi_times.(fn_sig{i})(startIdx);
            ts_aligned.(fn_sig{i}) = ts.(fn_sig{i}) - ts.(fn_sig{i})(locs.(fn_sig{i})(startIdx+1));
            continue
        else
            [c, mad_vect, mse_vect] = slideOp(ibi_low_noise, ibi.(fn_sig{i}), params.plot_switch);
            [~, corr_shift] = max(c);
            [~, mad_shift] = min(mad_vect);
            [~, mse_shift] = min(mse_vect); % represents how many indices our low var R needs to be moved to get aligned
            if min(mad_vect) < min(mse_vect)
                disp("   mad < mse")
            else
                disp("   mse < mad")
            end
            if params.method == "mad"
                shift.(fn_sig{i}).seg = mad_shift;
            elseif params.method == "mse" 
                shift.(fn_sig{i}).seg = mse_shift;
            elseif params.method == "corr"
                shift.(fn_sig{i}).seg = corr_shift;
            else
                error('Specified method not recognized. Use mad, mse, or corr.')
            end

            % Add a check to see if we have a P IBI spike in our signal
            spike_idx = find(ibi.(fn_sig{i})(shift.(fn_sig{i}).seg:shift.(fn_sig{i}).seg+params.low_var_len) > 1.52);
            if ~isempty(spike_idx)
                % check for multiple spikes or a spike with multiple points
                if length(spike_idx) > 1
                    if all(diff(spike_idx)==1) 
                        num_spikes = 1;
                    % If we have more than one spike, do it by hand
                    else
                        warning("@Align. Found two IBI spikes in the low noise matching")
                        ts_aligned = [];
                        shift = [];
                        return
                    end
                elseif length(spike_idx) == 1
                    num_spikes = 1;
                end
                % check if we have better alignment before/after spike
                if num_spikes == 1
                    mae1 = mean(abs(ibi_low_noise(1:spike_idx(1)) - ibi.(fn_sig{i})(shift.(fn_sig{i}).seg:shift.(fn_sig{i}).seg+spike_idx(1)-1)));
                    mae2 = mean(abs(ibi_low_noise(spike_idx(end):end) - ibi.(fn_sig{i})(shift.(fn_sig{i}).seg+spike_idx(end):shift.(fn_sig{i}).seg+spike_idx(end)+length(ibi_low_noise(spike_idx(end):end))-1)));
                    
                    % shift our segment to after spike
                    if mae1 > mae2
                        startIdx = startIdx + spike_idx(end); 
                        shift.(fn_sig{i}).seg = shift.(fn_sig{i}).seg+spike_idx(end);

                        % plot updated value
                        if params.plot_switch
                            figure
                            ibi_low_noise = ibi.(params.fixed_signal)(startIdx:endIdx);
                            plot(ibi.(fn_sig{i}), 'Marker', '.', 'markersize', 10)
                            short_sig = zeros(length(ibi.(fn_sig{i})), 1);
                            short_sig(shift.(fn_sig{i}).seg:shift.(fn_sig{i}).seg+length(ibi_low_noise)-1) = ibi_low_noise;
                            hold on
                            plot(short_sig, 'Marker', '.', 'markersize', 10)
                            title("Adjusted IBI segment after spike.")
                        end
                    
                    % don't need to shift segment!
                    else
                        % pass
                    end
                end
            end

            % set the time of our matching segments to be equal
            ibi_times_shifted.(fn_sig{i}) = ibi_times.(fn_sig{i}) - ibi_times.(fn_sig{i})(shift.(fn_sig{i}).seg);
            
            % carry this timeshift over to the raw signals
            ts_aligned.(fn_sig{i}) = ts.(fn_sig{i}) - ts.(fn_sig{i})(locs.(fn_sig{i})(shift.(fn_sig{i}).seg)+1);
        
            % save the matched time in our shift struct
            shift.(fn_sig{i}).time_zero = ts.(fn_sig{i})(locs.(fn_sig{i})(shift.(fn_sig{i}).seg)+1);
            
        end
    end
    
    if params.plot_switch
        f = figure; hold on;
        for i=1:length(fn_sig)
            plot(ibi_times_shifted.(fn_sig{i}), ibi.(fn_sig{i}), 'Marker', '.', 'markersize', 10)
        end
        legend(fn_sig)
        xline(ibi_times_shifted.(params.fixed_signal)(startIdx))
        xline(ibi_times_shifted.(params.fixed_signal)(endIdx))
        legend([fn_sig', "Start align segment", "End align segment"])
        title("Shifted IBI")
        
        figure;
        for i=1:length(fn_sig)
            ax(i) = subplot(length(fn_sig),1,i);
            plot(ts_aligned.(fn_sig{i}), signals.(fn_sig{i}), 'b-')
            title(fn_sig{i})
        end
        linkaxes(ax, 'x')
        title("Shifted Signals")
        figure(f)
        xlim([ibi_times_shifted.(params.fixed_signal)(startIdx)-40 ibi_times_shifted.(params.fixed_signal)(endIdx)+40])
        
%         figure(8); clf
%         plot(params.P_ibi_time, params.P_ibi, 'b', 'marker', '.')
%         hold on
%         plot(params.R_ibi_time-897.97, params.R_ibi, 'r', 'marker', '.')
%         legend({'P', 'R'})
    end
end

function [c, mad_vect, mae_vect] = slideOp(x, y, plot_switch)
% Performs sliding operations of x onto y (corr, MSE, median absolute
% difference)
% Assumes x is the low noise signal (x should be shorter than y)
    n = length(x);
    c = zeros(length(y)-n, 1);
    mad_vect = zeros(length(y)-n, 1);
    mae_vect = zeros(length(y)-n, 1);
    % slide x along the length of y
    for i = 1:length(y) - n
        v = zeros(length(y), 1);
        v(i:n+i-1) = x;
        c(i) = sum(v.*y);
        mad_vect(i) = median(abs(y(i:n+i-1) - x));
        mae_vect(i) = mean(abs(y(i:n+i-1) - x));

        %clf; plot(y); hold on; plot(v);

    end
    
    % plot results
    if plot_switch
        % plot sliding corr results and move to closest energy of input x
        figure; subplot(4, 1, 1)
        hold on;
        plot(c)
        [~, corr_idx] = min(abs(c - sum(x.^2)));
        scatter(corr_idx, c(corr_idx), 'rx')
        yline(sum(x.^2))
        title('Correlation (searching for matched energy of low var segment)')
        
        % plot median asbolute difference and mark optimal location
        subplot(4, 1, 2)
        plot(mad_vect); hold on;
        [~, mad_idx] = min(mad_vect);
        scatter(mad_idx, mad_vect(mad_idx), 'rx')
        title('Median absolute difference')
        
        % plot MSE and mark optimal location
        subplot(4, 1, 3);  hold on;
        plot(mae_vect)
        [~, mae_idx] = min(mae_vect);
        scatter(mae_idx, mae_vect(mae_idx), 'rx')
        title('Mean square error')

        % plot signals and move according to optimal locations
        subplot(4,1,4)
        plot(y, 'b-', 'marker', '.', 'markersize', 10)
        hold on
        x_x = 1:length(y);
        x_y = zeros(length(y), 1);
        x_y(corr_idx:corr_idx+length(x)-1) = x;
        plot(x_x, x_y, 'r-', 'marker', '.', 'markersize', 10)
        
        x_y = zeros(length(y), 1);
        x_y(mad_idx:mad_idx+length(x)-1) = x;
        plot(x_x, x_y, 'k-', 'marker', '.', 'markersize', 10)

        x_y = zeros(length(y), 1);
        x_y(mae_idx:mae_idx+length(x)-1) = x;
        plot(x_x, x_y, 'o-', 'marker', '.', 'markersize', 10)
        legend('y', 'x_corr', 'x_mad', 'x_mse')

        title('Shifted signals')

    end
end

function [startIdx, endIdx] = findLowVarSeg(signal, windowSize, lam, plot_switch)
    % Find the least noisy segment based on sliding window variance

    % Input:
    % - signal: The input signal
    % - windowSize: Size of the sliding window

    % Output:
    % - startIdx: Start index of the least noisy segment
    % - endIdx: End index of the least noisy segment

    % Ensure the window size is smaller than the signal length
    if windowSize >= numel(signal)
        error('Window size must be smaller than the signal length.');
    end

    % Initialize variables
    numPoints = numel(signal);
    SD = zeros(numPoints - windowSize + 1, 1);
    loc_cost = zeros(numPoints - windowSize + 1, 1);

    % Iterate through the signal with the sliding window
    for i = 1:(numPoints - windowSize + 1)
        window = signal(i:(i + windowSize - 1));
        SD(i) = std(window);
        loc_cost(i) = (abs(i - length(signal)/2)/length(signal)/2);
        
        % Update if the current window has lower variance
%         if cost < minCost
%             minCost = cost;
%             startIdx = i;
%             endIdx = i + windowSize - 1;
%         end
    end

    % scale SD to be between 0 and 1
    SD = (SD - min(SD)) / (max(SD) - min(SD));
    cost = lam*SD + (1-lam)*loc_cost;
    [~, startIdx] = min(cost);
    endIdx = startIdx + windowSize-1;

    if plot_switch
        figure
        plot(signal, 'b-', 'marker', '.', 'markersize', 10)
        hold on
        plot(startIdx:endIdx, signal(startIdx:endIdx), 'r-', 'marker', '.', 'markersize', 10)
        legend('Signal', 'Lowest Var')
    end
end