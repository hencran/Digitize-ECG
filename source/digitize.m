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
image_path = "";
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
