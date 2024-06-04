# Digitize ECG
A tool for digitizing black and white medical scans from the ICU. The MATLAB script is not automated nor is it intended for high throughput. 

# Repository Organization
## source
This folder houses the main script `digitize.m`

## resources
This folder contains a sample de-identified scan from the ICU.

# Instructions
Open the digitize script and update the hyperparameters. The script will have ask the user to define various aspects of the image, including the x-axis scale, y-axis scale, and defining regions of interest around the signals. The script will then try to remove the background grid points and extract the signal by identifying extrema in the image pixels. The method works, but is sensitive to stray pixels. The idea behind this script is to give researchers a tool to manually digitize ICU scans for data extraction or creating images.



