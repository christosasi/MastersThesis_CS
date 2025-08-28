#!/bin/bash

# Set working directory
wdir="/Users/gcooper/Downloads/splora_dmt"

# Create necessary directories
mkdir -p "$wdir"/splora/sliding_windows
mkdir -p "$wdir"/splora/outputs

# Get length of input timeseries
funcvol="$wdir"/pb3_dmt002_dmt.nii.gz
length=$(3dinfo -nv "$funcvol")
echo "Length of time series: $length"

# Define subject(s)
subjects="dmt002"

# Define sliding window parameters
window_size=25  # Length of each window
step_size=5     # Step size for overlapping windows

# Calculate number of windows
num_windows=$(( (length - window_size) / step_size + 1 ))
echo "Number of windows: $num_windows"

for subject in $subjects; do
    mask="$wdir"/binary_mask.nii.gz
    outdir="$wdir"/splora/sliding_windows

    for (( w=0; w<num_windows; w++ )); do 
        start=$(( w * step_size ))
        end=$(( start + window_size ))
        
        # Ensure end does not exceed length
        if [ $end -ge $length ]; then
            end=$(( length - 1 ))
        fi



        echo "Processing window $w: time points $start to $end"

        # Extract the windowed time series
        windowed_funcvol="$outdir/${subject}_window${w}.nii.gz"
        3dTcat -prefix "$windowed_funcvol" "$funcvol"[${start}..${end}]

        # Run splora on the windowed data
        splora -i "$windowed_funcvol" \
               -m "$mask" \
               -tr 1.5 \
               -te 0.0352 \
               -jobs 4 \
               -o "$wdir"/splora/outputs/"splora_window${w}"
    done
done
