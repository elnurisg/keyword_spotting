# execute command in /Home/HDA/project/speech_commands_v0.02
# bash ../data_organizer_for_AE.sh 120 train_data/_unknown_ extra_unknown/train 
# bash ../data_organizer_for_AE.sh 16 validation_data/_unknown_ extra_unknown/val

#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <number_to_keep> <source_directory> <destination_directory>"
    exit 1
fi

# Assigning arguments to variables
number_to_keep=$1
source_dir=$2
destination_dir=$3

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist."
    exit 1
fi

# Check if destination directory exists, create if it does not
if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
fi

# Find all unique classes in the filenames
classes=$(find "$source_dir" -type f -name "*.wav" | sed 's/.*_\(.*\)\.wav/\1/' | sort | uniq)

# Loop through each class and move files exceeding the specified number to keep
for class in $classes; do
    # Find all files of the current class
    files=($(find "$source_dir" -type f -name "*_${class}.wav" | awk 'BEGIN{srand()}{print rand(), $0}' | sort -k1,1n | cut -d' ' -f2-))
    total_files=${#files[@]}
    # Calculate the number of files to move
    num_to_move=$(($total_files-$number_to_keep))

    # Check if there are more files than the number to keep
    if [ $num_to_move -gt 0 ]; then
        echo "Moving $num_to_move files of class $class to $destination_dir"

        # Select the last $num_to_move files to move
        for ((i=$total_files-$num_to_move; i<$total_files; i++)); do
            mv "${files[$i]}" "$destination_dir"
        done
    else
        echo "Class $class has $total_files files, no files moved."
    fi
done

echo "Files moved successfully."