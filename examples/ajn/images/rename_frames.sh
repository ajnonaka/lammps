#!/bin/bash

# This script renames LAMMPS output frames for proper sorting.
# Example: frame_200.png -> frame_0000200.png

echo "Renaming files in the 'images' directory..."

# Go into the images directory
cd images

# Loop over every file that ends in .png
for f in *.png
do
  # Extract the number from the filename (e.g., "200" from "frame_200.png")
  num=$(echo $f | sed 's/frame_//' | sed 's/.png//')
  
  # Create the new filename with 7-digit zero-padding
  newname=$(printf "frame_%07d.png" $num)
  
  # Rename the file
  mv "$f" "$newname"
done

echo "Done."
