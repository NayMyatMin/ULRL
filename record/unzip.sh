for z in *.zip; do
  dir=$(basename "$z" .zip) # Remove the .zip extension to get the directory name
  mkdir "$dir"               # Create a directory with the name of the file
  unzip "$z" -d "$dir"       # Unzip the file into the created directory
done
