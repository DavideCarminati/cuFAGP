#!/usr/bin/env bash

# Remove old dataset folders, if any
rm -rf input_matrices_* &> /dev/null
# Remove old log files, if any
rm log_*.csv &> /dev/null
echo "Generating train datasets..."
cd matlab
{
    # Run the matlab script to generate training dataset
    octave example.m &> /dev/null 
} || 
{
    # If Octave is not installed, try to run the script in Matlab itself
    matlab -batch "example" &> /dev/null
} ||
{
    # If neither Matlab or Octave are installed, notify the user
    echo "Matlab/Octave not found. Exiting..."
    return 2
}
echo "Done"

# Enter in each folder and run both implementations multiple times
cd ..
for d in $(printf '%s\n' input_matrices_*/ | sort -V) ; do
    printf "\n------------------------------------------\nOpening folder $d...\n------------------------------------------\n"
    for sim in {1..20} ; do
        printf "Simulation #%d\n" $sim
        build/gp_cpu $d output_matrices/ ./
        build/gp_gpu $d output_matrices/ ./
        printf "Done\n\n"
    done
done

# Plot the execution times using Octave/Matlab
echo "Plotting the results..."
cd matlab
{
    octave plots.m 
} || 
{
    matlab -batch "plots"
} 