#! /bin/bash

# Compress SOAP catalogues

#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -J L2800N5040_HYDRO_FIDUCIAL-compression
#SBATCH -o logs/job_compression.%a.dump
#SBATCH -e logs/job_compression.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 72:00:00
#SBATCH --array=71-76%8

# location of the SOAP repository
soap=/snap8/scratch/dp004/dc-vand2/FLAMINGO/SOAP

# compression script
script="${soap}/compression/compress_fast_metadata.py"

# output directory
outdir=SOAP_compressed

# get the snapshot index from the array task ID
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# name of the input SOAP catalogue
input_filename="SOAP/halo_properties_${snapnum}.hdf5"
# name of the output SOAP catalogue
output_filename="${outdir}/halo_properties_${snapnum}.hdf5"
# directory used to store temporary files (preferably a /snap8 directory for
# faster writing and reading)
scratch_dir="SOAP_compressed"

# run the script using all available threads on the node
python3 ${script} --nproc 128 ${input_filename} ${output_filename} ${scratch_dir}
