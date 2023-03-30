#! /bin/bash

# Compress the membership files using h5repack, applying GZIP compression

#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -J L1000N1800_HYDRO_JETS-compression-membership
#SBATCH -o logs/job_compression_membership.%a.dump
#SBATCH -e logs/job_compression_membership.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#SBATCH --array=71-76%8

module load intel_comp/2018
module load hdf5
module load gnu-parallel

# get the snapshot number from the array task ID
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# create the output folder if it does not exist
mkdir -p "SOAP_compressed/membership_${snapnum}"

# uncompressed membership file basename
input_filename="SOAP/membership_${snapnum}/membership_${snapnum}"
# compressed membership file basename
output_filename="SOAP_compressed/membership_${snapnum}/membership_${snapnum}"

# run h5repack in parallel using 32 processes on files 0 to 63
# we could use more processes, but that causes a larger strain for the file
# system and is therefore not really more efficient
# make sure to update the 'seq' arguments when there are more/less membership
# files
seq 0 63 | parallel -j 32 \
  h5repack \
    -i "${input_filename}.{}.hdf5" \
    -o "${output_filename}.{}.hdf5" \
    -l CHUNK=10000 -f GZIP=9
