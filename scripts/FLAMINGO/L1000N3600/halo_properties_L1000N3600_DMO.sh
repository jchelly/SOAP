#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Job name determines which of the L1000N3600 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# sbatch -J DMO_FIDUCIAL --array=78 ./halo_properties_L1000N3600_DMO.sh
#
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_props
#SBATCH -o ./logs/halo_properties_L1000N3600_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 4:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which simulation to do
sim="L1000N3600/${SLURM_JOB_NAME}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the final output
outbase="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/"

# Location for temporary chunk output
scratchdir="/snap8/scratch/dp004/jch/SOAP-tmp/${sim}/"

# Generate file names for this snapshot
swift_filename="${basedir}/snapshots/flamingo_%(snap_nr)04d/flamingo_%(snap_nr)04d.%(file_nr)d.hdf5"
extra_filename="${outbase}/group_membership/group_membership_%(snap_nr)04d/vr_membership_%(snap_nr)04d.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_%(snap_nr)04d/vr_catalogue_%(snap_nr)04d"
outfile="${outbase}/halo_properties/halo_properties_%(snap_nr)04d.hdf5"

nr_chunks=32

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

# add the "--dmo" flag to prevent computing/outputting quantities that are
# not relevant for DMO runs!
mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${scratchdir} ${vr_basename} ${outfile} ${SLURM_ARRAY_TASK_ID} \
    --chunks=${nr_chunks} \
    --extra-input=${extra_filename} \
    --dmo
