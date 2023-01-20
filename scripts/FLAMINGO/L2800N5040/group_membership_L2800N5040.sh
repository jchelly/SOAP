#!/bin/bash -l
#
# Compute VR group membership for each particle in a snapshot.
# Job name determines which of the L2800N5040 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# sbatch -J HYDRO_FIDUCIAL --array=78 ./group_membership_L2800N5040.sh
#
#SBATCH --nodes=32
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/group_membership_L2800N5040_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 12:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L2800N5040/${SLURM_JOB_NAME}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the output
outbase="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/"

# Generate input and output file names
swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="${outbase}/group_membership/group_membership_${snapnum}/vr_membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Copy virtual file
#cp "${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.hdf5" ${outdir}
#vfile_out=${outdir}/flamingo_${snapnum}.hdf5
#chmod u+w ${vfile_out}

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}

 #--update-virtual-file=${vfile_out}
