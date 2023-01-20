#!/bin/bash -l
#
# Match halos between 1Gpc DMO and HYDRO runs
#
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH -J match_L1000N3600_DMO_FIDUCIAL
#SBATCH -o ./logs/matching/%x_%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 8:00:00
#SBATCH --mail-user=j.c.helly@durham.ac.uk
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

set -e

box=L1000N3600
sim1=DMO_FIDUCIAL
for sim2 in HYDRO_FIDUCIAL ; do

  # Snapshot to do
  snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

  # Location of the files
  basedir="/cosma8/data/dp004/flamingo/Runs/"
  vr_basename1="${basedir}/${box}/${sim1}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
  vr_basename2="${basedir}/${box}/${sim2}/VR_missing/catalogue_${snapnum}/vr_catalogue_${snapnum}"
  nr_particles=10

  # Where to put the output
  outdir="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${box}/${sim1}/matching/TO_${sim2}/"
  \mkdir -p ${outdir}
  lfs setstripe --stripe-size=32M --stripe-count=-1 ${outdir}
  outfile="${outdir}/match_${box}_${sim1}_${sim2}_${snapnum}.field_only.n${nr_particles}.hdf5"

  echo
  echo Matching $sim1 to $sim2, snapshot ${snapnum}
  echo

  mpirun python3 -u -m mpi4py \
      ./match_vr_halos.py ${vr_basename1} ${vr_basename2} ${nr_particles} ${outfile} --use-types 1 --to-field-halos-only

done

echo All matching done.

