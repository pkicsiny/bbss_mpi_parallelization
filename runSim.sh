#!/bin/bash

################################################################################
##CHANGE ME ACCORDING TO YOUR NEEDS
################################################################################
NTHREADS=10

#assign the command line arguments
scan_params=$1
JOBID=$2

# specify output dir and create it if it doesnt exist
OUTDIR=/eos/home-p/pkicsiny/xsuite_simulations/mpi_parallelization
mkdir -p ${OUTDIR}

#some sanity checks on the command line arguments
if test -z "$JOBID" 
then
   echo "No jobId number specified - default set to 0"
   JOBID=0
fi


################################################################################
#PRINT THE ARGUMENTS SUMMARY
################################################################################

echo '###########################################################################'
echo 'Configuration: '
echo 'scan_params: '${scan_params}
echo 'OutputDirectory: ' ${OUTDIR}
echo 'Number of Threads: ' ${NTHREADS}
echo '###########################################################################'

################################################################################
##RUN THE ACTUAL SIMULATION
################################################################################

echo "Activating miniconda python libraries"
command="source /afs/cern.ch/user/p/pkicsiny/miniconda3/bin/activate base"
echo 'Command: ' ${command}
${command}

python --version
which python
echo "Executing simulation"
command="python ./mpi_parallelization.py \
            --scan_params ${scan_params} \
            --nThreads ${NTHREADS} \
            --jobId ${JOBID} \
            --outputDirectory ${OUTDIR}"
echo 'Command: ' ${command}
${command}

echo "Done"
