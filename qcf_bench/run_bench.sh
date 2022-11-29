#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo "Processes:" $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM -hostfile mpi_host_file python fedml_bench.py --cf config/fedml_hiv.yaml
