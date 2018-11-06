#!/bin/bash
#
#PBS -l nodes=1:ppn=4:sl32g,walltime=3:00:00
#PBS -o /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.out -e /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.err
# first non-empty non-comment line ends PBS options

# Submit with 'qsub -t 1-10 submit_h5_data_to_h5_input.sh'
# This script uses the h5_data_to_h5_input.py file in order to convert all 600 (muon/elec/tau) .h5 raw files to .h5 2D/3D projection files (CNN input).
# The total amount of simulated files for each event type in ORCA is 600 -> file 1-600
# The files should be converted in batches of files_per_job=60 files per job

# load env
source activate /home/hpc/capn/mppi033h/.virtualenv/python_3_env/

n=${PBS_ARRAYID}

CodeFolder=/home/woody/capn/mppi033h/Code/OrcaSong
cd ${CodeFolder}

#ParticleType=muon-CC
#ParticleType=elec-CC
ParticleType=elec-NC
#ParticleType=tau-CC
# ----- 3-100GeV------
#FileName=JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016 #muon-CC
#FileName=JTE.KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-CC
#FileName=JTE.KM3Sim.gseagen.elec-NC.3-100GeV-3.4E6-1bin-3.0gspec.ORCA115_9m_2016 #elec-NC
#FileName=JTE.KM3Sim.gseagen.tau-CC.3.4-100GeV-2.0E8-1bin-3.0gspec.ORCA115_9m_2016 #tau-CC
#HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/with_jte_times/3-100GeV/${ParticleType}
# ----- 3-100GeV------
# ----- 1-5GeV------
#FileName=JTE.KM3Sim.gseagen.muon-CC.1-5GeV-9.2E5-1bin-1.0gspec.ORCA115_9m_2016 #muon-CC
#FileName=JTE.KM3Sim.gseagen.elec-CC.1-5GeV-2.7E5-1bin-1.0gspec.ORCA115_9m_2016 #elec-CC
FileName=JTE.KM3Sim.gseagen.elec-NC.1-5GeV-2.2E6-1bin-1.0gspec.ORCA115_9m_2016 #elec-NC
HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/with_jte_times/1-5GeV/${ParticleType}
# ----- 1-5GeV------

# -- denser detector study --
# elec-CC 170 files, muon-CC 400 files.
# 15m: 400 m-CC files, 170 e-CC files. time/file: m-CC 30s, e-CC 1m --> files_per_job_m-CC = 80, files_per_job_e-CC = 36, 5 jobs
# 12m: 400 m-CC files, 170 e-CC files. time/file: m-CC 30s, e-CC 1m10s --> files_per_job_m-CC = 80, files_per_job_e-CC = 36, 5 jobs
# 9m: 400 m-CC files, 170 e-CC files. time/file: m-CC 40s, e-CC - --> files_per_job_m-CC = 80, files_per_job_e-CC = 36, 5 jobs
# 6m: 400 m-CC files, 170 e-CC files. time/file: m-CC 53s, e-CC - --> files_per_job_m-CC = 80, files_per_job_e-CC = 36, 5 jobs
# 4.5m: 400 m-CC files, 170 e-CC files. time/file: m-CC 1m3s, e-CC - --> files_per_job_m-CC = 80, files_per_job_e-CC = 36, 5 jobs
#vert_space=4p5m # 15m fpj m: 80, e: 36 ; 12m fpj m: , e: ;
#FileName=JTE.KM3Sim.selectedEventsInCylinder_gseagen.muon-CC.1-20GeV-2.0E8-1bin-3.0gspec.ORCA115_h23v${vert_space}_2016 #muon-CC
#FileName=JTE.KM3Sim.selectedEventsInCylinder_gseagen.elec-CC.1-20GeV-1.0E6-1bin-3.0gspec.ORCA115_h23v${vert_space}_2016 #elec-CC
#HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/without_mc_time_fix/denser_detector_study/${vert_space}
#HDFFOLDER=/home/vault/capn/mppi033h/4p5m # 4p5m
# -- denser detector study --

files_per_job=60 # total number of files per job, e.g. 10 jobs for 600: 600/10 = 60

# run
no_of_loops=$((${files_per_job}/4)) # divide by 4 cores -> e.g, 15 4-core loops needed for files_per_job=60
file_no_start=$((1+((${n}-1) * ${files_per_job}))) # filenumber of the first file that is being processed by this script (depends on JobArray variable 'n')

for (( k=1; k<=${no_of_loops}; k++ ))
do
    file_no_loop_start=$((${file_no_start}+(k-1)*4))
    thread1=${file_no_loop_start}
    thread2=$((${file_no_loop_start} + 1))
    thread3=$((${file_no_loop_start} + 2))
    thread4=$((${file_no_loop_start} + 3))

    (time taskset -c 0  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread1}.h5 > ./logs/cout/${FileName}.${thread1}.txt) &
    (time taskset -c 1  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread2}.h5 > ./logs/cout/${FileName}.${thread2}.txt) &
    (time taskset -c 2  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread3}.h5 > ./logs/cout/${FileName}.${thread3}.txt) &
    (time taskset -c 3  python ${CodeFolder}/h5_data_to_h5_input.py ${HDFFOLDER}/${FileName}.${thread4}.h5 > ./logs/cout/${FileName}.${thread4}.txt) &
    wait
done

