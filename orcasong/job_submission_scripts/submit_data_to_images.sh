#!/bin/bash
#
#PBS -l nodes=1:ppn=4:sl32g,walltime=3:00:00
#PBS -o /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.out -e /home/woody/capn/mppi033h/logs/submit_h5_to_histo_${PBS_JOBID}_${PBS_ARRAYID}.err
# first non-empty non-comment line ends PBS options

# Submit with 'qsub -t 1-10 submit_h5_data_to_h5_input.sh'
# This script uses the h5_data_to_h5_input.py file in order to convert all 600 (muon/elec/tau) .h5 raw files to .h5 2D/3D projection files (CNN input).
# The total amount of simulated files for each event type in ORCA is 600 -> file 1-600
# The files should be converted in batches of files_per_job=60 files per job

#--- USER INPUT ---##

# load env, only working for conda env as of now
python_env_folder=/home/hpc/capn/mppi033h/.virtualenv/python_3_env/
code_folder=/home/woody/capn/mppi033h/Code/OrcaSong/orcasong

detx_filepath=${code_folder}/detx_files/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx
config_file=${code_folder}/config/orca_115l_regression/conf_ORCA_115l_3-100GeV_xyz-t.toml

particle_type=muon-CC
mc_prod=neutr_3-100GeV

files_per_job=60 # total number of files per job, e.g. 10 jobs for 600: 600/10 = 60

#--- USER INPUT ---##

n=${PBS_ARRAYID}
source activate ${python_env_folder}
cd ${code_folder}

declare -A filename_arr
declare -A folder_ip_files_arr

filename_arr=( ["muon-CC"]="JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016"
               ["elec-CC"]="JTE.KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016"
               ["elec-NC"]="JTE.KM3Sim.gseagen.elec-NC.3-100GeV-3.4E6-1bin-3.0gspec.ORCA115_9m_2016"
               ["tau-CC"]="JTE.KM3Sim.gseagen.tau-CC.3.4-100GeV-2.0E8-1bin-3.0gspec.ORCA115_9m_2016"
               ["mupage"]="JTE.ph.ph.mupage.ph.ph.ph.ORCA115_9m_2016")

folder_ip_files_arr=( ["neutr_3-100GeV"]="/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/${particle_type}"
                      ["neutr_1-5GeV"]="/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/${particle_type}"
                      ["mupage"]="/home/saturn/capn/mppi033h/Data/raw_data/mupage")

filename="${filename_arr[${particle_type}]}"
folder="${folder_ip_files_arr[${mc_prod}]}"

#ParticleType=muon-CC
#ParticleType=elec-CC
#ParticleType=elec-NC
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
#FileName=JTE.KM3Sim.gseagen.elec-NC.1-5GeV-2.2E6-1bin-1.0gspec.ORCA115_9m_2016 #elec-NC
#HDFFOLDER=/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/calibrated/with_jte_times/1-5GeV/${ParticleType}
# ----- 1-5GeV------

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

    (time taskset -c 0  python ${code_folder}/data_to_images.py -c ${config_file} ${folder}/${filename}.${thread1}.h5 ${detx_filepath} > ./job_logs/cout/${filename}.${thread1}.txt) &
    (time taskset -c 1  python ${code_folder}/data_to_images.py -c ${config_file} ${folder}/${filename}.${thread2}.h5 ${detx_filepath} > ./job_logs/cout/${filename}.${thread2}.txt) &
    (time taskset -c 2  python ${code_folder}/data_to_images.py -c ${config_file} ${folder}/${filename}.${thread3}.h5 ${detx_filepath} > ./job_logs/cout/${filename}.${thread3}.txt) &
    (time taskset -c 3  python ${code_folder}/data_to_images.py -c ${config_file} ${folder}/${filename}.${thread4}.h5 ${detx_filepath} > ./job_logs/cout/${filename}.${thread4}.txt) &
    wait
done

