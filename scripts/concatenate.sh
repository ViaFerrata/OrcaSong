#!/bin/bash

source activate orcasong

DIR=/data/arca/v6

concatenate --outfile $DIR/ready_for_training/train_nuecc_vs_nutaushower_triggered_small_v6.h5 $DIR/ml_ready_triggered/train/*
concatenate --outfile $DIR/ready_for_training/test_nuecc_vs_nutaushower_triggered_small_v6.h5 $DIR/ml_ready_triggered/test/*
