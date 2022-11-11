#!/bin/bash

for batch_i in {1..25} # 25 is the number of batches
    do 
    for participant_i in {0..10} # 10 is the number of individuals per batch
        do
            nohup python optimize-single-subject.py "$(((10 * ($batch_i-1)) + $participant_i))" | tee -a output.txt &
        done
        sleep 60
    done