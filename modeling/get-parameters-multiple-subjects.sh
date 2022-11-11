#!/bin/bash

for batch_i in {1..13} # 13 is the number of batches
    do 
    for participant_i in {0..20} # 20 is the number of individuals per batch
        do
            nohup python get-parameters.py "$(((20 * ($batch_i-1)) + $participant_i))" | tee -a output.txt &
        done
        sleep 10
    done