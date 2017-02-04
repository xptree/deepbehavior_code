#!/bin/bash

./main -threads 1 -cbow 0 -size 120 -window 5 -hs 1 -negative 0 -iter 15 -timestep 20 -train aminer.txt -output_entity aminer_entity.bin -output_action aminer_action.bin -output_entity_time aminer_entity_time.bin -output_action_time aminer_action_time.bin
