#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py unreal data/unreal/rawframes_train/ --level 3 --format rawframes --num_split 1 --subset train --shuffle
echo "Train filelist for rawframes generated."

#PYTHONPATH=. python data_tools/build_file_list.py unreal data/unreal/rawframes_val_1016/ --level 3 --format rawframes --num_split 1 --subset val --shuffle
#echo "Val filelist for rawframes generated."
cd data_tools/unreal/
