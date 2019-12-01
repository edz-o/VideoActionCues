#! /usr/bin/bash env

cd ../../
#PYTHONPATH=. python3 data_tools/build_file_list.py nturgbd data/nturgbd/rawframes/ --level 1 --format rawframes --num_split 2 --subset train --shuffle
#echo "Train filelist for rawframes generated."

PYTHONPATH=. python3 data_tools/build_file_list.py nturgbd data/nturgbd/denseflow/ --level 1 --format denseflow --num_split 3 --subset val --shuffle
echo "Val filelist for denseflow generated."
cd data_tools/nturgbd/
